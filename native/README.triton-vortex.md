# GAC 接入 Triton / Vortex

这条路径让 **C++ MLIR 从 Linalg 中选择符合条件的逐元素子图、静态 FP32 rank-2 / batched MatMul、rank-1 FP32 ReduceSum、最后一轴 row-wise FP32 Softmax 和完整 FP32 LayerNorm，交给 Triton/Vortex 编译**。其他子图继续使用 GAC 原生 Vortex Lowering。两种内核链接成同一个 `kernel.vxbin`，由 GAC Runtime 依据同一张 Dispatch DAG 执行，共享静态内存规划。

**已验证：** 2026-10-22，WSL SimX 上最新完整 GAC 集成检查 **118/118 通过**。新增 batched MatMul 覆盖 matched-prefix、shared-RHS、rank-4 batch 前缀、extent-1 前缀、M/N tail，以及 K48→BK64 的双输入零填充。代表性最大绝对误差为：matched/shared `1.430511474609375e-6`，rank-4 matched `7.152557373046875e-7`，rank-4 shared constant-RHS K48 `9.5367431640625e-7`。17 类非法 MatMul Manifest 被 Python driver 与 C++ Runtime 独立拒绝；K15、rank-1 MatMul、singleton broadcast、shared-LHS、别名和 shape-changing View 都完整回退 native。完整回归保留此前 row-wise Softmax 与 LayerNorm 检查。这是正确性和 ABI 验证，不是性能结果。

**Batched MatMul 当前状态：** rank-2 与静态 FP32 batched MatMul 已贯通 ONNX → Linalg → 标准 `tt.dot` → Triton/Vortex C++ Lowering → VOLT → SimX。batched A 必须携带与输出完全相同的 batch 前缀，B 可携带相同前缀或作为 rank-2 shared RHS；每个 Program 只含一个 rank-2 `tt.dot`。`program_id(0)` 严格分解 batch/M tile/N tile；A/C 始终使用经过验证的 batch base，matched-prefix B 使用 batch base，shared RHS B 固定从同一矩阵起始地址读取。非二次幂 K 使用 `BK=nextPowerOfTwo(K)`；A/B 的补齐 lane 在进入 `tt.dot` 前同时 mask 成零。

**Softmax 当前状态：** ONNX → Linalg → 标准 TTIR → Triton/Vortex C++ Lowering → VOLT → SimX 已实际贯通。静态 FP32 `rank >= 1` 张量的最后一轴由 row-wise kernel 处理；一个 `program_id(0)` 对应一行，地址为 `program_id(0) * N + offset`，grid 等于前缀维度乘积。独立后端专项已覆盖 legacy rank-1 与 row-wise exact graph；每个 kernel 只生成一对匹配的 `vx_split`/`vx_join`。这仍是 CTA leader 串行正确性模板，不是性能版本。width 1025 会正确选择并编译 native fallback，但该标量 fallback 在小型 SimX 配置下耗时很长，因此完整集成把它记录为 compile/selection-only。

**LayerNorm 当前状态：** ONNX opset 17+ 的单输出 `LayerNormalization` 会先完整合法化成五个内部 Linalg operation 和一个最终 root。只有整个闭合子图同时满足合同，才原子地生成一个 TTIR LayerNorm kernel；否则统计链完整保留为 3 个 native Vortex dispatch，内部 mean/reduction 不会被其他 Triton selector 半融合。TTIR 使用两次真实 add reduction，平方前把无效 tail lane 的 centered 值置零，再完成 `sqrt`、Scale 和 Bias。ABI 为 `X/Scale/Bias/Y/N/epsilon`，Runtime 按位传递 FP32 epsilon。

- [逐元素首版结果、中间 IR 和 Buffer 示例](../docs/validation/triton-vortex/2026-10-22/README.md)
- [`tt.dot` 的 plan、原始/解析后 TTIR、LLVM IR、Runtime 报告和拒绝日志](../docs/validation/triton-vortex/2026-10-22-dot/README.md)
- [batched MatMul、shared RHS、K48 padding、回退和 118 项最终证据](../docs/validation/triton-vortex/2026-10-22-batched-matmul/README.md)
- [`tt.reduce` 的 Linalg、TTIR、LLVM IR、Runtime 报告、回退和拒绝证据](../docs/validation/triton-vortex/2026-10-22-reduction/README.md)
- [Softmax rank-1 首版的 48 项历史验证](../docs/validation/triton-vortex/2026-10-22-softmax/README.md)
- [row-wise Softmax 的多维 shape、TTIR、Runtime、回退与拒绝证据](../docs/validation/triton-vortex/2026-10-22-rowwise-softmax/README.md)
- [含 33 项 LayerNorm 覆盖的 81 项完整集成、原子回退、TTIR、Runtime 与拒绝证据](../docs/validation/triton-vortex/2026-10-22-layernorm/README.md)

```text
ONNX 模型
  │ C++ OnnxImporter
  ▼
GAC MLIR → 图优化 / 合法化 → Linalg / 专用算子
  │ C++ 判断子图支持范围、融合、规划 Buffer
  ├─ 合法的逐元素 / MatMul / ReduceSum / Softmax / LayerNorm → 标准 TTIR → Triton/Vortex C++ Lowering → LLVM IR
  └─ 其他子图 ───────────→ GAC 原生 Vortex Lowering ───────→ LLVM IR
                                                            │
                                              VOLT llvm-link / clang / lld
                                                            ▼
                                               一个 kernel.vxbin + plan.json
                                                            │
                                             GAC Runtime → Vortex / SimX
```

**Python 构建脚本只调用编译器、保存中间文件和链接产物，不实现算子运算。** 已选择 Triton 的内核一旦编译失败，构建直接失败；不会悄悄换到 CPU、CUDA 或另一个实现。

## 当前边界

- Triton 逐元素路径要求输入/输出为**同形状、同类型的静态 FP32 或 i32 张量**，各维为正、无自定义编码，Linalg 所有迭代轴为 `parallel`、索引映射为恒等映射。逻辑张量可展平为一维 TTIR；相邻逐元素操作可在同一个内核里执行。
- 标量计算 Region 按白名单检查：支持浮点/整数加减乘、浮点除法/取负、ReLU 所需的 min/max，以及比较、位运算和 `select`；内部比较结果允许 i1。**不会仅凭 ONNX 算子名字就认定整个子图可用。** 精确白名单和融合生产者的递归检查在 [`TritonBridge.cpp`](lib/TritonBridge.cpp) 的 `arithmetic()` / `checkCandidate()`。
- MatMul 选择静态、无编码、行主序 FP32 张量。rank-2 使用 `A[M,K] × B[K,N] → C[M,N]`；batched 形式要求 `C` 的 rank 至少为 3，`A` 与 `C` 具有完全相同的 batch 前缀，`B` 可具有相同前缀，也可使用一个 rank-2 RHS 供所有 batch 共享。不支持 shared-LHS、逐轴 singleton broadcast、转置、Bias 或 epilogue。`BM=BN=16`，逻辑 `K` 满足 `16≤K≤65536` 且为 16 的倍数；物理 `BK=nextPowerOfTwo(K)`，补齐 lane 在 A/B load 前同时被 mask 为零。每个 Program 计算一个 16×16 输出块，`program_id(0)` 被严格分解为 batch、M tile 与 N tile；M/N/K 尾部均受掩码保护。TTIR 中只有一个标准 `tt.dot`，参数 ABI 严格为 `A、B、C` 三个 64 位指针槽。
- ReduceSum 首版只选择静态 rank-1 FP32 输入的完整 axis-0 求和，输出必须是 rank-0 标量或 shape `[1]`。Linalg Region 必须是 FP32 零初始化的规范加法归约；width 上限为 1024。TTIR 使用一个真实 `tt.reduce` Region，非二次幂宽度由掩码和显式零填充保护，参数 ABI 严格为输入、输出两个 64 位指针槽。
- Softmax 选择静态、无编码 FP32 `rank >= 1` 张量的最后一轴，输入和输出 shape 必须完全相同，width 范围为 1～1024。`rows=product(shape[:-1])` 且 `rows*width<=INT32_MAX`；shape-changing dense View 保守回退。`tile=max(16,nextPowerOfTwo(width))`，每个 `program_id(0)` 处理一行，尾部 masked load 使用 `-Inf`，随后执行 `maxnumf reduce → exp(x-max) → addf reduce → 除法 → masked store`。TTIR 保留真实第三参数 `N:i32`；Runtime 在输入、输出两个 64 位地址槽之后追加一个 8 字节参数槽，其低 32 位为 `N`、高 32 位为零。
- LayerNorm 首版只选择静态 FP32、`rank >= 2`、最后一轴、width 1～1024、Scale+Bias、`stash_type=1`、单输出的 ONNX LayerNormalization。epsilon 必须是有限正 FP32；X/Scale/Bias/Y 不得物理别名；dense View 根的物理 shape 必须与逻辑参数 shape 相同；`rows*width` 必须适合有符号 32 位索引。匹配器验证完整五节点内部统计链和所有 use，TTIR 使用两个真实 add reduction，ABI 为 `vortex-u64-pointers-i32-n-f32-epsilon`。
- 不符合上述严格合同的 MatMul、Bias/ReLU epilogue、整数除法、一般广播、一般 pointwise `math.*`、数据类型转换和不规则算子保留在原生 Vortex 路径；是否最终能编译仍取决于原生 Lowering 的支持范围。Softmax 合同内部的 `math.exp`、LayerNorm 合同内部的 `sqrt` 是各自独立、严格验证的例外。未选择 Triton 的原因保存在每个 Dispatch 的 `triton_exclusion` 中。
- 插件依赖独立的 Triton 仓库，**没有复制进 GAC**；GAC 调用其已安装的 `triton_vortex` backend。
- 本次接入不表示完整 BEVFormer 已经走 Triton，也不表示任意轴 Softmax、任意形状 LayerNorm、一般多轴归约、LMEM、DXA 或 TCU 已接入 Triton。MatMul、ReduceSum、Softmax 和 LayerNorm 首版是 FPU 正确性基线；SimX 正确性验证不能用来证明实际硬件的推理速度提升。
- 当前使用 TTIR 和已实现的 Vortex 线程映射；尚未接入 TritonGPU / TTGIR 的布局优化和 TCU 加速。

## 依赖

在 Linux 或 WSL 中使用以下组件：

1. 编译了此分支新功能的 `gac-compile` 与 `gac-vortex-run`。
2. VOLT LLVM / MLIR **20.1.8**，包含 `mlir-translate`、`llvm-link`、`clang`、`ld.lld`、`llvm-objdump`。
3. 已配置的 RV32 Vortex SDK / Runtime。
4. 已安装的 Triton/Vortex 插件及其 `triton-vortex-opt`。插件当前执行验证基于 Triton **3.8.0**；本地 Triton 3.9 源码没有因此被全量构建验证。

插件安装、原生工具构建和支持范围见其仓库的 `third_party/vortex/README.md`。`--triton-backend-root` 只用于定位链接 Makefile，不能替代 Python 插件安装。

## 构建 GAC 编译器与 Runtime

先激活安装了 Triton/Vortex、ONNX 的 Python 环境。宿主机还需要 CMake、Ninja、C++ 编译器、`protoc` 和 Protobuf C++ 开发库。以下命令在 **GAC 仓库根目录**执行；`/path/to/...` 替换成实际位置：

```bash
export GAC_NATIVE_BUILD=/path/to/gac-native-build
export VOLT_ROOT=/path/to/volt-llvm20/build-release
export VORTEX_BUILD=/path/to/vortex-rv32-build
export VORTEX_SOURCE=/path/to/vortex-source
export VORTEX_HOME="$VORTEX_SOURCE"
export ONNX_PROTO_DIR="$(python -c 'from pathlib import Path; import onnx; print(Path(onnx.__file__).parent)')"

cmake -S native -B "$GAC_NATIVE_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR="$VOLT_ROOT/lib/cmake/llvm" \
  -DMLIR_DIR="$VOLT_ROOT/lib/cmake/mlir" \
  -DONNX_PROTO_DIR="$ONNX_PROTO_DIR" \
  -DVORTEX_SOURCE="$VORTEX_SOURCE" \
  -DVORTEX_BUILD="$VORTEX_BUILD"

cmake --build "$GAC_NATIVE_BUILD" \
  --target gac-compile gac-opt gac-vortex-run \
           gac-triton-layernorm-selector-contract \
           gac-triton-softmax-selector-contract \
           gac-triton-batched-matmul-selector-contract -j4
```

`ONNX_PROTO_DIR` 必须包含官方 `onnx.proto`，也可以手动指向 ONNX 源码中的对应目录。Vortex 的 `libvortex.so`、生成的硬件头文件和 RV32 Kernel 库必须已经在 `VORTEX_BUILD` 中构建好；本命令只构建 GAC 的编译器与主机 Runtime。

## 编译一个模型

保持上面的环境变量，在同一 Python 环境里执行：

```bash
# 插件无法自动定位原生工具时再设置：
export TRITON_VORTEX_OPT=/path/to/triton/third_party/vortex/build/triton-vortex-opt

python native/tools/compile-triton-vortex.py model.onnx \
  --output build/model-triton-vortex \
  --block 128 --num-warps 1 --lanes 4
```

`--compiler` 可显式指定 `gac-compile`，覆盖 `GAC_NATIVE_BUILD`。编译 BEVFormer-base 大模型时可选择 `--constant-fold-profile bevformer-base`，这只是常量折叠资源配置，并不代表整网验证完成。

输出目录必须不存在，防止旧权重、中间 IR 或二进制混入本次产物。失败时保留中间文件和 `compile-report.json` 供排查，重试应使用新的目录。

| 参数 | 含义 |
|---|---|
| `--block 128` | 逐元素 Kernel 的一个 Triton Program 处理 128 个逻辑元素，范围为 16～1024 的二次幂；MatMul 固定使用 16×16×`nextPowerOfTwo(K)` 的物理 tile |
| `--num-warps 1` | 一个 CTA 使用的 Warp 数，当前为 1、2 或 4 |
| `--lanes 4` | 目标硬件每个 Warp 的物理 Lane 数，必须与实际配置一致，当前支持不超过 32 的二次幂 |

例如 `block=128, num_warps=1, lanes=4`，一个 CTA 有 4 个物理线程，每个线程在循环里处理多个元素。`grid=ceil(元素数/128)`；尾部不足 128 的元素由掩码保护。

## 输出文件与接口

```text
build/model-triton-vortex/
├── plan.json                    # Dispatch DAG、Buffer、常量和各内核的启动配置
├── tensor_<id>.bin              # GAC 导出的权重/常量，以 plan.json 为准
├── 01-*.mlir / 02-*.mlir ...   # GAC 各阶段中间表示
├── native.mlir / native.ll      # 原生 Vortex 部分
├── triton/
│   ├── gac_op_N.ttir            # C++ 生成的 TTIR
│   ├── gac_op_N.triton.ttir     # Triton 解析后的 TTIR
│   ├── gac_op_N.vxmlir          # 插件生成的 LLVM Dialect
│   └── gac_op_N.ll              # VOLT LLVM IR
├── kernel.ll                    # 两条路径共同链接的 LLVM IR
├── kernel.elf / kernel.vxbin    # 最终设备程序
├── kernel.dump                 # 反汇编
└── compile-report.json         # 编译状态、工具/模型校验值及各阶段耗时
```

构建会使用该目录下独立的 `triton-cache/`。统计是**主机端构建耗时**，包括插件为单个内核生成临时二进制的耗时；不是推理延时，也不是设备周期。

`plan.json` 中的 Triton Dispatch 包含：

```json
{
  "lowering": "linalg-to-triton",
  "triton": {
    "kind": "pointwise",
    "ttir_file": "gac_op_0.ttir",
    "block_elements": 128,
    "num_warps": 1,
    "lanes": 4,
    "block_size": 4,
    "grid_size": 8,
    "abi": "vortex-u64-pointers"
  }
}
```

这里 `grid_size=8` 只是示例，实际由元素数推导。普通内核的输入/输出设备地址按 64 位槽打包，再由设备入口转换成 RV32 指针；Softmax 另有一个携带 `N:i32` 的 8 字节槽；LayerNorm 再追加一个携带 epsilon 原始 FP32 位模式的槽。数学计算仍在设备内核中完成。原生路径和 Triton 路径共用 GAC 的 Buffer 生命周期与依赖检查。

MatMul 使用独立且可校验的合同：

```json
{
  "lowering": "linalg-to-triton",
  "inputs": ["A", "B"],
  "output": "C",
  "triton": {
    "kind": "matmul",
    "ttir_file": "gac_op_0.ttir",
    "m": 17,
    "n": 19,
    "k": 32,
    "batches": 2,
    "batch_mode": "matched-prefix",
    "a_shape": [2, 17, 32],
    "b_shape": [2, 32, 19],
    "c_shape": [2, 17, 19],
    "block_m": 16,
    "block_n": 16,
    "block_k": 32,
    "num_warps": 1,
    "lanes": 4,
    "block_size": 4,
    "grid_size": 8,
    "abi": "vortex-u64-pointers"
  }
}
```

构建驱动和 Runtime 都会从物理张量分别重算 batch 前缀、元素数、字节数、网格与 Tile，并核对 `batch_mode`、三个逻辑 shape、参数顺序和 Buffer 不别名。为兼容首版，旧 rank-2 Manifest 可以同时省略 `batches`、`batch_mode`、`a_shape`、`b_shape`、`c_shape`；batched 形式和只缺部分字段的计划会被拒绝。篡改 `kind`、M/N/K、batch、shape、Tile 或 grid 会在设备初始化前失败。

ReduceSum 使用独立且可校验的合同：

```json
{
  "lowering": "linalg-to-triton",
  "inputs": ["A"],
  "output": "Y",
  "triton": {
    "kind": "reduction",
    "ttir_file": "gac_op_0.ttir",
    "width": 37,
    "tile": 64,
    "num_warps": 1,
    "lanes": 4,
    "block_size": 4,
    "grid_size": 1,
    "abi": "vortex-u64-pointers"
  }
}
```

`tile` 是不小于 `width` 的最小二次幂，并且至少为 16。构建驱动和 Runtime 都会核对输入必须为 `FP32[width]`、输出元素数必须为 1、`grid_size` 必须为 1；篡改 width、tile 或 grid 会在设备初始化前失败。

Softmax 使用独立 ABI，避免把动态 `N` 伪装成编译期常量：

```json
{
  "lowering": "linalg-to-triton",
  "inputs": ["A"],
  "output": "Y",
  "triton": {
    "kind": "softmax",
    "ttir_file": "gac_op_0.ttir",
    "rows": 6,
    "width": 19,
    "n": 19,
    "tile": 32,
    "num_warps": 1,
    "lanes": 4,
    "block_size": 4,
    "grid_size": 6,
    "abi": "vortex-u64-pointers-i32-n"
  }
}
```

构建驱动和 Runtime 会各自从输入 shape 重新计算 `rows=product(shape[:-1])` 与 `width=shape[-1]`，核对 `N=width`、`grid_size=rows`、`rows*width<=INT32_MAX`、输入/输出为不同且完全同 shape 的张量，并重新计算 tile。篡改 ABI、rows、N、width、tile、grid、别名或张量形状会在设备初始化前失败。

LayerNorm 也使用独立、可逐层复核的合同：

```json
{
  "lowering": "linalg-to-triton",
  "inputs": ["X", "Scale", "Bias"],
  "output": "Y",
  "triton": {
    "kind": "layer_norm",
    "ttir_file": "gac_op_0.ttir",
    "rows": 6,
    "width": 19,
    "n": 19,
    "tile": 32,
    "epsilon_f32_bits": 925353388,
    "num_warps": 1,
    "lanes": 4,
    "block_size": 4,
    "grid_size": 6,
    "abi": "vortex-u64-pointers-i32-n-f32-epsilon"
  }
}
```

构建驱动和 Runtime 分别重算 rows、width、N、tile 与 grid，验证 X/Y 为相同 FP32 shape、Scale/Bias 为 `[width]`，四个 Buffer 互不别名，并按位复核有限正 epsilon。`epsilon_f32_bits=925353388` 即 FP32 `1e-5` 的 `0x3727c5ac`。任何检查失败都在打开设备之前终止。

## 执行与验证

按 `plan.json` 中输入张量的顺序准备 `input_<tensor-index>.bin`，类型、元素数量必须与计划一致。输入文件和编译产生的常量文件放在同一数据目录，然后执行：

```bash
export VORTEX_DRIVER=simx
export LD_LIBRARY_PATH="$VORTEX_BUILD/sw/runtime:${LD_LIBRARY_PATH:-}"

"$GAC_NATIVE_BUILD/gac-vortex-run" \
  build/model-triton-vortex/plan.json \
  build/model-triton-vortex/kernel.vxbin \
  build/model-triton-vortex \
  build/model-triton-vortex-results --trace-dispatches
```

集成检查脚本生成小型 ONNX 模型、构建共同设备模块、在 SimX 上执行，并用 ONNX ReferenceEvaluator 独立核对结果：

```bash
python native/tests/run_triton_vortex_tests.py \
  --compiler "$GAC_NATIVE_BUILD/gac-compile" \
  --runtime "$GAC_NATIVE_BUILD/gac-vortex-run" \
  --output build/triton-vortex-validation
```

覆盖方向包括逐元素融合、逐元素和 MatMul 尾部掩码、张量 View、原生与 Triton 的混合执行、K 不满足约束时的明确回退、非二次幂 ReduceSum、rank-2/超宽 reduction 回退，以及无效 pointwise/MatMul/reduction 启动信息的拒绝。MatMul 正例覆盖 rank-2、matched-prefix、shared-RHS、rank-4、extent-1 前缀和 K48 constant RHS；K48 TTIR 会精确检查 range64、双输入 signed K mask 与零填充，K16/K32 则验证没有多余 K mask。K15、rank-1 LHS/RHS、singleton broadcast、shared-LHS、别名和 shape-changing View 均完整保留为单个 native dispatch。17 项 MatMul Manifest 变异覆盖 batch/grid/shape/alias 与 K48 的错误 `block_k`、逻辑 K/shape 不一致。

Softmax 包含旧 rank-1、rank-2/rank-3、width 1/19/256/1024、负 last-axis、极大有限值、NaN、`+Inf`、全 `-Inf`、部分 `-Inf`，以及 axis、超宽、shape-changing View 的 native 回退和同图 sibling Triton dispatch。

LayerNorm 增加 BEV width 256、tail 19、width 1、width 1024，rank/axis/Scale/Bias/epsilon/View/alias 回退，完整 DeviceLowering internal-fanout 合同，以及同图 sibling pointwise 仍可选择 Triton。15 项篡改测试覆盖 grid、rows、`rows*width` 溢出、width、N、tile、epsilon、ABI、重复输入、输出别名和张量 shape。LayerNorm 阶段的 **81/81** 与加入 row-wise Softmax 后的 **92/92** 是冻结的历史检查点；加入 batched MatMul 后的最新完整结果为 **118/118**。逐项证据见 [LayerNorm 验证记录](../docs/validation/triton-vortex/2026-10-22-layernorm/README.md)、[row-wise Softmax 验证记录](../docs/validation/triton-vortex/2026-10-22-rowwise-softmax/README.md) 和 [batched MatMul 验证记录](../docs/validation/triton-vortex/2026-10-22-batched-matmul/README.md)。

## 主要代码入口

| 工作 | GAC 源码 |
|---|---|
| ONNX 解析并建立原生 MLIR | `lib/OnnxImporter.cpp` |
| 图优化 / 算子融合 | `lib/GraphPasses.cpp`、`lib/fusion/` |
| 规则算子转 Linalg | `lib/LinalgLowering.cpp` |
| 检查逐元素子图并创建 TTIR Operation | [`lib/TritonBridge.cpp`](lib/TritonBridge.cpp)：`selectTritonPointwise()` / `emitTritonPointwise()` |
| 检查 MatMul 并创建标准 `tt.dot` | [`lib/TritonBridge.cpp`](lib/TritonBridge.cpp)：`selectTritonDot()` / `emitTritonDot()` |
| 检查 ReduceSum 并创建标准 `tt.reduce` | [`lib/TritonBridge.cpp`](lib/TritonBridge.cpp)：`selectTritonReduction()` / `emitTritonReduction()` |
| 检查 Softmax 并创建 `max/exp/sum/div` 标准 TTIR | [`lib/TritonBridge.cpp`](lib/TritonBridge.cpp)：`selectTritonSoftmax()` / `emitTritonSoftmax()` |
| 原子检查完整 LayerNorm 子图并创建双 reduction TTIR | [`lib/TritonBridge.cpp`](lib/TritonBridge.cpp)：`selectTritonLayerNorm()` / `emitTritonLayerNorm()` |
| Dispatch 选择、静态 Buffer 与 Manifest | `lib/DeviceLowering.cpp` |
| 构建编排、Triton API、LLVM 链接 | `tools/compile-triton-vortex.py` |
| Manifest 校验和设备提交 | `runtime/dispatch-plan.h`、`runtime/vortex-run.cpp` |

TTIR 到 Vortex 机器码的专用后端源码位于独立 Triton 仓库的 `third_party/vortex/native/` 和 `third_party/vortex/python/triton_vortex/`。

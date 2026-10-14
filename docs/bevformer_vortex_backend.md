# BEVFormer FFN 到 Vortex 3.0 后端

## 当前状态

GAC 已实现一条可审阅、可构建的 BEVFormer FFN 垂直切片。下面先列出保留的
`cpp-template` compatibility 路径：

```text
GAC JSON / BEVFormer FFN builder
                │
                ▼
        Canonical Graph
                │  shape、dtype、常量和拓扑合同
                ▼
标准 MLIR：func + tensor + linalg + arith + math
                │
                ▼
第一次 mlir-opt --verify-each --canonicalize --cse
                │
                ▼
 optimized.mlir（Tensor/Linalg）
                │  严格解析 FFN op/type/shape pattern
                ▼
确定性 lowering：标准 func.call Runtime ABI
                │
                ▼
第二次 mlir-opt --verify-each --canonicalize --cse
                │
                ▼
 vortex.mlir（func + arith + 静态 memref）
                │  严格解析调用、类型、launch 和权重绑定
                ▼
固定 Vortex C++ 模板：3 个 __kernel + vortex2 host
                │
                ▼
VOLT clang++ → RISC-V/Vortex ELF → vxbin.py → kernel.vxbin
                │
                ▼
      vortex2 → SimX / RTL simulator / FPGA driver
```

在 `lg/bevformer-vortex-direct-mlir` 分支上，后端另提供
`--device-lowering direct-mlir`。它保留前面的 Graph、Tensor/Linalg MLIR 和严格
call-plan 审计，但把最后的设备实现替换为：

```text
面向 Vortex 的 LLVM Dialect kernel.mlir
  ├── RV32 data layout 与 a0 参数缓冲解包
  ├── CTA thread/block CSR inline asm
  ├── vortex-kernel 函数属性
  ├── llvm.global.annotations 中的 vortex.kernel
  └── llvm.used + llvm.compiler.used
                │
                ▼
同一个 VOLT LLVM 20.1.8：mlir-opt → mlir-translate
                │
                ▼
kernel.ll → VOLT RISC-V/Vortex 指令选择 → ld.lld → ELF → vxbin.py
```

这个模式的 artifact 不包含设备端 `kernel.cpp`。`main.cpp`、Vortex startup、
`libvortex2.a`、linker script、`vxbin.py` 和 host runtime 继续复用，因为它们分别
承担调度、设备启动、链接、封装和执行职责。更细的入口 ABI、CSR、参数字段偏移与
VectorAdd golden 见
[`designs/vortex_direct_mlir_abi.md`](designs/vortex_direct_mlir_abi.md)。
从输入 Graph、三层 MLIR、VOLT 机器码到 Host Runtime 的完整分层逻辑见
[`bevformer_vortex_direct_mlir_architecture.md`](bevformer_vortex_direct_mlir_architecture.md)。

这条路径当前只覆盖一个静态 FP32 BEVFormer Transformer FFN：

```text
query
  → MatMul(fc1_weight)
  → Bias + ReLU
  → MatMul(fc2_weight)
  → Bias + query residual
  → LayerNorm(scale, bias)
  → output
```

它不是完整 BEVFormer。ResNet-50、FPN、图像到 BEV 的几何准备、
MSDeformableAttention、时序/空间 Attention、Decoder、检测头和状态管理尚未
lowering 到 Vortex。当前代码也没有把原来的 CUDA、PTX 或 cuBLAS 实现在 Vortex
运行时中复用；Vortex 路径会生成独立的 RISC-V/Vortex Kernel。

实现入口是 [`src/gac/backends/vortex.py`](../src/gac/backends/vortex.py)，命令行
工具是 [`tools/build_bevformer_vortex.py`](../tools/build_bevformer_vortex.py)。

## MLIR 与 Vortex 的边界

上游 Vortex 3.0 当前没有 Vortex MLIR Dialect，也没有通用的 MLIR-to-Vortex
kernel translator。因此这一版采用两级 MLIR 和一份经过严格审计的 Runtime call
plan；前五步由两种设备 lowering 共用：

1. GAC 把 Canonical Graph 发射为标准 Tensor/Linalg MLIR。
2. 第一次外部 `mlir-opt` 真实验证、规范化并做 CSE，产生 `optimized.mlir`。
3. 后端从优化后的 Tensor/Linalg 文本重新解析函数 ABI、静态类型、两个 MatMul
   的 shape、ReLU 和 LayerNorm epsilon，并与 Canonical Graph 合同逐项核对；超出这份
   受支持 fingerprint 的模块会被拒绝。同时，六组 `dense<...>` FP32 常量按
   `(shape, little-endian f32 payload SHA-256)` 与 Graph 常量逐 bit 核对。
4. 解析结果确定性 lowering 为第二个标准方言 MLIR module。它不引入自定义
   Vortex op，而是用 `func.call` 表达三次 Vortex Runtime 调用。
5. 第二次外部 `mlir-opt` 验证并规范化这个 module，artifact 保存其输出为
   `vortex.mlir`。
6. 后端严格解析 `vortex.mlir`；只有调用声明、调用顺序、静态 memref ABI、launch
   size、epsilon 和外部权重绑定全部符合冻结合同，才允许进入设备 lowering。
7. `cpp-template` compatibility 模式用固定模板生成 Vortex C++；`direct-mlir` 模式
   根据同一份冻结 plan 直接生成三个 LLVM Dialect Kernel，并再次审计函数体、RV32
   参数解包、CTA CSR、入口 annotation、retention global 和 provenance hash。
8. compatibility 模式由 VOLT 编译 C++；direct 模式由固定的同一棵 LLVM/VOLT
   20.1.8 工具树执行 `mlir-opt → mlir-translate → VOLT clang → ld.lld`。两者最终
   都由 `vxbin.py` 把 ELF 打包为 `kernel.vxbin`。

`vortex.mlir` 的公开 plan 函数名包含 `optimized.mlir` 的 SHA-256。它有 10 个静态
FP32 memref 参数，分别承载 query、六个权重/偏置、activation、residual 和 output；
三次 `func.call` 的 launch size 依次为 `rows × hidden`、`rows × channels`、`rows`，
LayerNorm epsilon 是显式 `f32` 常量。函数属性还保存
`fc1_weight`、`fc1_bias`、`fc2_weight`、`fc2_bias`、`norm_weight`、`norm_bias` 六个
外部 `.bin` 文件的 SHA-256。

权重绑定分成两道检查。编译时，后端解析 `optimized.mlir` 中所有 dense FP32
常量；无论 MLIR 将常量打印成逐元素列表、scalar splat，还是 `dense<"0x...">`
raw bytes，都会还原成 FP32 bytes，再比较 shape 和 payload 摘要。后端还根据两次
`linalg.matmul` 和三个对应 `linalg.generic` 的 SSA 用途，把六个常量逐一绑定到
`fc1_weight`、`fc1_bias`、`fc2_weight`、`fc2_bias`、`norm_weight` 和 `norm_bias`；
同 shape 的 LayerNorm scale/bias 对调也会被拒绝。这样允许 `canonicalize`/CSE 合并
内容完全相同的常量，同时拒绝任意 bit、shape、语义角色、缺项或额外 dense 常量变化。发布 artifact 时，
六个外部 `.bin` 直接由同一组 Graph 常量按 little-endian FP32 写出；其摘要同时进入
`vortex.mlir` 属性、manifest lowering metadata 和逐文件记录，部署前校验再把三者
逐一核对。因此绑定链是 `optimized.mlir dense ↔ Canonical Graph ↔ 外部 .bin`。

Runtime-call parser 也校验每次调用的完整 SSA buffer 实参，而不只校验 callee 名称和
memref 类型。冻结的数据流必须恰好为：

```text
fc1(query, fc1_weight, fc1_bias -> activation)
fc2(query, activation, fc2_weight, fc2_bias -> residual)
layer_norm(residual, norm_weight, norm_bias -> output; epsilon)
```

解析器从 plan 函数的 10 个参数取得 SSA 名称，并逐调用比较上述输入、权重、scratch
和输出的准确顺序；即使两个 buffer 具有相同 shape/type，交换 SSA 实参也会被拒绝。
每次调用引用的 launch SSA 还必须分别解析为预期的 `index` 常量，LayerNorm 最后一个
实参必须引用已经核对的 `f32` epsilon 常量。

这里没有跨 LLVM 版本传递 LLVM bitcode。标准 MLIR 是 GAC 的中间层和审计边界，
VOLT 是 Vortex 的目标编译器。compatibility 模式属于 **MLIR 驱动的 Runtime-call
lowering**；direct 模式已经移除设备 C++，但它仍是专门覆盖三个静态 FFN Kernel 的
受限 LLVM Dialect lowering，并非通用 Vortex dialect，也不是 stock `mlir-opt`
自动生成 Vortex Kernel。`frontend.mlir`、`optimized.mlir`、`vortex.mlir`、设备源码
和文件摘要一起保存在 artifact 中，使这两条 lowering 都可复核。摘要用于发现意外
损坏和错配，不是发布者签名。

## Kernel 与 Runtime

后端只在 `vortex.mlir` 通过第二次 `mlir-opt` 和严格解析后生成三个 entry：

| Entry | 工作划分 | 中间结果 |
| --- | --- | --- |
| `bevformer_fc1_bias_relu` | 每个线程计算一个 `rows × hidden` 元素 | `activation` |
| `bevformer_fc2_bias_residual` | 每个线程计算一个 `rows × channels` 元素 | `residual` |
| `bevformer_layer_norm` | 每个线程处理一整行 | `output` |

主机程序使用 `vortex2.h` 的 device、queue、buffer、module、kernel 和 event API。
它上传常驻权重和每轮 query，按 event 依赖顺序提交三个 Kernel，再读取 output。
artifact 生成时由 GAC/NumPy reference 独立冻结 `query.bin` 和 `golden.bin`；远端
runner 加载 golden，并以 `1e-3` 最大绝对误差门验证结果。
输出 JSON 分开记录 H2D、三个 Kernel 的 host wall、可用时的 runtime event profile、
D2H、端到端均值和正确性结果。

首版 MatMul 是每个输出线程串行遍历 K，LayerNorm 也是每行单线程归约。这是验证
编译链、ABI 和数值正确性的基线，还没有使用 Vortex shared-memory tiling、tensor
core、向量化、双缓冲或 autotuning，不能视为性能完成版。

现在可选 `--device-lowering direct-mlir --matmul-schedule dxa-double-buffer`，
为两个 MatMul 生成真实 DXA/LMEM 双 Stage 和同步，不改变上述参数 ABI 与三 Kernel
调用关系。Tile、硬件要求和数值/周期对照见[双缓冲设计](designs/vortex_buffered_mlir.md)。

## 生成 artifact

Linux 上可以先生成最小正确性 workload：

```bash
PYTHONPATH=src python tools/build_bevformer_vortex.py \
  --synthetic-smoke \
  --output build/bevformer-vortex-smoke \
  --mlir-opt /path/to/mlir-opt \
  --report build/bevformer-vortex-smoke.json
```

Windows 调用 WSL 中的 `mlir-opt`：

```powershell
$env:PYTHONPATH = "src"
python tools/build_bevformer_vortex.py `
  --synthetic-smoke `
  --output build/bevformer-vortex-smoke `
  --mlir-opt /home/thomas/gac-mlir-build-lf/bin/mlir-opt `
  --wsl --wsl-distribution Ubuntu-24.04 `
  --report build/bevformer-vortex-smoke.json
```

生成不含设备 C++ 的 direct-MLIR artifact 时，先把本地 LLVM checkout 切到固定的
VOLT revision `b2ecfe086df2d00233a87f3d3ebe6a74aba129e2`，再在 WSL 构建统一工具链：

```bash
bash tools/build_volt_mlir_wsl.sh
```

随后显式选择 direct lowering；Makefile 会从同一个 `LLVM_PATH` 使用
`mlir-opt`、`mlir-translate` 和 VOLT clang：

```bash
PYTHONPATH=src python tools/build_bevformer_vortex.py \
  --synthetic-smoke \
  --device-lowering direct-mlir \
  --output build/bevformer-vortex-direct-smoke \
  --mlir-opt "$HOME/.cache/gac-volt-llvm20/current-build/bin/mlir-opt"
```

`--synthetic-smoke` 生成 `[1,2,4]` 小 workload，适合先跑 SimX；
`--synthetic-width-smoke` 使用一行、256 通道和 512 hidden，适合验证生产通道宽度；
`--synthetic-production` 使用 `[1,2500,256] → 512 → 256`。也可以用
`--model /path/to/ffn.json` 输入带真实 FFN 权重的 GAC graph。synthetic profile
只用于编译链和正确性测试，不能用于模型精度结论。

如果已经用安全 checkpoint 导出器得到 `.gacweights`，可以直接抽取一个真实
Encoder FFN，不必先生成大型 JSON：

```bash
PYTHONPATH=src python tools/build_bevformer_vortex.py \
  --weights build/bevformer_tiny.gacweights --layer 0 \
  --output build/bevformer-vortex-layer0 \
  --mlir-opt /path/to/mlir-opt
```

默认生成完整 2500-row profile。`--weights-rows 1` 使用相同真实权重只编译一行，
适合先在很慢的 SimX 上验证 256 → 512 → 256 的生产通道宽度；它不是完整 FFN
延时测试。

输出目录包含：

| 文件 | 用途 |
| --- | --- |
| `frontend.mlir` | 优化前的标准 MLIR |
| `optimized.mlir` | 第一次外部 `mlir-opt` 的 Tensor/Linalg 输出 |
| `vortex.mlir` | 第二次外部 `mlir-opt` 验证后的标准 `func.call` Runtime plan |
| `kernel.cpp` 或 `kernel.mlir` | compatibility C++ entry，或三个 direct LLVM-dialect MLIR entry |
| `main.cpp` | `vortex2` host、正确性检查和计时 |
| `common.h`、`model_config.h` | Host/Kernel ABI 与冻结 profile |
| `fc1_weight.bin` 等六个 `.bin` | FP32 FFN 常量 |
| `query.bin`、`golden.bin` | 冻结的确定性输入与 GAC/NumPy oracle |
| `verify_artifact.py` | 构建前校验每个源文件、权重和 golden 的大小与 SHA-256 |
| `Makefile` | Vortex regression build 接入 |
| `manifest.json` | 目标、工具链、profile、direct 参数 ABI 和逐文件摘要 |

目标目录必须不存在或为空，避免把旧 Kernel 或旧权重混入新 artifact。

`manifest.json.files` 必须精确包含下面 **17 个 payload 文件**；`manifest.json`
记录这些 payload，因此不把自身列入白名单：

```text
Makefile                 common.h               model_config.h
frontend.mlir            optimized.mlir         vortex.mlir
kernel.cpp/kernel.mlir   main.cpp               verify_artifact.py
fc1_weight.bin           fc1_bias.bin           fc2_weight.bin
fc2_bias.bin             norm_weight.bin        norm_bias.bin
query.bin                golden.bin
```

两种模式都恰好有 17 个 payload；上面斜线表示二选一，不能同时存在。
`verify_artifact.py` 拒绝缺失或额外的 manifest 文件记录、重复 JSON key、非有限 JSON
数、目录穿越名、symlink、大小或 SHA-256 不符。它还要求三份 MLIR 的摘要绑定到
lowering metadata，并逐一核对 `vortex.mlir` 中的六个权重 SHA 与对应 `.bin` 以及
manifest 一致。普通 build 输出（例如后续生成的 `.o`、`.elf`、`.vxbin`）不属于这
17 个输入 payload。

direct manifest 的 `lowering.device_argument_abi` 还逐 Kernel 记录参数缓冲区的
字段名、LLVM 类型、byte offset、字段大小与对齐、结构总大小/对齐，以及设备地址从
`i64` 存储值显式截断为 RV32 `i32` 后再 `inttoptr` 的转换合同。Host 在 launch 前
拒绝超过 `UINT32_MAX` 的设备地址，避免截断丢失有效地址位。

## 部署到已经配置好的 Vortex build 树

生成的 Makefile 使用 Vortex regression 的相对目录合同，因此把 artifact 内容放到：

```text
/home/robot/project/vortex/build/tests/regression/gac_bevformer_vortex/
```

例如，从本地同步目录；SSH 密码不应写进命令、脚本或仓库：

```bash
VORTEX_HOST=vortex-host.example
rsync -av --delete build/bevformer-vortex-smoke/ \
  "robot@$VORTEX_HOST:/home/robot/project/vortex/build/tests/regression/gac_bevformer_vortex/"
```

目标目录必须只用于这一份 artifact。`--delete` 会移除上一次构建留下的
`kernel.ll`、对象、ELF 和 `.vxbin`；如果目标目录还承载其他内容，应改用新的空目录，
不能把不同 artifact 增量覆盖到同一个目录。

登录远端后先构建，再用 SimX 做正确性 smoke：

```bash
cd /home/robot/project/vortex/build/tests/regression/gac_bevformer_vortex
make verify-artifact
make -j"$(nproc)"
make run-simx OPTS="--data-dir . --kernel kernel.vxbin --warmup 1 --repeat 3"
```

direct-MLIR artifact 还必须把 `LLVM_PATH` 指向上面固定 revision 构建出的统一
LLVM/VOLT 20.1.8 目录。这个目录同时提供 `mlir-opt`、`mlir-translate`、clang、
`ld.lld` 和 VOLT 后端；生成的链接参数还用 `--ld-path` 绑定到这个目录中的
`ld.lld`，不会依赖 `PATH` 里的其他 linker：

```bash
VOLT_BUILD="$HOME/.cache/gac-volt-llvm20/current-build"
make LLVM_PATH="$VOLT_BUILD" verify-artifact verify-toolchain
make LLVM_PATH="$VOLT_BUILD" -j"$(nproc)"
make LLVM_PATH="$VOLT_BUILD" run-simx \
  OPTS="--data-dir . --kernel kernel.vxbin --warmup 1 --repeat 3"
```

`verify-toolchain` 会检查 MLIR/Clang 工具的 `20.1.8` 版本、Vortex 构建实际使用的
`clang++`/`llvm-objdump`/`llvm-objcopy`，以及构建目录中的
`.gac-build-revision`。revision 不等于
`b2ecfe086df2d00233a87f3d3ebe6a74aba129e2` 时构建立即失败，防止 LLVM 24
生成的 IR 被悄悄交给 LLVM 20 VOLT 后端。

`make verify-artifact` 可显式检查部署是否完整。生成的 Makefile 还把这项检查设为
host executable、VOLT object/ELF、`kernel.vxbin`、dump 以及所有 `run-*` 目标的
order-only 前置条件，所以直接执行并行 `make` 或 `make run-simx` 也不能绕过校验。
direct-MLIR 模式进一步把 `kernel.opt.mlir`、`kernel.ll`、设备对象、host executable、
ELF、`.vxbin` 和 dump 声明为强制重建目标；即使目录中存在时间戳更新的旧产物，也会
从当前 manifest 绑定的 `kernel.mlir` 重新生成完整派生链。
Makefile 在解析时要求 Vortex build 的 `XLEN=32`；当前 artifact ABI 没有验证 RV64。

构建命令由 Vortex build 配置选择 VOLT `clang++`，生成 `kernel.elf` 和
`kernel.vxbin`，同时链接 host runner。运行命令构建/选择 SimX driver 并通过
`vortex2` 启动三个 entry。成功标准是程序输出的 JSON 同时满足
`"verified": true`、`"finite": true`，且 `max_abs_error <= tolerance`。

2026-10-13 的 compatibility C++ bridge 验证包含两层：合成 `[1,2,4]` smoke 连续
执行 3 次通过，最大绝对误差 `1.1920929e-7`；SHA-256 已核对的官方
BEVFormer-tiny checkpoint 第 0 层权重在 `[1,1,256] → 512 → 256` 上通过，最大
绝对误差 `1.90734863e-6`。真实权重的完整 `[1,2500,256]` artifact 也已由 VOLT
编译成 `.vxbin`。详细的 MLIR/VXBIN 摘要、SimX cycles 和计时边界保存在
[`docs/validation/vortex/bevformer-ffn-2026-10-13`](validation/vortex/bevformer-ffn-2026-10-13/README.md)。

direct-MLIR 路径已用固定 LLVM/VOLT 20.1.8 完成 RV32 Vortex 可重定位对象、最终
ELF、`.vxbin` 和 SimX 数值验收，并核对 Kernel alias、`.vx_entry`、CTA CSR 与
`vx_split_n/vx_join`。合成 `[1,2,4] → 8 → 4` smoke 的最大绝对误差为
`1.1920929e-7`；10 次 SimX 的 runtime event-profile 均值为 `4.447282 ms`、端到端均值为
`4.602381 ms`。这是软件模拟器的小尺寸计时，不能作为真实 Vortex 硬件或完整
BEVFormer 的推理速度。证据与复现步骤见
[`docs/validation/vortex/direct-mlir-2026-10-13`](validation/vortex/direct-mlir-2026-10-13/README.md)。

同一 artifact 可以在已经配置好相应环境的 build 树中使用 `make run-rtlsim`。
FPGA 运行还需要该 build 配置匹配的 bitstream、板卡驱动和运行环境；不能仅把
`kernel.vxbin` 复制到任意 FPGA 主机就视为可运行。

## 如何理解延时

`make run-simx` 的主要价值是验证 ISA、Kernel ABI、Runtime 调度和数值结果。SimX
是功能/指令级模拟器，它的 host wall 和 event profile 都不是 Vortex FPGA/ASIC 的
真实硬件延时，不能换算为 FPS，也不能与 RTX 3070 的 CUDA Event 延时比较。

性能验收必须在明确的 Vortex 实体目标上记录至少以下信息：Vortex commit 与 build
配置、core/warp/thread 数、时钟、内存配置、driver、bitstream、输入 profile、预热和
重复次数。只有相同 artifact 在 RTL/FPGA 目标上取得的 event/cycle 数据才能用于
硬件性能结论。

## 扩展到完整 BEVFormer

后续扩展应保持当前分层，每增加一类算子都同时完成 Canonical Graph 合同、MLIR
表达、Vortex lowering、reference 对照和目标测试。建议顺序是：

1. 用 shared-memory tiled SGEMM、向量化和 Vortex tensor core 替换 FFN 串行 K 循环。
2. 补齐 Softmax、transpose/layout、batched MatMul 和 Attention 基础算子。
3. 实现 MSDeformableAttention、GridSample 和 BEV geometry Kernel。
4. 接入 Conv/ResNet-50/FPN，或定义清楚由外部 accelerator 执行的图分区边界。
5. 接通 Encoder、Decoder、检测头和 cold/warm `prev_bev` 状态事务。
6. 在真实 checkpoint 与 nuScenes 输入上做逐层对齐、整网正确性、峰值内存和硬件
   端到端延时验收。

在这些阶段完成前，准确的支持声明是“BEVFormer FFN 已通过两级标准 MLIR 和严格
call-plan 审计；compatibility 路径生成 C++ Kernel，direct 路径直接生成 LLVM
Dialect Kernel 并已完成 VOLT、ELF、`.vxbin` 与 SimX 数值验证”，不是“完整
BEVFormer 已运行在 Vortex”。

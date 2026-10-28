# Triton / Vortex batched MatMul 验证记录

本目录保存 GAC 将 ONNX `MatMul` 合法化为 Linalg，并把符合严格合同的
静态 FP32 rank-2 / batched MatMul 交给 Triton/Vortex 后端的验证证据。
验证目标是数值正确性、地址计算、原子选择、ABI 一致性与失败边界；这些结果
不是硬件性能数据。

## 结论

2026-10-22 的最终完整 GAC 集成检查为 **118/118 通过**。正例全部由
Vortex SimX 执行设备 Kernel，并与 ONNX ReferenceEvaluator 独立比较；
报告中的 `host_operator_fallback` 均为 `false`。

同一份冻结后端还通过了 **20/20 batched MatMul 专项检查**和
**108/108 后端全量回归**。专项检查包含 matched-prefix、shared-RHS、
K48→BK64，以及缺失 A/B K mask、错误 K 谓词、错误 batch base/stride、
多 `tt.dot`、多 store 和 signed-i32 溢出等恶意图的拒绝。

```text
ONNX MatMul
    ↓ C++ importer / legalize-to-Linalg
规范 linalg.generic multiply-add reduction
    ↓ 静态 shape、索引映射与物理 Buffer 合同检查
一个 batched TTIR kernel（每 program 恰好一个 tt.dot）
    ↓ Triton/Vortex C++ lowering
VOLT → Vortex vxbin
    ↓ GAC Runtime
SimX 设备执行
```

## 选择合同

当前 Triton 路径只接受：

- 静态、无编码、行主序 FP32 张量；
- rank-2：`A[M,K] × B[K,N] → C[M,N]`；
- batched：`C` 的 rank 至少为 3，`A` 必须具有与 `C` 完全相同的 batch
  前缀；`B` 可具有同一前缀，也可为供所有 batch 共享的 rank-2 RHS；
- `M/N/K` 都为正，`16 <= K <= 65536`，且 `K` 为 16 的倍数；
- `batch*M*K`、实际 `B` 元素数、`batch*M*N` 和启动 grid 都在 signed-i32
  范围内；
- A、B、C 的逻辑 shape 与物理根 shape 完全一致，三个 Buffer 不别名；
- Linalg region 必须是 FP32 零初始化的规范 `acc + A*B`，且没有 Bias、
  激活或其他 epilogue。

shared-LHS、逐轴 singleton broadcast、转置、shape-changing View、rank-1
向量 MatMul 和不符合 K 合同的形式完整保留为一个 native Vortex dispatch。

## Program 与地址计算

`BM=BN=16`，`BK=nextPowerOfTwo(K)`。一个 `program_id(0)` 只负责一个
batch 的一个 16×16 输出 tile：

```text
tilesM        = ceil(M / 16)
tilesN        = ceil(N / 16)
tilesPerBatch = tilesM * tilesN

batchId    = pid / tilesPerBatch
tileProgram= pid % tilesPerBatch
tileM      = tileProgram / tilesN
tileN      = tileProgram % tilesN

A offset = batchId*(M*K) + row*K + reduction
B offset = batchId*(K*N) + reduction*N + col   # matched-prefix
B offset =                    reduction*N + col # shared-rhs
C offset = batchId*(M*N) + row*N + col
```

K=48 时 TTIR 使用 64 个物理 reduction lane。A 和 B 的 load mask 都额外
包含 `reduction < 48`，无效 lane 读取精确 FP32 零，再进入唯一的 `tt.dot`。
这保持逻辑 stride 为 K=48，同时满足 `tt.make_range` 的二次幂长度约束。

Manifest 明确记录：

```text
batches / batch_mode / a_shape / b_shape / c_shape
m / n / k / block_m / block_n / block_k / grid_size
```

Python 构建驱动与 C++ Runtime 各自从物理张量重新计算 shape、元素数、
字节数、batch、tile 和 grid，并在设备打开前拒绝不一致的计划。旧 rank-2
Manifest 仅在五个 batch 字段全部缺失时兼容；batched 或部分缺字段的计划
会被拒绝。

## 数值与边界结果

| 用例 | 路径 | 最大绝对误差 |
|---|---:|---:|
| rank-2 tail `[17,32] × [32,19]` | 1 Triton / 0 native | `9.5367431640625e-7` |
| matched batch `[2,17,32] × [2,32,19]` | 1 Triton / 0 native | `1.430511474609375e-6` |
| shared RHS `[2,17,32] × [32,19]` | 1 Triton / 0 native | `1.430511474609375e-6` |
| rank-4 matched `[2,3,5,16] × [2,3,16,7]` | 1 Triton / 0 native | `7.152557373046875e-7` |
| rank-4 shared constant RHS `[2,2,5,48] × [48,7]` | 1 Triton / 0 native | `9.5367431640625e-7` |
| unit prefix `[1,2,5,16] × [1,2,16,7]` | 1 Triton / 0 native | `7.152557373046875e-7` |

K15、rank-1 LHS/RHS、singleton broadcast、shared-LHS、A/B 输入别名和
shape-changing View 均验证为 **0 Triton / 1 native**，且小形状完成 SimX
数值对齐。

17 组 Manifest 变异覆盖 grid、K、tile、kind、batches、batch 乘积溢出、
batch mode、部分 batch 字段、A/B/C 逻辑 shape、物理 A shape、重复输入和
输出别名，并单独覆盖 K48 的 `block_k=48`、`block_k=128` 与逻辑 K/shape
不一致。Python driver 与 C++ Runtime 均独立拒绝这些错误。

`selector-contract.log` 独立覆盖 rank-2、matched-prefix、shared-RHS、
unit-prefix、K48→BK64，以及 singleton broadcast、shared-LHS、K15、输入
别名和 signed-i32 边界。

## 复现命令

```bash
cmake --build "$GAC_NATIVE_BUILD" \
  --target gac-compile gac-vortex-run \
           gac-triton-layernorm-selector-contract \
           gac-triton-softmax-selector-contract \
           gac-triton-batched-matmul-selector-contract -j4

export TRITON_VORTEX_OPT=/path/to/triton/third_party/vortex/build/triton-vortex-opt
export PYTHONPATH=/path/to/triton/third_party/vortex/python:$PWD/src:${PYTHONPATH:-}
export TRITON_CACHE_DIR=/tmp/gac-triton-batched-matmul-cache
export VORTEX_DRIVER=simx
export LD_LIBRARY_PATH="$VORTEX_BUILD/sw/runtime:${LD_LIBRARY_PATH:-}"

python native/tests/run_triton_vortex_tests.py \
  --compiler "$GAC_NATIVE_BUILD/gac-compile" \
  --runtime "$GAC_NATIVE_BUILD/gac-vortex-run" \
  --output /tmp/gac-triton-batched-matmul-validation
```

本次运行使用的工具和源码 SHA-256、原始输出路径见
[`evidence.json`](evidence.json)。

## 证据文件

- [`validation.json`](validation.json)：118 项逐项结果。
- [`backend-specialized-validation.json`](backend-specialized-validation.json)：
  冻结后端的 20 项 batched MatMul 专项检查。
- [`backend-full-validation.json`](backend-full-validation.json)：冻结后端的
  108 项全量回归。
- [`rank2-tail`](rank2-tail/)、[`matched-prefix`](matched-prefix/) 与
  [`shared-rhs`](shared-rhs/)：rank-2 与两种 batch 地址形式。
- [`rank4-matched`](rank4-matched/)、
  [`rank4-shared-k48-constant`](rank4-shared-k48-constant/) 与
  [`unit-prefix`](unit-prefix/)：多维前缀展平、常量共享 RHS、K padding 和
  extent-1 前缀。
- `native-*`：不支持形态完整回退后的 Linalg、Device plan 与 Runtime 报告。
- [`rejections`](rejections/)：17 组非法 Manifest 与 Runtime 拒绝日志。
- [`selector-contract.log`](selector-contract.log)：选择器边界。
- [`legacy-rank2-manifest.json`](legacy-rank2-manifest.json)：严格受限的旧
  rank-2 Manifest 兼容合同。

当前 `tt.dot` 后端是正确性模板；本目录不声称性能提升，也不代表完整
BEVFormer 已全部走 Triton。

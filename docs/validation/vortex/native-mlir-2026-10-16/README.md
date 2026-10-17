# Native C++ MLIR compiler validation — 2026-10-16

本目录记录真正的 C++ ONNX 导入、MLIR Operation/Pass、Vortex 机器码生成和 SimX 执行证据。测试没有调用旧 Python Graph 编译器，也没有生成承载算子实现的 C++ Kernel。

## 已验证范围

| 层级 | 测试结果 | 证据 |
| --- | --- | --- |
| C++ ONNX → 注册的 GAC Operation → MLIR Pass → LLVM → 主机验证 | 22 个测试通过：18 个有效模型数值验证、4 个错误路径拒绝 | `native-results.json` |
| MLIR → LLVM IR → VOLT → RV32 ELF object | 首批 6 个有效模型均通过，检查 ELF32/RISC-V、每个 Kernel 的 `__vx_kentry_` 符号及 `.vx_entry` 节 | `vortex-results.json`、`symbols/` |
| ELF → vxbin → C++ Runtime → Vortex SimX | MatMul+Bias+ReLU、Conv+Bias+ReLU、Softmax 三例与 ONNX 参考数值一致 | `vortex-results.json`、`simx/` |

首批案例还覆盖多消费者输出保留、Residual+ReLU 融合、Softmax 大数值稳定性、常量 shape 程序折叠，以及非法广播、未知 ONNX domain 的拒绝。未知算子只能通过显式 inspection 模式保留，不能据此执行。

额外案例覆盖批量/向量 MatMul、分组膨胀卷积、LayerNorm 多归约轴、ReduceSum/Mean、Transpose/Concat/负向 Slice、GridSample 双线性边界、FP16 舍入边界保留、BOOL Cast、输入输出别名、纯常量输出，以及运行时 shape 参数和常量除零拒绝。FP16 的 MatMul→Add→ReLU 保留三个 Dispatch，避免融合消除中间舍入。

另有两例验证重复 Add 的真实 MLIR CSE，以及内部 SSA 名称与公开输出名称碰撞时的正确性。CSE 前会规范化内部操作的非语义溯源属性，避免源节点名字不同阻止等价操作消除；公开输入输出名字仍由函数 ABI 保留。

参考值来自 `onnx.reference.ReferenceEvaluator`。GridSample 的参考求值使用官方 ONNX version converter 将参考模型转换到 opset 20，以匹配已安装 ReferenceEvaluator 的 mode 拼写；实际交给 C++ 编译器的 fixture 仍为 opset 18。

## 数值结果和边界

三例 SimX 的最大绝对误差分别为：MatMul+Bias+ReLU **0**，Conv+Bias+ReLU **1.9073486e-6**，Softmax **2.9802322e-8**；测试阈值为 `rtol=2e-5, atol=2e-5`。具体值以保存的 JSON 为准。

这验证了小规模静态模型的编译链和数值正确性，不代表完整 BEVFormer 已能编译或运行，不代表任意 ONNX 算子语义已覆盖。主机 native target 仅用于明确选择的数值测试；SimX Runtime 不做主机算子回退。

当前设备模板为每个 SIMT 线程负责一个输出元素，归约在该线程内串行执行。Dispatch 顺序提交并等待，按 SSA 最后使用位置规划 DDR arena 和复用 Buffer。`lmem_bytes=0`、`double_buffer=false`：本轮没有启用片上 LMEM 分块、DXA、双 Buffer、TCU 或异步流水。Buffer 复用证据不能当作这些优化已经实现。

`simx/*.json` 中的计时是 **host wall clock**，包括模拟器及 Runtime 开销；上传/下载还包括文件 I/O。它不是芯片推理延时，不能换算为完整 BEVFormer FPS。本轮没有 FPGA/物理芯片性能测试。

## 编译链示例

`matmul-example/` 保存同一个 MatMul+Bias+ReLU 小模型的关键阶段：

```text
ONNX ModelProto（C++ protobuf 读取）
  → 01-import.mlir：gac.constant / gac.matmul / gac.add / gac.relu
  → 02-optimized.mlir：gac.matmul_bias_relu（MLIR RewritePattern 融合）
  → 03-device.mlir：SCF/Arith/Math/LLVM + Vortex Operation
  → kernel.mlir：LLVM Dialect
  → kernel.ll：LLVM IR
  → VOLT RV32 object → ELF → vxbin
  → C++ gac-vortex-run → SimX
```

该例融合后只有一个 Dispatch。`plan.json` 是从 MLIR SSA 提取的执行计划，而不是 Python Graph 的拷贝。常量仍在单独的二进制数据文件中交给 Runtime；本证据目录不保存二进制、模型权重或可执行文件。

## 本次环境与复现

本次使用 WSL Ubuntu 24.04、GCC 13.3、CMake 3.28.3、Protobuf 3.21.12；MLIR、LLVM translation 与 VOLT 都来自同一个 **LLVM 20.1.8** 开发树。`provenance.json` 记录工具路径、版本、可执行文件摘要、源码摘要和模拟器执行的 ELF/vxbin 摘要。其 Git commit 是工作区基线；本轮新增原生实现尚未提交，以源码摘要定位测试内容。

本机 Linux 路径示例：

```bash
export REPO=/mnt/c/Users/Thomas/Desktop/universal_ai_compile/General-AI-Compiler
export LLVM_PATH=/home/thomas/gac-volt-llvm20/build-release
export GAC_NATIVE_BIN=/home/thomas/gac-native-build
export VORTEX_BUILD=/home/thomas/vortex-gac-buffered-build32
export VORTEX_SOURCE=/mnt/c/Users/Thomas/Desktop/project/vortex-3.0
export ONNX_PROTO_DIR=/mnt/c/Users/Thomas/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages/onnx

cmake -S "$REPO/native" -B "$GAC_NATIVE_BIN" \
  -DMLIR_DIR="$LLVM_PATH/lib/cmake/mlir" \
  -DLLVM_DIR="$LLVM_PATH/lib/cmake/llvm" \
  -DONNX_PROTO_DIR="$ONNX_PROTO_DIR" -DCMAKE_BUILD_TYPE=Release \
  -DVORTEX_BUILD="$VORTEX_BUILD" -DVORTEX_SOURCE="$VORTEX_SOURCE"
cmake --build "$GAC_NATIVE_BIN" --parallel 2
```

夹具生成需要安装 NumPy 与 ONNX 的 Python。Python 仅生成测试模型、参考值和测试编排，不承担编译器语义转换。本次夹具由 Windows bundled Python 生成，WSL 测试只读取已经生成的 `.onnx` 和 `.npz`。使用任意具备上述依赖的 Python 可生成相同夹具：

```bash
python native/tests/generate_models.py --output build/native-tests
```

生成器目前包含全部 22 个案例；本次保存的 `native-results.json` 来自最终统一构建后的完整回归（原始产物在 `build/native-cse-tests/results.json`）。VOLT/SimX 则在同一最终构建上重新验证原先六个案例。完整主机回归命令：

```bash
python3 "$REPO/native/tests/run_tests.py" \
  --compiler "$GAC_NATIVE_BIN/gac-compile" --optimizer "$GAC_NATIVE_BIN/gac-opt" \
  --mlir-translate "$LLVM_PATH/bin/mlir-translate" --clang "$LLVM_PATH/bin/clang" \
  --fixtures "$REPO/build/native-tests"
```

VOLT 对象与 SimX 验证（默认执行 MatMul、Conv、Softmax 三个 SimX 案例）：

```bash
python3 "$REPO/native/tests/run_vortex_tests.py" \
  --compiler "$GAC_NATIVE_BIN/gac-compile" --llvm-bin "$LLVM_PATH/bin" \
  --fixtures "$REPO/build/native-tests" \
  --compile-script "$REPO/native/tools/compile-vortex.sh" \
  --runner "$GAC_NATIVE_BIN/gac-vortex-run" --vortex-build "$VORTEX_BUILD" \
  --case matmul_bias_relu --case multi_consumer --case conv_bias_relu \
  --case residual_add_relu --case softmax --case shape_chain
```

`--device-case NAME` 可指定要在 SimX 执行的案例；`--case NAME` 限定对象编译范围。上例精确复现本目录保存的 6 个对象与 3 个 SimX 案例。命令不需要运行或安装旧 Python GAC 编译器。

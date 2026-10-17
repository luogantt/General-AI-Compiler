# 全模型机器码执行与验证

`run_full_model.py` 是测试驱动。它读取 C++ 编译器的 dispatch manifest，依次调用 LLVM 编译出的 CPU 共享库函数。NumPy 只负责存储、文件读写和结果比较，脚本不实现 Conv、MatMul、Attention 等算子，不调用 BLAS、ONNX Runtime 或 PyTorch 来替代生成的 Kernel。

这条 CPU 路径用于排查编译器语义。Vortex 执行必须另外经过 VOLT、vxbin 和 C++ `gac-vortex-run`。两者的时间不能互相替代，SimX 的主机运行时间也不是物理 GPU 推理延时。

## 执行完整的 native artifact

先由 `gac-compile --target=native --emit=llvm` 生成 `kernel.mlir`、`plan.json` 和 `constants/`，再生成共享库：

```sh
mlir-translate --mlir-to-llvmir kernel.mlir -o kernel.ll
clang -shared -fPIC -O2 kernel.ll -lm -o kernels.so

python3 native/tests/run_full_model.py \
  --manifest build/native-bevformer-full/plan.json \
  --constants-dir build/native-bevformer-full/constants \
  --library build/native-bevformer-full/kernels.so \
  --inputs build/bevformer-onnx-inspect/images.npy \
  --reference build/bevformer-onnx-inspect/pytorch-outputs.npz \
  --output-dir build/native-bevformer-full/host-results \
  --checkpoint-every 100 --progress-every 10
```

输入必须与 manifest 的 dtype、shape、名称完全匹配，不自动裁剪、缩放或转精度。单输入可以使用 `.npy`，多输入用按输入名称索引的 `.npz`。BEVFormer 现有固定 fixture 是 `float32[1,6,3,480,800]`，校验三个原始输出 `bev_embed`、`all_cls_scores`、`all_bbox_preds`。这是固定标定参数、cold 帧的数值回归，不代表整个数据集精度或 warm 时序路径已经验证。

生成的验证文件：

- `dispatch-progress.jsonl`：每个 dispatch 的 begin/end、来源节点、主机 Kernel 时间；卡在某个 Kernel 时，最后一个 begin 可定位它。
- `outputs.npz`：完整执行后的输出，输出名称来自 C++ manifest。
- `full-model-report.json`：artifact SHA-256、完整执行状态、数值误差、内存需求及逐 dispatch 时间。
- `checkpoint.json`、`checkpoint-arena-{0,1}.npy`：可选断点；保留整个 arena，恢复前核对 manifest、共享库、输入和权重哈希。

使用同样参数加 `--resume` 从最近完整断点恢复。`--stop-after 100` 在累计完成 100 个 dispatch 后保存断点停止；这时报告 `full_model_executed=false`，不能称为整网通过。断点读写和中间结果校验不包含在 `summed_host_kernel_wall_seconds` 中，但会计入本次执行 wall time。恢复后的 Kernel 时间是分段执行之和，不是一次连续执行的端到端延时。

## 逐层定位

```sh
# 在中间 tensor 被 arena 复用覆盖之前保存指定层的真实结果。
python3 native/tests/run_full_model.py ... \
  --dump-intermediates 'img_backbone|img_neck'

# 可选参考文件仅用于比较；参考不能作为生成 Kernel 的输入替代物。
python3 native/tests/run_full_model.py ... \
  --intermediate-reference selected-layer-references.npz
```

`--dump-intermediates` 是匹配 `source_output`、`source_node` 和内部 tensor 名称的正则表达式。文件 `intermediates.json` 将 dispatch、源节点和保存的 `.npy` 对应起来。逐层参考 `.npz` 也按这些名称索引；只比较匹配项，发现第一处不一致后保存断点停止。大张量全部落盘会明显增加磁盘占用和执行时间，通常先选 Backbone/FPN/Encoder 等边界，再缩小到有差异的层。

如需独立 ONNX Runtime 参考，可以在安装 `onnx`、`onnxruntime` 的单独测试环境中执行：

```sh
python3 native/tests/collect_onnx_references.py \
  --model build/bevformer-onnx-inspect/bevformer-tiny-cold.onnx \
  --inputs build/bevformer-onnx-inspect/images.npy \
  --node-regex '^/img_neck/|^/img_backbone/layer4/layer4.2/relu_2/Relu$' \
  --output-dir build/native-bevformer-full/reference --threads 1
```

该工具读取原始 ONNX，由 ORT 独立推导 shape/type，生成 `node-references.npz` 和 `model-reference.npz`；没有把 GAC 输出形状复制进参考。当前按节点名收集只接受单输出节点，避免把 Split 的多个结果误当作同一个 tensor。ORT 仅是参考实现，生成的 Kernel 执行路径始终读取原始输入及权重。

## Vortex 输入和结果

```sh
python3 native/tests/run_full_model.py \
  --manifest build/vortex-bevformer-full/plan.json \
  --inputs build/bevformer-onnx-inspect/images.npy \
  --output-dir build/vortex-bevformer-full \
  --prepare-vortex-inputs

# constants、input_<tensor index>.bin 必须按 manifest 放在 runtime 数据目录。
gac-vortex-run build/vortex-bevformer-full/plan.json \
  build/vortex-bevformer-full/kernel.vxbin \
  build/vortex-bevformer-full build/vortex-bevformer-full/device-results --trace-dispatches

python3 native/tests/run_full_model.py \
  --manifest build/vortex-bevformer-full/plan.json \
  --vortex-results build/vortex-bevformer-full/device-results \
  --reference build/bevformer-onnx-inspect/pytorch-outputs.npz \
  --output-dir build/vortex-bevformer-full/validation
```

结果文件本身不能证明是在 SimX、FPGA 还是 ASIC 上运行，必须连同 runtime 日志、驱动选择和硬件配置保存。工具始终保留 `physical_device_latency_ms=null`，不会把主机运行时间改称 GPU 延时。

可选 `--trace-dispatches` 将每个 Kernel 的开始和实际完成写到输出目录的
`dispatch-progress.jsonl`。完成记录在设备 event wait 返回后写入；它不会把 enqueue
成功当成计算完成。该诊断模式在每个 dispatch 处进行主机等待，不能用于性能结论。

## 执行成本与容量

`--audit-only` 只检查 manifest 和资源预算，不执行模型：

```sh
python3 native/tests/run_full_model.py --manifest path/to/plan.json \
  --output-dir path/to/audit --audit-only
```

`arena_bytes + input_bytes + constant_bytes + embedded_constant_bytes` 是不包含机器指令、栈、runtime 元数据的最低存储需求；`embedded_constant_bytes` 是 Kernel 模块内嵌常量，不能因不在外部权重文件中而遗漏。应比较设备实际可分配空间，不能只看某一个 tensor 是否放得下。工具还核对每个 dispatch 的依赖、读前定义，以及 arena 地址复用与 tensor 活跃区间是否冲突。

完整六相机 fixture 的历史 shape inspection 显示，55 个 Conv 的名义计算量约为 **190.181 GMAC**，其中第一层约 **5.419 GMAC**；最大已知 activation 为 **147,456,000 字节**。这些是形状估算，不是运行测量；边界 padding 会影响实际执行次数，其他未解析的 MatMul/Attention 也未包含在 Conv 合计中。正式容量以新编译器最终 manifest 为准。

当前本地 SimX 构建配置为 1 core、4 warps、4 threads；标量 Kernel 每个输出执行归约和寻址，尚无 TCU/shared-memory tiling。因此全图仿真成本远高于数个小算子验证，不应根据小案例的启动时间或 CPU 吞吐率推算硬件 FPS。先确保同一 Lowering 生成的主机代码完整数值对齐，再生成完整 Vortex 机器码，并记录实际仿真进度与资源限制。

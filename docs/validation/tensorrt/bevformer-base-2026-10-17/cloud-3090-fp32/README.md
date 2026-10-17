# RTX 3090：完整 BEVFormer-base FP32 对照

2026-10-17，同一 RTX 3090 24GB、TensorRT 10.13.3.9、源代码提交
`cf34273c6d1853a302bbfc24c59d4afce46fbd16`，完整 Base 固定 cold raw-head 图。
两组使用相同 ONNX、同一 mini_val 六相机输入 `[1,6,3,928,1600]`，预热 5 次、测量 20 次。

| 指标 | FP32 | 混合 INT8/FP16 |
|---|---:|---:|
| CUDA event 延时中位数 | **373.821 ms** | **204.073 ms** |
| 延时中位数倒数 | **2.675 FPS** | **4.900 FPS** |
| CUDA event P95 | 377.045 ms | 206.887 ms |
| 推理期间整卡显存采样峰值 | 4348 MiB（4.25 GiB） | 2570 MiB（2.51 GiB） |
| 构建 workspace 上限 | 4096 MiB | 1024 MiB |
| 构建优化级别 / aux streams | 3 / 0 | 3 / 0 |

这两组配置中，混合精度的速度约为 FP32 的 **1.832 倍**，均未达到 30 FPS。
workspace 不同、所选 tactic 可能不同，不把差异完全归因于量化位宽。
混合结果复用同日同 GPU 的上一轮测试，不是交替执行的温度控制实验。
计时排除加载、预热、H2D/D2H 和预后处理，没有 CUDA Graph replay，FPS 为延时倒数。
显存每 250 ms 采样一次，是整卡占用且可能漏过瞬时峰值，不是本进程独占显存。

FP32 构建关闭 FP16/INT8/TF32，插件浮点边界约束 FP32；构建、执行均设置
`NVIDIA_TF32_OVERRIDE=0`。整数 shape/index 和 Bool 保持整数/布尔类型。
实际 inspector 的浮点 tensor/weight 格式以及 156 个 DCNv2、18 个 MSDA 的插件边界检查
见 [精度审查](full-precision-audit.json)。检查覆盖公开格式和 tactic 名称，不等于反汇编全部内部指令。

三路输出均有限。与同帧独立 GAC native MLIR FP32 参考的
`rtol=1e-3, atol=1e-4` 逐元素检查全部通过；相对 RMSE 分别约
`8.34e-7 / 2.00e-7 / 3.93e-7`。这不是新一次 PyTorch 真值或 nuScenes 检测指标验收。
混合输出对同 GPU TensorRT FP32 的 BEV/分类/框相对 RMSE 为 2.419%/0.539%/0.475%，
严格逐元素阈值未通过。图内仍使用合成全可见投影、零 CAN-bus，无历史帧或框解码，
真实几何/时序与 mAP/NDS 待验证。

## 构建失败与修正

| 尝试 | workspace / 搜索级别 / tactic DRAM | 结果 |
|---|---|---|
| 1 | 1024 MiB / 3 / 默认 | 构建 OOM；请求 1,247,040,000 B |
| 2 | 512 MiB / 0 / 8192 MiB | 同一请求处 OOM |
| 3 | 4096 MiB / 3 / 默认 | 成功构建并执行；构建进程约 162.4 秒 |

前两轮整卡显存采样峰值仅为 2948/1134 MiB。日志中的 OOM 不能直接证明 24GB 显存已满，
也不能当作 FP32 推理显存需求。扩大 workspace 后完成构建；不再把该现象称为“3090 放不下 FP32”。
原始配置、退出码和采样在 `attempt-1.json`、`attempt-2.json`、`attempt-3.json` 与对应 CSV/日志。

证据：[运行](run.json)、[构建](full-build.json)、[比较汇总](comparison-summary.json)、
[FP32 对独立参考](fp32-vs-native.json)、[混合对 FP32](mixed-vs-fp32.json)。
模型、输入和 engine 的 SHA-256 在报告中绑定；原始 engine/outputs 保留于云端
`/root/gac-tensorrt-20261017/results/fp32-workspace-01/`，没有用估算值替代失败实验的 FPS。

## 复现

```sh
cd /root/gac-tensorrt-20261017
NVIDIA_TF32_OVERRIDE=0 venv/bin/python tools/tensorrt/run_engine.py build \
  --onnx models/full.onnx --precision fp32 --plugins-fp32 \
  --engine results/fp32-workspace-01/full.engine \
  --report results/fp32-workspace-01/full-build.json \
  --workspace-mib 4096 --builder-optimization-level 3 --max-aux-streams 0 --verbose
NVIDIA_TF32_OVERRIDE=0 venv/bin/python tools/tensorrt/run_engine.py run \
  --engine results/fp32-workspace-01/full.engine --inputs models/held-out-images.npy \
  --output results/fp32-workspace-01/heldout --warmup 5 --repeat 20
```

重新运行须使用新输出目录，避免覆盖已有证据。

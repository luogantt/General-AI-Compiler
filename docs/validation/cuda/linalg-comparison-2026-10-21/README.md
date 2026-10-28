# BEVFormer-base：Linalg 与原 CUDA 路径对照

2026-10-21 实测。三种路径都执行完整固定 cold-frame 图，三路最终输出均通过同一份 PyTorch 参考验证。

本报告分别比较模型编译、首次加载/JIT、预热后执行。以下为本次短时实验，频率未锁定，不能将中位数比值当成固定硬件加速比。

| 路径 | 模型编译 ONNX→PTX / 秒 | 首次模块加载/JIT / 秒 | 预热后每帧 / 秒 | 推理样本范围 / 秒 |
|---|---:|---:|---:|---:|
| 旧标量路径 | 87.351 | 17.012 | 20.824 | 17.619–30.084 |
| 新 Linalg 路径 | 96.342 | 17.037 | 20.108 | 17.148–22.311 |
| 旧优化路径 | 100.517 | 22.380 | 4.362 | 4.299–4.422 |

表中均为中位数。模型编译在 Core Ultra 5 125H / WSL2 上完成；加载和执行在 RTX 3070 Ti Laptop 8GB / SM86、驱动 580.178.04 上完成。两个阶段不是在同一台主机上测量，但每个阶段的三条路径使用同一台主机。

## 本次结论

- 引入 Linalg 没有缩短本次模型编译：约96.34秒，旧标量为87.35秒；首次模块加载/JIT均约17秒。
- 未测得稳定的整网推理提升。Linalg与旧标量的样本范围重叠明显；频率随负载变化，不宜用20.11/20.82秒的中位数差作加速结论。末次标量三帧为21.997→24.917→30.084秒，Linalg也逐帧变慢；高负载温度最高89°C、SM频率最低450MHz。本次是预热后样本，并非已达到热稳定状态的基准。
- 旧优化路径本次最快，六个样本为4.30–4.42秒/帧，约0.23 FPS；首次加载/JIT更长，约22.38秒。这个结果不能与历史6.43秒直接比较后宣称新增优化收益。
- 当前 Linalg 提供结构化变换基础，仍需实现与迁移实际的分块、融合、线程映射和协作归约。此次测量不能证明仅更换IR便能提高速度，也未达到30 FPS。

若把驱动初始化、Context/Stream建立、PTX读取、模块加载、Buffer分配和Kernel查找一起算，三条路径初始化总计分别约17.302、17.390、22.729秒。该总计已经包含表中的加载/JIT，不能再次相加；输入和权重上传仍另计。

## 测量边界

- 模型：ResNet-101＋DCNv2，四级 FPN，200×200 BEV，六层 Encoder，FP32。固定随机图像张量 `[1,6,3,928,1600]`，固定合成标定；输出 `bev_embed`、`all_cls_scores`、`all_bbox_preds`。数值检查不等于真实 nuScenes 数据集检测精度验证。
- 重复同一 cold-frame 图。预热 GPU 不等于加入上一帧 BEV 的历史帧推理；不包含图像预处理、输入/输出传输和框后处理。
- 三条路径使用同一 C++ 编译器、ONNX、859 个相同常量文件、libdevice、LLVM 20.1.8、sm_86/PTX80。禁用 FMA contraction。
- 旧优化路径为 `MatMul=auto + reduction=cooperative + Conv channel8`。新 Linalg 尚未迁入这些调度；旧标量路径用于区分 IR 接入与既有优化的影响。
- 编译含 ONNX 导入、MLIR、权重导出、LLVM 翻译/链接/O2 和 PTX 生成；不含 C++ 编译器构建、ONNX 导出、网络传输、哈希和 Driver JIT。OS 文件缓存未清空。首轮标量与 Runtime 构建重叠，已排除并补测；每种保留两个有效样本。
- GPU 顺序为标量→Linalg→优化→优化→Linalg→标量。每次新进程、禁用驱动磁盘缓存、EAGER 加载；每进程预热1次、计时3次，每条路径共6个计时样本。
- 首次加载/JIT列为 `cuModuleLoadDataEx` API 墙钟时间，包含模块加载；驱动报告的 `CU_JIT_WALL_TIME` 可能为0，不能据此断言没有JIT。Context创建、Buffer分配和Kernel查找分别记录，不与模块加载重复相加。
- 推理使用 CUDA Events 包围整个 dispatch 序列，包含主机提交间隙；没有开启逐 Kernel profiling/同步。另保存逐帧同步主机墙钟时间。

## 数值与运行状态

六个进程的最终输出均逐元素检查 `abs(actual-reference) <= 1e-4 + 1e-3*abs(reference)`；同时检查shape、dtype及有限值。每条路径前后两次的最终输出哈希一致。没有逐帧下载每个中间计时样本的输出。

| 路径 | dispatch 数 | arena / MiB | 高负载采样温度 / °C | 高负载采样 SM 频率 / MHz |
|---|---:|---:|---|---|
| 旧标量路径 | 3224 | 1482.91 | [59.0, 89.0] | [450.0, 1530.0] |
| 新 Linalg 路径 | 3752 | 1563.40 | [67.0, 89.0] | [915.0, 1530.0] |
| 旧优化路径 | 3224 | 1482.91 | [67.0, 86.0] | [1200.0, 1545.0] |

高负载范围来自每秒采样且 GPU utilization≥90% 的时点，包含预热；不是每个Kernel的频率轨迹。每轮前尝试降到55°C，最长等待60秒，未达到阈值的情况保留在日志。arena仅为静态中间张量区，不等于总显存或峰值显存。

## 代码与原始数据

- [汇总与输入/产物身份](summary.json)
- [全部编译阶段计时（含排除样本）](compile-timings.json)
- [全部GPU样本、启动阶段、设备状态](gpu-timings.json)
- [逐输出数值比较](numerical-validation.json)
- [部署哈希核对](deployment-validation.json)
- [编译基准脚本](../../../../native/tests/benchmark_cuda_lowering.py)
- [设备基准脚本](../../../../native/tests/benchmark_cuda_device.py)
- [CUDA Runtime 分阶段计时](../../../../native/runtime/cuda-run.cpp)

完整未压缩报告和GPU逐秒JSONL在本地 `build/linalg-cuda-comparison/remote-reports`；服务器实验目录为 `/home/robot/project/gac-linalg-cuda-20261021`。

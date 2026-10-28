# 拆分前的历史性能记录

本文件保留原 TensorRT 路径在 **2026-10-17** 的实测摘要，便于理解预期范围。它不代表独立源码包在新设备上的测试结果。

## RTX 3090 24 GB

模型为完整固定形状 BEVFormer-base cold raw-head 图，六相机输入 `[1,6,3,928,1600]`、200×200 BEV。Python 3.11.15，TensorRT 10.13.3.9，PyTorch 2.7.0+cu126，驱动 580.126.18；预热 5 次，测量 20 次。

| 指标 | FP32 | 混合 INT8/FP16 |
|---|---:|---:|
| CUDA event 延时中位数 | 373.821 ms | 204.073 ms |
| P95 | 377.045 ms | 206.887 ms |
| 中位延时倒数 | 2.675 FPS | 4.900 FPS |
| workspace | 4096 MiB | 1024 MiB |
| builder 优化级别 / aux streams | 3 / 0 | 3 / 0 |
| 推理期间整卡显存采样峰值 | 4348 MiB | 2570 MiB |

显存每 250 ms 采样一次，包括其他进程/基线占用，可能漏掉瞬时峰值；不是进程独占分配量，也不能作为构建显存需求。混合图历史构建期间整卡采样峰值达 9346 MiB，明显高于运行时。

计时不含输入预处理、H2D/D2H、加载、构建、预热和最终框解码；没有 CUDA Graph replay。CUDA event 区间可能包含主机提交产生的设备空闲，不等于逐个 kernel 的纯执行时间之和。FPS 是延时倒数，不是多帧流水吞吐测试。

FP32 与同帧独立原生编译器 FP32 参考在 `rtol=1e-3, atol=1e-4` 下通过逐元素比较。混合图相对于同 GPU TensorRT FP32 的 BEV/分类/框相对 RMSE 约 2.419%/0.539%/0.475%，未通过严格逐元素门限。误差不直接等同于检测精度下降百分比。

两组使用同一模型和真实 mini_val 图像输入，但投影仍是图内固定合成几何、CAN-bus 为零，无历史帧。没有真实几何/时序与 nuScenes mAP/NDS 验收。workspace 和 tactic 不同，不将约 1.832 倍的速度差全部归因于位宽。

## RTX 3070 Ti Laptop 8 GB 的历史范围

历史机器上完成过 backbone 混合精度校准与执行，backbone 中位延时 146.72 ms。**该结果没有包括 FPN、Transformer 和检测头，不能作为完整 Base 的 FPS。** 当时完整图构建失败，必须由新一次完整构建、运行及报告确认当前结果。

## 原始证据来源

原项目提交 `cf34273c6d1853a302bbfc24c59d4afce46fbd16`；历史报告保存在：

- [General-AI-Compiler TensorRT 验证目录](https://github.com/luogantt/General-AI-Compiler/tree/main/docs/validation/tensorrt/bevformer-base-2026-10-17)
- 其中 `cloud-3090/run.json`、`cloud-3090/full-build.json`。
- `cloud-3090-fp32/run.json`、`cloud-3090-fp32/full-build.json`、`cloud-3090-fp32/comparison-summary.json`。

独立项目的模型资产、结果哈希和测试环境以其本次交付清单及运行报告为准。不能因为沿用了相同代码就把历史速度标成重新实测。

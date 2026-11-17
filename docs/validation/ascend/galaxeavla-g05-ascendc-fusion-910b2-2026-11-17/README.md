# GalaxeaVLA G0.5 LIBERO：Ascend C Q/K 融合实测

2026-11-17，单张 Ascend 910B2 的完整 Vision/VLM + FM10 推理达到 **p50 135.22 ms，p95 135.70 ms**。同一会话重新测量原 matrix RoPE + native Norm 组合为 **153.48 ms**；本次融合使整模型 p50 降低 **18.26 ms / 11.90%**，Action Expert FM10 p50 降低 **25.44%**。这是两组独立完整请求的实测结果，没有拼接阶段最小值。

用户 RTX 3090 的归档结果为 **120.48 ms**。本次仍高出 **14.75 ms / 12.24%**，尚未达到目标。相对历史 native-noise eager 1011.55 ms，本次为 **7.48×**；同轮融合收益以本页的 153.48 ms 对照计算，不使用前一天的 155.57 ms 代替。

## 测量范围与对照

| 配置 | Prefill p50 | FM10 p50 | 完整模型 p50 | 完整模型 p95 |
| --- | ---: | ---: | ---: | ---: |
| 2026-11-16 原组合，历史记录 | 83.36 ms | 72.18 ms | 155.57 ms | 156.79 ms |
| 本轮 matrix RoPE + native Norm 对照 | 82.07 ms | 71.34 ms | 153.48 ms | 154.41 ms |
| 本轮 Ascend C Q/K Norm + RoPE + 布局融合 | 81.97 ms | 53.19 ms | **135.22 ms** | **135.70 ms** |
| 用户 RTX 3090 BF16 + Inductor AE，已有记录 | 75.42 ms | 45.03 ms | 120.48 ms | 125.33 ms |

新两组均为 5 次预热、20 次正式测量，`timing-noise=native`，计入原设备 `torch.randn`。计时覆盖已准备的设备输入 → 双相机 Vision/VLM prefill → 完整 10 步连续 FM → 归一化 `[1,32,20]` 动作；不含外层图像加载/预处理、H2D、输出 D2H、反归一化、网络、冷编译及独立精度请求。这是 LIBERO checkpoint 的完整连续动作推理路径，无 CoT/离散 AR；不与三相机 R1 Lite 双头路径混用。

官方 `g05-libero`、真实 episode 0/frame 0、双 256×256 相机、159 tokens、batch 1、32 个动作位置及 10 步 FM 保持一致。权重为 BF16 常驻并保留官方 FP32 区域，总参数 **2,859,995,988**，参数存储 **6,099,991,888 bytes**。融合版正式阶段峰值 allocated 为 **6,638,633,472 bytes（约 6.18 GiB）**，reserved 为 **7,105,150,976 bytes（约 6.62 GiB）**。

两组 `comparison_identity` 完全相等，含 checkpoint、sidecars、原始输入、tokens、共享验证噪声的 SHA-256 及模型契约；参数 dtype/数量和 benchmark 源文件哈希也相等。两组都是先独立加载、编译、验证，再正式计时，先候选后对照，并非随机交错的多轮 A/B 统计试验。冷请求分别为 **152.70 s（融合）**和 **138.39 s（对照）**，均未计入上表。

- [融合版完整报告](libero-bf16-native-ascendc-qknorm-v4/report.json) · [运行日志](libero-bf16-native-ascendc-qknorm-v4/run.log)
- [同轮对照完整报告](libero-bf16-native-matrix-control-20261117/report.json) · [运行日志](libero-bf16-native-matrix-control-20261117/run.log)
- [机器可读对比及逐项精度](experiments/full-model-comparison.json)
- [历史优化与 3090 原始证据](../galaxeavla-g05-optimization-910b2-2026-11-16/README.md)
- [可切换配置的离线交互图](../../../visualizations/g05-ascend-optimization.html)

每组目录也保存共享噪声下的 `action.npy`、`eager_action.npy`、`noise.npy`、`input_ids.npy` 和 padding mask。它们用于数值验证；正式计时使用原生随机数。

## 融合做了什么

新增 Ascend C 算子 `GacQkNormRope`，由 Torch operator `gac_g05::qk_norm_rope` 和 TorchAir GE converter 接入。24 个 Action Expert 层，每层 Q/K 各一次、FM 共 10 步，整请求对应 **480 次融合调用**。输入直接读取 Q/gate 打包投影的 Q 区域或 K 投影，在一个 Vector kernel 内完成 RMSNorm、原有 BF16 舍入、前 64 维 partial RoPE 和连续 BHSD 输出布局。

```text
显存 GM：Q/gate 或 K 投影、FP32 norm weight、当前 cos/sin
    ↓
片上 UB：FP32 RMSNorm → BF16 RNE → FP32 partial RoPE
         同核各行复用已读入的 1 + weight
    ↓
显存 GM：连续 BF16 [1,H,32,256]，直接供后续路径消费
```

本核使用的是 **UB**。L1 和 L0A/L0B/L0C 属于 Cube 矩阵计算相关的数据通路，本轮没有编写 Cube GEMM，也没有宣称 L0/L1 驻留、双缓冲或把模型权重全放入 SRAM。权重、输入和最终输出仍访问设备显存。关于 A2 的存储层次，可参阅 [Ascend C 官方硬件架构](https://asc.gitcode.com/guide/programming_guide/advanced_programming/hardware_implementation/architecture_spec/npu_arch_2201.html)。

保留 Q gate、V 投影、KV 拼接、attention、MLP、残差、49 个 Adaptive norm、VLM 和动作步数。已有 GE/ACLGraph、Triton GDN、fused attention、时间条件预计算、固定 Vision 几何和 functional MRoPE 继续使用。Q/K 原来的独立 native Norm 和 matrix RoPE 被本核接管；并未把整个模型重写为 Ascend C。

必须保留 RMSNorm 后的 BF16 RNE 舍入再转回 FP32 的边界。RoPE 的两个乘法与相加不合为 FMA，构建设置 `-ffp-contract=off`。v4 使用 `Sqrt` 再 `Div` 计算 inverse RMS，修正早期直接 `Rsqrt` 的精度问题；FP32 归约树仍可能不同，不能声称逐位等价。官方 [Rsqrt 精度说明](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/82RC1alpha001/API/ascendcopapi/atlasascendc_api_07_0030.html)也建议高精度场景考虑 Sqrt/Div。

这组结果证明当前 Q/K 路径的碎片化计算可以通过 Ascend C 融合提速。它没有单独比较等价的 Triton 与 Ascend C 融合核，因此不能据此把整个瓶颈归因于 Triton-Ascend，或认定手写 Ascend C 在所有算子上必然更快。

## Trace：减少了哪些调用

正式计时后各采集一个独立 Level1 请求，`wait=0,warmup=0,active=1,repeat=1`。以下为原始 `kernel_details.csv` 的 Type 精确聚合：

| Type | 同轮对照次数 | 融合版次数 | 对照 duration 合计 | 融合 duration 合计 |
| --- | ---: | ---: | ---: | ---: |
| 所有记录 | 19,206 | 15,826 | 141.792 ms | 123.462 ms |
| GacQkNormRope | 0 | 480 | 0 | 4.730 ms |
| RmsNorm | 490 | 10 | 5.347 ms | 0.115 ms |
| RotaryPositionEmbedding | 480 | 0 | 6.220 ms | 0 |
| Cast | 3,476 | 2,516 | 16.071 ms | 11.294 ms |
| Transpose | 1,659 | 1,179 | 18.213 ms | 14.968 ms |
| InplaceAddRmsNorm | 480 | 480 | 3.192 ms | 2.313 ms |

减少 **3,380 条记录（17.60%）**。两组仍各有 **4 个 AI_CPU** 任务。未改动类型的 duration 也可能随调度变化，不能把每个差值独立认作优化收益；算子时长可能重叠，合计不是请求 wall time。

[完整按类型/核心聚合](experiments/kernel-type-comparison-matrix-vs-ascendc.json)保存远端原 CSV 的路径、字节数和 SHA-256；[聚合脚本](experiments/summarize_kernel_traces.py)可重算。原大型 profiler 文件保留在报告记录的服务器目录，本仓库保存聚合结果。仍有显著的 transpose/cast 和矩阵相关工作，后续应根据这些剩余阶段继续定位，而不是只按峰值 BF16 吞吐推算。

## 数值与动态输入验证

整模型采用已有工程门槛：有效 7 维归一化动作的 relative-L2 ≤ 1%、max-abs ≤ 0.05，并要求有限值及上下文匹配。这些门槛没有因新核而放宽，不代表机器人闭环成功率。

融合版完成 A/B/A/B、状态 C、任务 D、外部相机 E、腕部相机 F 各两次，共 **12 次**同设备 eager 对照。最坏 relative-L2 为 **0.002466373546（0.246637%）**，max-abs 为 **0.004699468613**，均出现在 E；padding 为 0，C/D/E/F 的 tokens/mask 均匹配对应 eager。

动作符号并非全部相同：B 的两次和 D 的两次为 **223/224 = 99.55357%**，其他检查为 100%；夹爪符号全部一致。保存的 A 动作相对 eager 的 L2 为 0.131553%，max-abs 0.00252950。相对历史 155.57 ms 保存动作，A 的 L2 为 0.0851872%，max-abs 0.00123590。同轮原组合的 12 次最坏误差为 0.251121% / 0.00458613。没有跨 CUDA 逐元素对齐或闭环任务验收。

独立算子探针覆盖 Q 8 头/K 2 头、RoPE 长度 32/159 四种组合，改变输入、weight、cos/sin，包含零/小/大输入与权重、gate-only、未消费后缀和返回 A 回放。Ascend C eager/GE 共 **208 个输出、416 次对两个 eager 参考的比较**通过原门槛；最坏 relative-L2 **1.27844e-5**、max-abs **0.00390625**。输入未修改，每个 GE shape 在动态输入/weight 回放中只编译一次。探针动态权重使用 `frozen_parameter=False`，整模型固定权重使用 `True`。

观察性 original GE/native GE 控制组的部分压力用例存在偏差，原始 `passed=false` 明细保留，不作为 eager 参考的替代，也不称全部路径通过。早期复用缓存导致的候选失败也保留：切换新核后首次 Q32 GE 仍出现旧核误差，独立新缓存/新进程的对应检查恢复。两份 JSON 未记录相同输入张量逐字节哈希，不把它们描述成独立证明的输入字节一致实验。

- [最终独立缓存微实验](experiments/qknorm-micro-v4-fresh.json) · [日志](experiments/qknorm-micro-v4-fresh.log)
- [旧缓存失败报告](experiments/qknorm-micro-v4-old-cache.json) · [独立 GE 缓存检查](experiments/ge-fresh-cache-check.json)
- [微实验计数、来源哈希与证据边界](experiments/micro-summary.json)

## 构建与复现

实测环境：Ubuntu 22.04 aarch64、Ascend 910B2 / A2、CANN 9.0.0、Python 3.11.15、PyTorch 2.7.1、TorchNPU 2.7.1.post4、Triton-Ascend 3.2.1。原上游 revision、overlay diff、helper 哈希及资产 SHA 全部见完整报告。

先按 [Ascend C 算子构建说明](../../../../tools/galaxeavla/ascendc_qknorm_rope/README.md)构建 OPP 和 C++ bridge，使用独立安装目录。设置 `ASCEND_CUSTOM_OPP_PATH`、`GAC_QKNORM_ROPE_LIBRARY`，每次更换候选新建 `ASCEND_CACHE_PATH` 并启动新 Python 进程；保留 CANN 环境脚本提供的 `PYTHONPATH`。不要把旧缓存或旧进程中的同名算子视为新核已生效。

本轮归档命令：[融合运行脚本](experiments/run-full-ascendc-v4.sh)、[同轮对照脚本](experiments/run-matrix-control-20261117.sh)。脚本中的服务器资产、源码和输出路径需按实际位置修改，并先激活 CANN/venv；输出目录选用新路径。核心差异是：

```bash
# 融合版：在共同参数后添加
--ae-qknorm-rope ascendc --ae-norm npu --functional-mrope

# 对照版：替换上述 Q/K 融合参数
--ae-rope matrix --ae-norm npu --functional-mrope
```

两组共同参数包括 BF16 resident、AE GE/VLM ACLGraph frozen、Triton GDN、NPU attention、`--precompute-time-conditioning --time-conditioning-lookup request --static-vision`、跳过未消费 codec、`--verify --timing-noise native --warmup 5 --iterations 20`。`--ae-qknorm-rope ascendc` 默认关闭，要求验证，和独立 `--ae-rope matrix/npu` 互斥。没有静默 fallback。

[OPP 源码清单](experiments/opp-source-manifest.json)、[bridge 构建清单](experiments/bridge-build.json)及[运行环境补充清单](experiments/runtime-manifest-v4.json)记录 SHA-256。补充清单记录实际 `CustomFunctions.h`、已安装 kernel/object、tiling/opapi 和 bridge；旧 bridge 清单的头文件元数据未回写。文件哈希用于追溯选定文件，并非独立动态加载跟踪。

本地 `python -m unittest discover -s tests -p 'test_g05*.py'` 共 **161 项通过**，包括 Python 契约/实例恢复、CLI 组合、探针验收分组和可视化构建检查；本地测试不代替上面的 NPU 硬件验证。17 配置页面通过 Chrome/Playwright 的桌面及 390px 窄屏检查，配置切换、阶段分位数、符号一致率、JSON 下载与离线运行正常，无 JavaScript 错误或页面横向溢出。[浏览器验收记录](experiments/visualization-qa.json)

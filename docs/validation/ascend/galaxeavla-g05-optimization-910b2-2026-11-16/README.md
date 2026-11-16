# G0.5 在 Ascend 910B2 上的优化证据，2026-11-16

本次优化以用户已经验证的 RTX 3090 **120.4772 ms** 为目标。当前已归档的最快 Ascend 结果为 **155.5743 ms p50、156.7861 ms p95**：组合使用 BF16 GE AE / ACLGraph VLM、Triton GDN、fused AE attention/Norm、FP32 matrix partial RoPE、functional MRoPE、未消费 codec 消除、request 时间条件预计算和固定几何 Vision 图。正式计时采用 **native 设备随机数**，相对同模式 eager **1011.5460 ms** 加速 **6.5020×**，不与早期 shared-noise 基线混算。A/B/C/D/E/F 共 **12 次**同设备核验最坏 relative-L2 **0.2511210%**，最大绝对误差 **0.00458613**；不宣称位级一致或闭环任务准确率。163.9053/172.4950/200.6956 ms 对照、single_stream 的 173.0910 ms 无收益实验及历史 shared-noise 原公式 **297.8925 ms** 的零误差版本继续保留。仍未达到 120 ms 目标；跨 CUDA/NPU 数值一致性和闭环任务成功率尚未验证。

3090 原始数据来自用户 PPT 配套的本地 GalaxeaVLA 实验仓库，按字节复制到 [reference-3090](reference-3090/provenance.json)。原数据未修改，本次任务没有重跑这些历史 CUDA 实验。[120.4772 ms 原始 JSON](reference-3090/bf16resident_steps10_compileae.json)、[A/B 汇总](reference-3090/OPTIMIZATION_RESULTS.json)和[121.7879 ms 独立复测](reference-3090/bf16_compile_recheck.json)分别保留。

**固定的工作负载与计时边界**

| 项目 | 对齐条件 |
|---|---|
| 模型 | `OpenGalaxea/G05/g05-libero/model.pt`；2,859,995,988 个参数 |
| 样本 | LIBERO `libero_goal_no_noops_lerobot`，episode 0 / frame 0 / global index 0 |
| 指令 | `open the middle drawer of the cabinet` |
| 视觉输入 | 原始 `image.png`、`wrist_image.png` 各 RGB 512×512；官方 processor 转为两相机 256×256 |
| 状态 | 原始 `observation.state[0:6]` 为 `right_ee_pose`，`[6:7]` 为 `right_gripper`；频率 20 Hz |
| VLM 序列 | batch 1，159 tokens：128 image + 1 proprio + 30 text |
| 动作 | 连续 FM，10 次 Euler，`pi_convention`；输出 `[1,32,20]` |
| 有效动作维 | `[10,11,12,13,14,15,19]`；其余 13 维是 padding |
| 生成分支 | `predict_cot=false`、`discrete_action=false`、`continuous_action=true` |
| 精度 | BF16 常驻权重 + BF16 autocast；官方 `apply_fp32_params()` 保留规定的 FP32 参数 |
| 统计 | 5 次 warmup，20 次正式测量，设备同步后读 wall clock；报告 p50 / p95 |

总时间从已经准备并传入设备的 batch 开始，覆盖 `policy.prefill`（Vision + VLM）和 `policy.generate_action`（完整 FM10），到归一化动作块产生为止。外层图像读取/预处理、batch H2D、输出 D2H、反归一化、网络和机器人执行时间不在该范围。`prefill` 内部的 tokenizer、`encode_inference`、`build_proprio_batch` 及其设备传输仍在计时内，与原 3090 脚本一致。两个阶段各自的 p50 不要求相加等于总时间 p50。

3090 的 120.4772 ms 使用 Inductor `reduce-overhead`，只编译 `action_expert.forward`；首次 warmup 的 16.536 秒编译开销单列。早期 Ascend 基线使用 **shared noise**：CPU 预先生成并保存 float32 `[1,32,20]` 噪声，每次 FM 内计入一次设备 `clone`；历史 CUDA 测试计入的是设备 `torch.randn`。现在已补齐 `--timing-noise native` 的 Ascend eager 与优化测量，正式计入原设备随机数路径。两种模式分别归档，优化倍数只在相同模式内计算；native 边界对齐不代表 CUDA/NPU 产生了相同随机张量或全部算子实现相同。

`noise.npy`、归一化 `action.npy`、padding mask 和输入身份用于数值对照。相同整数 seed 不能代替跨 CUDA/NPU 共享同一噪声张量。即使选择 native 计时，新 runner 保存的 `action.npy` 也来自计时外的独立 shared-noise 请求，并与报告中的噪声 SHA 对应。

**Native 随机数计时：当前正式对照**

| Ascend 910B2 BF16 配置 | Prefill p50 (ms) | FM10 p50 (ms) | Total p50 (ms) | Total p95 (ms) | 证据 |
|---|---:|---:|---:|---:|---|
| Eager，native | 265.2891 | 746.7601 | 1011.5460 | 1022.6976 | [report.json](libero-bf16-eager-native/report.json) |
| GE AE / ACLGraph VLM + triangular + 未消费 codec 消除，native | 119.0303 | 100.0980 | 219.1758 | 221.8684 | [report.json](libero-bf16-ge-acl-native/report.json) |
| 上一行 + exact 时间条件预计算，native，性能退化 | 117.1958 | 111.3011 | 228.4562 | 229.2494 | [report.json](libero-bf16-ge-acl-native-time/report.json) |
| GE AE / ACLGraph VLM + Triton GDN + fused AE + 未消费 codec 消除，native | 103.0380 | 97.6717 | 200.6956 | 203.0980 | [report.json](libero-bf16-native-triton-fused/report.json)、[日志](libero-bf16-native-triton-fused/run.log) |
| 上一行 + request 时间条件预计算 + 固定几何 Vision 图，native | 82.7173 | 89.7357 | 172.4950 | 173.5759 | [report.json](libero-bf16-native-triton-fused-time-vision/report.json)、[日志](libero-bf16-native-triton-fused-time-vision/run.log) |
| 172.4950 ms 配置 + GE single_stream，native，未提速 | 83.4088 | 89.6339 | 173.0910 | 173.5825 | [report.json](libero-bf16-native-triton-time-vision-single/report.json) |
| 172.4950 ms 配置 + AE FP32 matrix partial RoPE，native | 82.6356 | 81.1980 | 163.9053 | 164.6181 | [report.json](libero-bf16-native-vision-rope-matrix/report.json) |
| 上一行 + fused AE Norm + functional MRoPE，native | 83.3635 | 72.1844 | **155.5743** | **156.7861** | [report.json](libero-bf16-native-vision-rope-norm-mrope/report.json)、[日志](libero-bf16-native-vision-rope-norm-mrope/run.log) |

这八份报告均为 5 warmup + 20 次测量，其 `comparison_identity` 与同输入历史参考相同。最新同模式倍数为 `1011.5459698718041 / 155.574289849028 = 6.5020124524×`；163.9053/172.4950/200.6956 ms 对照的倍率分别保留为 6.1715×/5.8642×/5.0402×。正式 `native` 请求不使用固定噪声替代 `randn`；数值验证与保存的动作另用固定 shared noise。最新最短单次正式样本 **155.1270 ms**，不能把它写成 p50。

200.6956 ms 对照及后续组合均保留同一 TorchAir 包中的 GE AE / ACLGraph VLM，并开启 frozen 参数；GDN 逆矩阵由 `gac_g05::row_inverse32_fp32` 执行 **31 次按行递推再加 I**，只允许 NPU FP32 contiguous `[N,32,32]`，关闭 FP fusion，不使用 `tl.dot`、TF32/HF32 或 doubling。行内 `tl.sum` 的归约树与原 Torch 可以不同。AE 的 24 层注意力使用 `torch_npu.npu_fusion_attention`，保留 2→8 KV repeat、scale `0.0625`、非因果 mask 和 BF16/FP32 区域；每个请求的 mask 验证开销计入 FM10。200.6956/172.4950 ms 对照保留原 RoPE，163.9053 ms 另行启用下述 matrix 变体；它们的原 QK norm 在最终 155.5743 ms 组合中才替换为保持 FP32 数学的 fused Norm。组合收益由完整报告证明，不能把局部微实验比例相加。

200.6956 ms 对照的 A/B/A/B 最坏 relative-L2 为 **0.0019247327（0.1924733%）**、max-abs **0.0029824376**；C/D 各两次检查的最坏 relative-L2 **0.0015413666（0.1541367%）**、max-abs **0.0031360686**，编码 tokens/mask 与未修改 eager 一致。该对照最低动作符号一致率 **223/224（99.5536%）**，gripper 100%，padding 0。所有这些检查使用共享显式噪声；[action.npy](libero-bf16-native-triton-fused/action.npy)与[noise.npy](libero-bf16-native-triton-fused/noise.npy)一并归档，`external_comparison=null`。

200.6956 ms 对照的首次含编译请求 **136.4497 秒**、模型加载 **69.0516 秒**另列；正式测量峰值 allocated **6,679,006,208 字节**。计时后单次 trace 使用 `Level1` 和显式 `wait=0,warmup=0,active=1,repeat=1`，不把 trace 开销计入正式样本。该对照未启用时间条件预计算或固定 Vision 图，下面的新结果独立归档。

**172.4950 ms 对照：request 时间条件与固定几何 Vision 的完整组合**

[该对照完整报告](libero-bf16-native-triton-fused-time-vision/report.json)为 `status=success`，p50 **172.4950 ms**、p95 **173.5759 ms**，prefill **82.7173 ms**、FM10 **89.7357 ms**。相对同模式 200.6956 ms 对照总延迟下降 **14.0514%**；这是时间条件与 Vision 同时启用的组合测量，尚未单独拆分两项的整模型贡献。

时间条件使用 `lookup_mode=request`：固定源文件的原 `FMHelper.infer` 请求内推进游标，每请求检查一次契约，严格消费原 FP32 `t -= delta_t` 的 10 个时间值；没有逐步 D2H 查表，也没有用 `linspace` 更换时间值。仍保留 49 个独立 AdaLN conditioning 投影和动态 RMS/affine/cast 路径。报告记录 **39 次请求、39 次完整结束、0 次失败、390 次时间条件调用**；表大小 6,021,120 字节，建立一次耗时 **121.7454 ms**，不与稳态时延混合。

Vision 使用 frozen TorchAir ACLGraph，固定 grid 为 `[[1,16,16],[1,16,16]]`，输入 patches `[512,1536]`，输出 `[512,1024]` 与 `[128,2048]`。只缓存插值索引/权重、RoPE 和 `cu_seqlens=[0,256,512]`，学习位置嵌入在图内读取；每请求在图外核验当前几何，核验仍计入请求时间。24 层各自保持 **两次独立 256-token SDPA**，不形成跨相机的 512-token attention，patch/merger 的 FP32 区域保持。记录 39 次真实 Vision 调用；图像、状态、KV、动作与模型结果均不跨请求缓存。

| 12 次有效 7 维核验 | 最坏 relative-L2 | 最大绝对误差 |
|---|---:|---:|
| A/B/A/B：原样本与图像/噪声变化 | 0.0017382901（0.1738290%） | 0.0029405057 |
| C/D 各两次：状态与任务变化 | 0.0018082610（0.1808261%） | 0.0033214688 |
| E 两次：仅外部相机变化 | **0.0026672077（0.2667208%）** | **0.0048633516** |
| F 两次：仅 wrist 相机变化 | 0.0014910638（0.1491064%） | 0.0027596802 |

12 次结果均有限，动作和 gripper 符号一致率均 **100%**，padding 为 **0**；C/D/E/F 的 tokens/mask 与各自未修改 eager 参考一致。E/F 的 eager 参考相对 A 明确可区分（relative-L2 分别 0.2180088 与 0.0197201），因此相机验证确实改变了输入。所有比较都在正式计时外使用共享显式噪声，正式计时只使用原始 A 与 native 随机数。[该对照 action.npy](libero-bf16-native-triton-fused-time-vision/action.npy)、[noise.npy](libero-bf16-native-triton-fused-time-vision/noise.npy)保留；`external_comparison=null`，不能表述为跨卡一致或闭环任务准确率。

本轮首次含编译请求 **138.1496 秒**、模型加载 **57.8049 秒**单列；正式测量峰值 allocated **6,638,657,024 字节**。单次 Level1 trace 位于正式测量后，不作为时延统计来源。`workload=null`，原公式含 codec 的部分 FLOPs 仍不能冒充该优化路径的执行计数。

**163.9053 ms 对照：AE matrix partial RoPE 完整实测与单流对照**

[matrix RoPE 完整报告](libero-bf16-native-vision-rope-matrix/report.json)为 `status=success`，在 172.4950 ms 配置上仅增加 `--ae-rope matrix`，实测 **163.9053 ms p50、164.6181 ms p95**，prefill **82.6356 ms**、FM10 **81.1980 ms**。使用 `torch_npu.npu_rotary_mul` 的固定非学习 FP32 256×256 旋转矩阵，保持 head256 中前 64 维 partial RoPE、其余维度直通及原 Q/K 输出 dtype；每次读取当前 cos/sin，没有缓存位置或模型结果。该报告未启用 fused Norm 或 functional MRoPE。

12 次 A/B/C/D/E/F 检查全部通过，最坏 relative-L2 **0.0026672077（0.2667208%）**、max-abs **0.0048633516**，均在 E；动作和 gripper 符号一致率 100%、padding 0、C/D/E/F 编码 tokens/mask 均与各自 eager 一致。首个含编译请求 **135.4737 秒**、模型加载 **57.5548 秒**单列，测量峰值 allocated **6,652,557,312 字节**。time lookup 完成 **38 次请求 / 380 步**、0 失败。该轮 `trace_profile=null`、`workload=null`；部分算量和其他版本 trace 不冒充该版本实测。

[GE single_stream 完整报告](libero-bf16-native-triton-time-vision-single/report.json)保持 172.4950 ms 配置的原 RoPE，只把 AE GE 的 `single_stream` 设为 true，测得 **173.0910 ms p50、173.5825 ms p95**，prefill **83.4088 ms**、FM10 **89.6339 ms**。12 次数值检查通过，但没有性能收益，故当前最佳组合不启用此选项。

**当前最快：matrix RoPE、fused Norm 与 functional MRoPE 的完整组合**

[最终组合完整报告](libero-bf16-native-vision-rope-norm-mrope/report.json)为 `status=success`。在 163.9053 ms matrix 配置上增加 `--ae-norm npu --functional-mrope`，5 warmup + 20 次 native 测量得到 **155.5743 ms p50、156.7861 ms p95**，prefill **83.3635 ms**、FM10 **72.1844 ms**；相对 matrix 对照总延迟下降 **5.0828%**。这是两项同时启用的完整组合收益，不拆分或相加局部微实验比例。

Norm 仅替换 AE 的 **49 个 AdaptiveRMSNorm._norm 与 48 个 Q/K RMSNorm.forward**，使用 `torch_npu.npu_rms_norm`，FP32 数学、原 epsilon 和输出 cast 保留；Q/K 的 `1 + weight.float()` 每次读取当前参数。functional MRoPE 只替换 VLM 第 3 层、AE 第 0 层实际调用的 rotary 实例，以两个固定布尔索引和 `torch.where` 选择当前 `[3,B,S,32]` 频率，消除原地切片写入。它不缓存 position、freqs、KV 或模型结果。源码按完整规范化文本 SHA 锁定，避免 Python 版本的 AST 空字段差异；`ListConfig[11,11,10]` 只在实例上规范为 tuple，原配置对象不变，卸载后恢复。独立[真实配置/rotary 微检查](experiments/functional-mrope-ge-acl.json)及[探针源码](experiments/probe_functional_mrope.py)覆盖 eager/GE/ACLGraph 的 **24 次 A/B/A/B**，误差全为 0，恢复后也精确一致；这不替代以下整模型检查。

| 最终 12 次有效 7 维核验 | 最坏 relative-L2 | 最大绝对误差 |
|---|---:|---:|
| A/B/A/B | 0.0019409310（0.1940931%） | 0.0033299923 |
| C/D 各两次 | 0.0018464964（0.1846496%） | 0.0034386814 |
| E 两次，仅外部相机变化 | **0.0025112100（0.2511210%）** | **0.0045861304** |
| F 两次，仅 wrist 相机变化 | 0.0014837760（0.1483776%） | 0.0024656653 |

全部结果有限，动作和 gripper 符号一致率 **100%**、padding **0**，C/D/E/F 编码 tokens/mask 与各自 eager 一致。正式计时仍只用真实 A；核验及归档 [action.npy](libero-bf16-native-vision-rope-norm-mrope/action.npy)使用 [noise.npy](libero-bf16-native-vision-rope-norm-mrope/noise.npy)对应的共享噪声。`external_comparison=null`，无跨 CUDA 或闭环任务精度结论。首次含编译请求 **153.6495 秒**、模型加载 **57.6715 秒**另列，正式测量峰值 allocated **6,652,559,872 字节**。39 次 request 全部消费 10 个时间条件，共 390 步、0 失败；`workload=null`。

最终单次 Level1 trace 仍在正式计时后采集。[155.5743 ms 算子汇总](experiments/profile-155ms-level1-summary.json)中 `kernel_details.csv` 有 **19,206 行、duration 合计 143.3601 ms**，172.4950 ms 对照是 **21,713 行、161.8965 ms**；两者都不是 wall-clock p50。[原 CSV 的 Type 精确统计](experiments/kernel-type-comparison-172-vs-155.json)保留两份 CSV 的路径、字节数和 SHA-256、全部 Type/core 计数，并显式记录零计数：`ScatterElements` **40→0**、`StridedSliceD` **2120→180**（19.1036→0.5023 ms）、`Square` **960→0**；新路径记录 `RmsNorm` **490 次 / 5.3649 ms**、`RotaryPositionEmbedding` **480 次 / 6.0656 ms**。仍有 **4 个 AI_CPU** 任务，不能宣称全部 CPU 工作消除。算子 duration 只解释执行路径变化，不能逐项相加当组合收益。本轮相关本地测试、可视化 builder 测试及 overlay 检查均通过。

**Shared-noise 历史测量与用户 3090 参考**

| 设备与配置 | Prefill p50 (ms) | FM action p50 (ms) | Total p50 (ms) | Total p95 (ms) | 证据 |
|---|---:|---:|---:|---:|---|
| RTX 3090，BF16，Inductor AE，原设备随机数 | 75.4166 | 45.0250 | **120.4772** | 125.3290 | [用户原始 JSON](reference-3090/bf16resident_steps10_compileae.json) |
| Ascend 910B2，BF16 eager，shared noise | 266.3246 | 747.0695 | **1013.8698** | 1026.2450 | [report.json](libero-bf16-eager/report.json)、[运行日志](libero-bf16-eager/libero-bf16-eager.log) |
| Ascend 910B2，BF16，TorchAir GE AE，shared noise | 257.3871 | 108.0256 | **365.3891** | 369.5766 | [report.json](libero-bf16-torchair-ae/report.json)、[运行日志](libero-bf16-torchair-ae/libero-bf16-torchair-ae-env.log) |
| Ascend 910B2，BF16，`npugraph_ex` frozen AE + VLM 原公式，shared noise | 155.0403 | 142.8234 | 297.8925 | 300.6155 | [report.json](libero-bf16-npugraph-frozen-ae-vlm/report.json)、[运行日志](libero-bf16-npugraph-frozen-ae-vlm/run.log) |
| Ascend 910B2，BF16，TorchAir frozen AE + VLM + 兼容分解，shared noise | 1490.6200 | 100.5431 | 1590.8996 | 1597.4734 | [report.json](libero-bf16-ge-frozen-compatible/report.json)、[运行日志](libero-bf16-ge-frozen-compatible/run.log) |
| Ascend 910B2，BF16，frozen graph AE/VLM + triangular + 未消费 codec 消除，shared noise | 118.1189 | 146.3754 | 264.4622 | 265.4022 | [report.json](libero-bf16-graph-triangular-context/report.json)、[运行日志](libero-bf16-graph-triangular-context/run.log) |
| Ascend 910B2，BF16，frozen GE AE / ACLGraph VLM + triangular + 未消费 codec 消除，shared noise | 117.6110 | 108.0122 | **225.5800** | **226.3952** | [report.json](libero-bf16-ge-ae-acl-vlm-triangular-context/report.json)、[运行日志](libero-bf16-ge-ae-acl-vlm-triangular-context/run.log) |

该历史表的七份原始报告与新增八份 native 报告分别保留，共十五份；不覆盖历史 CUDA、Ascend eager 或各编译变体证据。历史表中的倍率只与 shared 基线比较，不将其套到 native 行。已完成测量但性能退化的变体也保留，编译失败和 helper 微实验不进入整模型延迟表。

Ascend 基线实际环境为 Python 3.11.15、PyTorch 版本字符串 `2.7.1+cpu`，执行设备记录为 `npu:0 / Ascend910B2`。模型加载 79.679 秒、外层输入准备与传输 26.312 ms，均单列；测量阶段峰值 allocated 为 6,744,743,936 字节。当前基线的 `status=success` 表示该执行完成并通过形状/finite 检查，不表示已经通过 CUDA/NPU 数值一致性或闭环任务成功率验证。

TorchAir GE 使用 `max-autotune`，`fullgraph=true`、`dynamic=false`，仅作用于 `policy.model.action_expert.forward`；VLM 和 GDN helper 保持 eager。两份 Ascend 报告的 `comparison_identity` 完全相同，确认 checkpoint、sidecar、原始图像/状态/指令、输入 token、显式噪声、BF16 精度策略和 FM10 均一致。总时间比为 `1013.8698 / 365.3891 = 2.7748×`；FM action 阶段比为 `747.0695 / 108.0256 = 6.9157×`。这里的收益没有通过减少采样步数、相机、动作维数或更换数据获得。

TorchAir 的首次 `cold_A` 验证请求为 **78.9350 秒**，其中 action 为 **78.6838 秒**，包含惰性编译与首次执行，并非单独测得的纯编译时间；这次请求及后续 A/B/A/B 验证不计入 5 warmup 后的 20 次正式统计。模型加载另计 **136.5682 秒**，测量阶段峰值 allocated 为 **6,758,453,248 字节**。

上述 TorchAir AE 变体的同设备、同权重 eager/compiled 数值验证覆盖原始 A 与变化图像/噪声的 B，正式计时始终使用原始 A。四次比较都只统计有效的 7 维、共 224 个动作值：

| A/B/A/B 最坏指标 | 实测值 |
|---|---:|
| Relative L2 | 0.0013367668（0.1336767%） |
| 最大绝对误差 | 0.0032060146 |
| 最低 cosine similarity | 0.9999991066 |
| 动作符号一致率 / gripper 符号一致率 | 100% / 100% |
| Padding 最大绝对值 | 0 |

报告中 `reference_inputs_distinguishable=true`；A 与 B 的 eager 参考 relative-L2 为 0.1986426，确认验证确实改变了输入内容。归档的 [action.npy](libero-bf16-torchair-ae/action.npy) 对应 [noise.npy](libero-bf16-torchair-ae/noise.npy)。本次 `external_comparison=null`，上述误差来自该次运行的同设备 eager 参考，不能表述为已经完成跨卡或独立 CUDA 参考验证。

**保留原公式的零误差变体：`npugraph_ex` frozen AE + VLM**

AE 和 VLM 都使用 `fullgraph=true`、`dynamic=false`、`frozen_parameter=true`，GDN helper 未另行编译，也没有启用 triangular 算法。其 `comparison_identity` 与 eager 基线完全相同，保持相同权重、双相机真实输入、159 tokens、BF16/FP32 区域、FM10 和 shared noise。总延迟相对 eager 为 **3.4035×**，相对上一轮 TorchAir AE 的总延迟降低 **18.4725%**。本轮 prefill 降至 155.0403 ms，FM 为 142.8234 ms；FM 单阶段比 TorchAir AE 的 108.0256 ms 慢，所以不能将不同后端各自最好的阶段拼成未经测量的整模型结果。

本轮 `cold_A` 含捕获/编译的首次请求为 **153.8206 秒**（prefill 123.2242 秒、action 30.5965 秒），排除在稳态统计外。模型加载另计 **58.1048 秒**；测量阶段峰值 allocated 为 **6,592,889,344 字节**。

四次 A/B/A/B 对照的 MAE、RMSE、最大绝对误差和 relative-L2 均为 **0**，有效动作维与 gripper 符号一致率均为 100%，padding 为 0。A/B 参考可区分；这是已测试内容与形状下的同设备结果。[输出 action.npy](libero-bf16-npugraph-frozen-ae-vlm/action.npy)、[对应 noise.npy](libero-bf16-npugraph-frozen-ae-vlm/noise.npy)、[完整验证与源码身份](libero-bf16-npugraph-frozen-ae-vlm/report.json)一并保留。该报告的 `external_comparison=null`，仍未包含跨 CUDA 参考。

报告内 `module_profile` 是计时外另一次同步模块诊断，内部时间不可相加代替 297.8925 ms 正式测量。这份结果未采集 `workload`；下述 GE 兼容实验已另行采集真实 LIBERO 算量，不使用 R1 Lite 的 FLOPs 代替。

**上一轮完整组合：frozen graph + FP32 triangular + 未消费 codec 消除**

[本轮完整报告](libero-bf16-graph-triangular-context/report.json)保持相同 checkpoint、双相机真实输入、159 tokens、20D 动作、FM10 和 shared noise，`comparison_identity` 与 eager 基线相同。AE/VLM 均为 `npugraph_ex`、`fullgraph=true`、`dynamic=false`、`frozen_parameter=true`；GDN chunk 改为 FP32 三角求解，求解时关闭 autocast，仍保留原 chunk size 与其他精度策略，但**改变浮点累加顺序**。

FM-only context 优化只在保留的上下文没有 action 消费者时跳过 ActionCodec 编码。记录显示 **34 次调用实际移除 action 字段**；图片、状态和指令每次重新处理，没有输入缓存或模型结果缓存。未知 processor 被拒绝，CoT/AR 路径不允许启用；被省略的副作用仅是未消费的 codec `_last_encode_state` 更新。本次是三角求解与 context 消除的组合测量，尚不能分别归因它们各自的整模型收益。

正式 p50 为 prefill **118.1189 ms**、FM10 **146.3754 ms**、总计 **264.4622 ms**，p95 **265.4022 ms**，相对 eager **3.8337×**。首个含编译请求 **104.2415 秒**、模型加载 **58.3453 秒**均单列；测量阶段峰值 allocated 为 **6,658,791,424 字节**。

| 同设备验证 | 实际结果 |
|---|---|
| A/B/A/B 有效 7 维最坏 relative-L2 / max-abs | 0.0018828365（0.1882836%）/ 0.0034265518 |
| A/B/A/B 最低动作符号一致率 | 223/224（99.5536%）；不能写成 100% |
| Gripper 符号一致率 / padding | 100% / 0 |
| C：proprio pose +0.01；D：task open → close | 各重复两次，input_ids 和 attention_mask 均与原 eager 一致 |
| C/D 动作最坏 relative-L2 / max-abs | 0.0015558859（0.1555886%）/ 0.0031617507 |

所有验证均在正式计时外，正式计时只用原始 A。最坏误差通过声明的 regression gate，但不等于逐元素一致；`external_comparison=null`，没有跨卡或闭环结论。[action.npy](libero-bf16-graph-triangular-context/action.npy)、[noise.npy](libero-bf16-graph-triangular-context/noise.npy)保留。该变体 `workload=null`，下述包含 codec 的原公式计数不能冒充此路径的执行算量。

**Shared-noise 最快归档：同一 TorchAir 包的 GE AE + ACLGraph VLM 混合后端**

[完整报告](libero-bf16-ge-ae-acl-vlm-triangular-context/report.json)保持上一轮的真实输入、BF16/FP32 策略、FM10、triangular/context 优化和共享噪声，`comparison_identity` 与 eager 相同。AE 使用 `--ae-backend torchair` 的 GE 路径；VLM 使用 `--vlm-backend torchair_aclgraph`，由同一 `torchair.get_npu_backend` 通过 `mode=npugraph_ex` 生成 `AclConcreteGraph`。两端均为 fullgraph、静态 shape、frozen 参数；没有启用 Softplus/eye 分解，也没有请求独立 `npugraph_ex` 后端包。这样避免了两套后端包共同加载时的重复注册冲突。

5 warmup + 20 次完整请求实测为 **225.5800 ms p50、226.3952 ms p95**，prefill **117.6110 ms**、FM **108.0122 ms**，相对 eager **4.4945×**。它是独立完整测量，没有拼接不同实验的最快阶段。首次含编译请求 **127.8029 秒**、模型加载 **82.1848 秒**均单列；正式测量峰值 allocated 为 **6,661,815,808 字节**。

A/B/A/B 最坏 relative-L2 为 **0.0019946460（0.1994646%）**，max-abs **0.0038019419**；最低动作符号一致率 **223/224（99.5536%）**，gripper 100%，padding 0。变化状态/指令的 C/D 各检查两次，编码 tokens/mask 与原 eager 全部一致；其动作最坏 relative-L2 **0.0016331926（0.1633193%）**，max-abs **0.0035376549**。同设备 regression gate 通过，仍非逐元素一致、跨 CUDA 验证或闭环成功率结论。

context 优化记录 **35 次**真实移除 action 字段，包含计时后的 **1 次 trace 诊断**，没有缓存输入或模型结果。[run.log](libero-bf16-ge-ae-acl-vlm-triangular-context/run.log)提示 profiler 在 RECORD 状态停止，默认 schedule 可能产生不完整的解析结果；另提示未采集 AIC metrics。该 trace 仅用于诊断，不替代 20 次正式计时，不据此计算模块正式耗时或硬件利用率。该报告 `workload=null`；原公式包含 codec 的算量仍不能作为此优化路径的执行计数。

**TorchAir 整 VLM 兼容实验：数值通过，性能退化**

[完整报告](libero-bf16-ge-frozen-compatible/report.json)的 AE/VLM 均启用 TorchAir `max-autotune`、`fullgraph=true`、`dynamic=false`、`frozen_parameter=true`，仅 VLM 启用后端局部 Softplus/eye 分解，GDN 保持原公式。`comparison_identity` 与最佳 `npugraph_ex` 结果完全相同。5 warmup + 20 次测量为 **1590.8996 ms p50、1597.4734 ms p95**；prefill 升至 1490.6200 ms，FM 为 100.5431 ms。该路径不选为最佳，较快的 FM 单阶段也不能拼入其他报告的总延迟。

同设备 A/B/A/B 最坏 relative-L2 为 **0.0019562659（0.1956266%）**，最大绝对误差 **0.0030649900**，动作与 gripper 符号一致率均为 100%，padding 为 0；通过本次 regression gate。`cold_A` 含编译的首次请求为 **464.7847 秒**，单列在正式计时外。此处 `external_comparison=null`，不表示跨卡或闭环质量验证完成。

**真实 LIBERO 的部分算术计数**

该 GE 兼容运行在正式测量完成、AE/VLM/GDN 后端恢复 eager 后，通过 `--count-workload` 另做一次固定输入、共享显式噪声请求。其耗时不计入正式样本或峰值显存；hook 只用于计数，随后关闭。[workload.json](libero-bf16-ge-frozen-compatible/workload.json)和报告 `workload` 字段保留了每个算子、参数根、实际 token shape、模型调用次数及 checkpoint/input/noise 身份。该诊断输出与同次 eager 参考的有效 7 维误差为 0。

| 已覆盖部分 | FLOPs / 请求 |
|---|---:|
| Prefill：codec | 1,210,843,136 |
| Prefill：Vision | 317,290,708,992 |
| Prefill：proprio | 8,470,528 |
| Prefill：VLM | 436,667,351,040 |
| FM10：Action Expert | 309,167,390,720 |
| **已覆盖合计** | **1,064,344,764,416（1.0643 TFLOP）** |

计数约定为乘法和累加各 1 FLOP，只覆盖被调用的 Linear/Conv 模块及 AdaLN conditioning 投影。不含 attention QK/AV、SDPA 内部、GDN chunk/recurrent 功能运算、绕过模块的卷积、ConvTranspose、归一化/激活/逐元素运算、RoPE、Euler 更新及数据搬运，也不代表编译后填充或重计算。因此 **1.064344764416 TFLOP 是原公式 eager 诊断路径的已覆盖算量下界**，不能称为整模型总 FLOPs、实际硬件指令数、硬件利用率或部署所需峰值算力。该路径还执行 codec，含上表的 1.2108 GFLOP，并逐请求执行时间条件投影；当前删除未消费 codec 或预计算投影的变体需要重新计数。不能从原数粗减这些项目后声称已实测新路径的完整 FLOPs。

峰值 BF16 吞吐不直接决定 batch 1 / 159-token 短序列推理时间。172.4950 ms 对照的正式计时外 trace 记录了大量 vector/core 的 slice、transpose、cast、小算子及 AI_CPU 工作，说明矩阵吞吐之外仍有访存、调度与算子实现成本；这些 duration 不能直接当作可扣除的总延迟。Eager 与编译路径都依赖设备显存，图编译主要减少启动和部分中间结果搬运，并不使整个模型权重全部常驻 SRAM。这里不引用未经核对的硬件峰值，也不由部分 FLOPs 推导必达时延。

计数 helper SHA-256 为 `e53580ae99f8623f46c07992464f6fc39a357ca37075baf852d19f331ad4f9ed`，runner SHA-256 为 `d52535b06422fd9b47dacd85254e80ed50b918d834c1b7d2675c09f4c79a039e`；原始记录同时说明参数统计包含加载的 policy/codec 根及未执行的参数，不能把参数存在当成算子已执行。

**R1 Lite 诊断单独记录**

[opt-fm-bf16-npugraph-ae.json](opt-fm-bf16-npugraph-ae.json) 使用 `g05-base`、合成 R1 Lite 三相机输入、27D 动作、FM-only。该项只用于定位 Ascend 编译/提交开销，不能与上述 LIBERO eager 行计算“同模型优化倍数”。更早约 3 秒的 R1 Lite 完整 CoT/FM/AR 路径也属于不同工作负载。

| R1 Lite BF16 + `npugraph_ex` AE 诊断指标 | p50 |
|---|---:|
| Prefill | 348.8048 ms |
| FM action | 145.1135 ms |
| `forward_inference_total_ms` | 494.2372 ms |
| `model_predict_action_ms` | 506.6275 ms |
| 含外层预处理等工作的 request total | 548.8614 ms |

该诊断为 3 warmup + 10 测量，首次编译请求 31.046 秒单列。旧报告的单次同 seed eager/compiled 输出差异为零，但没有覆盖变化的请求内容及后续 replay；不能把它升级表述为新的 A/B/A/B 验证已通过。

**权重与源码身份**

模型仓库固定 revision 为 `aca2916ba2adaba3ce05bfa279cdc6b98da35cad`。已核对该 revision 与 `e312be81e90c56a55bcb26b57429bd39a335b449` 的 LIBERO 三文件、共享 codec 和 processor 八文件的 Git object ID/size 相同。完整路径、大小及公开校验字段见 [download_manifest.json](download_manifest.json)；实际下载文件的全部 SHA-256 见 [LIBERO report 的 assets 字段](libero-bf16-eager/report.json)。模型文件下载需要已获得该 gated 仓库访问权的 Hugging Face 凭据，报告和命令不包含凭据。

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `g05-libero/model.pt` | 11,440,381,065 | `432f225c7647bdab77bfa2c2b397dc38cf8631bd70c92b85b261ff2817c1e795` |
| `g05-libero/.hydra/config.yaml` | 20,644 | `544c185b903f94d76617b419c4d3d395444d8d3b342409465645e7d7386a001a` |
| `g05-libero/dataset_stats.json` | 95,006 | `b3953f24d94d08e75134ce63fb77c69a09ef2ed46f2dffc0397d2f2b2bebc756` |
| `action_tokenizer.pt` | 506,886,775 | `5088f64a5452a60bbc8cac90ee7d79c156f1d14540bc3139aa7060f862dddace` |

Ascend 上游源码基于 `89f2322b4ad016e192437adc1a2c253b05bab246`，带本项目的 Ascend overlay。首份 LIBERO 基线 JSON 仅记录上游 revision，没有运行期 dirty diff 哈希；该历史文件保持原样，不补写未采集的证据。TorchAir 成功报告已记录 `source.tree_state` 和关键模型源文件 SHA，tracked diff SHA-256 为 `bcbf8968073712c7214c47ae9322cf51534f84898a396598ceb8ce0906f4e24a`。该报告还保留了编译器生成的 `kernel_meta` 等 untracked 文件记录；它们是编译产物，模型源码身份应查看 tracked diff 与关键源文件 SHA。

**已定位的后端问题**

| 尝试 | 原始失败与处理 | 当前证据边界 |
|---|---|---|
| 整个 VLM 的 `npugraph_ex` | [Dynamo `Observed exception` 日志](opt-fm-bf16-npugraph-ae-vlm.log)。GDN 初次 prefill 访问缺失 cache key 时，原 `_DefaultNoneDict.__missing__` 路径不适合追踪；overlay 改成等价的 `.get(layer_idx)`。 | 修复后已完成 [frozen AE + VLM](libero-bf16-npugraph-frozen-ae-vlm/report.json) 的完整测量及 A/B/A/B 检查；早期失败日志仍保留。 |
| GDN helper 图捕获 | [`aclrtMemcpy` 107030 日志](opt-fm-bf16-ae-gdn.log)：capture 中出现 CPU→NPU 同步复制。将 `torch.zeros(...).to(value)` 改为直接在 `value.device/value.dtype` 分配，覆盖 chunk/recurrent 状态及 recurrent 输出。 | 保留原公式、FP32 recurrent state、chunk size；修复已用于整个 VLM 的成功路径，单独 GDN helper 编译尚无本表中的完整测量。 |
| AE `super_kernel_optimize` | [日志](opt-fm-bf16-ae-super.log)显示 `libascendsk.so: cannot open shared object file`。 | 当前环境的 SK 共享库加载失败；没有成功测量，不报告加速收益，默认关闭该选项。 |
| TorchAir GE 首次启动 | [日志](libero-torchair-tbe-path-failure.log)显示 `No module named 'tbe'`。远程命令覆盖了 `PYTHONPATH`，丢失 CANN 路径；已改为先 source 环境再 prepend 源码路径。 | **已修复并成功验证**：[修正环境后的日志](libero-bf16-torchair-ae/libero-bf16-torchair-ae-env.log)及[成功报告](libero-bf16-torchair-ae/report.json)记录了 365.3891 ms 与 A/B/A/B 检查。这是启动命令问题，不是镜像未安装 TBE。 |
| TorchAir GE 整 VLM converter | 首次遇到 `aten.softplus.default ge_converter is not implemented`；加入后端局部 Softplus 分解后，下一处为 `aten.eye.default` 未实现。[两轮失败摘要与原日志 SHA](ge-vlm-converter-failures.json)。 | 两次均失败于正式测量前；补齐两项局部分解后的[整模型运行](libero-bf16-ge-frozen-compatible/report.json)已通过同设备检查，但 1590.8996 ms 性能退化。原失败记录和成功执行的退化结果分别保留。 |
| GE 外层手动 NPUGraph 捕获 | [小图探针](probe-ge-capture.json)在已预热的 GE 调用外包 NPUGraph 时，报 capture stage 不允许 model execution。 | 仅为 `[8,64]` 合成小图的失败，不是 G0.5 整模型结果，也没有证明该组合可以重放。 |

这些源码修复在 [Ascend overlay](../../../../integrations/galaxeavla/patches/0001-ascend-torchnpu-runtime.patch) 中；编译安装与选项在 [g05_npu_graphs.py](../../../../tools/galaxeavla/g05_npu_graphs.py) 中。后端失败保持失败，不允许自动转 eager 后仍报告为 compiled。

**局部实验的边界**

- [Softplus 分解 CPU 检查](probe-softplus-decomp.json)：FP32，10 组/48,892 个值，最大绝对误差 `9.5367e-7`，在声明容差内通过；舍入顺序改变。这只验证了后端局部分解的 CPU 数学行为。
- [方阵 eye 分解 CPU 检查](probe-eye-decomp.json)：244 例覆盖值、dtype、layout、stride 等支持边界；有效输出为精确 0/1。不代表 NPU GE 转换或整模型已通过。
- [GDN triangular CPU 检查](gdn-triangular-cpu.json)：84 例，全部有限，最大输出绝对误差 `0.0078125`、最大 FP32 state 误差 `0.0117176`；变换改变了累加顺序，需要整模型验证。
- [GDN triangular 单 helper NPU eager 微实验](gdn-triangular-npu-eager.json)：仅 `[1,159,16,128,128]` 输入的 chunk helper，chunk size 32、BF16 输入/FP32 状态，3 warmup + 10 测量；p50 `9.7202 → 4.9213 ms`，该个案输出和状态误差均为 0。计时不包含投影、卷积、Vision、VLM 其他层或完整 FM，不能外推成整模型倍数；上述 297.8925 ms 结果没有启用它。
- [GDN triangular 单 helper NPU graph 微实验](gdn-triangular-npu-graph.json)：同类输入及 3 warmup + 10 测量，`npugraph_ex` 下 p50 `4.2875 → 3.1413 ms`，该个案输出/state 误差为 0。这项局部结果不能代替上述 264.4622 ms 组合的整模型测量，也不能证明三角求解在所有输入上逐元素相同。
- [FP32 Triton 顺序行递推 NPU 探针](experiments/row-inverse-npu.json)：8 组 × 80 个 32×32 矩阵，包括负 lower-ones、随机和近共线压力输入，均通过声明容差，部分不是位级一致。单 kernel p50 约 0.2126 ms，对照原 Torch eager 约 4.0450 ms、graph 2.8599 ms、triangular solve 1.2214 ms；Triton 输出预分配，而 Torch eager 包含临时分配。这是逆矩阵局部测量，不是整层或整模型收益。
- [Opaque custom op 的 ACLGraph 检查](experiments/gdn-triton-custom-op-aclgraph.json)：A/B/A/B 通过，输入未修改，eager 输出彼此独立；图返回缓冲确有重用，因此每次结果在下一请求前独立快照。p50 0.4950 ms 是含 custom-op/图调用的独立微计时，不能用 0.2126 ms 替代，也不能从总延迟直接减去某个旧算子时间。
- [Doubling 反例](experiments/gdn_doubling_counterexample.json)：仅 CPU NumPy FP32 数学检验。`L=-strict_lower_ones` 的原递推精确，而 doubling 最大误差达 8，故未采用 doubling；不含性能测量。
- [AE 私有 namespace 修复检查](experiments/probe-ae-namespace.json)：旧路径触发 Dynamo `AttentionInterface` guard 错误；为复制函数建立并注册真实私有 `ModuleType` 后，小型 CPU fullgraph A/B/A/B 单图重放与恢复通过、原 registry 未改。该小实验不证明 NPU 性能；200.6956 ms 对照及后续完整报告另行记录了修复后的 NPU fused 路径。
- [AE partial RoPE NPU/GE 微实验](experiments/ae-rotary-npu.json)、[探针源码](experiments/probe_ae_rotary_npu.py)：Q/K 的 8/2 heads 各测 original、sliced、full_matrix，共 6 项，FP32、rotary64/head256；eager 与编译 A/B/A/B 检查的最大绝对误差均为 0。局部调用含 Python dispatch，不能将这些微计时乘层数后宣称整模型收益；matrix 的 163.9053 ms 由独立完整测量得出。
- [AE Norm NPU/GE 微实验](experiments/ae-norm-ge.json)、[探针源码](experiments/probe_ae_fused_norm.py)：Adaptive RMS、Q norm、K norm 的 BF16/FP32 共 6 个 shape/dtype case，包含 Q 的非连续 stride；NPU eager 的变化输入/零/小/大值及 GE A/B/A/B 均通过声明容差。最大绝对误差 **9.5367e-7**、relative-L2 **4.6216e-8**；这只是局部数学及编译验证，155.5743 ms 整模组合另行测得。
- [Functional MRoPE NPU/GE/ACLGraph 微检查](experiments/functional-mrope-ge-acl.json)、[探针源码](experiments/probe_functional_mrope.py)：直接使用真实配置和两个 rotary 模块，不加载模型权重。24 次变化位置输入检查及卸载恢复均精确一致，证明该实例替换的数学、配置兼容与图执行；不单独宣称整模型性能收益。
- [172.4950 ms 对照的 Level1 算子汇总](experiments/profile-172ms-level1-summary.json)、[汇总脚本](experiments/summarize_cann_ops.py)：计时后一个请求，`kernel_details.csv` 中 40 个 AI_CPU `ScatterElements` 的算子 duration 合计 **5.1150 ms**，另有大量 vector/core 的 slice、transpose、cast 和小算子。各 CSV 独立汇总，不跨表求和，也不把算子 duration 总和当正式 p50 或可直接扣除的关键路径；这些记录是后续优化的定位证据。

**时间条件预计算的已归档退化实验**

[native exact-lookup 报告](libero-bf16-ge-acl-native-time/report.json)按原 FP32 `t -= delta_t` 递推保存 10 个时间值，没有用 `linspace` 替代；预计算 `encode_time` 和 49 个独立 AdaLN conditioning 投影，表大小 6,021,120 字节、建立耗时 118.1110 ms。动态 RMS、affine 顺序及输出 cast 保留。但是 exact lookup 每一步执行一次 `t.detach().cpu().tolist()`，同步包含在推理时间内。实测 **228.4562 ms** 比同 native 的 **219.1758 ms** 对照慢，故该 exact 版本未纳入当前最快组合。后来的 request lookup 取消每步 D2H，并与固定 Vision 图共同得到上述 172.4950 ms 完整结果；不能把两个不同 lookup 版本合并归因。

**复现命令**

下列命令对应本次远端已准备的路径。`/root/work` 中需同时有 `benchmark_g05_libero.py`、`benchmark_g05_full.py` 和 `g05_npu_graphs.py`；启用 `--count-workload` 时还需同目录的 `count_g05_workload.py`。源码先应用本仓库的 Ascend overlay。`/root/work/libero_input/` 包含原始 `case.json`、`image.png`、`wrist_image.png`，runner 会校验三个文件的固定 SHA。输出使用新目录，保留归档基线。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="/root/work/GalaxeaVLA-upstream/src:/root/work/GalaxeaVLA-upstream:/root/work:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python /root/work/benchmark_g05_libero.py \
  --source-dir /root/work/GalaxeaVLA-upstream \
  --asset-root /root/work/models/galaxeavla-g05 \
  --checkpoint /root/work/models/galaxeavla-g05/g05-libero/model.pt \
  --input-dir /root/work/libero_input \
  --output-dir /root/work/libero-bf16-eager-rerun \
  --device npu:0 --weights bf16 --ae-backend eager \
  --timing-noise shared --seed 0 --warmup 5 --iterations 20
```

重跑 AE TorchAir，使用基线的同一显式噪声，并额外启用独立基线文件对照。已归档的成功运行完成了 `--verify` 同进程 eager/compiled 对照，未填写 `external_comparison`；下面的 `--reference-report` 是额外复核选项：

```bash
python /root/work/benchmark_g05_libero.py \
  --source-dir /root/work/GalaxeaVLA-upstream \
  --asset-root /root/work/models/galaxeavla-g05 \
  --checkpoint /root/work/models/galaxeavla-g05/g05-libero/model.pt \
  --input-dir /root/work/libero_input \
  --output-dir /root/work/libero-bf16-torchair-ae-rerun \
  --device npu:0 --weights bf16 --ae-backend torchair \
  --noise /root/work/libero-bf16-eager/noise.npy \
  --reference-report /root/work/libero-bf16-eager/report.json \
  --timing-noise shared --seed 0 --warmup 5 --iterations 20 --verify
```

其他独立实验可把 AE 改为 `--ae-backend npugraph_ex`；整 VLM 用 `--vlm-backend npugraph_ex`；GDN helper 用 `--gdn-backend npugraph_ex --gdn-helpers chunk,recurrent`。整 VLM 与 GDN helper 编译不能同时打开，以便归因。图选项使用 `--ae-options`、`--vlm-options` 或 `--gdn-options` JSON 对象；当前不启用失败的 `super_kernel_optimize`。每个变体独立进程、独立输出目录，并保留同一 noise、精度、shape 和 FM10。

重跑保留原公式的 frozen AE + VLM 组合，在同一已初始化环境中执行：

```bash
python /root/work/benchmark_g05_libero.py \
  --source-dir /root/work/GalaxeaVLA-upstream \
  --asset-root /root/work/models/galaxeavla-g05 \
  --checkpoint /root/work/models/galaxeavla-g05/g05-libero/model.pt \
  --input-dir /root/work/libero_input \
  --output-dir /root/work/libero-bf16-npugraph-frozen-ae-vlm-rerun \
  --device npu:0 --weights bf16 \
  --ae-backend npugraph_ex --ae-options '{"frozen_parameter":true}' \
  --vlm-backend npugraph_ex --vlm-options '{"frozen_parameter":true}' \
  --noise /root/work/libero-bf16-eager/noise.npy \
  --timing-noise shared --seed 0 --warmup 5 --iterations 20 \
  --verify --profile-modules
```

新 `--verify` 在计时外建立 eager A/B 参考，再执行 selected A/B/A/B。A 是原始真实样本；B 为同 shape 图像水平翻转和负噪声，仅用于验证变化的图像、KV 与 action buffer，正式计时始终使用 A。结果包括 7 维 MAE/RMSE/max-abs/relative-L2、cosine、符号一致率和 padding 检查；默认 regression gate 为 relative-L2 ≤ 1%、max-abs ≤ 0.05，报告需保留实际误差，不能把这些门槛当作闭环任务准确率标准。

复现 GE 兼容实验时，以上命令改为独立输出目录，将 AE/VLM 后端改为 `torchair`，保留 AE 的 `--ae-options '{"frozen_parameter":true}'`，并设置 `--vlm-options '{"frozen_parameter":true,"softplus_decomposition":true,"eye_decomposition":true}' --count-workload`。`--count-workload` 是计时后的独立原公式 eager 请求，不会把计数 hook 带入正式测量。

复现已归档的 264.4622 ms 组合时，保留以上 frozen `npugraph_ex` AE/VLM 选项，换用新输出目录，并追加 `--gdn-algorithm triangular --skip-unused-action-codec --verify`。同目录需同时有 `g05_gdn_triangular.py` 和 `g05_context_only.py`；报告记录实现 SHA。该命令会额外执行变化 state/task 的 C/D 对照，不允许省略验证或将此优化用于 CoT/AR。此变体尚未采集对应执行路径的算量。

复现 shared-noise 225.5800 ms 混合后端组合时，在上述 triangular/context 配置上改为 `--ae-backend torchair --vlm-backend torchair_aclgraph`，AE/VLM options 都保持 `'{"frozen_parameter":true}'`，并使用独立输出目录。无需启用 GE VLM 的 Softplus/eye 分解。`--trace-dir` 为可选的单次计时后诊断；该旧版归档 trace 有 schedule 警告，不作为正式时延来源。

复现保留的 native 200.6956 ms 对照，除上述共用 helper 外，同目录还需 `g05_gdn_triton.py`、带 inverse 接缝的 `g05_gdn_triangular.py`、`g05_ae_fused_attention.py` 与 `g05_context_only.py`。保留前述 CANN/PYTHONPATH 初始化，运行：

```bash
python /root/work/benchmark_g05_libero.py \
  --source-dir /root/work/GalaxeaVLA-upstream \
  --asset-root /root/work/models/galaxeavla-g05 \
  --checkpoint /root/work/models/galaxeavla-g05/g05-libero/model.pt \
  --input-dir /root/work/libero_input \
  --output-dir /root/work/libero-bf16-native-triton-fused-rerun \
  --device npu:0 --weights bf16 \
  --ae-backend torchair --ae-options '{"frozen_parameter":true}' \
  --vlm-backend torchair_aclgraph --vlm-options '{"frozen_parameter":true}' \
  --gdn-algorithm triton --ae-attention npu --skip-unused-action-codec \
  --noise /root/work/libero-bf16-eager/noise.npy \
  --timing-noise native --seed 0 --warmup 5 --iterations 20 --verify
```

复现保留的 **172.4950 ms** 组合，在上面命令中使用新输出目录 `/root/work/libero-bf16-native-triton-fused-time-vision-rerun`，并追加：

```bash
--precompute-time-conditioning --time-conditioning-lookup request --static-vision
```

同目录还需 `g05_time_conditioning.py` 和 `g05_static_vision.py`。`--verify` 必须保留，静态 Vision 会增加独立相机 E/F 各两次检查；模型、噪声计时、FM10 及 batch/geometry 约束不变。源码和 helper SHA 以归档报告为准，后续代码修改不能追溯改变已有实测。

复现保留的 **163.9053 ms** 结果，在该 172.4950 ms 配置上再追加 `--ae-rope matrix`，输出到独立目录 `/root/work/libero-bf16-native-vision-rope-matrix-rerun`，同目录加入 `g05_ae_fused_rope.py`。保持默认关闭的 Norm、functional MRoPE 和 GE single_stream；这些配置不能混入此报告的复现身份。复现单流对照则保持原 RoPE，把 AE options 改为 `'{"frozen_parameter":true,"single_stream":true}'`，使用另一个输出目录。

复现当前 **155.5743 ms** 组合，保持前述 CANN/PYTHONPATH 初始化，确保同目录具备以上全部 helper 及 `g05_ae_fused_norm.py`、`g05_functional_mrope.py`，运行：

```bash
python /root/work/benchmark_g05_libero.py \
  --source-dir /root/work/GalaxeaVLA-upstream \
  --asset-root /root/work/models/galaxeavla-g05 \
  --checkpoint /root/work/models/galaxeavla-g05/g05-libero/model.pt \
  --input-dir /root/work/libero_input \
  --output-dir /root/work/libero-bf16-native-vision-rope-norm-mrope-rerun \
  --device npu:0 --weights bf16 \
  --ae-backend torchair --ae-options '{"frozen_parameter":true}' \
  --vlm-backend torchair_aclgraph --vlm-options '{"frozen_parameter":true}' \
  --gdn-algorithm triton --ae-attention npu --skip-unused-action-codec \
  --precompute-time-conditioning --time-conditioning-lookup request --static-vision \
  --ae-rope matrix --ae-norm npu --functional-mrope \
  --noise /root/work/libero-bf16-eager/noise.npy \
  --timing-noise native --seed 0 --warmup 5 --iterations 20 --verify \
  --trace-dir /root/work/profile-libero-norm-mrope-rerun
```

上述选项默认关闭，保留 `--verify` 的完整变化输入检查。`--trace-dir` 只增加计时后诊断请求，不把 trace 开销纳入 20 次样本。报告包含 benchmark/helper SHA、上游源码身份与全部模型/输入哈希；复现结果需生成自己的独立报告。

**待补齐的证据**

- TorchAir AE、原公式 `npugraph_ex` frozen AE + VLM、带兼容分解的整 VLM GE、triangular/context 组合及 GE AE / ACLGraph VLM 混合组合均已完成真实 LIBERO 测量和同设备检查；GE 整 VLM 有性能退化。其他候选及单独各项消融需各自追加正式结果，不能复用微实验结论。
- 当前已归档最快为 native **155.5743 ms**，同 native eager 加速 **6.5020×**，尚未达到 3090 的 120.4772 ms。matrix RoPE、fused Norm 与 functional MRoPE 已在 request 时间条件与固定 Vision 组合上完成整模实测及 12 次检查；不把微实验收益相加或宣称已达到目标。
- LIBERO 的原公式 `--count-workload` 诊断已归档。后续若跳过 codec 或改变算子路径，需要重新计数并保留执行身份；遗漏的 attention/GDN 功能运算等仍不能宣称为已覆盖，不能称为完整 FLOPs 或硬件利用率。
- 用相同 `noise.npy` 在 CUDA 与 NPU 生成新参考，完成跨卡有效 7 维误差验证；历史 CUDA seed-0 输出不能直接冒充该参考。
- 独立 `native` 基线和优化测量已补齐，shared/native 历史结果继续分别标注；原设备随机数计时路径对齐不等于跨卡随机张量一致。
- 实际部署质量需要另外记录 LIBERO 闭环成功率；当前仅有单帧推理及数值检查。

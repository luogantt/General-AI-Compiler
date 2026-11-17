这是 G0.5 LIBERO 的 **AE Q/K RMSNorm + partial RoPE + 输出布局融合实验**，默认关闭。它通过 `gac_g05::qk_norm_rope` 调用 Ascend C 算子 `GacQkNormRope`，仅替换 24 个 Action Expert 层中的 Q/K 路径。VLM、Q 的 gate 分支、V 投影、KV 拼接、注意力及动作步数保持原有计算。

当前开发环境为 Ascend 910B2 / A2、CANN 9.0.0、PyTorch 2.7.1、TorchNPU 2.7.1.post4、Python 3.11、Triton-Ascend 3.2.1。这个算子使用 Ascend C；Triton-Ascend 用于同一实验环境中的其他优化，不是本算子的编译器。

**融合范围与精度**

每个 Vector 核从设备显存 GM 读取一行 256 维 Q/K 及当前 RoPE 数据，在片上 UB 中完成归一化和旋转，直接把结果写成连续 `[1,H,32,256]` 布局。同一次调用中，该核处理的各行复用已读入 UB 的 `1 + weight`。输入、权重和输出仍需访问设备显存；没有把整个模型或跨请求动作/KV 缓存在片上。

UB 是这段 Vector 归约和逐元素计算使用的片上存储。L1、L0A/L0B/L0C 主要服务 Cube 矩阵计算的数据通路，本核没有 Cube 运算，因此不采用它们。当前实现使用深度为 1 的队列，没有实现双缓冲或宣称搬运与计算已重叠。

计算顺序必须保留：

```text
BF16 投影值 → FP32 square / mean / epsilon / inverse RMS
            → FP32 × 原值 × (1 + FP32 weight)
            → BF16 RNE 舍入 → 转回 FP32
            → 前 64 维 partial RoPE；后 192 维保留归一化结果
            → BF16 RNE 舍入 → 连续 BHSD 输出
```

RNE 为最近偶数舍入。两次 BF16 舍入都是模型精度边界，不能删掉中间一次。RoPE 保留两个 FP32 乘法再相加；构建脚本设置 `-ffp-contract=off`。当前 v4 使用 `Sqrt` 后 `Div(1, sqrt)` 计算 inverse RMS，目的是改善 A2 上直接 `Rsqrt` 近似带来的精度问题。FP32 归约树仍可能与 PyTorch/native RMSNorm 不同，因此需要数值回归，不声明逐位等价。

| 参数 | 约束 |
| --- | --- |
| `x`（Q） | 连续 BF16 `[1,32,4096]`，每头 512 个元素中前 256 个是 Q，后 256 个是 gate |
| `x`（K） | 连续 BF16 `[1,32,512]`，2 个头，每头 256 维 |
| `weight` | 当前 FP32 `[256]` 原始 norm 参数，算子内部加 1 |
| `cos` / `sin` | 当前连续 FP32 `[1,1,S,64]`，`S >= 32`，只消费前 32 个位置 |
| `head_stride` | 元素单位；Q 为 `512`，K 为 `256` |
| `epsilon` | 有限正数，转换到 FP32 后仍须有限且大于零 |
| 输出 | 新分配、连续 BF16 `[1,8,32,256]` 或 `[1,2,32,256]` |

所有张量必须在同一 NPU。桥和 tiling 只接受未分块的 dense ND/NCHW 存储，并检查实际形状；NCHW 标签不意味着执行额外重排。不接受 blocked format，不隐式补做 contiguous/format cast。仅允许 eval 加 `no_grad`/`inference_mode`；正常模型 Parameter 可以保留 `requires_grad=True`。

**构建与加载**

以下命令在 Linux Ascend 环境、仓库根目录执行。按实际安装位置调整 CANN 环境脚本。所有构建和 OPP 安装使用独立目录，不覆盖 CANN 自带 OPP。`build_opp.py --output` 要求目录尚不存在；重建候选时使用新的目录名。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export GAC_QK_BUILD=/root/work/build/gac-qk-opp-v4
export GAC_QK_INSTALL=/root/work/opp/gac-qk-v4
export GAC_QK_BRIDGE_BUILD=/root/work/build/gac-qk-bridge-v4

python tools/galaxeavla/ascendc_qknorm_rope/build_opp.py \
  --output "$GAC_QK_BUILD" --soc Ascend910B2 --jobs 4

# 选择这次独立构建产生的唯一 .run 包；若不唯一则停止并核对路径。
GAC_QK_RUN="$(python -c 'import pathlib,sys; p=list((pathlib.Path(sys.argv[1])/"build_out").glob("*.run")); assert len(p)==1,p; print(p[0])' "$GAC_QK_BUILD")"
bash "$GAC_QK_RUN" --install-path="$GAC_QK_INSTALL"

export GAC_QK_VENDOR="$GAC_QK_INSTALL/vendors/gacqknorm"
source "$GAC_QK_VENDOR/bin/set_env.bash"
export ASCEND_CUSTOM_OPP_PATH="$GAC_QK_VENDOR"

# 每次更换构建版本都使用新缓存目录，并启动新的 Python 进程。
mkdir -p /root/work/cache
export ASCEND_CACHE_PATH="$(mktemp -d /root/work/cache/gac-qk-v4-XXXXXXXX)"

python tools/galaxeavla/ascendc_qknorm_rope/build_bridge.py \
  --opapi-include-dir "$GAC_QK_VENDOR/op_api/include" \
  --opapi-lib-dir "$GAC_QK_VENDOR/op_api/lib" \
  --custom-opp-path "$GAC_QK_VENDOR" \
  --build-dir "$GAC_QK_BRIDGE_BUILD" --jobs 2 --load

export GAC_QKNORM_ROPE_LIBRARY="$(python -c 'import json,pathlib,sys; print(json.loads((pathlib.Path(sys.argv[1])/"bridge-build.json").read_text())["artifact"])' "$GAC_QK_BRIDGE_BUILD")"
```

`build_opp.py` 使用 `msopgen` 和当前 `ASCEND_HOME_PATH`，保存源文件 SHA-256 到 `source-manifest.json`。`build_bridge.py` 保存 `.so` 路径、ABI、依赖和哈希到 `bridge-build.json`；`--load` 仅检查加载及 dispatcher 注册，不执行 NPU 计算。

`ASCEND_CUSTOM_OPP_PATH` 和 `ASCEND_CACHE_PATH` 必须在新的 Python 进程导入 TorchNPU/TorchAir **之前**设置。这里显式选择一个 `gacqknorm` vendor，避免旧候选重复注册同名算子。每次重建或重新选择 OPP/bridge，都使用新的缓存目录并启动新进程；不要在已加载旧库的进程中替换它，也不要复用旧候选的编译缓存。同名自定义算子复用旧缓存会使 GE 结果对应旧核，不能仅凭新 `.so` 的路径判断已运行新实现。探针记录缓存/OPP 路径及源码、bridge 哈希，供复核。

桥调用直接 ACLNN `GetWorkspaceSize` / launch，并通过已导出的 `OpCommand::RunOpApi` 在当前 NPU stream 排队。它持有张量和 ACL descriptor 至 host launch 完成，严格要求 workspace 为 **0**，不在 queue callback 中分配 ATen workspace，也不执行每次调用的设备同步。当前验证范围聚焦同一 stream 的 eager 和 GE；跨 stream 生命周期不作已验证承诺。

Python 装配入口位于 [g05_ae_qknorm_rope.py](../g05_ae_qknorm_rope.py)。它加载上述 `.so`，注册 FakeTensor 和 `ge.custom_op` converter，按锁定的 AST 片段仅重写 AE 层实例。实例修改可恢复；算子/fake/converter 注册保留到进程结束。遇到契约、编译或注册错误会报错，不静默回退。

**微测试与完整模型验证**

先在已经设置上述环境的全新进程运行 [probe_ae_qknorm_rope.py](../probe_ae_qknorm_rope.py)：

```bash
python tools/galaxeavla/probe_ae_qknorm_rope.py \
  --device npu:0 --backend torchair --rope-lengths 32,159 \
  --warmup 5 --iterations 100 --timing-groups 5 \
  --output /root/work/results/gac-qk-v4/probe.json
```

探针对比原公式、native RMSNorm + matrix RoPE、Ascend C 路径的 eager/GE 输出。它分别改变 x、weight、cos、sin，包含零/小/大输入和权重、gate-only 与 RoPE 后缀变化，并检查重复回放、输入未修改、新输出存储及后 192 维误差。每次验证立即复制输出，避免图输出缓冲区复用遮蔽错误。逐位相等只是记录项；归约差异按报告中的绝对误差、相对 L2 门槛判定。正式微测试批次与编译、验证开销分离，批次耗时包含 Python 提交成本，不能直接当作整模型收益。

Ascend C eager/GE 是必须通过的候选，均对 original eager 和 native eager 两个基准使用原有误差门槛；eager 参考本身也必须正常且有限。original GE/native GE 是观察性对照，其超门槛结果仍保留 `passed=false` 和逐项失败明细，并单列 `control_observations`；它们不替代 eager 基准，也不改变候选验收门槛。

微测试明确使用 `frozen_parameter=False`，用于验证**动态权重输入**，并要求同一 shape 的变体序列只触发一次 Dynamo backend 编译。完整模型则维持固定权重的 `frozen_parameter=True` 契约，不在冻结后更换或修改权重。

完整模型用 `--ae-qknorm-rope ascendc` 替换原来的 `--ae-rope matrix`；二者互斥。可保留 `--ae-norm npu`：49 个 Adaptive norm 继续使用 native 优化，48 个 Q/K norm 调用由新融合算子接管。下面给出同一 LIBERO 配置的组合示例，路径需指向已校验的源码、官方权重和原始观测样本：

```bash
python tools/galaxeavla/benchmark_g05_libero.py \
  --source-dir /path/to/GalaxeaVLA \
  --asset-root /path/to/assets \
  --checkpoint /path/to/assets/g05-libero/model.pt \
  --input-dir /path/to/libero-input \
  --output-dir /root/work/results/gac-qk-v4/full-model \
  --device npu:0 --weights bf16 --verify \
  --ae-backend torchair --ae-options '{"frozen_parameter":true}' \
  --vlm-backend torchair_aclgraph --vlm-options '{"frozen_parameter":true}' \
  --gdn-algorithm triton --ae-attention npu --ae-norm npu \
  --ae-qknorm-rope ascendc --functional-mrope \
  --skip-unused-action-codec --precompute-time-conditioning \
  --time-conditioning-lookup request --static-vision \
  --timing-noise native --warmup 5 --iterations 20
```

保持 10 个 FM 步、32 个动作位置及原输入契约。每次复现仍需通过独立 eager 参考下的 A/B/C/D/E/F 动作回归，再接受正式性能结果。

**验证状态**

本地 `test_g05*.py` 共 161 项测试通过，覆盖 Python 契约、源码替换、恢复、CLI 组合与可视化；本算子的 helper 8 项和 probe 4 项使用 CPU NumPy/mocks，包含观察性对照与候选验收分离、原有门槛、缺失结果及非有限值处理。实际 NPU 验证另见下述微实验和整模型报告。

2026-11-17 的 [v4 独立缓存微实验报告](../../../docs/validation/ascend/galaxeavla-g05-ascendc-fusion-910b2-2026-11-17/experiments/qknorm-micro-v4-fresh.json) 已完成：`status=success`，`candidate_validation.passed=true`，`reference_validation.passed=true`。H=8/2、RoPE S=32/159 四种组合的 Ascend C eager 与 GE 候选均通过原有门槛，每个 GE 候选在动态输入/权重回放期间只编译一次。报告记录本次独立缓存目录 `/root/work/cache-qknorm-v4-46c1a432`、OPP/bridge 路径与源文件哈希；复现时仍应新建自己的缓存目录。

同一报告保留 `control_observations.status=deviations_observed`：original GE/native GE 的部分动态权重压力用例超出观察门槛，其 `passed=false` 与失败明细没有删除。这些观察性对照没有替代 eager 参考，也没有放宽 Ascend C 候选门槛。

完整 LIBERO FM10 的 [2026-11-17 验收报告](../../../docs/validation/ascend/galaxeavla-g05-ascendc-fusion-910b2-2026-11-17/README.md)现已归档：native 随机数计时，预热 5 次、正式测量 20 次，整模 **p50 135.22 ms / p95 135.70 ms**，prefill p50 **81.97 ms**、FM10 p50 **53.19 ms**。同轮 matrix RoPE 对照为 **153.48 ms / 154.41 ms**，prefill **82.07 ms**、FM10 **71.34 ms**；整模 p50 减少 **18.26 ms（11.90%）**，FM10 减少 **25.44%**。历史 155.57 ms 记录保留，不用它代替同轮对照。

A/B/C/D/E/F 共 12 次同设备动作检查均通过原有工程门槛，最坏 relative-L2 **0.246637%**、max-abs **0.004699469**，均来自外部相机 E；最低动作符号一致率 **99.5536%**，gripper 符号一致率均 100%，padding 为 0，C/D/E/F tokens/mask 与各自 eager 参考一致。首次含编译请求 **152.70 秒**单列，不计入稳态结果。这些检查不代表跨 CUDA 对齐或机器人闭环任务准确率，120.48 ms 目标仍未达到。

该收益来自 **UB 矢量融合**，没有新增 L0/L1 或 Cube 实现。[同轮计时后 Level1 诊断](../../../docs/validation/ascend/galaxeavla-g05-ascendc-fusion-910b2-2026-11-17/experiments/kernel-type-comparison-matrix-vs-ascendc.json)显示 kernel 行数 **19206→15826**，新增 480 次 `GacQkNormRope`，RmsNorm **490→10**、RotaryPositionEmbedding **480→0**、Cast **3476→2516**、Transpose **1659→1179**；两边仍各有 4 个 AI_CPU 任务。诊断的算子 duration 合计不等于 wall-clock 延迟，整模收益只引用上述正式测量。

# BEVFormer-Base：完整分辨率 FP16 训练

**2026-10-19 状态：完整 Base 的单卡与三卡 DDP 真实训练预检均已通过，正式 trainval 训练尚未开始。** 单卡带指纹复测于 **09:48:12**、三卡预检于 **09:50:17（均为北京时间）** 正常结束。三卡每个 rank 完成 4 次实际 AdamW 更新，并通过模块梯度、参数探针同步、3 帧预测和检查点验证，见[三卡实测报告](../../../docs/training/bevformer-base/three-gpu-smoke.json)。全量数据仍在提取；后台继续流程已启动（PID `7377`），核查通过后等待数据准备完成，再自动启动三卡正式训练。当前没有正式全量训练精度结果。

**09:51:15** 的[状态快照](../../../docs/training/bevformer-base/pipeline-status-20261019.json)已确认流程为 `running / await_full_data`，数据准备仍处于 `extract`。实际生成的正式配置见[配置快照](../../../docs/training/bevformer-base/base-trainval-config.py)；以下结果与配置均对应本次实验。

这套工具调用官方 BEVFormer 的 PyTorch FP16 训练入口。模型结构保留完整 Base，增加 Backbone 与 Encoder 激活重计算、数据校验、训练门控与状态记录。本次反向传播不经过 GAC 的 MLIR 推理编译器。

## 1. 机器与模型

| 项目 | 配置 |
| --- | --- |
| SSH | `ssh -p 34280 root@connect.nmb2.seetacloud.com` |
| GPU | 3×RTX 3080 Ti，每卡 12GB |
| 主机内存 / 私有数据盘 | 约 251 GiB / 650GB |
| BEVFormer 源码 | `/root/src/BEVFormer` |
| Python 环境 | `/root/miniconda3/envs/bevformer`，通过任务的 `env.sh` 载入 |
| 实验目录 | `/root/autodl-tmp/bevformer-base-trainval-20261019` |
| 只读公共数据 | `/root/autodl-pub/nuScenes` |

三卡采用数据并行：每卡维护模型副本并处理不同样本，梯度通过 NCCL 同步。**3×12GB 不等于单个样本可以使用 36GB 显存**；每卡 batch 1 仍须单独容纳其模型、激活、优化器等开销。三卡短预检已经通过，后续仍需在正式全量训练中持续检查显存峰值与更新状态。

| 模型 / 训练参数 | 设置 |
| --- | --- |
| 基础配置 | `projects/configs/bevformer/bevformer_base.py` |
| Backbone | ResNet-101，Caffe 风格，后两阶段 DCNv2 |
| 图像 | 6 路原始 900×1600 图像，按 32 对齐后为 928×1600；不缩小图像 |
| FPN | 4 个特征层级，256 通道 |
| BEV | 200×200 |
| Encoder / Decoder | 6 层 / 6 层 |
| 时序队列 / 检测 query | 4 帧 / 900 个 |
| 初始化 | 官方 `r101_dcn_fcos3d_pretrain.pth` |
| 执行方式 | 官方 `BEVFormer_fp16` 与 `EpochBasedRunner_video`，静态 loss scale 512 |
| 激活重计算 | Backbone：`img_backbone.with_cp=True`；6 层 Encoder：`CheckpointedBEVFormerLayer` |
| 正式训练 | 3 卡 DDP，每卡 batch 1、workers 2，全局 batch 3 |
| 优化器 | AdamW，weight decay 0.01，梯度裁剪 max norm 35 |
| 正式初始学习率 | `7.5e-5`；Backbone 沿用 0.1 倍学习率 |
| 学习率策略 | 500 步线性 warmup、余弦衰减 |
| 轮次 / 验证 | 24 epochs，每轮完整验证 |
| 检查点 | 每轮保存，保留最近 3 轮 |

`with_cp` 使用 MMDetection ResNet 的实现，在反向阶段重新计算部分 Backbone 激活。新增 `CheckpointedBEVFormerLayer` 对每层 Encoder 使用相同策略；所有 Tensor 输入（包括嵌套参数和图像特征）显式传入 checkpoint，保留 dropout RNG，历史帧的无梯度计算直接执行。两者不会降低输入分辨率或减少模型层数。小型 CPU 测试已检查输出、输入与参数梯度、嵌套 Tensor、RNG、层绑定和直通行为；单卡真实训练结果记录如下。FP16 也不意味着所有算子、归约和参数状态都统一使用半精度。

**每 GPU 的 batch_size 仍为 1。** 官方 8 卡配置的全局 batch 为 8，本次 3 卡配置的全局 batch 为 3；模型尺寸、六路输入分辨率和网络层数不变。正式学习率按 `2e-4 × 3 / 8 = 7.5e-5` 线性缩放。这是实验起点，尚未经超参数搜索，不保证复现官方多卡训练精度；不要再叠加 `--autoscale-lr`。单卡 mini 预检使用 `2.5e-5`，正式实验从 FCOS3D 重新初始化，**不载入 mini 或预检产生的权重**。

预训练文件为 225,215,819 字节，已检查可读；SHA-256：

```text
91d8a649e007663121a64c0511d486fff1d421b77d61359399dd0a821fe633e8
```

权重检查记录在 `$RUN/weights/weight-check.json`，三卡通信检查记录在 `$RUN/nccl-check.json`。通信检查成功不能替代模型训练预检。

当前预检已处理两个旧环境兼容问题：将 YAPF 从 `0.43` 调整到 `0.40.1`，兼容旧 MMCV 的配置格式化接口；在 Base 专用数据集中把 nuScenes `DetectionConfig.class_names` 的 `dict_keys` 转为列表，使 DataLoader 能使用 `spawn` 序列化。这些改动没有切换全局 multiprocessing 启动方式，也没有修改旧 Tiny 数据集。

随后实际训练前向在 Encoder 的空间注意力处发生 OOM。报错时请求再分配约 **182 MiB**；设备报告总量 **11.63 GiB**、PyTorch allocated **9.68 GiB**、reserved **10.02 GiB**，剩余约 **67 MiB**。这些是失败瞬间的不同口径统计，不能相加；此轮尚未完成一次优化器更新。证据保存在 `$RUN/evidence/backbone-checkpoint-oom.json`，本地副本见[失败配置记录](../../../docs/training/bevformer-base/backbone-checkpoint-oom.json)。

加入 Encoder 逐层重计算后的首次单卡完整预检结果：

| 检查项 | 实测结果 |
| --- | --- |
| 真实训练更新 | 4 次 AdamW 更新，每步均更新一次 |
| Loss | `24.398527 → 24.008858 → 23.677174 → 23.050928` |
| 模块梯度与参数 | 7 类模块均观察到有限梯度和参数探针变化，包括 DCNv2 offset |
| 历史模型 | 5 次同步检查通过 |
| 预测 | 3 个真实验证帧、共 900 个检测框，预测数值有限 |
| 检查点 | `work_dirs/smoke/epoch_1.pth` 已保存 |
| PyTorch 峰值 allocated | 7,124,145,664 字节，约 **6.63 GiB** |
| PyTorch 峰值 reserved | 8,350,859,264 字节，约 **7.78 GiB** |

以上内存数值不等于 `nvidia-smi` 的设备总显存占用。这证明该完整尺寸配置能够在这批真实数据上完成单卡训练预检；不保证任意批次、三卡通信和全量训练都没有显存峰值，也不代表模型已收敛。预检未计算官方 mAP/NDS，原始证据见[单卡实测报告](../../../docs/training/bevformer-base/single-gpu-smoke.json)。

加入运行指纹后的单卡复测也完成 4 次实际更新，Loss 为 `24.398527 → 24.000387 → 23.694605 → 23.036591`，峰值 allocated/reserved 与首次结果相同，并生成通过的运行证据。首次报告与[带门控的复测报告](../../../docs/training/bevformer-base/single-gpu-gated-smoke.json)分别保留，未用新结果覆盖旧结果。

正式三卡执行方式的短预检也已完成：

| 检查项 | 三卡实测结果 |
| --- | --- |
| 更新 | 每个 rank 都完成 4 次 AdamW 更新；是 4 次全局同步更新，不是 12 次全局更新 |
| 日志 Loss | `24.173092 → 23.891727 → 22.470551 → 21.555061` |
| 模块检查 | 每个 rank 的 7 类模块均有参数探针变化和有限梯度 |
| 历史模型 | 每个 rank 均通过 5 次同步检查 |
| 跨 rank 一致性 | 命名参数探针 SHA-256 相同 |
| 预测与检查点 | rank 0 汇总 3 帧、900 个有限预测框；检查点已保存并核对内容 |
| PyTorch 峰值 allocated | 每卡 7,124,906,496 字节，约 **6.64 GiB** |
| PyTorch 峰值 reserved | rank 0 为 8,394,899,456 字节，rank 1/2 为 8,396,996,608 字节，均约 **7.82 GiB** |

这仍是 mini 数据上的三卡执行预检，不是完成了一轮 trainval，也没有计算官方 mAP/NDS。参数一致性采用指定参数片段的 SHA-256，不等同于整个模型状态逐字节比较。详细结果与门控证据分别见[三卡实测报告](../../../docs/training/bevformer-base/three-gpu-smoke.json)、[三卡运行指纹](../../../docs/training/bevformer-base/three-gpu-runtime-evidence.json)。

## 2. 文件职责

| 文件 | 作用 |
| --- | --- |
| [prepare_data.py](prepare_data.py) | 从只读公共归档选择性提取训练所需数据，校验六相机、标定、标注、CAN 与划分，调用官方时序转换器生成索引 |
| [make_configs.py](make_configs.py) | 生成完整 Base 正式配置和单卡 mini 预检配置，不启动训练 |
| [make_ddp_config.py](make_ddp_config.py) | 在单卡预检配置基础上生成独立三卡 DDP 预检配置 |
| [base_training_support.py](base_training_support.py) | 检查模型和图像形状；构建含真实历史的短预检；检查梯度、参数探针更新和预测；按 rank 写健康记录 |
| [ddp_preflight.py](ddp_preflight.py) | 检查三卡真实更新、跨 rank 参数探针一致性、验证预测与检查点 |
| [checkpoint_layers.py](checkpoint_layers.py) | 包装完整 Encoder 层进行激活重计算，保留其数学运算与输入依赖 |
| [check_checkpoint_layers.py](check_checkpoint_layers.py) | 用小型可微算例检查重计算包装器的输出、梯度、RNG 和执行语义；不作为完整 Base 验收 |
| [launch.py](launch.py) | 检查前置报告与运行指纹、互斥启动、后台运行、状态与恢复；正式训练必须通过单卡和三卡预检 |
| [continue_train.py](continue_train.py) | 两道预检通过后等待已有全量数据准备任务，再启动三卡训练；当前已在后台启动 |
| [process_support.py](process_support.py) | 管理训练子进程及停止清理，避免 supervisor 退出后遗留训练 worker |
| [共享 training_support.py](../bevformer_tiny/training_support.py) | 复用历史模型同步、数据集评估与训练健康检查等公共逻辑 |

部署时将本目录的运行脚本和共享的 `training_support.py` 放到 `$RUN` 根目录。配置只导入 `base_training_support`，它会导入共享实现。第三方 BEVFormer 源码与 Python 环境单独安装，不包含在本目录中。

## 3. 数据范围与目录

读取公共数据的原始压缩包，直接把选定文件解压到私有实验目录，**不复制压缩包**。此数据配置用于完整相机训练：保留所有官方训练/验证帧，提取元数据、地图、六相机关键帧、`LIDAR_TOP` 关键帧和所需 CAN-bus，不提取雷达与 sweeps。LiDAR 关键帧用于兼容原转换器的检查，模型输入仍是六相机。

| 数据 | 划分 |
| --- | --- |
| mini | 8 个训练场景 / 323 帧；2 个验证场景 / 81 帧 |
| 正式 trainval | 700 个训练场景 / 28,130 帧；150 个验证场景 / 6,019 帧 |
| 预检训练 | 同一真实场景 8 个连续有效帧：前 4 个作为历史上下文，只暴露后 4 个为训练样本 |
| 预检验证 | 3 个连续真实验证帧，只检查预测结构与数值 |

8 帧预检 PKL 必须配合 `BaseSmokeTrainDataset` 使用，不能直接解释为 8 次训练更新。保留完整历史是为了实际执行 Base 的时序路径。正式训练使用全量 PKL，与预检索引分开。

```text
$RUN/
├── env.sh
├── weights/
│   ├── r101_dcn_fcos3d_pretrain.pth
│   └── weight-check.json
├── nccl-check.json
├── data/
│   ├── mini/                        mini 所需数据与 CAN-bus
│   └── trainval/                    全量相机训练所需数据与 CAN-bus
├── infos/
│   ├── nuscenes_infos_temporal_mini_{train,val}.pkl
│   ├── nuscenes_infos_temporal_smoke_{train,val}.pkl
│   └── nuscenes_infos_temporal_{train,val}.pkl
├── preparation_{mini,trainval}_report.json
├── data-preparation-{mini,trainval}-status.json
├── configs/
│   ├── base_smoke.py
│   ├── base_ddp_smoke.py
│   └── base_trainval.py
├── logs/
│   ├── prepare-mini.log
│   ├── smoke.log
│   ├── ddp-smoke.log
│   └── train.log
├── {smoke,ddp-smoke,train}-status.json
└── work_dirs/
    ├── smoke/                       预检报告与检查点
    ├── ddp-smoke/                   三卡独立预检报告与检查点
    └── trainval/                    三卡正式训练输出
```

准备报告的 `profile` 为 `camera_training`，不代表通过了全传感器、全 sweeps 的归档完整性验收。CAN 仅允许官方缺失的场景 `0161–0168`、`0170–0176`；其他所需场景缺失应失败。提取脚本保留磁盘空间余量并检查输出路径，原始公共数据保持只读。

## 4. 准备与单卡预检

以下命令适用于已部署的服务器。mini 准备、单卡和三卡预检均已通过，全量准备与后台继续流程正在运行；应先查看已有状态，不要重复执行相同阶段。原 OOM 记录保留为基线。

```bash
ssh -p 34280 root@connect.nmb2.seetacloud.com

RUN=/root/autodl-tmp/bevformer-base-trainval-20261019
REPO=/root/src/BEVFormer
PUBLIC=/root/autodl-pub/nuScenes
source "$RUN/env.sh"
export PYTHONPATH="$RUN:$REPO:${PYTHONPATH:-}"

cat "$RUN/data-preparation-mini-status.json"
tail -n 30 "$RUN/logs/prepare-mini.log"
```

首次准备 mini 的命令为：

```bash
python "$RUN/prepare_data.py" \
  --public-root "$PUBLIC" --output-root "$RUN" \
  --repo-root "$REPO" --mode mini
```

生成两个配置文件。`--data-root` 指向正式数据目录，预检配置会单独使用 `$RUN/data/mini`：

```bash
python "$RUN/make_configs.py" \
  --repo-root "$REPO" --task-root "$RUN" \
  --data-root "$RUN/data/trainval" \
  --weights "$RUN/weights/r101_dcn_fcos3d_pretrain.pth" \
  --world-size 3
```

mini 准备报告和权重检查的 `passed` 必须为 `true`，然后执行单卡预检：

```bash
python "$RUN/launch.py" --task-root "$RUN" --repo-root "$REPO" \
  --stage smoke --gpus 1 --detach

cat "$RUN/smoke-status.json"
tail -n 50 "$RUN/logs/smoke.log"
nvidia-smi
```

预检保持 Base 的全部尺寸与模块，要求 4 次实际 AdamW 更新、有限 Loss/梯度、各主要模块参数探针变化、检查点保存和 3 帧有限预测。模块覆盖 Backbone、DCNv2 offset、FPN、时序注意力、空间注意力、Decoder 和检测 Head。

完成后检查：

```bash
cat "$RUN/work_dirs/smoke/base_smoke_check.json"
cat "$RUN/work_dirs/smoke/base-architecture.json"
cat "$RUN/work_dirs/smoke/training-health-rank0.json"
```

必须同时看到 `smoke-status.json` 的状态为 `completed` 和 `base_smoke_check.json` 的 `passed=true`。进程启动、能做一次前向、甚至仅有检查点，都不能替代这些验收。如果出现 OOM，说明当前执行配置尚未适配每卡显存；不能把模型或分辨率缩小后仍称作同一完整 Base 验收。3 帧验证不报告官方 mAP/NDS。

新版启动器在预检成功后保存同目录的 `runtime-evidence.json`，记录配置及其继承文件、运行代码、权重、预检 PKL 和依赖版本的指纹。启动后续阶段时重新比对；相关内容改变必须重新预检，不能拿旧的 `passed=true` 代替新版本证据。本次新版单卡门控已经完成；[对应运行证据](../../../docs/training/bevformer-base/single-gpu-runtime-evidence.json)单独保存。

单卡通过后，再检查三卡真实训练；这与单独的 NCCL all-reduce 测试不同：

```bash
python "$RUN/make_ddp_config.py" --task-root "$RUN"

python "$RUN/launch.py" --task-root "$RUN" --repo-root "$REPO" \
  --stage ddp-smoke --gpus 3 --detach

cat "$RUN/ddp-smoke-status.json"
tail -n 50 "$RUN/logs/ddp-smoke.log"
cat "$RUN/work_dirs/ddp-smoke/base_ddp_smoke_check.json"
```

三卡预检把相同的 4 个真实训练目标重复 3 次，形成 12 个逻辑目标；每个 rank 执行 4 次更新，历史仍取自同一组 8 个连续真实帧。它是执行验证，不是增加了 12 个不同帧或进行全量训练。每个 rank 保存 `base_ddp_smoke_rank{rank}.json`；rank 0 汇总报告要求所有 rank 的真实更新、7 类模块梯度与参数变化、历史同步通过，并比较命名参数探针 SHA-256，检查 3 帧有限预测和保存的检查点。探针相同不是整个模型状态逐字节相同的证明。

本次三卡预检已满足上述检查：`ddp-smoke-status.json` 为 `completed`、退出码为 0、汇总 `passed=true`，并生成与当前环境匹配的 `runtime-evidence.json`。此结果满足当前版本正式训练的三卡门控。

## 5. 完整数据与三卡正式训练

准备全量数据及全量时序索引：

```bash
python "$RUN/prepare_data.py" \
  --public-root "$PUBLIC" --output-root "$RUN" \
  --repo-root "$REPO" --mode trainval --workers 3

cat "$RUN/preparation_trainval_report.json"
```

`--workers 3` 按归档并行提取；默认值为 1。中断后重新执行相同准备命令，可复用已完成归档的状态及已提取文件，最终仍须通过相机训练数据的完整检查；这不表示已提取 nuScenes 的全部传感器文件。

全量准备仍在后台运行，已使用 `continue_train.py` 等待它完成，再自动进入正式训练；本次后台继续流程 PID 为 `7377`，无需再次启动。本次使用的命令如下，其中准备进程 PID 为 `2090`；以后重启准备任务必须使用新的真实 PID：

```bash
python "$RUN/continue_train.py" --task-root "$RUN" --repo-root "$REPO" \
  --preparation-pid 2090 --gpus 3 --detach

cat "$RUN/pipeline-status.json"
```

继续流程核对准备进程身份及最终完整数据报告；准备失败或未完成时不会切换为 mini 训练。当前已启动该流程，不要再手动执行下面的正式启动命令。全量数据就绪后会自动从 FCOS3D 权重初始化正式模型，不载入预检训练权重。

全量准备、权重检查及与当前运行指纹匹配的单卡和三卡 DDP 预检全部通过后，才能启动正式训练：

```bash
python "$RUN/launch.py" --task-root "$RUN" --repo-root "$REPO" \
  --stage train --gpus 3 --detach
```

启动器使用 `torch.distributed.launch` 调用官方 `tools/fp16/train.py`，要求实际 GPU 进程数与配置中的 `base_world_size` 相同；`--detach` 使训练不依赖当前 SSH 会话。返回 supervisor PID 仅表示派发，训练是否正常需要确认步数与各 rank 的健康报告持续更新。

```bash
cat "$RUN/train-status.json"
tail -n 50 "$RUN/logs/train.log"
cat "$RUN/work_dirs/trainval/base-architecture.json"
cat "$RUN/work_dirs/trainval/training-health-rank0.json"
cat "$RUN/work_dirs/trainval/training-health-rank1.json"
cat "$RUN/work_dirs/trainval/training-health-rank2.json"
nvidia-smi
```

每个 rank 独立记录 CUDA 峰值 allocated/reserved、有限 Loss 与优化器更新等信息，避免多个进程相互覆盖报告。正式训练每 50 步更新健康记录；短预检每步更新。显存采样、PyTorch allocated 和 reserved 是不同统计口径。

## 6. 中断与恢复

正式训练中断后，选择完整保存的检查点恢复，例如：

```bash
python "$RUN/launch.py" --task-root "$RUN" --repo-root "$REPO" \
  --stage train --gpus 3 \
  --resume-from "$RUN/work_dirs/trainval/latest.pth" --detach
```

恢复加载模型、优化器与轮次，目标总量仍是 24 轮。检查点按轮保存，未保存的半轮进度需要重做。已有正式训练日志时，启动器拒绝不带 `--resume-from` 的重复从头训练；预检阶段不允许恢复优化器。

当前已有仅 Backbone 重计算的 OOM 基线，以及增加 Encoder 重计算后的单卡、三卡真实预检成功记录。正式全量训练仍等待数据准备，尚无其 Loss 曲线、完整验证 mAP/NDS 或完成时间结论。后续结果应对应实际配置、完整验证集与保存的报告；不能沿用之前 Tiny 实验的精度、吞吐或显存数据。

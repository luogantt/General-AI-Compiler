# BEVFormer-Tiny：完整 nuScenes trainval 单卡训练

这组脚本使用官方 BEVFormer 的 PyTorch FP16 训练路径，负责数据索引、实验配置、训练预检和进程状态。模型训练不经过本项目的 MLIR 推理编译器。

当前部署机器为 `robot@192.168.3.24`，RTX 3070 Ti Laptop 8 GB。原 mini 实验保留在 `/home/robot/project/bevformer-tiny-train-20261018`；全量实验单独使用 `/home/robot/project/bevformer-tiny-trainval-20261019`，复用已经验证的 Python 环境和 BEVFormer 源码。

## 文件职责

| 文件 | 功能 |
| --- | --- |
| `prepare_trainval.py` | 调用官方时序转换器；核对完整划分、相机、标定、CAN 和标注；生成全量与短预检 PKL |
| `make_configs.py` | 生成正式训练和短预检配置，避免修改 mini 配置 |
| `training_support.py` | 同步历史模型、验证真实参数更新、记录健康状态、释放评估元数据 |
| `launch.py` | 单卡分布式启动、互斥锁、后台日志、退出状态和检查点恢复 |
| `pipeline.py` | 串行执行数据准备、配置、真实训练预检与正式训练；任一步失败就停止 |

## 模型和超参数

使用完整 Tiny：ResNet-50 ImageNet 初始化、六相机、图像缩放 0.5、BEV 50×50、3 层 Encoder、6 层 Decoder、3 帧时序队列。正式实验从 ImageNet Backbone 开始，没有载入 mini 或短预检检查点。

| 参数 | 设置 |
| --- | --- |
| 数据版本 | `v1.0-trainval` |
| 训练 | 700 场景，28,130 帧 |
| 验证 | 150 场景，6,019 帧；每轮完整验证 |
| batch / workers | 单卡 1 / 2 |
| 训练轮次 | 24 |
| 优化器 | AdamW，weight decay 0.01，梯度裁剪 35 |
| 学习率 | `1.75e-5`；500 步 warmup，余弦衰减 |
| FP16 | 官方专用 FP16 入口，静态 loss scale 512 |
| 检查点 | 每轮保存，保留最近 3 轮 |
| 随机种子 | 0 |

学习率按官方 FP16 的全局 batch 16、学习率 `2.8e-4` 线性缩放到单卡 batch 1；这是起始设置，并非已调优的最佳配置，也不保证复现官方多卡精度。不要再加 `--autoscale-lr`。

## 准备和启动

以下命令针对现有服务器。`BEVFormer` 源码及其旧版依赖已单独安装，不包含在本目录中。不要在未经验证的普通 Python 环境直接运行。

```bash
ssh robot@192.168.3.24
source /home/robot/project/bevformer-tiny-train-20261018/env.sh

RUN=/home/robot/project/bevformer-tiny-trainval-20261019
REPO=/home/robot/project/bevformer-tiny-train-20261018/BEVFormer
DATA=/home/robot/datasets/nuscenes-trainval
export PYTHONPATH="$RUN:$REPO:$PYTHONPATH"
```

将本目录的五个运行脚本部署到 `$RUN`。原始数据位于 `$DATA`；已有下载校验报告必须通过。首次全量实验可以用一个后台任务自动执行全部阶段：

```bash
python "$RUN/pipeline.py" --task-root "$RUN" --repo-root "$REPO" \
  --data-root "$DATA" --detach
cat "$RUN/pipeline-status.json"
```

阶段依次为 `prepare → configs → smoke → train`。`prepare` 可能持续数十分钟，此时尚未进行正式参数更新；前一阶段退出成功后才执行下一阶段。索引准备完成还必须通过真实训练预检，正式训练才能启动。不要在自动流程运行期间另外执行下面的手动阶段命令，以免重复任务。

也可以分阶段手动执行。生成索引的过程只读原始数据，PKL 写入独立实验目录：

```bash
python "$RUN/prepare_trainval.py" \
  --repo-root "$REPO" --data-root "$DATA" --output-dir "$RUN/infos"

python "$RUN/make_configs.py" \
  --repo-root "$REPO" --task-root "$RUN" --data-root "$DATA"
```

索引已存在时，用 `--reuse-infos` 重新核对；不会默认覆盖已有的全量索引。CAN 的官方缺失场景仅允许 `0161–0168`、`0170–0176`，其他场景缺失或异常不得静默填零。转换器原有 CAN 数值选择逻辑保持不变；缓存返回新的字典，避免原转换器的 `pop()` 修改下一次读取结果。

先执行 4 步真实训练及 3 帧预测检查：

```bash
python "$RUN/launch.py" --task-root "$RUN" --repo-root "$REPO" \
  --stage smoke --detach
```

等待 `$RUN/smoke-status.json` 为 `completed`，并检查 `$RUN/work_dirs/smoke/smoke_training_check.json` 的 `passed` 为 `true`。该检查实际验证 loss、4 次 AdamW 更新、参数变化、检查点和预测数值，不使用少量样本报告官方 mAP/NDS。

检查通过后启动完整训练：

```bash
python "$RUN/launch.py" --task-root "$RUN" --repo-root "$REPO" \
  --stage train --detach
```

`--detach` 使训练不依赖 SSH 会话。返回 supervisor PID 只说明已派发；启动成功应同时看到 `train-status.json` 为 `running`、训练日志持续推进、健康报告中的 AdamW 步数增加。

## 查看进度和恢复

```bash
cat "$RUN/train-status.json"
cat "$RUN/work_dirs/trainval/training-health.json"
tail -n 30 "$RUN/logs/train.log"
nvidia-smi
```

健康报告每 50 步更新，记录有限 loss、实际优化器步数、FP16 跳步次数及 CUDA 峰值分配/预留。单次 FP16 overflow 跳步会被记录；反复跳步需调查，不能只看进程存活判断训练正常。

如任务中断，选择正式训练目录中已完整保存的检查点恢复，例如：

```bash
python "$RUN/launch.py" --task-root "$RUN" --repo-root "$REPO" \
  --stage train --resume-from "$RUN/work_dirs/trainval/latest.pth" --detach
```

恢复会加载模型、优化器和轮次，总目标仍为 24 轮。检查点按轮次保存，未保存的半轮进度需要重做；不支持宣称精确恢复到任意 batch。已有正式日志时，启动器拒绝不带 `--resume-from` 的重复从头训练。

训练指标以完整验证集产生的 mAP/NDS 为准。数据校验、短预检和前几十步正常，都不能证明模型已经收敛。

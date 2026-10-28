# BEVFormer-Tiny 单卡训练记录与恢复说明

**2026-10-19 更新：已按用户要求启动独立的完整 trainval 训练流程。** 启动记录处于 `prepare` 索引准备阶段，尚未开始正式优化器更新；准备和真实训练预检通过后会自动进入训练，无需再次手动启动。新任务不载入 mini 检查点。详见[第 8 节](#8-2026-10-19完整-trainval-训练任务)和[训练工具说明](../../../tools/training/bevformer_tiny/README.md)。

**原 mini 实验状态：已停止，检查点保留。** 停止时间为 **2026-10-18 21:09:44（北京时间）**。已完成 6 个 mini 训练轮次及对应验证；第 7 轮最后一条训练日志为 `150/323`。保留的最新检查点是 `epoch_6.pth`，恢复该旧实验时从第 7 轮开始，未保存的第 7 轮进度不会恢复。新全量任务不会修改此检查点。

以下第 1～7 节记录原 mini 实验；第 8 节记录新全量任务。两者使用官方 BEVFormer-Tiny FP16 模型和 PyTorch，覆盖图像 Backbone、时序 BEV Encoder 与检测 Decoder 的训练流程。本项目的 MLIR/Vortex/CUDA 编译器用于推理，不承担这些训练的反向传播。

## 1. 已完成什么

- 建立独立 Python/CUDA 构建环境，未替换原 TensorRT 环境或系统驱动。
- 完成 4 次真实小样本反向传播，逐次确认 AdamW 更新；保存检查点并检查真实验证预测。
- 完成全部 81 帧 mini_val 的官方评估预检查。
- 正式训练完成 6 轮，每轮使用 323 帧训练样本，并在 81 帧验证集上计算 mAP/NDS。
- 停止后确认训练进程已全部退出；GPU 计算进程列表为空，显存占用降至约 15 MiB。
- 使用 CPU 重新读取 `epoch_6.pth`：检查点可读，包含模型和优化器状态。

**尚未收敛。** Loss 在下降，但验证 mAP 仍很低；不能把流程跑通、Loss 下降或 mini 结果当作完整 nuScenes 上的精度验收。

## 2. 模型与训练配置

基础配置为 BEVFormer 源码中的 `projects/configs/bevformer_fp16/bevformer_tiny_fp16.py`，本次覆盖配置为 `local_training/bevformer_tiny_fp16_mini.py`。

| 项目 | 本次设置 |
| --- | --- |
| 模型 | 完整 BEVFormer-Tiny FP16 |
| 图像 Backbone | ResNet-50，ImageNet 预训练初始化；沿用基础配置的冻结阶段与 BN 设置 |
| 图像输入 | 6 路相机；原图缩放 0.5 至 450×800，按 32 对齐后为 480×800 |
| 图像特征 | C5 单尺度，经 FPN 转为 256 通道 |
| 时序队列 | 3 帧；使用真实标定、历史输入和 CAN-bus 信息 |
| BEV | 50×50，特征维度 256 |
| Encoder / Decoder | 3 层 / 6 层；FFN 中间维度 512 |
| 检测 | 900 个 query，nuScenes 10 类 |
| 单卡 batch / workers | 1 / 2 |
| 优化器 | AdamW，weight decay 0.01，梯度裁剪 max norm 35 |
| 初始学习率 | `1.75e-5` |
| FP16 | 官方 FP16 训练入口，静态 loss scale 512 |
| 计划训练量 | 24 epochs；实际完成 6 轮后按要求停止 |
| 验证 / 保存 | 每轮一次；保留最近 3 个检查点 |
| 随机种子 | 0 |

学习率由原 FP16 配置的 `2.8e-4`、8 卡×每卡 2 样本，按总 batch 线性缩放到 1 卡×1 样本。这只是保守起点，尚未搜索最佳超参数。训练从 ImageNet Backbone 权重开始，没有载入训练好的完整 BEVFormer 权重，也没有使用小样本预检查产生的模型作为正式训练初始化。

官方 FP16 路径使用独立的历史帧模型进行无梯度历史特征计算。本次增加的 `SyncHistoryBeforeRunHook` 在初始化或恢复检查点后同步历史模型权重，避免恢复后的第一批次使用旧历史权重；后续沿用 `TransferWeight` 更新机制。

## 3. 数据在哪里

| 内容 | 服务器路径 / 数量 |
| --- | --- |
| 原始 mini 数据 | `/home/robot/datasets/nuscenes` |
| 训练集 | `mini_train`：8 个 scene，323 帧 |
| 验证集 | `mini_val`：2 个 scene，81 帧 |
| 训练任务的数据入口 | `/home/robot/project/bevformer-tiny-train-20261018/BEVFormer/data/nuscenes/` |
| 时序训练索引 | 上述入口中的 `nuscenes_infos_temporal_train.pkl` |
| 时序验证索引 | 上述入口中的 `nuscenes_infos_temporal_val.pkl` |
| CAN-bus | `BEVFormer/data/can_bus/`，已准备 mini 的 10 个 scene |

任务中的 `samples`、`sweeps`、`maps` 和 `v1.0-mini` 通过软链接指向原始数据，生成的时序 PKL 保存在任务目录。图像和 CAN-bus 均为真实数据，没有用随机图像或全零 CAN-bus 代替本次训练输入。

完整 trainval 数据另行下载至 `/home/robot/datasets/nuscenes-trainval`，与现有 mini 分开。本节的 PKL 和已报告训练结果均属于 mini。新全量任务重新生成对应时序索引、检查 CAN-bus 并生成独立配置，不沿用 mini PKL。

**2026-10-19 数据更新：完整 trainval 已下载并解压。** 11 个归档共 314,886,603,672 字节（314.89 GB），均通过 MD5 校验；逐条检查的 2,631,083 个传感器文件引用均存在且非空。官方划分为 700 个训练场景 / 28,130 帧、150 个验证场景 / 6,019 帧，全部 34,149 帧均覆盖六相机。CAN-bus 已解压；除官方缺失的 15 个场景外，其余 835 个场景的 pose 数据检查通过。

首次完整性检查因校验脚本把官方 `nbr_samples` 字段误写为 `nb_samples` 而报错；修正后已重新执行检查，结果为零错误，见 [trainval 校验报告](trainval-validation.json)。检查涵盖归档完整性、引用文件、六相机、CAN 与划分，不包含逐张图像解码或全部标注关系的语义验证。全量时序 PKL 由第 8 节的新任务生成。

## 4. 实测结果

以下指标来自每轮全部 81 帧 mini_val 的官方评估。mAP、NDS 在原始 JSON 中为 0～1 的小数，下表统一转换为百分比。

| 完成轮次 | 日志区间平均 Loss | 验证 mAP | 验证 NDS |
| ---: | ---: | ---: | ---: |
| 1 | 20.6009 | 0.000% | 2.127% |
| 2 | 17.7543 | 0.005% | 1.981% |
| 3 | 16.7175 | 0.020% | 3.136% |
| 4 | 15.7989 | 0.136% | 3.745% |
| 5 | 14.9432 | 0.156% | 3.880% |
| 6 | 14.1637 | **0.177%** | **4.155%** |

Loss 每 10 步记录一次，表中平均值覆盖每轮记录到的 320/323 步，不是精确的整轮 323 步均值。第 7 轮未完成，不作为完整轮次加入表格。

训练日志中后期约 **1.1～1.2 秒/步**，包含数据等待、前向与反向等训练开销，不能解释成推理延时。早期采样的 GPU 总显存约 **4307 MiB**；训练日志记录的 PyTorch 峰值分配约 **2696～2697 MiB**，两者统计口径不同，也不是对所有训练设置的显存保证。

已记录的 Loss 没有出现非有限值。停止时的 `KeyboardInterrupt`、退出码 `-2` 与相关 traceback 来自用户要求的 SIGINT 停止，状态已明确标为 `stopped_by_user`，不能据此判定此前训练失败。

可核查的记录：

- [训练与验证统计、停止状态](convergence-check.json)
- [最新检查点读取验证](checkpoint-check.json)

## 5. 机器与环境

| 项目 | 实际版本 / 配置 |
| --- | --- |
| SSH | `robot@192.168.3.24` |
| 操作系统 | Ubuntu 24.04.3 |
| GPU | RTX 3070 Ti Laptop GPU，8192 MiB |
| 驱动 | 580.178.04 |
| Python | 3.8.20，任务内独立环境 |
| PyTorch / torchvision | 1.9.1+cu111 / 0.10.1+cu111 |
| MMCV-full | 1.4.0 |
| MMDetection / MMSegmentation | 2.14.0 / 0.14.1 |
| MMDetection3D | 0.17.1，本机编译扩展 |
| Detectron2 | 0.6，cu111 版本 |
| nuScenes-devkit | 1.1.9 |
| NumPy / Numba / llvmlite | 1.19.5 / 0.48 / 0.31 |
| 扩展构建工具链 | CUDA 11.3、GCC 9，目标 `sm_86` |

PyTorch 的 CUDA 11.1 wheel 运行库与用于扩展构建的 CUDA 11.3 工具链是两个不同版本来源。本表记录已验证的环境，不要求系统全局 CUDA 恰好等于某个版本。

`env.sh` 设置本任务的 PATH、CUDA_HOME、PYTHONPATH 和线程数。旧版 Numba 所带 LLVM 在此 CPU 上的特性识别会崩溃，因此使用 `NUMBA_CPU_NAME=generic`；这是 CPU JIT 设置，不会关闭 CUDA 训练。普通 Python/Conda 依赖采用清华镜像，特定 CUDA wheel 与数据使用官方来源。

完整安装列表保存在服务器任务目录的 `dependencies/requirements-installed.txt`；源码快照和下载权重的校验分别见 `source-manifest.json` 与 `dependencies/pretrained-manifest.json`。

## 6. 日志、代码与检查点

服务器任务根目录：

```text
/home/robot/project/bevformer-tiny-train-20261018/
├── env/                         Python 环境
├── toolchain/                   CUDA / GCC 构建工具链
├── env.sh                       进入本任务环境
├── launch_training.py           首次后台启动与状态记录
├── training-status.json         当前状态：stopped_by_user
├── logs/train.log               完整训练输出，含人工停止记录
├── dependencies/                安装版本、CAN-bus、权重下载记录
└── BEVFormer/
    ├── tools/fp16/train.py      官方训练入口
    ├── local_training/
    │   ├── prepare_tiny_training.py
    │   ├── bevformer_tiny_fp16_mini.py
    │   ├── tiny_training_support.py
    │   ├── run_smoke.sh
    │   └── run_train.sh
    ├── data/                    数据软链接与时序 PKL
    └── work_dirs/
        ├── bevformer_tiny_fp16_smoke/   小样本流程验证
        └── bevformer_tiny_fp16_mini/
            ├── 20261018_202326.log.json
            ├── epoch_4.pth
            ├── epoch_5.pth
            └── epoch_6.pth      最近的可恢复检查点
```

`epoch_6.pth` 大小为 **400,785,914 字节**；读取出的 `epoch=6`、`iter=1938`，包含 643 个 `state_dict` 键和优化器状态。该文件保留在服务器，不包含在此 Git 文档目录中。本文命令面向上述已配置服务器；这里只保存说明与结果证据，没有把第三方 BEVFormer 源码、Python 环境或训练权重复制到文档目录。

## 7. 查看旧 mini 状态与手动恢复

**以下是旧 mini 实验的恢复命令，不是新全量任务的启动命令。旧实验已停止，没有执行恢复；全量自动流程运行期间，不要同时启动此实验争用 GPU。**

登录后先载入环境：

```bash
ssh robot@192.168.3.24
cd /home/robot/project/bevformer-tiny-train-20261018
source env.sh
cd BEVFormer
```

查看现有记录：

```bash
cat ../training-status.json
tail -n 60 ../logs/train.log
ls -lh work_dirs/bevformer_tiny_fp16_mini/epoch_*.pth
nvidia-smi
```

**从第 6 轮检查点恢复，在前台继续第 7～24 轮：**

```bash
bash local_training/run_train.sh \
  --resume-from "$PWD/work_dirs/bevformer_tiny_fp16_mini/epoch_6.pth"
```

`run_train.sh` 会先确认小样本预检查已通过，然后以单卡分布式入口启动官方 FP16 训练。使用 `--resume-from` 恢复模型、优化器和轮次；不加该参数会从 ImageNet Backbone 初始化重新开始。恢复不是额外训练 24 轮，目标总轮数仍为配置中的 24。

这个命令在前台运行，SSH 会话退出可能影响训练。它也不会更新旧的 supervisor 状态文件 `training-status.json`，新训练以终端及 `work_dirs` 中新生成的日志为准。现有 `launch_training.py` 是首次后台启动器，**不支持通过 CLI 转发 `--resume-from`**，不要把恢复参数直接加到它后面。

已有数据索引和预检查结果可直接用于同配置恢复。若环境或数据路径改变，应先检查配置，再按需重新执行：

```bash
python local_training/prepare_tiny_training.py
bash local_training/run_smoke.sh
```

小样本检查会实际更新其独立模型并保存到 `bevformer_tiny_fp16_smoke/`；不会替代正式 `epoch_6.pth`。更换为全量 trainval、变更模型结构或重做数据划分，属于新的训练实验，应使用独立配置与输出目录。

## 8. 2026-10-19：完整 trainval 训练任务

用户已授权使用新下载的完整数据，在同一台 `192.168.3.24` 的 RTX 3070 Ti Laptop 8GB 上训练 Tiny。后台流程已启动，启动记录的 pipeline PID 为 `37080`，阶段为 **`prepare`**。这个 PID 表示准备与训练的管理进程已启动，不代表已执行正式反向传播；后续状态以服务器 JSON 和日志为准。

| 项目 | 新任务设置 |
| --- | --- |
| 独立任务目录 | `/home/robot/project/bevformer-tiny-trainval-20261019` |
| 数据根目录 | `/home/robot/datasets/nuscenes-trainval` |
| 完整训练集 | 700 场景，28,130 帧 |
| 完整验证集 | 150 场景，6,019 帧 |
| 复用环境与源码 | 原 `/home/robot/project/bevformer-tiny-train-20261018/` 内的环境及 BEVFormer 源码 |
| 模型与初始化 | 完整 Tiny FP16；从 ImageNet ResNet-50 初始化，不恢复 mini 或预检权重 |
| 单卡 batch / workers | 1 / 2 |
| 初始学习率 / 目标轮次 | `1.75e-5` / 24 epochs |
| 验证 / 检查点 | 每轮完整验证、每轮保存，保留最近 3 轮 |
| 正式输出目录 | 新任务下的 `work_dirs/trainval/` |

后台流程按依赖顺序执行：

```text
prepare：读取完整数据，生成并核对时序索引
    ↓ 成功
configs：生成独立正式配置与预检配置
    ↓ 成功
smoke：4 步真实训练更新 + 3 帧预测数值检查
    ↓ 检查通过
train：从 ImageNet 初始化，完整 trainval 正式训练
```

任一阶段失败，流程停止并记录错误；只有前一阶段成功才执行下一阶段。预检会检查有限 Loss、实际 AdamW 更新、参数变化、检查点与预测数值，不用 3 帧结果宣称官方精度。正式训练自动进入，无需等待用户再执行启动命令，也不要在自动流程运行期间另外启动一份训练。

可通过 SSH 查看新任务状态：

```bash
ssh robot@192.168.3.24
RUN=/home/robot/project/bevformer-tiny-trainval-20261019
cat "$RUN/pipeline-status.json"
```

进入正式训练后再查看以下文件；`prepare` 阶段没有这些文件属于正常情况：

```bash
cat "$RUN/train-status.json"
cat "$RUN/work_dirs/trainval/training-health.json"
tail -n 30 "$RUN/logs/train.log"
nvidia-smi
```

健康报告记录 Loss、优化器更新和 FP16 跳步等信息。正式训练是否开始需要看到实际步数推进，是否收敛需要完整验证集的 mAP/NDS；当前启动记录不包含全量训练精度或完成时间结论。

相关文件：

- [全量任务启动记录](trainval-startup.json)
- [脚本职责、分阶段命令与后台恢复说明](../../../tools/training/bevformer_tiny/README.md)
- [完整数据校验报告](trainval-validation.json)

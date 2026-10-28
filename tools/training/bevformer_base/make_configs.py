"""Generate full BEVFormer-Base configurations; no training is launched."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo-root", "task-root", "data-root", "weights"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=3)
    args = parser.parse_args()
    if args.world_size < 1:
        raise ValueError("world-size must be positive")
    repo, task, data, weights = (path.expanduser().resolve() for path in
                                (args.repo_root, args.task_root, args.data_root, args.weights))
    base = repo / "projects/configs/bevformer/bevformer_base.py"
    for path in (base, weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    directory = task / "configs"
    directory.mkdir(parents=True, exist_ok=True)
    train = f'''# Full Base: R101 caffe/DCNv2, 4 FPN levels, BEV200, encoder6/decoder6,
# six full-resolution cameras, queue4. FP16 and recomputation change execution only.
_base_ = [{str(base)!r}]
custom_imports = dict(imports=['base_training_support'], allow_failed_imports=False)
base_world_size = {args.world_size}
model = dict(
    type='BEVFormer_fp16', img_backbone=dict(with_cp=True),
    pts_bbox_head=dict(transformer=dict(encoder=dict(
        transformerlayers=dict(type='CheckpointedBEVFormerLayer')))))
data_root = {str(data) + '/'!r}
info_root = {str(task / 'infos') + '/'!r}
data = dict(
    samples_per_gpu=1, workers_per_gpu=2,
    train=dict(type='BaseTrainvalNuScenesDataset', data_root=data_root,
               ann_file=info_root + 'nuscenes_infos_temporal_train.pkl'),
    val=dict(type='BaseTrainvalNuScenesDataset', data_root=data_root, samples_per_gpu=1,
             ann_file=info_root + 'nuscenes_infos_temporal_val.pkl'),
    test=dict(type='BaseTrainvalNuScenesDataset', data_root=data_root, samples_per_gpu=1,
              ann_file=info_root + 'nuscenes_infos_temporal_val.pkl'))
# Original Base uses lr=2e-4 with 8 GPUs x 1 sample. Do not also pass --autoscale-lr.
optimizer = dict(lr={2e-4 / 8 * args.world_size!r})
total_epochs = 24
runner = dict(type='EpochBasedRunner_video', max_epochs=24)
fp16 = dict(loss_scale=512.)
evaluation = dict(interval=1)
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
log_config = dict(interval=50)
work_dir = {str(task / 'work_dirs/trainval')!r}
load_from = {str(weights)!r}
resume_from = None
custom_hooks = [
    dict(type='TransferWeight', priority='LOWEST'),
    dict(type='SyncHistoryBeforeRunHook', priority='LOWEST'),
    dict(type='BaseArchitectureCheckHook', expected_world_size={args.world_size},
         expected_version='v1.0-trainval', expected_train_samples=28130,
         priority='LOWEST'),
    dict(type='BaseTrainingHealthHook', interval=50, priority='LOWEST')]
'''
    smoke = f'''# Mini preflight ONLY: 8 stored frames, 4 context + 4 training targets.
# Full Base tensor sizes and all modules are preserved; no full-dataset claim.
_base_ = ['./base_trainval.py']
base_world_size = 1
data_root = {str(task / 'data/mini') + '/'!r}
data = dict(
    train=dict(type='BaseSmokeTrainDataset', context_frames=4, data_root=data_root,
               ann_file={str(task / 'infos/nuscenes_infos_temporal_smoke_train.pkl')!r}),
    val=dict(type='BaseSmokeNuScenesDataset', data_root=data_root,
             ann_file={str(task / 'infos/nuscenes_infos_temporal_smoke_val.pkl')!r}),
    test=dict(type='BaseSmokeNuScenesDataset', data_root=data_root,
              ann_file={str(task / 'infos/nuscenes_infos_temporal_smoke_val.pkl')!r}))
optimizer = dict(lr=2.5e-5)
total_epochs = 1
runner = dict(max_epochs=1)
lr_config = dict(warmup_iters=1)
log_config = dict(interval=1)
work_dir = {str(task / 'work_dirs/smoke')!r}
custom_hooks = [
    dict(type='TransferWeight', priority='LOWEST'),
    dict(type='SyncHistoryBeforeRunHook', priority='LOWEST'),
    dict(type='BaseArchitectureCheckHook', expected_world_size=1,
         expected_version='v1.0-mini', expected_train_samples=4, priority='LOWEST'),
    dict(type='BaseTrainingHealthHook', interval=1, priority='LOWEST'),
    dict(type='BaseSmokeCheckHook', expected_steps=4, priority='LOWEST')]
'''
    for name, text in (("base_trainval.py", train), ("base_smoke.py", smoke)):
        path = directory / name
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise FileExistsError("Refusing to overwrite a different configuration: " + str(path))
        path.write_text(text, encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()

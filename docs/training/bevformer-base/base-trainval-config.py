# Full Base: R101 caffe/DCNv2, 4 FPN levels, BEV200, encoder6/decoder6,
# six full-resolution cameras, queue4. FP16 and recomputation change execution only.
_base_ = ['/root/src/BEVFormer/projects/configs/bevformer/bevformer_base.py']
custom_imports = dict(imports=['base_training_support'], allow_failed_imports=False)
base_world_size = 3
model = dict(type='BEVFormer_fp16', img_backbone=dict(with_cp=True))
model['pts_bbox_head'] = dict(transformer=dict(encoder=dict(transformerlayers=dict(type='CheckpointedBEVFormerLayer'))))
data_root = '/root/autodl-tmp/bevformer-base-trainval-20261019/data/trainval/'
info_root = '/root/autodl-tmp/bevformer-base-trainval-20261019/infos/'
data = dict(
    samples_per_gpu=1, workers_per_gpu=2,
    train=dict(type='BaseTrainvalNuScenesDataset', data_root=data_root,
               ann_file=info_root + 'nuscenes_infos_temporal_train.pkl'),
    val=dict(type='BaseTrainvalNuScenesDataset', data_root=data_root, samples_per_gpu=1,
             ann_file=info_root + 'nuscenes_infos_temporal_val.pkl'),
    test=dict(type='BaseTrainvalNuScenesDataset', data_root=data_root, samples_per_gpu=1,
              ann_file=info_root + 'nuscenes_infos_temporal_val.pkl'))
# Original Base uses lr=2e-4 with 8 GPUs x 1 sample. Do not also pass --autoscale-lr.
optimizer = dict(lr=7.500000000000001e-05)
total_epochs = 24
runner = dict(type='EpochBasedRunner_video', max_epochs=24)
fp16 = dict(loss_scale=512.)
evaluation = dict(interval=1)
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
log_config = dict(interval=50)
work_dir = '/root/autodl-tmp/bevformer-base-trainval-20261019/work_dirs/trainval'
load_from = '/root/autodl-tmp/bevformer-base-trainval-20261019/weights/r101_dcn_fcos3d_pretrain.pth'
resume_from = None
custom_hooks = [
    dict(type='TransferWeight', priority='LOWEST'),
    dict(type='SyncHistoryBeforeRunHook', priority='LOWEST'),
    dict(type='BaseArchitectureCheckHook', expected_world_size=3,
         expected_version='v1.0-trainval', expected_train_samples=28130,
         priority='LOWEST'),
    dict(type='BaseTrainingHealthHook', interval=50, priority='LOWEST')]

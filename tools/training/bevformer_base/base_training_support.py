"""Base-specific checks, retaining the upstream full-resolution architecture.

Deploy training_support.py from the Tiny tooling beside this file. Import only
base_training_support in the config: it imports and registers shared hooks once.
"""

import json
from pathlib import Path

import torch
from mmcv.runner import HOOKS, Hook
from mmdet.datasets import DATASETS

from checkpoint_layers import CheckpointedBEVFormerLayer
from training_support import (
    SyncHistoryBeforeRunHook, TrainingHealthHook, TrainvalNuScenesDataset,
    SmokeTrainingCheckHook, TinySmokeNuScenesDataset, _write_json,
)


def _world_size():
    return (torch.distributed.get_world_size()
            if torch.distributed.is_initialized() else 1)


def _model(wrapper):
    return getattr(wrapper, "module", wrapper)


def _normalize_evaluation_config(dataset):
    # nuScenes DetectionConfig stores class_range.keys(), a dict view that
    # cannot be pickled when DataLoader starts workers with spawn. A list keeps
    # the same class order and evaluation meaning without changing the SDK.
    config = dataset.eval_detection_configs
    config.class_names = list(config.class_names)


@DATASETS.register_module()
class BaseTrainvalNuScenesDataset(TrainvalNuScenesDataset):
    """Base-local spawn compatibility, with unchanged full nuScenes metrics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _normalize_evaluation_config(self)


@DATASETS.register_module()
class BaseSmokeTrainDataset(BaseTrainvalNuScenesDataset):
    """Expose four targets while keeping four earlier frames for real history.

    Sampler indices and flag are 0..3. Only prepare_train_data shifts the index;
    upstream history/annotation lookups still address all eight data_infos.
    Inherited __getitem__ retries remain in the sampler's 0..3 index space.
    """

    def __init__(self, context_frames=4, *args, **kwargs):
        self._context_offset = 0
        super().__init__(*args, **kwargs)
        if self.version != "v1.0-mini" or context_frames != 4 or len(self.data_infos) != 8:
            raise ValueError("Base smoke requires eight mini frames: four context and four targets.")
        infos = self.data_infos
        if len({info["scene_token"] for info in infos}) != 1 or not all(
                a["next"] == b["token"] and b["prev"] == a["token"]
                for a, b in zip(infos, infos[1:])):
            raise ValueError("Base smoke frames must be contiguous within one real scene.")
        if self.queue_length != 4:
            raise ValueError("Full Base requires queue_length=4.")
        self.flag = self.flag[context_frames:].copy()
        self._context_offset = context_frames

    def __len__(self):
        return super().__len__() - self._context_offset

    def prepare_train_data(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        return super().prepare_train_data(index + self._context_offset)


@DATASETS.register_module()
class BaseSmokeNuScenesDataset(TinySmokeNuScenesDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _normalize_evaluation_config(self)

    def evaluate(self, results, **kwargs):
        if self.version != "v1.0-mini" or len(self.data_infos) != 3:
            raise ValueError("Base smoke validation must contain exactly three mini frames.")
        return super().evaluate(results, **kwargs)


@HOOKS.register_module()
class BaseTrainingHealthHook(TrainingHealthHook):
    """Each rank owns its report; one rank cannot overwrite another's health."""

    def _save(self, runner, report):
        report = dict(report, rank=getattr(runner, "rank", 0), world_size=_world_size())
        if torch.cuda.is_available():
            report["cuda_max_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["cuda_max_reserved_bytes"] = torch.cuda.max_memory_reserved()
        _write_json(Path(runner.work_dir) / ("training-health-rank%d.json" % report["rank"]), report)


@HOOKS.register_module()
class BaseArchitectureCheckHook(Hook):
    """Reject reduced models or image sizes before accepting a Base experiment."""

    def __init__(self, expected_world_size, expected_version, expected_train_samples):
        self.world_size = expected_world_size
        self.version = expected_version
        self.samples = expected_train_samples

    @staticmethod
    def _architecture(model):
        backbone, neck, head = model.img_backbone, model.img_neck, model.pts_bbox_head
        checks = {
            "fp16_model": type(model).__name__ == "BEVFormer_fp16",
            "resnet101": backbone.depth == 101 and
                [len(getattr(backbone, name)) for name in backbone.res_layers] == [3, 4, 23, 3],
            "caffe_backbone": backbone.style == "caffe",
            "dcn_stages": tuple(backbone.stage_with_dcn) == (False, False, True, True),
            "dcn_offset_parameters": any("conv_offset" in name for name, _ in backbone.named_parameters()),
            "checkpointing": backbone.with_cp is True,
            "fpn_four_levels": neck.num_outs == 4 and list(neck.in_channels) == [512, 1024, 2048],
            "bev_200": head.bev_h == 200 and head.bev_w == 200,
            "queries_900": head.num_query == 900,
            "encoder_six": len(head.transformer.encoder.layers) == 6,
            "encoder_checkpointing": all(isinstance(layer, CheckpointedBEVFormerLayer)
                                         for layer in head.transformer.encoder.layers),
            "decoder_six": len(head.transformer.decoder.layers) == 6,
        }
        if not all(checks.values()):
            raise RuntimeError("Full Base architecture mismatch: " + json.dumps(checks))
        return checks

    def before_run(self, runner):
        if _world_size() != self.world_size:
            raise RuntimeError("Configuration requires %d GPU processes, found %d." %
                               (self.world_size, _world_size()))
        self.checks = self._architecture(_model(runner.model))
        if getattr(runner, "eval_model", None) is None:
            raise RuntimeError("FP16 Base requires the dedicated history model.")
        self._architecture(_model(runner.eval_model))
        self.calls = {"training_backbone": 0, "history_backbone": 0}
        self.handles = []
        for label, wrapper in (("training_backbone", runner.model),
                               ("history_backbone", runner.eval_model)):
            def check_image(module, inputs, label=label):
                if tuple(inputs[0].shape) != (6, 3, 928, 1600):
                    raise RuntimeError("Full Base expects six 928x1600 padded images, got %s." %
                                       (tuple(inputs[0].shape),))
                self.calls[label] += 1
            self.handles.append(_model(wrapper).img_backbone.register_forward_pre_hook(check_image))
        self._save(runner)

    def before_train_epoch(self, runner):
        dataset = runner.data_loader.dataset
        if (dataset.version != self.version or len(dataset) != self.samples
                or dataset.queue_length != 4 or tuple(dataset.bev_size) != (200, 200)):
            raise RuntimeError("Base dataset version, sample count, queue or BEV shape mismatch.")

    def _save(self, runner):
        if getattr(runner, "rank", 0) == 0:
            _write_json(Path(runner.work_dir) / "base-architecture.json", dict(
                architecture_checks=self.checks, image_shape=[6, 3, 928, 1600],
                observed_backbone_calls=self.calls, world_size=_world_size(),
                dataset_version=self.version, training_samples=self.samples))

    def after_train_iter(self, runner):
        if runner.inner_iter == 0 or (runner.iter + 1) % 50 == 0:
            self._save(runner)

    def after_run(self, runner):
        self._save(runner)
        for handle in self.handles:
            handle.remove()


@HOOKS.register_module()
class BaseSmokeCheckHook(SmokeTrainingCheckHook):
    """Require real updates in DCNv2 and each major Base module on one GPU."""

    def before_run(self, runner):
        if _world_size() != 1:
            raise RuntimeError("BaseSmokeCheckHook is a single-GPU preflight; use a separate DDP check.")
        super().before_run(runner)
        params = dict((name, param) for name, param in runner.model.named_parameters()
                      if param.requires_grad and param.numel())
        patterns = {
            "backbone": lambda n: n.startswith("img_backbone.layer2.") and "conv_offset" not in n,
            "dcn_offset": lambda n: n.startswith("img_backbone.") and "conv_offset" in n,
            "fpn": lambda n: n.startswith("img_neck."),
            "temporal_attention": lambda n: ".encoder.layers.0.attentions.0." in n,
            "spatial_attention": lambda n: ".encoder.layers.0.attentions.1." in n,
            "decoder": lambda n: ".transformer.decoder." in n,
            "head": lambda n: ".cls_branches." in n or ".reg_branches." in n,
        }
        self.groups = {}
        for group, matches in patterns.items():
            names = [name for name in params
                     if matches(name[7:] if name.startswith("module.") else name)]
            if not names:
                raise RuntimeError("No trainable Base parameters for smoke group: " + group)
            self.groups[group] = list(dict.fromkeys(names[:4] + names[-4:]))
        self.probes = {name: params[name] for names in self.groups.values() for name in names}
        self.initial = {name: param.detach().reshape(-1)[:256].float().cpu().clone()
                        for name, param in self.probes.items()}
        self.gradient_observations = {name: 0 for name in self.groups}
        self.nonfinite_gradients = set()
        self.history_sync_checks = 0
        self._check_history(runner)
        _write_json(Path(runner.work_dir) / "base_smoke_check.json", dict(
            passed=False, status="running", scope="single-GPU mini preflight of full Base"))

    def _check_history(self, runner):
        history = dict(runner.eval_model.named_parameters())
        for name, parameter in self.probes.items():
            target = history[name].detach().reshape(-1)[:256]
            current = parameter.detach().reshape(-1)[:256].to(dtype=target.dtype)
            if not torch.equal(current, target):
                raise RuntimeError("History model is not synchronized: " + name)
        self.history_sync_checks += 1

    def after_train_iter(self, runner):
        super().after_train_iter(runner)
        self._check_history(runner)
        for group, names in self.groups.items():
            for name in names:
                grad = self.probes[name].grad
                if grad is not None:
                    self.gradient_observations[group] += 1
                    if not torch.isfinite(grad).all().item():
                        self.nonfinite_gradients.add(name)

    def after_run(self, runner):
        failure = None
        try:
            super().after_run(runner)
        except RuntimeError as exc:
            failure = str(exc)
        report = json.loads((Path(runner.work_dir) / "smoke_training_check.json").read_text())
        changed = set(report.get("changed_parameter_probes", []))
        report["module_checks"] = {group: dict(
            changed_parameters=[name for name in names if name in changed],
            gradient_observations=self.gradient_observations[group])
            for group, names in self.groups.items()}
        report["nonfinite_gradient_parameters"] = sorted(self.nonfinite_gradients)
        report["history_sync_checks"] = self.history_sync_checks
        report["scope"] = "single-GPU mini preflight of the full Base architecture; not full training"
        report["cuda_max_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["cuda_max_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["world_size"] = _world_size()
        report["passed"] = bool(report["passed"] and not self.nonfinite_gradients and all(
            item["changed_parameters"] and item["gradient_observations"]
            for item in report["module_checks"].values()))
        report["status"] = "passed" if report["passed"] else "failed"
        if failure:
            report["error"] = failure
        _write_json(Path(runner.work_dir) / "base_smoke_check.json", report)
        if not report["passed"]:
            raise RuntimeError("Base smoke failed module/optimizer/validation checks; see base_smoke_check.json")

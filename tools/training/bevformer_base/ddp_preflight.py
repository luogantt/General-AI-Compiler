"""Three-rank preflight hooks; loaded by the normal upstream FP16 trainer.

This module is independent of the single-GPU preflight so its recorded evidence
remains valid. Repeated mini targets test distributed execution, not accuracy.
"""

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from mmcv.runner import HOOKS, Hook
from mmdet.datasets import DATASETS

import training_support
from training_support import _optimizer_step, _write_json
from base_training_support import BaseSmokeTrainDataset


@DATASETS.register_module()
class RepeatedBaseSmokeDataset(BaseSmokeTrainDataset):
    """Repeat four targets, retaining the same eight real history/target frames."""

    def __init__(self, repeats=3, *args, **kwargs):
        if repeats != 3:
            raise ValueError("DDP preflight requires three repetitions of four targets.")
        # Base initialization uses virtual __len__ when constructing its flags.
        self.repeats = 1
        super().__init__(*args, **kwargs)
        self.target_count = super().__len__()
        if self.target_count != 4 or len(self.flag) != 4:
            raise ValueError("DDP preflight requires exactly four Base smoke targets.")
        self.flag = np.tile(self.flag, repeats)
        self.repeats = repeats

    def __len__(self):
        return super().__len__() * self.repeats

    def prepare_train_data(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        # Base then adds its context offset. Do not slice data_infos: upstream
        # history lookup must still see all eight consecutive original frames.
        return super().prepare_train_data(index % self.target_count)


def summarize_rank_reports(reports, expected_world_size=3, expected_steps=4):
    """Pure aggregation shared by the hook and a CPU communication fixture."""
    ranks = [report.get("rank") for report in reports]
    checks = {
        "all_ranks_present": sorted(ranks) == list(range(expected_world_size)),
        "all_rank_checks_passed": all(report.get("passed") is True for report in reports),
        "four_updates_per_rank": all(
            report.get("training_steps") == expected_steps
            and report.get("optimizer_updates_per_iteration") == [1] * expected_steps
            for report in reports),
        "parameter_probes_equal_across_ranks": len({
            report.get("parameter_probe_sha256") for report in reports}) == 1
            and all(report.get("parameter_probe_sha256") for report in reports),
    }
    rank_zero = next((report for report in reports if report.get("rank") == 0), {})
    validation = rank_zero.get("validation") or {}
    checks["three_finite_validation_samples"] = (
        validation.get("validation_samples") == 3
        and validation.get("finite_predictions") is True)
    checks["checkpoint_saved_and_matches"] = rank_zero.get("checkpoint_verified") is True
    return dict(
        passed=all(checks.values()), status="passed" if all(checks.values()) else "failed",
        scope="Three-GPU mini preflight of full Base; not full trainval or an accuracy benchmark",
        world_size=expected_world_size, checks=checks, ranks=reports,
        parameter_comparison="SHA-256 of named first-256-value probes; not a full-state checksum",
        validation=validation, checkpoint=rank_zero.get("checkpoint"))


@HOOKS.register_module()
class BaseDDPPreflightHook(Hook):
    """Require genuine AdamW steps, module gradients, synchronization and inference."""

    def __init__(self, expected_steps=4, expected_world_size=3):
        self.expected_steps = expected_steps
        self.expected_world_size = expected_world_size

    def before_run(self, runner):
        dist = torch.distributed
        if not dist.is_initialized() or dist.get_world_size() != self.expected_world_size:
            raise RuntimeError("Base DDP preflight requires exactly three initialized ranks.")
        self.rank = dist.get_rank()
        training_support._last_smoke_validation = None
        params = {name: param for name, param in runner.model.named_parameters()
                  if param.requires_grad and param.numel()}
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
                raise RuntimeError("No trainable parameters for DDP probe group: " + group)
            self.groups[group] = list(dict.fromkeys(names[:4] + names[-4:]))
        self.probes = {name: params[name] for names in self.groups.values() for name in names}
        self.initial = {name: self._sample(param) for name, param in self.probes.items()}
        self.losses = []
        self.optimizer_steps = [_optimizer_step(runner)]
        self.gradient_observations = {group: 0 for group in self.groups}
        self.nonfinite_gradients = set()
        self.history_sync_checks = 0
        self.history_mismatches = set()
        self._check_history(runner)
        _write_json(self._path(runner), dict(passed=False, status="running", rank=self.rank))
        if self.rank == 0:
            _write_json(Path(runner.work_dir) / "base_ddp_smoke_check.json",
                        dict(passed=False, status="running", world_size=self.expected_world_size))

    @staticmethod
    def _sample(parameter):
        return parameter.detach().reshape(-1)[:256].float().cpu().clone()

    def _path(self, runner):
        return Path(runner.work_dir) / ("base_ddp_smoke_rank%d.json" % self.rank)

    def _check_history(self, runner):
        history = dict(runner.eval_model.named_parameters())
        for name, parameter in self.probes.items():
            target = history[name].detach().reshape(-1)[:256]
            current = parameter.detach().reshape(-1)[:256].to(dtype=target.dtype)
            if not torch.equal(current, target):
                self.history_mismatches.add(name)
        self.history_sync_checks += 1

    def after_train_iter(self, runner):
        loss = float(runner.outputs["log_vars"]["loss"])
        self.losses.append(loss if math.isfinite(loss) else None)
        self.optimizer_steps.append(_optimizer_step(runner))
        self._check_history(runner)
        for group, names in self.groups.items():
            for name in names:
                gradient = self.probes[name].grad
                if gradient is not None:
                    self.gradient_observations[group] += 1
                    if not torch.isfinite(gradient).all().item():
                        self.nonfinite_gradients.add(name)

    def _verify_checkpoint(self, path, samples):
        # Read on CPU. CheckpointHook ran before this after_run hook; confirm
        # optimizer progress and saved probes, not merely a stale file's name.
        saved = torch.load(str(path), map_location="cpu")
        meta, optimizer = saved["meta"], saved["optimizer"]
        if meta["epoch"] != 1 or meta["iter"] != self.expected_steps:
            return False
        steps = [int(state.get("step", 0)) for state in optimizer["state"].values()]
        if not steps or max(steps) != self.optimizer_steps[-1]:
            return False
        for name, sample in samples.items():
            key = name[7:] if name.startswith("module.") else name
            if key not in saved["state_dict"] or not torch.equal(
                    sample, self._sample(saved["state_dict"][key])):
                return False
        return True

    def after_run(self, runner):
        samples = {name: self._sample(param) for name, param in self.probes.items()}
        digest = hashlib.sha256()
        for name, sample in sorted(samples.items()):
            digest.update(name.encode("utf-8") + b"\0")
            digest.update(sample.numpy().tobytes())
        changed = {name for name, sample in samples.items()
                   if not torch.equal(self.initial[name], sample)}
        modules = {group: dict(changed_parameters=[name for name in names if name in changed],
                               gradient_observations=self.gradient_observations[group])
                   for group, names in self.groups.items()}
        increments = [after - before for before, after in
                      zip(self.optimizer_steps, self.optimizer_steps[1:])]
        checks = dict(
            fresh_optimizer=self.optimizer_steps[0] == 0,
            four_real_updates=len(self.losses) == self.expected_steps
                and increments == [1] * self.expected_steps,
            finite_losses=all(loss is not None for loss in self.losses),
            finite_parameter_probes=all(torch.isfinite(value).all().item() for value in samples.values()),
            all_modules_updated=all(item["changed_parameters"] and item["gradient_observations"]
                                    for item in modules.values()),
            finite_gradients=not self.nonfinite_gradients,
            history_synchronized=not self.history_mismatches
                and self.history_sync_checks == self.expected_steps + 1)
        report = dict(
            passed=all(checks.values()), rank=self.rank, world_size=self.expected_world_size,
            checks=checks, training_steps=len(self.losses), losses=self.losses,
            optimizer_steps=self.optimizer_steps, optimizer_updates_per_iteration=increments,
            module_checks=modules, parameter_probe_sha256=digest.hexdigest(),
            nonfinite_gradient_parameters=sorted(self.nonfinite_gradients),
            history_sync_checks=self.history_sync_checks,
            history_mismatched_probes=sorted(self.history_mismatches))
        if torch.cuda.is_available():
            report.update(cuda_max_allocated_bytes=torch.cuda.max_memory_allocated(),
                          cuda_max_reserved_bytes=torch.cuda.max_memory_reserved())
        if self.rank == 0:
            checkpoint = Path(runner.work_dir) / "epoch_1.pth"
            report.update(validation=training_support._last_smoke_validation,
                          checkpoint=str(checkpoint), checkpoint_verified=False)
            try:
                report["checkpoint_verified"] = self._verify_checkpoint(checkpoint, samples)
            except Exception as exc:
                # Let all ranks reach the collective and emit one failed report.
                report["checkpoint_error"] = type(exc).__name__ + ": " + str(exc)
        _write_json(self._path(runner), report)
        reports = [None] * self.expected_world_size
        torch.distributed.all_gather_object(reports, report)
        summary = summarize_rank_reports(reports, self.expected_world_size, self.expected_steps)
        if self.rank == 0:
            _write_json(Path(runner.work_dir) / "base_ddp_smoke_check.json", summary)
        torch.distributed.barrier()
        if not summary["passed"]:
            raise RuntimeError("Three-GPU Base preflight failed; see base_ddp_smoke_check.json")
        runner.logger.info("Three-GPU preflight passed: real updates and matching parameter probes.")

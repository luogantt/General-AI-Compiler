"""Small BEVFormer training hooks; deploy beside the training configuration.

The model, loss, optimizer and nuScenes metric implementation remain upstream.
TrainingHealthHook must run at LOWEST priority, after the optimizer hook.
"""

from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path

import torch
from mmdet.datasets import DATASETS
from mmcv.runner import HOOKS, Hook

from projects.mmdet3d_plugin.datasets.nuscenes_dataset import CustomNuScenesDataset


_last_smoke_validation = None


def _write_json(path, payload):
    """Replace a report atomically so a reader never sees half-written JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(str(temporary), str(path))


def _optimizer_step(runner):
    # AdamW in the pinned PyTorch environment stores an integer per parameter.
    # Reading state does not change the optimizer or assume a fixed parameter name.
    return max([int(state.get("step", 0)) for state in runner.optimizer.state.values()]
               or [0])


@HOOKS.register_module()
class SyncHistoryBeforeRunHook(Hook):
    """Copy loaded/resumed weights into the separate, inference-only history model."""

    def before_run(self, runner):
        if getattr(runner, "eval_model", None) is None:
            raise RuntimeError("This configuration requires the dedicated FP16 runner.")
        runner.eval_model.load_state_dict(runner.model.state_dict())
        runner.eval_model.eval()
        runner.logger.info("Synchronized FP16 history model before the first iteration.")


@DATASETS.register_module()
class TrainvalNuScenesDataset(CustomNuScenesDataset):
    """Run upstream metrics without retaining a complete SDK between evaluations.

    Upstream _evaluate_single stores both the full NuScenes object and its
    evaluator on self. The next evaluation otherwise constructs another SDK
    while the previous evaluator still retains the old one.
    """

    def _release_evaluation_state(self):
        self.__dict__.pop("nusc_eval", None)
        self.__dict__.pop("nusc", None)
        gc.collect()

    def _evaluate_single(self, *args, **kwargs):
        self._release_evaluation_state()
        try:
            return super()._evaluate_single(*args, **kwargs)
        finally:
            self._release_evaluation_state()


@HOOKS.register_module()
class TrainingHealthHook(Hook):
    """Report actual AdamW updates; FP16 overflow skips are visible, not fatal."""

    def __init__(self, interval=50):
        if interval < 1:
            raise ValueError("TrainingHealthHook interval must be positive.")
        self.interval = interval

    def before_run(self, runner):
        self.previous_step = _optimizer_step(runner)
        self.skipped_steps = 0
        self.iterations_observed = 0
        self.last_report = {}

    def after_train_iter(self, runner):
        loss = float(runner.outputs["log_vars"]["loss"])
        step = _optimizer_step(runner)
        increment = step - self.previous_step
        self.previous_step = step
        self.iterations_observed += 1
        self.skipped_steps += int(increment == 0)
        finite = math.isfinite(loss)
        report = dict(
            status="running" if finite else "failed",
            updated_at=datetime.now(timezone.utc).isoformat(),
            epoch=runner.epoch + 1,
            inner_iter=runner.inner_iter + 1,
            global_iter=runner.iter + 1,
            loss=loss if finite else None,
            finite_loss=finite,
            optimizer_step=step,
            optimizer_step_increment_last_iter=increment,
            skipped_optimizer_steps=self.skipped_steps,
            iterations_observed=self.iterations_observed,
            counters_scope="Current process since before_run; epoch/iteration are one-based.",
        )
        self.last_report = report
        if (self.iterations_observed == 1 or (runner.iter + 1) % self.interval == 0
                or not finite):
            self._save(runner, report)
        if not finite:
            raise RuntimeError("Training produced a nonfinite loss: %r" % loss)

    def _save(self, runner, report):
        if getattr(runner, "rank", 0) != 0:
            return
        if torch.cuda.is_available():
            report["cuda_max_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["cuda_max_reserved_bytes"] = torch.cuda.max_memory_reserved()
        _write_json(Path(runner.work_dir) / "training-health.json", report)

    def after_run(self, runner):
        report = dict(self.last_report)
        report.update(status="completed", updated_at=datetime.now(timezone.utc).isoformat())
        self._save(runner, report)


@HOOKS.register_module()
class SmokeTrainingCheckHook(Hook):
    """Check a short real training run, including optimizer updates and validation."""

    def __init__(self, expected_steps=4):
        self.expected_steps = expected_steps

    def before_run(self, runner):
        global _last_smoke_validation
        _last_smoke_validation = None
        params = [(name, param) for name, param in runner.model.named_parameters()
                  if param.requires_grad and param.numel()]
        self.probes = dict(params[:8] + params[-8:])
        self.initial = {name: param.detach().reshape(-1)[:256].float().cpu().clone()
                        for name, param in self.probes.items()}
        self.losses = []
        self.optimizer_steps = [_optimizer_step(runner)]
        _write_json(Path(runner.work_dir) / "smoke_training_check.json",
                    dict(passed=False, status="running"))

    def after_train_iter(self, runner):
        loss = float(runner.outputs["log_vars"]["loss"])
        if not math.isfinite(loss):
            raise RuntimeError("Smoke training produced a nonfinite loss.")
        self.losses.append(loss)
        self.optimizer_steps.append(_optimizer_step(runner))

    def after_run(self, runner):
        updated = [name for name, param in self.probes.items()
                   if not torch.equal(self.initial[name],
                                      param.detach().reshape(-1)[:256].float().cpu())]
        increments = [after - before for before, after in
                      zip(self.optimizer_steps, self.optimizer_steps[1:])]
        checkpoint = Path(runner.work_dir) / "epoch_1.pth"
        report = dict(
            training_steps=len(self.losses), expected_steps=self.expected_steps,
            losses=self.losses, optimizer_steps=self.optimizer_steps[-1],
            optimizer_updates_per_iteration=increments,
            changed_parameter_probes=updated, checkpoint=str(checkpoint),
            checkpoint_exists=checkpoint.is_file(), validation=_last_smoke_validation,
            note="Real nuScenes samples; a short smoke run is not an accuracy benchmark.")
        report["passed"] = bool(
            len(self.losses) == self.expected_steps
            and all(increment == 1 for increment in increments)
            and updated and checkpoint.is_file() and _last_smoke_validation)
        _write_json(Path(runner.work_dir) / "smoke_training_check.json", report)
        if not report["passed"]:
            raise RuntimeError("Smoke training checks failed: " + json.dumps(report))
        runner.logger.info("Smoke passed: finite losses, optimizer updates, saved checkpoint.")


@DATASETS.register_module()
class TinySmokeNuScenesDataset(CustomNuScenesDataset):
    """Check real validation predictions without pretending a subset has full AP/NDS."""

    def evaluate(self, results, logger=None, **kwargs):
        global _last_smoke_validation
        if len(results) != len(self.data_infos):
            raise RuntimeError("Smoke validation prediction count does not match samples.")
        total_boxes = 0
        for result in results:
            prediction = result.get("pts_bbox", result)
            boxes = prediction["boxes_3d"].tensor
            scores = prediction["scores_3d"]
            labels = prediction["labels_3d"]
            if not (len(boxes) == len(scores) == len(labels)):
                raise RuntimeError("Smoke prediction boxes/scores/labels differ in length.")
            if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
                raise RuntimeError("Smoke validation contains nonfinite predictions.")
            if len(scores) and (torch.any(scores < 0) or torch.any(scores > 1)):
                raise RuntimeError("Smoke prediction scores are outside [0, 1].")
            if len(labels) and (torch.any(labels < 0) or torch.any(labels >= len(self.CLASSES))):
                raise RuntimeError("Smoke prediction labels are outside the class range.")
            total_boxes += len(boxes)
        summary = dict(
            validation_samples=len(results), predicted_boxes=total_boxes,
            finite_predictions=True,
            sample_tokens=[info["token"] for info in self.data_infos],
            accuracy_metrics_computed=False,
            note="Subset inference only. Full official validation is required for AP/NDS.")
        _last_smoke_validation = summary
        if kwargs.get("jsonfile_prefix"):
            _write_json(Path(kwargs["jsonfile_prefix"]) / "smoke_validation_summary.json", summary)
        if logger is not None and hasattr(logger, "info"):
            logger.info(json.dumps(summary))
        return dict(smoke_validation_samples=len(results),
                    smoke_predicted_boxes=total_boxes, smoke_finite_predictions=1.0)

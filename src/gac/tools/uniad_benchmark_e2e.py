#!/usr/bin/env python3
"""Benchmark the native PyTorch UniAD stage-2 end-to-end evaluation forward.

The measured boundary starts immediately before ``MMDataParallel`` is called
and ends after a CUDA synchronization.  It therefore includes wrapper scatter,
host-to-device copies, every UniAD task head, Python work inside ``forward_test``
and CUDA execution.  Dataset loading and preprocessing are measured separately.

This tool intentionally imports Torch/MMCV only while running ``main``.  Its
argument, statistics and report-writing helpers remain testable in a lightweight
Python environment.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
from importlib import import_module
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


GAC_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _nonnegative_int(value: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return result


def _parse_cfg_value(text: str) -> Any:
    """Parse one MMCV-style override value without importing MMCV."""

    if text == "":
        return ""
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in ("none", "null"):
        return None
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text


def _parse_cfg_options(items: Optional[Sequence[str]]) -> Dict[str, Any]:
    """Turn ``KEY=VALUE`` items into a mapping accepted by Config.merge_from_dict."""

    result: Dict[str, Any] = {}
    for item in items or ():
        if "=" not in item:
            raise ValueError("cfg option must use KEY=VALUE: {!r}".format(item))
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("cfg option has an empty key: {!r}".format(item))
        if key in result:
            raise ValueError("cfg option is repeated: {!r}".format(key))
        result[key] = _parse_cfg_value(raw_value)
    return result


def _load_checkpoint_strict(
    loader: Any, model: Any, checkpoint_path: Path
) -> Any:
    """Load a checkpoint while rejecting every model/state-dict mismatch."""

    return loader(
        model,
        str(checkpoint_path),
        map_location="cpu",
        strict=True,
    )


def _track_query_counts(
    object_indexes: Sequence[Any],
    scores: Sequence[Any],
    filter_score_threshold: float,
) -> Dict[str, int]:
    """Count tracker-state queries and score-filtered active track queries."""

    if len(object_indexes) != len(scores):
        raise ValueError("track object indexes and scores must have equal lengths")
    threshold = float(filter_score_threshold)
    if not math.isfinite(threshold):
        raise ValueError("track filter score threshold must be finite")
    object_index_values = [int(value) for value in object_indexes]
    score_values = [float(value) for value in scores]
    return {
        "track_query_count": len(object_index_values),
        "active_track_query_count": sum(
            object_index >= 0 and score >= threshold
            for object_index, score in zip(object_index_values, score_values)
        ),
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one sample")
    if percentile < 0.0 or percentile > 100.0:
        raise ValueError("percentile must be in [0, 100]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float], include_fps: bool = False) -> Dict[str, Any]:
    samples = [float(value) for value in values]
    if not samples:
        empty: Dict[str, Any] = {
            "count": 0,
            "median": None,
            "p90": None,
            "p99": None,
            "mean": None,
            "min": None,
            "max": None,
        }
        if include_fps:
            empty["fps_from_mean_latency"] = None
        return empty
    if any(not math.isfinite(value) or value < 0.0 for value in samples):
        raise ValueError("timing samples must be finite and non-negative")
    mean = float(statistics.fmean(samples))
    result = {
        "count": len(samples),
        "median": float(statistics.median(samples)),
        "p90": _percentile(samples, 90.0),
        "p99": _percentile(samples, 99.0),
        "mean": mean,
        "min": min(samples),
        "max": max(samples),
    }
    if include_fps:
        result["fps_from_mean_latency"] = (
            None if mean == 0.0 else 1000.0 / mean
        )
    return result


def _timing_groups(samples: Sequence[Mapping[str, Any]], field: str) -> Dict[str, Any]:
    all_values = [float(sample[field]) for sample in samples]
    first_values = [
        float(sample[field])
        for sample in samples
        if sample["frame_kind"] == "first-frame"
    ]
    steady_values = [
        float(sample[field])
        for sample in samples
        if sample["frame_kind"] == "steady-state"
    ]
    include_fps = field in (
        "model_call_wall_ms",
        "observed_next_plus_model_wall_ms",
    )
    return {
        "all": _distribution(all_values, include_fps=include_fps),
        "first_frame": _distribution(first_values, include_fps=include_fps),
        "steady_state": _distribution(steady_values, include_fps=include_fps),
    }


def _sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _repository_git_state(root: Path) -> Dict[str, Any]:
    """Return Git provenance when the benchmark runs from a checkout."""

    result: Dict[str, Any] = {
        "root": str(root.resolve()),
        "head": None,
        "tracked_files_dirty": None,
    }
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return result
    revision = head.stdout.strip().lower()
    if head.returncode != 0 or len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        return result
    result["head"] = revision
    try:
        status = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return result
    if status.returncode == 0:
        result["tracked_files_dirty"] = bool(status.stdout.strip())
    return result


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    rendered = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=".{}-".format(destination.name),
            suffix=".tmp",
            dir=str(destination.parent),
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(rendered)
            temporary.flush()
            os.fsync(temporary.fileno())
        with open(temporary_name, "r", encoding="utf-8") as written:
            json.load(written)
        os.replace(temporary_name, str(destination))
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _find_key(value: Any, key: str, seen: Optional[set] = None) -> Any:
    """Find a metadata key through dict/list/MMCV DataContainer nesting."""

    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return None
    seen.add(identity)
    if isinstance(value, Mapping):
        if key in value:
            return value[key]
        for nested in value.values():
            found = _find_key(nested, key, seen)
            if found is not None:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for nested in value:
            found = _find_key(nested, key, seen)
            if found is not None:
                return found
        return None
    if value.__class__.__name__ == "DataContainer" and hasattr(value, "data"):
        return _find_key(value.data, key, seen)
    return None


def _scene_token(dataset: Any, index: int, batch: Any) -> str:
    infos = getattr(dataset, "data_infos", None)
    if isinstance(infos, Sequence) and not isinstance(infos, (str, bytes)):
        if 0 <= index < len(infos) and isinstance(infos[index], Mapping):
            token = infos[index].get("scene_token")
            if token is not None:
                return str(token)
    token = _find_key(batch, "scene_token")
    if token is None:
        raise RuntimeError(
            "cannot determine scene_token; chronological first-frame grouping is unavailable"
        )
    return str(token)


def _annotation_records(dataset_cfg: Any) -> List[Dict[str, Any]]:
    if isinstance(dataset_cfg, Mapping):
        raw = dataset_cfg.get("ann_file")
    else:
        raw = getattr(dataset_cfg, "ann_file", None)
    paths: Iterable[Any]
    if isinstance(raw, (list, tuple)):
        paths = raw
    elif raw is None:
        paths = ()
    else:
        paths = (raw,)
    records = []
    for item in paths:
        path = Path(str(item)).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        record: Dict[str, Any] = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            record["size_bytes"] = path.stat().st_size
            record["sha256"] = _sha256_file(path)
        records.append(record)
    return records


def _import_config_plugin(
    cfg: Any, config_path: Path, uniad_root: Path
) -> None:
    if not bool(cfg.get("plugin", False)):
        return
    plugin_dir = cfg.get("plugin_dir", None)
    if plugin_dir:
        normalized = str(plugin_dir).replace("\\", "/").strip("/")
        module_name = normalized.replace("/", ".")
    else:
        try:
            relative_parent = config_path.resolve().parent.relative_to(
                uniad_root.resolve()
            )
        except ValueError as exc:
            raise RuntimeError(
                "plugin config must be inside --uniad-root when plugin_dir is absent"
            ) from exc
        module_name = ".".join(relative_parent.parts)
    import_module(module_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--uniad-root",
        type=Path,
        required=True,
        help="external UniAD 2.0 checkout containing projects/mmdet3d_plugin",
    )
    parser.add_argument("config", type=Path, help="UniAD stage-2 config")
    parser.add_argument("checkpoint", type=Path, help="real UniAD checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="atomic JSON report path")
    parser.add_argument("--device", type=_nonnegative_int, default=0)
    parser.add_argument("--warmup", type=_nonnegative_int, default=10)
    parser.add_argument("--measure", type=_positive_int, default=100)
    parser.add_argument("--workers", type=_nonnegative_int, default=0)
    parser.add_argument("--log-interval", type=_positive_int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="request deterministic cuDNN behavior (may reduce performance)",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        metavar="KEY=VALUE",
        help=(
            "config overrides, for example data.test.ann_file=/data/mini.pkl "
            "data.test.data_root="
        ),
    )
    return parser


def _validate_files(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    args.uniad_root = args.uniad_root.resolve()
    args.config = args.config.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    plugin = args.uniad_root / "projects" / "mmdet3d_plugin" / "__init__.py"
    if not plugin.is_file():
        parser.error(
            "--uniad-root does not contain projects/mmdet3d_plugin: {}".format(
                args.uniad_root
            )
        )
    if not args.config.is_file():
        parser.error("config does not exist: {}".format(args.config))
    if not args.checkpoint.is_file():
        parser.error("checkpoint does not exist: {}".format(args.checkpoint))
    if args.output.resolve() in (args.config.resolve(), args.checkpoint.resolve()):
        parser.error("--output must not replace the config or checkpoint")


def _benchmark(args: argparse.Namespace, cfg_options: Mapping[str, Any]) -> Dict[str, Any]:
    uniad_root = args.uniad_root.resolve()
    root_text = str(uniad_root)
    while root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    # UniAD configs use paths relative to their checkout (for example
    # ``data/nuscenes``), so reproduce the upstream command's working directory.
    os.chdir(str(uniad_root))

    import numpy as np
    import torch
    import mmcv
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet3d.models import build_detector

    from projects.mmdet3d_plugin.datasets import custom_build_dataset
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the UniAD end-to-end benchmark")
    if args.device >= torch.cuda.device_count():
        raise RuntimeError(
            "CUDA device {} is unavailable; visible device count is {}".format(
                args.device, torch.cuda.device_count()
            )
        )

    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    cfg = Config.fromfile(str(config_path))
    if cfg_options:
        cfg.merge_from_dict(dict(cfg_options))
    _import_config_plugin(cfg, config_path, uniad_root)

    model_type = cfg.model.get("type", None)
    if model_type != "UniAD":
        raise RuntimeError(
            "benchmark_e2e requires a full UniAD config, got model.type={!r}".format(
                model_type
            )
        )
    if not isinstance(cfg.data.test, Mapping):
        raise RuntimeError("benchmark_e2e requires one sequential test dataset")
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True
    cfg.data.test.pop("samples_per_gpu", None)
    cfg.data.workers_per_gpu = args.workers

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    elif cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    dataset_setup_started = time.perf_counter_ns()
    dataset = custom_build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        num_gpus=1,
        dist=False,
        shuffle=False,
    )
    dataset_setup_ms = (time.perf_counter_ns() - dataset_setup_started) / 1.0e6
    required_samples = args.warmup + args.measure
    if len(dataset) < required_samples:
        raise RuntimeError(
            "dataset has {} samples, but warmup + measure requires {}".format(
                len(dataset), required_samples
            )
        )
    if getattr(data_loader, "batch_size", None) != 1:
        raise RuntimeError("end-to-end benchmark requires batch size 1")

    torch.cuda.set_device(args.device)
    model_setup_started = time.perf_counter_ns()
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint_metadata = _load_checkpoint_strict(
        load_checkpoint, model, checkpoint_path
    )
    model = MMDataParallel(model.cuda(args.device), device_ids=[args.device])
    model.eval()
    model_setup_ms = (time.perf_counter_ns() - model_setup_started) / 1.0e6
    unwrapped_model = model.module
    task_heads = {
        "tracking": True,
        "mapping": bool(unwrapped_model.with_seg_head),
        "motion": bool(unwrapped_model.with_motion_head),
        "occupancy": bool(unwrapped_model.with_occ_head),
        "planning": bool(unwrapped_model.with_planning_head),
    }
    if not all(task_heads.values()):
        raise RuntimeError(
            "full stage-2 benchmark requires every UniAD task head; got {}".format(
                task_heads
            )
        )
    track_base = getattr(unwrapped_model, "track_base", None)
    active_track_score_threshold = getattr(
        track_base, "filter_score_thresh", None
    )
    if active_track_score_threshold is None:
        raise RuntimeError(
            "full stage-2 benchmark requires track_base.filter_score_thresh"
        )
    active_track_score_threshold = float(active_track_score_threshold)
    if not math.isfinite(active_track_score_threshold):
        raise RuntimeError("track_base.filter_score_thresh must be finite")

    iterator_started = time.perf_counter_ns()
    iterator = iter(data_loader)
    iterator_creation_ms = (time.perf_counter_ns() - iterator_started) / 1.0e6
    previous_scene: Optional[str] = None
    warmup_waits: List[float] = []
    samples: List[Dict[str, Any]] = []

    def next_batch(index: int) -> Any:
        wait_started = time.perf_counter_ns()
        try:
            batch = next(iterator)
        except StopIteration as exc:
            raise RuntimeError(
                "dataset ended at sample {} before benchmark completed".format(index)
            ) from exc
        wait_ms = (time.perf_counter_ns() - wait_started) / 1.0e6
        return batch, wait_ms

    def run_model(batch: Any) -> float:
        torch.cuda.synchronize(args.device)
        started = time.perf_counter_ns()
        with torch.no_grad():
            output = model(return_loss=False, rescale=True, **batch)
        torch.cuda.synchronize(args.device)
        elapsed_ms = (time.perf_counter_ns() - started) / 1.0e6
        del output
        return elapsed_ms

    def track_state() -> Dict[str, int]:
        instances = getattr(unwrapped_model, "test_track_instances", None)
        if instances is None:
            raise RuntimeError("UniAD did not retain test_track_instances")
        object_indexes = getattr(instances, "obj_idxes", None)
        if object_indexes is None:
            raise RuntimeError("test_track_instances has no obj_idxes")
        scores = getattr(instances, "scores", None)
        if scores is None:
            raise RuntimeError("test_track_instances has no scores")
        # Copy the two small telemetry vectors to CPU after the synchronized
        # timing boundary.  All comparisons and reductions then happen on CPU,
        # so query accounting does not add CUDA allocations to the peak metric.
        object_index_values = object_indexes.detach().cpu().reshape(-1).tolist()
        score_values = scores.detach().cpu().reshape(-1).tolist()
        return _track_query_counts(
            object_index_values,
            score_values,
            active_track_score_threshold,
        )

    for index in range(args.warmup):
        batch, wait_ms = next_batch(index)
        token = _scene_token(dataset, index, batch)
        previous_scene = token
        run_model(batch)
        warmup_waits.append(wait_ms)
        del batch

    torch.cuda.synchronize(args.device)
    baseline_allocated = int(torch.cuda.memory_allocated(args.device))
    baseline_reserved = int(torch.cuda.memory_reserved(args.device))
    torch.cuda.reset_peak_memory_stats(args.device)

    for measurement_index in range(args.measure):
        dataset_index = args.warmup + measurement_index
        batch, wait_ms = next_batch(dataset_index)
        token = _scene_token(dataset, dataset_index, batch)
        first_frame = previous_scene is None or token != previous_scene
        previous_scene = token
        elapsed_ms = run_model(batch)
        sample = {
            "measurement_index": measurement_index,
            "dataset_index": dataset_index,
            "scene_token": token,
            "frame_kind": "first-frame" if first_frame else "steady-state",
            "dataloader_wait_ms": wait_ms,
            "model_call_wall_ms": elapsed_ms,
            "observed_next_plus_model_wall_ms": wait_ms + elapsed_ms,
        }
        sample.update(track_state())
        samples.append(sample)
        del batch
        if (measurement_index + 1) % args.log_interval == 0:
            current = _distribution(
                [sample["model_call_wall_ms"] for sample in samples],
                include_fps=True,
            )
            print(
                "Measured {}/{} frames: median={:.3f} ms, mean={:.3f} ms, FPS={:.3f}".format(
                    measurement_index + 1,
                    args.measure,
                    current["median"],
                    current["mean"],
                    current["fps_from_mean_latency"],
                ),
                flush=True,
            )

    torch.cuda.synchronize(args.device)
    peak_allocated = int(torch.cuda.max_memory_allocated(args.device))
    peak_reserved = int(torch.cuda.max_memory_reserved(args.device))
    final_allocated = int(torch.cuda.memory_allocated(args.device))
    final_reserved = int(torch.cuda.memory_reserved(args.device))
    properties = torch.cuda.get_device_properties(args.device)

    checkpoint_meta = (
        checkpoint_metadata.get("meta", {})
        if isinstance(checkpoint_metadata, Mapping)
        else {}
    )
    if not isinstance(checkpoint_meta, Mapping):
        checkpoint_meta = {}
    report = {
        "schema_version": 1,
        "kind": "uniad-native-pytorch-end-to-end-benchmark",
        "scope": "full UniAD stage2 evaluation forward on real dataset samples",
        "runtime": "native-pytorch",
        "compiled_subgraphs": [],
        "boundary": (
            "dataloader-produced CPU batch -> MMDataParallel scatter/H2D -> "
            "full model forward_test -> CUDA-synchronized return"
        ),
        "excluded_from_model_latency": [
            "dataloader wait and input pipeline",
            "multi-sample metric aggregation",
            "result serialization",
        ],
        "notes": [
            "model_call_wall_ms is synchronized wall time, not GPU-resident latency",
            "the stock UniAD evaluation forward includes GT-dependent evaluation helpers",
            "temporal state is preserved across chronological batch-1 samples",
            "checkpoint loading rejects missing, unexpected, and incompatible state-dict keys",
            "track query telemetry is copied to CPU after the timed model call",
            (
                "observed_next_plus_model_wall_ms adds the blocking next(data_loader) "
                "wait to model_call_wall_ms; with workers=0 this is the serialized "
                "input-pipeline plus model latency"
            ),
        ],
        "provenance": {
            "gac_repository": _repository_git_state(GAC_REPOSITORY_ROOT),
            "uniad_repository": _repository_git_state(uniad_root),
            "benchmark_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
        },
        "inputs": {
            "config": {
                "path": str(config_path),
                "sha256": _sha256_file(config_path),
                "cfg_options": _jsonable(cfg_options),
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "size_bytes": checkpoint_path.stat().st_size,
                "sha256": _sha256_file(checkpoint_path),
                "meta_keys": sorted(str(key) for key in checkpoint_meta),
            },
            "dataset": {
                "class": dataset.__class__.__name__,
                "length": len(dataset),
                "test_mode": bool(getattr(dataset, "test_mode", True)),
                "data_root": _jsonable(cfg.data.test.get("data_root", None)),
                "annotation_files": _annotation_records(cfg.data.test),
                "source": "configured real dataset pipeline",
            },
        },
        "model": {
            "type": model_type,
            "class": unwrapped_model.__class__.__name__,
            "task_heads_executed": task_heads,
            "video_test_mode": bool(getattr(unwrapped_model, "video_test_mode", False)),
            "track_query_metrics": {
                "track_query_count": (
                    "size of next-frame test_track_instances after query interaction"
                ),
                "active_track_query_count": (
                    "obj_idxes >= 0 and scores >= filter_score_threshold"
                ),
                "filter_score_threshold": active_track_score_threshold,
            },
        },
        "protocol": {
            "device_index": args.device,
            "batch_size": 1,
            "shuffle": False,
            "distributed": False,
            "workers": args.workers,
            "warmup_frames": args.warmup,
            "measured_frames": args.measure,
            "seed": args.seed,
            "deterministic": args.deterministic,
            "chronological": True,
            "temporal_state_preserved": True,
            "strict_checkpoint_load": True,
            "cuda_synchronize_before_and_after_model_call": True,
            "includes_mmdata_parallel_scatter_and_h2d": True,
            "query_telemetry_after_timed_boundary": True,
            "dataloader_wait_definition": (
                "blocking wall time spent in next(data_loader); with workers=0 "
                "this includes the configured dataset input pipeline"
            ),
            "iterator_creation_ms": iterator_creation_ms,
            "dataset_setup_ms": dataset_setup_ms,
            "model_and_checkpoint_setup_ms": model_setup_ms,
        },
        "timing_ms": {
            "model_call_wall": _timing_groups(samples, "model_call_wall_ms"),
            "dataloader_wait": _timing_groups(samples, "dataloader_wait_ms"),
            "observed_next_plus_model_wall": _timing_groups(
                samples, "observed_next_plus_model_wall_ms"
            ),
            "warmup_dataloader_wait": _distribution(warmup_waits),
        },
        "memory": {
            "definition": (
                "PyTorch CUDA allocator totals; peaks are reset after warmup "
                "and query reductions run on CPU"
            ),
            "baseline_after_warmup": {
                "allocated_bytes": baseline_allocated,
                "reserved_bytes": baseline_reserved,
            },
            "measured_peak": {
                "allocated_bytes": peak_allocated,
                "reserved_bytes": peak_reserved,
            },
            "after_measurement": {
                "allocated_bytes": final_allocated,
                "reserved_bytes": final_reserved,
            },
        },
        "device": {
            "name": torch.cuda.get_device_name(args.device),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "total_memory_bytes": int(properties.total_memory),
        },
        "software": {
            "working_directory": str(Path.cwd().resolve()),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "cudnn": torch.backends.cudnn.version(),
            "mmcv": str(mmcv.__version__),
            "mmdet": _package_version("mmdet"),
            "mmdet3d": _package_version("mmdet3d"),
            "numpy": str(np.__version__),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "fp16_configured": fp16_cfg is not None,
        },
        "samples": samples,
    }
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_files(parser, args)
    try:
        cfg_options = _parse_cfg_options(args.cfg_options)
    except ValueError as exc:
        parser.error(str(exc))
    report = _benchmark(args, cfg_options)
    _atomic_write_json(args.output, report)
    summary = report["timing_ms"]["model_call_wall"]["all"]
    print(
        "Wrote {}: median={:.3f} ms, p90={:.3f} ms, p99={:.3f} ms, FPS={:.3f}".format(
            args.output.resolve(),
            summary["median"],
            summary["p90"],
            summary["p99"],
            summary["fps_from_mean_latency"],
        )
    )


if __name__ == "__main__":
    main()

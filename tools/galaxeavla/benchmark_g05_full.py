#!/usr/bin/env python3
"""Benchmark the released-default G0.5 R1 Lite policy on one Ascend device.

This is deliberately an end-to-end *synthetic-input* benchmark.  It loads the
official ``g05-base`` checkpoint, runs the official processor and
``PolicyInferencer`` path, and keeps both shipped action heads enabled:

* checkpoint-configured chain-of-thought generation,
* continuous flow matching (10 Euler steps), and
* discrete autoregressive action generation plus ActionCodec decode.

The generated camera/state request has the real R1 Lite shapes but does not
represent a robot episode.  Results from this tool must therefore be labelled
synthetic throughput, not task accuracy or real-robot control frequency.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_SOURCE_REVISION = "89f2322b4ad016e192437adc1a2c253b05bab246"
EXPECTED_CHECKPOINT_SHA256 = "072211e5b2f5ef036729bae673f3f44da40adbea5c0044af55fe2fb8af654327"
EXPECTED_CHECKPOINT_SIZE = 11_440_519_964
EXPECTED_ACTION_TOKENIZER_SIZE = 506_886_775
DEFAULT_TASK = "Pick up the object and place it in the target area."
DEFAULT_EMBODIMENT = "galaxea_r1lite"
CAMERA_KEYS = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
STATE_DIMS = {
    "left_arm": 6,
    "left_gripper": 1,
    "right_arm": 6,
    "right_gripper": 1,
}


def percentile(values: Sequence[float], q: float) -> float:
    """Return a linearly interpolated percentile without NumPy."""

    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be in [0, 100]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    """Summarize one non-empty sequence of millisecond measurements."""

    if not values:
        raise ValueError("summarize requires at least one value")
    floats = [float(value) for value in values]
    return {
        "count": len(floats),
        "mean": statistics.fmean(floats),
        "p50": percentile(floats, 50.0),
        "p95": percentile(floats, 95.0),
        "min": min(floats),
        "max": max(floats),
    }


def summarize_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Aggregate all numeric timing fields found in request samples."""

    keys = sorted(
        {
            key
            for sample in samples
            for key, value in sample.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    return {
        key: summarize(
            [float(sample[key]) for sample in samples if isinstance(sample.get(key), (int, float))]
        )
        for key in keys
    }


def _select(config: Any, dotted_path: str) -> Any:
    value = config
    for component in dotted_path.split("."):
        if isinstance(value, Mapping):
            if component not in value:
                return None
            value = value[component]
        else:
            if not hasattr(value, component):
                return None
            value = getattr(value, component)
    return value


def validate_shipped_default_contract(config: Any) -> dict[str, Any]:
    """Fail rather than accidentally headline a reduced-path benchmark."""

    expected = {
        "model.use_torch_compile": False,
        "model.model_arch.continuous_action": True,
        "model.model_arch.discrete_action": True,
        "model.model_arch.return_continuous_action": True,
        # The pinned g05-base checkpoint stores predict_cot=true.  This is also
        # what scripts/serve_policy.py executes when invoked with the official
        # README command, because that entry point loads the checkpoint's
        # .hydra/config.yaml rather than recomposing configs/task/r1lite.yaml.
        "model.model_arch.predict_cot": True,
        "model.model_arch.max_text_tokens": 160,
        "model.model_arch.input_preprocessor.pred_eov": True,
        "model.model_arch.attn_implementation": "eager",
        "model.model_arch.fm.num_inference_steps": 10,
        # The checkpoint's unavailable Cholesky file is scoped only to
        # behavior_r1pro.  R1 Lite takes the standard-Gaussian branch either
        # way, so disabling the unused loader preserves its inference math.
        "model.model_arch.fm.use_correlated_noise": False,
        "model.model_arch.ar.do_sample": False,
        "model.model_arch.ar.max_new_tokens": 300,
        "data.action_size": 32,
        "data.obs_size": 1,
        "model.model_arch.action_dim": 27,
        "model.model_arch.cond_steps": 1,
        "model.processor.num_output_cameras": 3,
    }
    observed = {path: _select(config, path) for path in expected}
    mismatches = {
        path: {"expected": wanted, "observed": observed[path]}
        for path, wanted in expected.items()
        if observed[path] != wanted
    }
    if mismatches:
        raise RuntimeError(
            "Checkpoint config does not match the released-default full-model contract: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return observed


def deployment_config_paths(source_dir: Path) -> dict[str, Path]:
    """Return the current R1 Lite deployment fragments used for config repair."""

    return {
        "data": source_dir / "configs" / "data" / "r1lite.yaml",
        "task": source_dir / "configs" / "task" / "r1lite.yaml",
    }


def load_r1lite_checkpoint_config(
    *,
    source_dir: Path,
    run_dir: Path,
    checkpoint: Path,
    overrides: list[str],
    omega_conf: Any,
    load_config_from_run_dir: Any,
) -> tuple[Any, dict[str, Any]]:
    """Load g05-base while repairing removed training-data config references.

    The released checkpoint's ``.hydra/config.yaml`` is authoritative for model
    execution switches (including ``predict_cot=true``), but its ``data`` node
    references private/legacy pretraining YAML paths that are absent from the
    pinned public source revision. The upstream loader resolves every
    interpolation before applying ``eval_embodiment``, so it otherwise fails
    before inference starts.

    This compatibility step replaces only the deployment data, processor, and
    tokenizer pieces with the public R1 Lite fragments. It deliberately keeps
    the checkpoint's model architecture and CoT/FM/AR settings intact.
    """

    paths = deployment_config_paths(source_dir)
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing public R1 Lite {label} config: {path}")

    checkpoint_config = run_dir / ".hydra" / "config.yaml"
    raw = omega_conf.load(checkpoint_config)
    task = omega_conf.load(paths["task"])
    omega_conf.set_struct(raw, False)
    raw.data = omega_conf.load(paths["data"])
    raw.model.processor = omega_conf.merge(raw.model.processor, task.model.processor)
    raw.tokenizer = omega_conf.merge(raw.tokenizer, task.tokenizer)

    with tempfile.TemporaryDirectory(prefix="g05-r1lite-config-") as temporary:
        shadow_run_dir = Path(temporary)
        shadow_hydra = shadow_run_dir / ".hydra"
        shadow_hydra.mkdir(parents=True)
        shadow_config = shadow_hydra / "config.yaml"
        omega_conf.save(raw, shadow_config, resolve=False)
        cfg = load_config_from_run_dir(shadow_run_dir, str(checkpoint), overrides)

    # Keep runtime sidecar/stat resolution anchored to the real checkpoint.
    omega_conf.set_struct(cfg, False)
    cfg.run_dir = str(run_dir)
    cfg.output_dir = str(run_dir / f"eval_{checkpoint.stem}")
    metadata = {
        "mode": "checkpoint-model-plus-public-r1lite-deployment-config",
        "checkpoint_config": str(checkpoint_config),
        "checkpoint_config_sha256": sha256_file(checkpoint_config),
        "replacement_data_config": str(paths["data"]),
        "replacement_data_config_sha256": sha256_file(paths["data"]),
        "replacement_task_config": str(paths["task"]),
        "replacement_task_config_sha256": sha256_file(paths["task"]),
        "reason": (
            "checkpoint data node references legacy oc.load files absent from the pinned "
            "public source; model execution settings remain checkpoint-authored"
        ),
    }
    return cfg, metadata


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resolve_asset_layout(checkpoint: Path, asset_root: Path | None = None) -> dict[str, Path]:
    checkpoint = checkpoint.expanduser().resolve()
    if asset_root is None:
        # .../<snapshot>/g05-base/checkpoints/model_state_dict.pt
        try:
            asset_root = checkpoint.parents[2]
        except IndexError as exc:
            raise ValueError("Cannot infer the G05 asset root from checkpoint path") from exc
    asset_root = asset_root.expanduser().resolve()
    return {
        "checkpoint": checkpoint,
        "asset_root": asset_root,
        "run_dir": checkpoint.parent.parent,
        "hydra_config": checkpoint.parent.parent / ".hydra" / "config.yaml",
        "dataset_stats": checkpoint.parent.parent / "dataset_stats.json",
        "action_tokenizer": asset_root / "action_tokenizer.pt",
        "processor": asset_root / "qwen3_5_2b_base_processor",
    }


def validate_asset_layout(layout: Mapping[str, Path], verify_checkpoint_hash: bool) -> str | None:
    required_files = (
        "checkpoint",
        "hydra_config",
        "dataset_stats",
        "action_tokenizer",
    )
    for key in required_files:
        if not layout[key].is_file():
            raise FileNotFoundError(f"Missing required G0.5 asset ({key}): {layout[key]}")
    if not layout["processor"].is_dir():
        raise FileNotFoundError(f"Missing Qwen processor directory: {layout['processor']}")

    checkpoint_size = layout["checkpoint"].stat().st_size
    if checkpoint_size != EXPECTED_CHECKPOINT_SIZE:
        raise RuntimeError(
            f"Checkpoint size mismatch: got {checkpoint_size}, expected {EXPECTED_CHECKPOINT_SIZE}"
        )
    tokenizer_size = layout["action_tokenizer"].stat().st_size
    if tokenizer_size != EXPECTED_ACTION_TOKENIZER_SIZE:
        raise RuntimeError(
            f"Action tokenizer size mismatch: got {tokenizer_size}, "
            f"expected {EXPECTED_ACTION_TOKENIZER_SIZE}"
        )

    if not verify_checkpoint_hash:
        return None
    observed_hash = sha256_file(layout["checkpoint"])
    if observed_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Checkpoint SHA-256 mismatch: got {observed_hash}, "
            f"expected {EXPECTED_CHECKPOINT_SHA256}"
        )
    return observed_hash


def source_revision(source_dir: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def source_tree_state(source_dir: Path) -> dict[str, Any]:
    """Record whether a reviewed Ascend overlay changed the pinned source."""

    status = subprocess.run(
        ["git", "-C", str(source_dir), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    diff = subprocess.run(
        ["git", "-C", str(source_dir), "diff", "--binary", "HEAD", "--"],
        check=True,
        capture_output=True,
    ).stdout
    untracked_names = subprocess.run(
        ["git", "-C", str(source_dir), "ls-files", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    untracked = []
    for relative_name in untracked_names:
        path = source_dir / relative_name
        if path.is_file():
            untracked.append(
                {
                    "path": relative_name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "dirty": bool(status.strip()),
        "status": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "tracked_diff_bytes": len(diff),
        "untracked_files": untracked,
    }


def build_synthetic_raw_observation(seed: int, task: str = DEFAULT_TASK) -> dict[str, Any]:
    """Build one deterministic request with the exact raw R1 Lite schema."""

    import numpy as np

    rng = np.random.default_rng(seed)
    images = {
        key: rng.integers(0, 256, size=(3, 720, 1280), dtype=np.uint8)
        for key in CAMERA_KEYS
    }
    state = {}
    cursor = 0
    state_vector = np.linspace(-0.25, 0.25, sum(STATE_DIMS.values()), dtype=np.float32)
    for key, width in STATE_DIMS.items():
        state[key] = state_vector[cursor : cursor + width].copy()
        cursor += width
    return {
        "images": images,
        "state": state,
        "task": task,
        "frequency": 15.0,
        "embodiment_type": DEFAULT_EMBODIMENT,
    }


def observation_sha256(raw_obs: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in CAMERA_KEYS:
        array = raw_obs["images"][key]
        digest.update(key.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    for key in STATE_DIMS:
        array = raw_obs["state"][key]
        digest.update(key.encode("utf-8"))
        digest.update(array.tobytes(order="C"))
    digest.update(raw_obs["task"].encode("utf-8"))
    return digest.hexdigest()


@contextlib.contextmanager
def working_directory(path: Path) -> Iterable[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class DeviceRuntime:
    """Small CUDA/NPU/CPU synchronization and memory adapter."""

    def __init__(self, torch_module: Any, device: str):
        self.torch = torch_module
        self.device = torch_module.device(device)
        self.kind = self.device.type
        self.backend = getattr(torch_module, self.kind, None)

    def synchronize(self) -> None:
        if self.backend is None or not hasattr(self.backend, "synchronize"):
            return
        self.backend.synchronize(self.device)

    def reset_peak_memory(self) -> None:
        if self.backend is not None and hasattr(self.backend, "reset_peak_memory_stats"):
            self.backend.reset_peak_memory_stats(self.device)

    def memory(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for output_key, method_name in (
            ("memory_allocated_bytes", "memory_allocated"),
            ("memory_reserved_bytes", "memory_reserved"),
            ("max_memory_allocated_bytes", "max_memory_allocated"),
            ("max_memory_reserved_bytes", "max_memory_reserved"),
        ):
            method = getattr(self.backend, method_name, None) if self.backend is not None else None
            if method is not None:
                result[output_key] = int(method(self.device))
        return result

    def info(self) -> dict[str, Any]:
        info: dict[str, Any] = {"device": str(self.device), "device_type": self.kind}
        if self.backend is not None and hasattr(self.backend, "get_device_name"):
            info["device_name"] = str(self.backend.get_device_name(self.device))
        return info


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _token_lengths(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        lengths = []
        for item in value:
            if hasattr(item, "numel"):
                lengths.append(int(item.numel()))
            elif hasattr(item, "__len__"):
                lengths.append(len(item))
        return lengths
    if hasattr(value, "shape") and len(value.shape) >= 2:
        return [int(value.shape[-1])] * int(value.shape[0])
    return []


@contextlib.contextmanager
def record_ar_generation(model: Any):
    """Observe returned sequence lengths without changing AR arguments or outputs.

    The pinned full-policy path calls AR for CoT first and action second. Lengths
    include the terminal stop token, and with batch size one have no row padding.
    The first token is sampled from the existing hidden state, so these counts
    are not counts of transformer forward passes.
    """
    calls: list[dict[str, Any]] = []
    original = model.inference_ar
    had_instance_override = "inference_ar" in vars(model)

    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        lengths = _token_lengths(result.get("generated_ids"))
        stages = ("cot", "action")
        calls.append({
            "stage": stages[len(calls)] if len(calls) < len(stages) else "unexpected",
            "generated_token_lengths": lengths,
            "generated_tokens": sum(lengths),
        })
        return result

    model.inference_ar = observed
    try:
        yield calls
    finally:
        if had_instance_override:
            model.inference_ar = original
        else:
            delattr(model, "inference_ar")


def validate_action_outputs(core: Mapping[str, Any], served: Mapping[str, Any]) -> dict[str, Any]:
    """Fail on nonfinite actions, after the measured request has completed."""
    import numpy as np

    checked: dict[str, Any] = {}

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
        elif hasattr(value, "shape"):
            if hasattr(value, "isfinite"):
                finite = bool(value.isfinite().all().item())
            else:
                array = np.asarray(value)
                if not np.issubdtype(array.dtype, np.number):
                    return
                finite = bool(np.isfinite(array).all())
            if not finite:
                raise RuntimeError(f"Nonfinite action output: {path}")
            checked[path] = {"shape": [int(n) for n in value.shape], "all_finite": True}

    for key in ("action", "ar_action"):
        if key not in core or not hasattr(core[key], "shape"):
            raise RuntimeError(f"Missing full-model core action tensor: {key}")
    visit(core, "core")
    visit(served, "served")
    if not any(key.startswith("served.") for key in checked):
        raise RuntimeError("No served action tensors were returned")
    return checked


def make_instrumented_inferencer(base_class: type, torch_module: Any, dict_apply: Any, runtime: DeviceRuntime):
    """Instrument transfers around the official ``predict_action`` call."""

    class InstrumentedPolicyInferencer(base_class):
        last_transfer_timing: dict[str, float]
        last_model_metadata: dict[str, Any]
        last_core_actions: dict[str, Any]

        def forward_batch(self, batch: dict) -> dict:
            runtime.synchronize()
            started = time.monotonic()
            batch = dict_apply(
                batch,
                lambda value: value.to(self.device)
                if isinstance(value, torch_module.Tensor)
                else value,
            )
            runtime.synchronize()
            after_h2d = time.monotonic()
            with record_ar_generation(self.policy.model) as ar_calls, torch_module.no_grad(), torch_module.autocast(
                runtime.kind,
                dtype=torch_module.bfloat16,
            ):
                batch = self.policy.predict_action(batch)
            runtime.synchronize()
            after_model = time.monotonic()
            self.last_model_metadata = {
                "ar_token_lengths": _token_lengths(batch.get("decoded_action_tokens")),
                "ar_generation_calls": ar_calls,
            }
            batch = dict_apply(
                batch,
                lambda value: value.cpu() if isinstance(value, torch_module.Tensor) else value,
            )
            runtime.synchronize()
            after_d2h = time.monotonic()
            self.last_core_actions = {
                key: batch[key] for key in ("action", "ar_action") if key in batch
            }
            self.last_transfer_timing = {
                "h2d_ms": (after_h2d - started) * 1000.0,
                "model_predict_action_ms": (after_model - after_h2d) * 1000.0,
                "d2h_ms": (after_d2h - after_model) * 1000.0,
            }
            return batch

    return InstrumentedPolicyInferencer


def _override_paths(layout: Mapping[str, Path]) -> list[str]:
    processor = str(layout["processor"])
    tokenizer = str(layout["action_tokenizer"])
    # load_config_from_run_dir applies overrides after resolving the saved Hydra
    # graph, so patch every supported expanded/legacy location.
    return [
        f"eval_embodiment={DEFAULT_EMBODIMENT}",
        "model.use_torch_compile=false",
        "model.model_arch.vlm.linear_attn_backend=torch",
        "model.model_arch.action_expert.linear_attn_backend=torch",
        "model.model_arch.fm.use_correlated_noise=false",
        f"model.model_arch.hf_processor_path={processor}",
        f"model.processor.tokenizer_params.pretrained_model_name_or_path={processor}",
        f"tokenizer.vq_config.ckpt_dir={tokenizer}",
        f"model.tokenizer.vq_config.ckpt_dir={tokenizer}",
        f"model.model_arch.AT_CONFIG.ckpt_dir={tokenizer}",
    ]


def _output_shapes(result: Mapping[str, Any]) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for key, value in result.items():
        shape = getattr(value, "shape", None)
        if shape is not None:
            shapes[key] = [int(dimension) for dimension in shape]
    return shapes


def _run_one(
    *,
    raw_obs: dict[str, Any],
    processor: Any,
    build_obs_dict: Any,
    inferencer: Any,
    runtime: DeviceRuntime,
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime.synchronize()
    request_start = time.monotonic()
    build_start = request_start
    obs_dict = build_obs_dict(raw_obs, processor)
    build_end = time.monotonic()
    results, timing = inferencer.infer_with_timing([obs_dict])
    runtime.synchronize()
    request_end = time.monotonic()
    output_validation = validate_action_outputs(inferencer.last_core_actions, results[0])

    sample: dict[str, Any] = {
        "raw_build_obs_ms": (build_end - build_start) * 1000.0,
        "request_total_ms": (request_end - request_start) * 1000.0,
    }
    for key, value in timing.items():
        numeric = _numeric(value)
        if numeric is not None:
            sample[key] = numeric
    sample.update(inferencer.last_transfer_timing)
    sample["cpu_preprocess_total_ms"] = sample["raw_build_obs_ms"] + sample.get(
        "preprocess_ms", 0.0
    )
    core_ms = sample.get("forward_inference_total_ms")
    token_lengths = inferencer.last_model_metadata.get("ar_token_lengths", [])
    if token_lengths:
        # The upstream policy exposes the action tokens parsed back out of the
        # generated string, not the complete raw AR sequence. Keep the label
        # precise so this is not mistaken for decoder token throughput.
        sample["ar_decoded_action_tokens"] = int(sum(token_lengths))
    calls = inferencer.last_model_metadata["ar_generation_calls"]
    if [call["stage"] for call in calls] != ["cot", "action"]:
        raise RuntimeError(f"Expected CoT and action AR calls, observed: {calls}")
    sample["ar_generation_calls"] = calls
    sample["cot_raw_generated_tokens"] = calls[0]["generated_tokens"]
    sample["action_ar_raw_generated_tokens"] = calls[1]["generated_tokens"]
    sample["output_validation"] = output_validation
    if core_ms is not None:
        sample["non_core_infer_ms"] = max(0.0, sample.get("infer_ms", core_ms) - core_ms)
    return sample, results[0]


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    print("[progress] Validating source and checkpoint assets", flush=True)
    source_dir = args.source_dir.expanduser().resolve()
    if not (source_dir / "src" / "g05").is_dir():
        raise FileNotFoundError(f"Not a GalaxeaVLA source tree: {source_dir}")
    revision = source_revision(source_dir)
    tree_state = source_tree_state(source_dir)
    if revision != EXPECTED_SOURCE_REVISION and not args.allow_source_revision:
        raise RuntimeError(
            f"GalaxeaVLA revision mismatch: got {revision}, expected {EXPECTED_SOURCE_REVISION}. "
            "Use --allow-source-revision only for an explicitly reviewed source patch."
        )

    layout = resolve_asset_layout(args.checkpoint, args.asset_root)
    observed_checkpoint_hash = validate_asset_layout(
        layout,
        verify_checkpoint_hash=not args.skip_checkpoint_sha256,
    )

    sys.path.insert(0, str(source_dir / "src"))
    sys.path.insert(0, str(source_dir))
    if args.device.split(":", 1)[0] == "npu":
        importlib.import_module("torch_npu")
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if hasattr(torch, "npu") and hasattr(torch.npu, "manual_seed_all"):
        torch.npu.manual_seed_all(args.seed)

    runtime = DeviceRuntime(torch, args.device)
    from g05.models.g05.inferencer import PolicyInferencer
    from g05.utils.checkpoint.ckpt_utils import find_run_dir, load_config_from_run_dir
    from g05.utils.common.pytorch_utils import dict_apply
    from g05.utils.eval.eval_utils import filter_embodiment
    from scripts.serve_policy import build_obs_dict, setup

    raw_obs = build_synthetic_raw_observation(args.seed, args.task)
    raw_obs_hash = observation_sha256(raw_obs)
    overrides = _override_paths(layout) + list(args.override)

    with working_directory(source_dir):
        print("[progress] Loading full G0.5 policy", flush=True)
        load_started = time.monotonic()
        run_dir = find_run_dir(str(layout["checkpoint"]))
        cfg, config_compatibility = load_r1lite_checkpoint_config(
            source_dir=source_dir,
            run_dir=run_dir,
            checkpoint=layout["checkpoint"],
            overrides=overrides,
            omega_conf=OmegaConf,
            load_config_from_run_dir=load_config_from_run_dir,
        )
        if "embodiment_datasets" in cfg.data:
            filter_embodiment(cfg, DEFAULT_EMBODIMENT)
        contract = validate_shipped_default_contract(cfg)
        policy, processor = setup(cfg, device=args.device)
        effective_action_ar_max_new_tokens = int(
            policy._get_action_generation_max_new_tokens()
        )
        runtime.synchronize()
        load_ms = (time.monotonic() - load_started) * 1000.0
        print(f"[progress] Model loaded in {load_ms:.3f} ms", flush=True)

        InstrumentedInferencer = make_instrumented_inferencer(
            PolicyInferencer,
            torch,
            dict_apply,
            runtime,
        )
        inferencer = InstrumentedInferencer(policy, processor, device=args.device)

        runtime.reset_peak_memory()
        print("[progress] Cold request started", flush=True)
        cold_sample, cold_result = _run_one(
            raw_obs=raw_obs,
            processor=processor,
            build_obs_dict=build_obs_dict,
            inferencer=inferencer,
            runtime=runtime,
        )
        cold_memory = runtime.memory()
        print(f"[progress] Cold request complete: {cold_sample['request_total_ms']:.3f} ms; outputs finite", flush=True)

        warmup_iterations = 0 if args.smoke_only else args.warmup
        measured_iterations = 0 if args.smoke_only else args.iterations
        warmup_samples = []
        for index in range(warmup_iterations):
            print(f"[progress] Warmup {index + 1}/{warmup_iterations} started", flush=True)
            sample, _ = _run_one(
                raw_obs=raw_obs,
                processor=processor,
                build_obs_dict=build_obs_dict,
                inferencer=inferencer,
                runtime=runtime,
            )
            warmup_samples.append(sample)
            print(f"[progress] Warmup {index + 1}/{warmup_iterations} complete: {sample['request_total_ms']:.3f} ms", flush=True)

        measured_samples = []
        last_result = cold_result
        runtime.reset_peak_memory()
        for index in range(measured_iterations):
            print(f"[progress] Measurement {index + 1}/{measured_iterations} started", flush=True)
            sample, result = _run_one(
                raw_obs=raw_obs,
                processor=processor,
                build_obs_dict=build_obs_dict,
                inferencer=inferencer,
                runtime=runtime,
            )
            measured_samples.append(sample)
            last_result = result
            print(f"[progress] Measurement {index + 1}/{measured_iterations} complete: {sample['request_total_ms']:.3f} ms", flush=True)
        measured_memory = runtime.memory()

    summaries = summarize_samples(measured_samples) if measured_samples else {}
    recompute_p50_ms = (
        summaries["request_total_ms"]["p50"] if "request_total_ms" in summaries else None
    )
    report = {
        "schema_version": 1,
        "benchmark": "galaxeavla-g05-base-r1lite-full-model",
        "result_scope": "synthetic-input full-model throughput",
        "accuracy_scope": "none; synthetic observations are not a robot task accuracy evaluation",
        "released_default_path": {
            "chain_of_thought_autoregressive": True,
            "prompt_max_text_tokens": 160,
            "configured_ar_max_new_tokens": 300,
            "effective_action_ar_max_new_tokens": effective_action_ar_max_new_tokens,
            "continuous_flow_matching": True,
            "flow_matching_steps": 10,
            "discrete_autoregressive": True,
            "predict_cot": True,
            "action_horizon": 32,
            "served_action_steps": args.action_steps,
        },
        "source": {
            "path": str(source_dir),
            "revision": revision,
            "expected_revision": EXPECTED_SOURCE_REVISION,
            "revision_override_used": revision != EXPECTED_SOURCE_REVISION,
            "tree_state": tree_state,
        },
        "checkpoint": {
            "path": str(layout["checkpoint"]),
            "size_bytes": layout["checkpoint"].stat().st_size,
            "sha256": observed_checkpoint_hash,
            "expected_sha256": EXPECTED_CHECKPOINT_SHA256,
            "hash_verification_skipped": args.skip_checkpoint_sha256,
        },
        "input": {
            "kind": "deterministic synthetic R1 Lite raw observation",
            "sha256": raw_obs_hash,
            "seed": args.seed,
            "batch_size": 1,
            "cameras": {key: [3, 720, 1280] for key in CAMERA_KEYS},
            "model_camera_shape": [3, 256, 256],
            "state_parts": STATE_DIMS,
            "raw_state_dim": sum(STATE_DIMS.values()),
            "model_state_dim": 27,
            "task": args.task,
        },
        "configuration_contract": contract,
        "configuration_compatibility": config_compatibility,
        "ascend_adaptation": {
            "linear_attention_backend": "torch fallback",
            "torch_compile": False,
            "vision_attention": "SDPA fallback when CUDA FlashAttention is absent",
            "overrides": overrides,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": runtime.info(),
            "cpu_threads": {
                "torch_intraop": torch.get_num_threads(),
                "torch_interop": torch.get_num_interop_threads(),
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            },
        },
        "load_ms": load_ms,
        "cold_request": {"timing_ms": cold_sample, "memory": cold_memory},
        "warmup": {"iterations": warmup_iterations, "samples": warmup_samples},
        "measurement": {
            "iterations": measured_iterations,
            "summary": summaries,
            "samples": measured_samples,
            "memory": measured_memory,
            "theoretical_action_capacity_hz_at_p50": (
                args.action_steps * 1000.0 / recompute_p50_ms
                if recompute_p50_ms is not None
                else None
            ),
            "recomputes_per_second_at_p50": (
                1000.0 / recompute_p50_ms if recompute_p50_ms is not None else None
            ),
        },
        "output_contract": {
            "cold_action_shapes": _output_shapes(cold_result),
            "last_action_shapes": _output_shapes(last_result),
            "all_requests_actions_finite": True,
            "validation_timing": "CPU validation after each measured request timer stops",
            "token_count_semantics": (
                "Raw AR counts are returned generated_ids lengths including stop tokens; "
                "batch size is one. The first token uses the existing hidden state. "
                "Decoded action tokens may originate in CoT; their count is not action AR throughput."
            ),
        },
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the complete released-default G0.5 R1 Lite policy on Ascend."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--asset-root",
        type=Path,
        help="Directory containing action_tokenizer.pt and qwen3_5_2b_base_processor/; "
        "defaults to the g05-base parent.",
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run exactly one cold full-model request after loading; skip warmup/statistics.",
    )
    parser.add_argument("--action-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional reviewed Hydra override (repeatable).",
    )
    parser.add_argument("--skip-checkpoint-sha256", action="store_true")
    parser.add_argument("--allow-source-revision", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.iterations < 1 and not args.smoke_only:
        parser.error("--iterations must be >= 1")
    if args.action_steps != 16:
        parser.error("the released R1 Lite service contract requires --action-steps 16")

    report = run_benchmark(args)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".part")
    partial.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(output)
    print(f"Wrote synthetic full-model benchmark: {output}")
    request_summary = report["measurement"]["summary"].get("request_total_ms")
    if request_summary is None:
        print(
            "Smoke request: "
            f"{report['cold_request']['timing_ms']['request_total_ms']:.3f} ms"
        )
    else:
        print(
            "p50 recompute: "
            f"{request_summary['p50']:.3f} ms; "
            "theoretical 16-action capacity: "
            f"{report['measurement']['theoretical_action_capacity_hz_at_p50']:.3f} Hz"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Real LIBERO episode 0/frame 0, aligned with the RTX 3090 120 ms benchmark.

Times prepared device inputs -> prefill -> ten-step continuous FM -> normalized
action. Image loading/preprocessing, transfer, compilation and output copying are
reported/excluded explicitly. This is separate from the R1 Lite request benchmark.
No dataset/video packages are needed: use the original lossless PNGs and case.json.
The saved CPU-generated noise must be shared for CUDA/NPU numerical comparisons;
historical CUDA seed-0 actions alone are not a valid cross-device reference.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Any

import benchmark_g05_full as base
from g05_npu_graphs import install_ae_backend, install_module_backend, install_gdn_backends, validate_graph_options


ASSET_REVISION = "aca2916ba2adaba3ce05bfa279cdc6b98da35cad"
CHECKPOINT_SIZE = 11_440_381_065
CHECKPOINT_SHA256 = "432f225c7647bdab77bfa2c2b397dc38cf8631bd70c92b85b261ff2817c1e795"
TASK = "open the middle drawer of the cabinet"
VALID_DIMS = [10, 11, 12, 13, 14, 15, 19]
ACTION_SHAPE = (1, 32, 20)
INPUT_SHA256 = {
    "case.json": "88125ba0258013aae556b18a179b6693288eb61dc388c6cf7ce97bb779740bd4",
    "image.png": "0f6008542bd572e28d555d804b2c3c78bcef9a99cbe4092362bd6c8232cf5632",
    "wrist_image.png": "341401f9ed834a8745678d432ed7dfc1be69437cf2e107bc1cb6cfc89516d62a",
}
# Immutable Git blob IDs from the official HF tree, identical at aca2916 and
# e312be81e90c56a55bcb26b57429bd39a335b449. LFS files use content SHA-256.
ASSET_BLOBS = {
    "g05-libero/.hydra/config.yaml": (20644, "2802cbf122f06ca9a860af17fafb5f0edd3bfaae"),
    "g05-libero/dataset_stats.json": (95006, "0bf668a172218b517c51034a3c4a3cacfa13f2c1"),
    "qwen3_5_2b_base_processor/config.json": (2908, "d30a15be1bbfd610f58ad80f1fbeb6778f58a80f"),
    "qwen3_5_2b_base_processor/configuration.json": (51, "3a6d425685de8896b2bc8b59b671e41aea1d7bf3"),
    "qwen3_5_2b_base_processor/merges.txt": (3353259, "a494e019ca1502219fd0128658b979e5f05ae8e8"),
    "qwen3_5_2b_base_processor/preprocessor_config.json": (390, "2ea84a437d448ff71b08df68fdd949d5cc4ebb64"),
    "qwen3_5_2b_base_processor/tokenizer_config.json": (16712, "34e1f7a59c40a7a4197ce3dbb242b94dc5bef1dd"),
    "qwen3_5_2b_base_processor/video_preprocessor_config.json": (386, "37900b3ff9295e1aa7e211378466356b52e64e55"),
    "qwen3_5_2b_base_processor/vocab.json": (6722759, "0aa0ce0658d60ac4a5d609f4eadb0e8e43514176"),
}
ASSET_LFS = {
    "action_tokenizer.pt": (506886775, "5088f64a5452a60bbc8cac90ee7d79c156f1d14540bc3139aa7060f862dddace"),
    "qwen3_5_2b_base_processor/tokenizer.json": (12807196, "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927"),
}


def validate_assets(checkpoint: Path, root: Path, checkpoint_sha256: str) -> dict:
    """Verify model and every reused base-checkpoint sidecar before torch.load."""
    rows = {}
    files = dict(ASSET_LFS)
    files["g05-libero/model.pt"] = (CHECKPOINT_SIZE, checkpoint_sha256)
    for relative, (size, expected) in {**ASSET_BLOBS, **files}.items():
        path = checkpoint if relative == "g05-libero/model.pt" else root / relative
        if path.stat().st_size != size:
            raise ValueError(f"Asset size mismatch: {path}")
        sha256 = base.sha256_file(path)
        if relative in ASSET_BLOBS:
            payload = path.read_bytes()
            observed = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
        else:
            observed = sha256
        if observed != expected:
            raise ValueError(f"Asset hash mismatch: {path}")
        rows[relative] = {"size": size, "sha256": sha256}
    return rows


def validate_case(case: dict) -> dict:
    data = case["dataset"]
    expected = {"episode": 0, "frame": 0, "global_index": 0, "task": TASK, "fps": 20.0}
    if any(data.get(k) != v for k, v in expected.items()):
        raise ValueError("Expected the original LIBERO episode 0/frame 0/task/fps")
    if len(data.get("raw_state", [])) != 8:
        raise ValueError("Expected original eight-element LIBERO observation.state")
    if case["checkpoint"]["bytes"] != CHECKPOINT_SIZE:
        raise ValueError("Case is not the original g05-libero checkpoint")
    return data


def load_observation(directory: Path) -> tuple[dict, dict]:
    import numpy as np
    from PIL import Image
    hashes = {name: base.sha256_file(directory / name) for name in INPUT_SHA256}
    if hashes != INPUT_SHA256:
        raise ValueError("Input files do not match the archived real 3090 episode")
    case = json.loads((directory / "case.json").read_text(encoding="utf-8"))
    data = validate_case(case)
    images = {}
    for key in ("image", "wrist_image"):
        with Image.open(directory / f"{key}.png") as image:
            if image.mode != "RGB" or image.size != (512, 512):
                raise ValueError(f"Unexpected real image format: {key}")
            images[key] = np.ascontiguousarray(np.asarray(image).transpose(2, 0, 1))
    state = np.asarray(data["raw_state"], dtype=np.float32)
    return {
        "images": images,
        "state": {"right_ee_pose": state[:6].copy(), "right_gripper": state[6:7].copy()},
        "task": TASK, "frequency": 20.0, "embodiment_type": "libero",
    }, {"files_sha256": hashes, "episode": 0, "frame": 0, "task": TASK, "frequency": 20.0}


def prepare_noise(source: Path | None, destination: Path, seed: int) -> tuple[Any, dict]:
    import numpy as np
    if source is None:
        noise = np.random.Generator(np.random.PCG64(seed)).standard_normal(ACTION_SHAPE).astype(np.float32)
        origin = "NumPy PCG64 CPU Gaussian; not the historical CUDA seed-0 noise"
    else:
        noise = np.load(source, allow_pickle=False)
        origin = str(source)
    if noise.shape != ACTION_SHAPE or noise.dtype != np.float32 or not np.isfinite(noise).all():
        raise ValueError("Noise must be finite float32 [1,32,20]")
    noise = np.ascontiguousarray(noise)
    np.save(destination, noise, allow_pickle=False)
    return noise, {"path": str(destination), "sha256": base.sha256_file(destination),
                   "tensor_sha256": hashlib.sha256(noise.tobytes()).hexdigest(),
                   "shape": list(noise.shape), "origin": origin, "seed": seed}


@contextlib.contextmanager
def explicit_noise(helper: Any, noise: Any):
    """Clone the immutable noise because FM masks/updates its input in place."""
    original = helper._sample_noise
    had_override = "_sample_noise" in vars(helper)
    calls = {"count": 0, "tensor": noise, "mode": "shared"}

    def sample(actions, dtype, embodiment_types=None):
        if tuple(actions.shape) != ACTION_SHAPE or list(embodiment_types or []) != ["libero"]:
            raise RuntimeError("Unexpected FM noise request")
        calls["count"] += 1
        if calls["mode"] == "native":
            return original(actions, dtype, embodiment_types)
        selected = calls["tensor"]
        if actions.device != selected.device or dtype != selected.dtype:
            raise RuntimeError("Explicit noise device/dtype mismatch; convert before timing")
        return selected.clone()

    helper._sample_noise = sample
    try:
        yield calls
    finally:
        if had_override:
            helper._sample_noise = original
        else:
            del helper._sample_noise


def validate_contract(cfg: Any) -> dict:
    expected = {
        "model.use_torch_compile": False,
        "model.model_arch.action_dim": 20,
        "model.model_arch.proprio_dim": 20,
        "model.model_arch.predict_cot": False,
        "model.model_arch.discrete_action": False,
        "model.model_arch.continuous_action": True,
        "model.model_arch.return_continuous_action": True,
        "model.model_arch.attn_implementation": "eager",
        "model.model_arch.fm.num_inference_steps": 10,
        "model.model_arch.fm.time_convention": "pi_convention",
        "model.model_arch.fm.use_correlated_noise": False,
        "model.processor.num_output_cameras": 2,
        "model.processor.num_input_cameras": 2,
        "model.processor.camera_size_config.exterior": [256, 256],
        "model.processor.camera_size_config.wrist_right": [256, 256],
        "data.action_size": 32, "data.obs_size": 1,
    }
    actual = {k: base._select(cfg, k) for k in expected}
    for key, wanted in expected.items():
        if isinstance(wanted, list) and actual[key] is not None:
            actual[key] = list(actual[key])
    mismatches = {k: {"expected": v, "actual": actual[k]} for k, v in expected.items() if actual[k] != v}
    if mismatches:
        raise ValueError("LIBERO workload mismatch: " + json.dumps(mismatches, default=str))
    return actual


def load_config(source, root, checkpoint, weights, OmegaConf, load_config_from_run_dir):
    """Reanchor only sidecar paths; preserve released LIBERO model/data/processor."""
    run_dir = root / "g05-libero"
    raw = OmegaConf.load(run_dir / ".hydra/config.yaml")
    OmegaConf.set_struct(raw, False)
    patches = {
        "eval_embodiment": "libero", "model.use_torch_compile": False,
        "model.model_weights_to_bf16": weights == "bf16",
        "model.model_arch.pretrained_model_path": None,
        "model.model_arch.hf_processor_path": str(root / "qwen3_5_2b_base_processor"),
        "model.processor.tokenizer_params.pretrained_model_name_or_path": str(root / "qwen3_5_2b_base_processor"),
        "tokenizer.vq_config.ckpt_dir": str(root / "action_tokenizer.pt"),
        "model.tokenizer.vq_config.ckpt_dir": str(root / "action_tokenizer.pt"),
        "model.model_arch.AT_CONFIG.ckpt_dir": str(root / "action_tokenizer.pt"),
        # Absent in the released LIBERO config: this is FMHelper's default.
        "model.model_arch.fm.use_correlated_noise": False,
        "model.model_arch.vlm.linear_attn_backend": "torch",
        "model.model_arch.action_expert.linear_attn_backend": "torch",
    }
    for key, value in patches.items():
        OmegaConf.update(raw, key, value, merge=False, force_add=True)
    with tempfile.TemporaryDirectory(prefix="g05-libero-config-") as temporary:
        shadow = Path(temporary)
        (shadow / ".hydra").mkdir()
        OmegaConf.save(raw, shadow / ".hydra/config.yaml", resolve=False)
        cfg = load_config_from_run_dir(shadow, str(checkpoint), [])
    cfg.run_dir = str(run_dir)
    cfg.output_dir = str(run_dir / "eval_model")
    cfg.ckpt_path = str(checkpoint)
    cfg.datastatics_path = str(run_dir / "dataset_stats.json")
    return cfg, patches


def action_comparison(reference, candidate, pad_mask) -> dict:
    import numpy as np
    ref, got = np.asarray(reference), np.asarray(candidate)
    mask = np.asarray(pad_mask, dtype=bool)
    if ref.shape != ACTION_SHAPE or got.shape != ACTION_SHAPE or mask.shape != (1, 20):
        raise ValueError("Expected [1,32,20] actions and [1,20] padding mask")
    valid = np.flatnonzero(~mask[0]).tolist()
    if valid != VALID_DIMS:
        raise ValueError(f"Unexpected LIBERO valid dimensions: {valid}")
    if not np.isfinite(ref).all() or not np.isfinite(got).all():
        raise ValueError("Nonfinite action, including padded dimensions")
    a, b = ref[..., valid].astype(np.float64), got[..., valid].astype(np.float64)
    diff = b - a
    norm = float(np.linalg.norm(a.ravel()))
    return {
        "scope": "normalized action chunk, seven valid LIBERO dimensions",
        "valid_action_dimensions": valid, "sample_count": int(diff.size), "all_finite": True,
        "mae": float(np.mean(np.abs(diff))), "rmse": float(np.sqrt(np.mean(diff**2))),
        "max_abs_error": float(np.max(np.abs(diff))),
        "p95_abs_error": float(np.percentile(np.abs(diff), 95)),
        "relative_l2": float(np.linalg.norm(diff.ravel()) / max(norm, 1e-12)),
        "cosine_similarity": float(np.dot(a.ravel(), b.ravel()) / max(norm * np.linalg.norm(b.ravel()), 1e-12)),
        "sign_agreement": float(np.mean(np.signbit(a) == np.signbit(b))),
        "gripper_sign_agreement": float(np.mean(np.signbit(a[..., -1]) == np.signbit(b[..., -1]))),
        "mae_per_valid_dimension": np.mean(np.abs(diff), axis=(0, 1)).tolist(),
        "padded_dimensions_max_abs_candidate": float(np.max(np.abs(got[..., mask[0]]))),
    }


def check_reference_identity(reference: dict, actual: dict) -> None:
    if reference != actual:
        changed = [key for key in set(reference) | set(actual) if reference.get(key) != actual.get(key)]
        raise ValueError("Cross-device reference identity mismatch: " + ", ".join(sorted(changed)))


def context_verification_observations(raw: dict, *, independent_cameras=False) -> dict:
    """Separate state/task perturbations; never mutate the timed real observation."""
    state_changed = copy.deepcopy(raw)
    state_changed["state"]["right_ee_pose"] += 0.01
    task_changed = copy.deepcopy(raw)
    task_changed["task"] = "close the middle drawer of the cabinet"
    observations = {"state_C": state_changed, "task_D": task_changed}
    if independent_cameras:
        import numpy as np
        for label, key in (("camera_E", "image"), ("wrist_F", "wrist_image")):
            changed = copy.deepcopy(raw)
            changed["images"][key] = np.ascontiguousarray(np.flip(changed["images"][key], axis=-1))
            observations[label] = changed
    return observations


def encoded_context(state) -> dict:
    return {name: getattr(state, name).detach().cpu().numpy().copy()
            for name in ("input_ids", "attention_mask")}


def compare_encoded_context(reference, candidate) -> dict:
    import numpy as np
    result = {name: bool(np.array_equal(reference[name], candidate[name]))
              for name in ("input_ids", "attention_mask")}
    if not all(result.values()):
        raise RuntimeError("Context optimization changed encoded tokens or semantic mask")
    return result


def install_backend(torch, policy, name, options=None):
    if name != "inductor":
        return install_ae_backend(torch, policy, name, options=options)
    original = policy.model.action_expert.forward
    metadata = {"target": "action_expert.forward", "backend": name, "fallback_allowed": False}
    if name == "eager":
        return original, metadata
    torch._dynamo.config.suppress_errors = False
    if name == "inductor":
        kwargs = {"backend": "inductor", "mode": "reduce-overhead", "dynamic": False, "fullgraph": False}
    elif name == "torchair":
        import torchair
        from torchair.configs.compiler_config import CompilerConfig
        kwargs = {"backend": torchair.get_npu_backend(compiler_config=CompilerConfig()), "dynamic": False, "fullgraph": True}
    else:
        kwargs = {"backend": "npugraph_ex", "dynamic": False, "fullgraph": True}
    policy.model.action_expert.forward = torch.compile(original, **kwargs)
    metadata.update({k: v for k, v in kwargs.items() if k != "backend"})
    return original, metadata


def collect_counted_workload(torch_module, policy, inference, pad_mask, *, counter_module=None) -> dict:
    """Observe one restored-eager request; never retain its hook-perturbed timing.

    The caller owns backend restoration and the fixed-noise context. Keeping
    attachment inside the try also cleans up a partially failed attach().
    """
    if counter_module is None:
        counter_module = importlib.import_module("count_g05_workload")
    counter = counter_module.WorkloadCounter(torch_module)
    try:
        counter.attach(policy)
        _, action, _discarded_diagnostic_timing = inference()
        observed = action.detach().float().cpu().numpy().copy()
        validation = action_comparison(observed, observed, pad_mask)
        if validation["padded_dimensions_max_abs_candidate"] != 0.0:
            raise RuntimeError("Counted LIBERO diagnostic produced nonzero padded actions")
    finally:
        counter.close()
    counted = counter.summary()
    return {
        "schema_version": 1,
        "result_scope": "one real LIBERO episode 0/frame 0 restored-eager request; partial operator arithmetic",
        "diagnostic_requests": 1,
        "execution": {"ae_backend": "eager", "vlm_backend": "eager", "gdn_backend": "eager",
                      "gdn_algorithm": "original", "noise": "shared explicit tensor"},
        "arithmetic_convention": "One multiplication and one accumulation count as 2 FLOPs; called dense Linear/Conv modules and AdaLN functional conditioning projections only",
        "not_counted": [
            "attention QK/AV matmuls and SDPA internals",
            "GatedDeltaNet chunk/recurrent rule functional tensor operations",
            "functional convolutions bypassing Conv module forward (including GDN decode)",
            "ConvTranspose operations, including ActionCodec decoder upsampling",
            "bias adds, activations, normalization, elementwise operations, reductions, RoPE and Euler updates",
            "embedding/codebook lookups, data movement, preprocessing and backend padding/recomputation",
        ],
        "interpretation": "Incomplete mathematical operator count, not total model FLOPs, measured hardware instructions, hardware utilization, or a compute-capacity requirement",
        "timing_scope": "Hooks perturb execution. This request runs after formal timing and backend restoration; its latency is discarded and not added to benchmark samples or memory peaks.",
        "parameter_scope": "Loaded policy and ActionCodec parameter roots, including inactive parameters; parameter presence does not imply an operator ran",
        "instrumentation_sha256": base.sha256_file(Path(counter_module.__file__)),
        "benchmark_sha256": base.sha256_file(Path(__file__)),
        "roots": counter.roots,
        "summary": {"counted_flops": counted["counted_flops"], "stages": counted["stages"]},
        "operators": counted["operators"],
        "parameters": counter.parameters,
        "prefill": counter.prefill,
        "model_calls": counter.model_calls,
        "output_validation": validation,
    }


def run(args) -> dict:
    import numpy as np
    print("[progress] Validating source revision, model SHA-256 and all sidecar hashes", flush=True)
    source, root = args.source_dir.resolve(), args.asset_root.resolve()
    checkpoint, output = args.checkpoint.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    revision = base.source_revision(source)
    if revision != base.EXPECTED_SOURCE_REVISION and not args.allow_source_revision:
        raise ValueError("Unexpected source revision")
    assets = validate_assets(checkpoint, root, args.checkpoint_sha256)
    print("[progress] Asset hashes verified; loading archived real LIBERO input", flush=True)
    raw, input_meta = load_observation(args.input_dir.resolve())
    noise_np, noise_meta = prepare_noise(args.noise, output / "noise.npy", args.seed)
    sys.path[:0] = [str(source / "src"), str(source)]
    if args.device.startswith("npu"):
        importlib.import_module("torch_npu")
    import torch
    from omegaconf import OmegaConf
    from g05.models.g05.inferencer import PolicyInferencer
    from g05.utils.checkpoint.ckpt_utils import load_config_from_run_dir
    from g05.utils.common.pytorch_utils import dict_apply
    from g05.utils.config.config_resolvers import register_default_resolvers
    from g05.utils.eval.eval_utils import filter_embodiment
    from scripts.serve_policy import setup, build_obs_dict
    register_default_resolvers()
    runtime = base.DeviceRuntime(torch, args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    with base.working_directory(source):
        print("[progress] Loading checkpoint-authored LIBERO configuration", flush=True)
        cfg, patches = load_config(source, root, checkpoint, args.weights, OmegaConf, load_config_from_run_dir)
        filter_embodiment(cfg, "libero")
        contract = validate_contract(cfg)
        print(f"[progress] Loading complete g05-libero model ({args.weights} resident weights)", flush=True)
        started = time.perf_counter()
        policy, processor = setup(cfg, device=args.device)
        policy.eval()
        runtime.synchronize()
        load_ms = (time.perf_counter() - started) * 1000
        print(f"[progress] Model loaded in {load_ms:.3f} ms; preparing real two-camera request", flush=True)
        started = time.perf_counter()
        obs = build_obs_dict(raw, processor)
        inferencer = PolicyInferencer(policy, processor, device=args.device)
        prepared = inferencer._prepare(obs)
        batch = inferencer._collate([prepared.sample], padding_input_id=prepared.sub_processor.pad_token_id)
        batch = dict_apply(batch, lambda x: x.to(args.device) if isinstance(x, torch.Tensor) else x)
        runtime.synchronize()
        preparation_ms = (time.perf_counter() - started) * 1000
        print(f"[progress] Input prepared/transferred in {preparation_ms:.3f} ms", flush=True)
        pad_np = batch["action_dim_is_pad"].detach().cpu().numpy().astype(bool)
        action_comparison(np.zeros(ACTION_SHAPE), np.zeros(ACTION_SHAPE), pad_np)
        np.save(output / "action_dim_is_pad.npy", pad_np)
        first_image = next(iter(batch["pixel_values"].values())) if isinstance(batch["pixel_values"], dict) else batch["pixel_values"]
        noise = torch.from_numpy(noise_np).to(device=args.device, dtype=first_image.dtype)

        def inference(selected_batch=None):
            selected_batch = batch if selected_batch is None else selected_batch
            with torch.inference_mode(), torch.autocast(runtime.kind, dtype=torch.bfloat16):
                runtime.synchronize()
                start = time.perf_counter()
                state = policy.prefill(selected_batch["samples"], selected_batch["pixel_values"])
                runtime.synchronize()
                middle = time.perf_counter()
                generated = policy.generate_action(state, selected_batch["samples"], action_dim_is_pad=selected_batch["action_dim_is_pad"])
                runtime.synchronize()
                end = time.perf_counter()
            if "ar_action" in generated:
                raise RuntimeError("Unexpected discrete AR execution")
            return state, generated["action"], {"prefill_ms": (middle-start)*1000,
                "action_ms": (end-middle)*1000, "total_ms": (end-start)*1000}

        eager_action, eager_b = None, None
        replay_comparisons = {}
        verification_timings = {}
        noise_b = -noise
        verification_batch = None
        reference_contrast = None
        workload = None
        context_batches, context_references, context_checks, context_contrasts = {}, {}, {}, {}
        eager_contexts = {}
        # Changed camera content with identical noise is required whenever a
        # prefix/cache path changes; changing both cameras and noise together
        # cannot independently detect a graph replaying stale visual KV data.
        changed_prefix = (args.static_vision or args.functional_mrope or args.vlm_backend != "eager"
                          or args.gdn_backend != "eager" or args.gdn_algorithm != "original")
        if changed_prefix or args.skip_unused_action_codec or args.ae_attention != "eager" or args.ae_norm != "eager" or args.ae_rope != "eager" or args.ae_qknorm_rope != "eager" or args.precompute_time_conditioning:
            for label, observation in context_verification_observations(raw, independent_cameras=changed_prefix).items():
                item = inferencer._prepare(build_obs_dict(observation, processor))
                collated = inferencer._collate([item.sample], padding_input_id=item.sub_processor.pad_token_id)
                context_batches[label] = dict_apply(
                    collated, lambda x: x.to(args.device) if isinstance(x, torch.Tensor) else x)
        with explicit_noise(policy.model.fm_helper, noise) as calls:
            if args.verify:
                print("[progress] Eager numerical references A/B (outside benchmark timings)", flush=True)
                # B is a controlled image perturbation used only to test changed
                # VLM/KV/AE graph inputs. The benchmark still uses real sample A.
                verification_batch = dict(batch)
                verification_batch["pixel_values"] = dict_apply(
                    batch["pixel_values"], lambda value: torch.flip(value, dims=[-1]) if isinstance(value, torch.Tensor) else value)
                ref_state, eager, _ = inference()
                eager_contexts["A"] = encoded_context(ref_state)
                eager_action = eager.detach().float().cpu().numpy().copy()
                np.save(output / "eager_action.npy", eager_action)
                calls["tensor"] = noise_b
                ref_state, eager, _ = inference(verification_batch)
                eager_contexts["B"] = encoded_context(ref_state)
                eager_b = eager.detach().float().cpu().numpy().copy()
                reference_contrast = action_comparison(eager_action, eager_b, pad_np)
                calls["tensor"] = noise
                for label, selected_batch in context_batches.items():
                    print(f"[progress] Eager context reference {label}", flush=True)
                    ref_state, ref_action, _ = inference(selected_batch)
                    context_references[label] = {
                        "context": encoded_context(ref_state),
                        "action": ref_action.detach().float().cpu().numpy().copy(),
                    }
                    context_contrasts[label] = action_comparison(eager_action, context_references[label]["action"], pad_np)
                    if context_contrasts[label]["max_abs_error"] <= 1e-6:
                        raise RuntimeError(f"Eager context reference {label} is not distinguishable from A")
            triangular_installation = None
            attention_installation = None
            time_installation = None
            vision_installation = None
            norm_installation = None
            rope_installation = None
            qknorm_rope_installation = None
            mrope_installation = None
            if args.gdn_algorithm == "triangular":
                from g05_gdn_triangular import install_triangular_chunk
                triangular_installation = install_triangular_chunk()
                triangular_installation.metadata["implementation_sha256"] = base.sha256_file(
                    Path(__file__).with_name("g05_gdn_triangular.py"))
            elif args.gdn_algorithm == "triton":
                from g05_gdn_triton import install_triton_chunk
                triangular_installation = install_triton_chunk(verify=args.verify)
                triangular_installation.metadata["implementation_files_sha256"] = {
                    name: base.sha256_file(Path(__file__).with_name(name))
                    for name in ("g05_gdn_triton.py", "g05_gdn_triangular.py")}
            try:
                if args.functional_mrope:
                    from g05_functional_mrope import install_functional_mrope
                    mrope_installation = install_functional_mrope(policy, verify=args.verify)
                    mrope_installation.metadata["implementation_sha256"] = base.sha256_file(
                        Path(__file__).with_name("g05_functional_mrope.py"))
                if args.ae_attention != "eager":
                    from g05_ae_fused_attention import install_ae_fused_attention
                    attention_installation = install_ae_fused_attention(policy, backend=args.ae_attention)
                    attention_installation.metadata["implementation_sha256"] = base.sha256_file(
                        Path(__file__).with_name("g05_ae_fused_attention.py"))
                if args.precompute_time_conditioning:
                    from g05_time_conditioning import install_time_conditioning
                    time_installation = install_time_conditioning(policy, lookup=args.time_conditioning_lookup)
                    time_installation.metadata["implementation_sha256"] = base.sha256_file(
                        Path(__file__).with_name("g05_time_conditioning.py"))
                if args.ae_norm != "eager":
                    from g05_ae_fused_norm import install_ae_fused_norm
                    norm_installation = install_ae_fused_norm(policy, verify=args.verify)
                    norm_installation.metadata["implementation_sha256"] = base.sha256_file(
                        Path(__file__).with_name("g05_ae_fused_norm.py"))
                if args.ae_rope != "eager":
                    from g05_ae_fused_rope import install_ae_fused_rope
                    rope_installation = install_ae_fused_rope(policy, verify=args.verify,
                        mode="matrix" if args.ae_rope == "matrix" else "sliced")
                    rope_installation.metadata["implementation_sha256"] = base.sha256_file(
                        Path(__file__).with_name("g05_ae_fused_rope.py"))
                if args.ae_qknorm_rope != "eager":
                    from g05_ae_qknorm_rope import install_ae_qknorm_rope
                    qknorm_rope_installation = install_ae_qknorm_rope(policy, verify=args.verify)
                    qknorm_rope_installation.metadata["implementation_sha256"] = base.sha256_file(
                        Path(__file__).with_name("g05_ae_qknorm_rope.py"))
                if args.static_vision:
                    from g05_static_vision import install_static_vision
                    vision_installation = install_static_vision(policy, verify=args.verify)
                    vision_installation.metadata["enabled"] = True
                original, compilation = install_backend(torch, policy, args.ae_backend, args.ae_options)
            except Exception:
                if vision_installation is not None:
                    vision_installation.restore()
                if qknorm_rope_installation is not None:
                    qknorm_rope_installation.restore()
                if rope_installation is not None:
                    rope_installation.restore()
                if norm_installation is not None:
                    norm_installation.restore()
                if time_installation is not None:
                    time_installation.restore()
                if attention_installation is not None:
                    attention_installation.restore()
                if mrope_installation is not None:
                    mrope_installation.restore()
                if triangular_installation is not None:
                    triangular_installation.restore()
                raise
            original_vlm = policy.model.vlm.forward
            gdn_installation = None
            context_installation = None
            try:
                if args.skip_unused_action_codec:
                    from g05_context_only import install_context_only_optimization
                    context_installation = install_context_only_optimization(policy)
                    context_installation.metadata["implementation_sha256"] = base.sha256_file(
                        Path(__file__).with_name("g05_context_only.py"))
                original_vlm, vlm_compilation = install_module_backend(
                    torch, policy.model.vlm, args.vlm_backend, "policy.model.vlm.forward only", options=args.vlm_options)
                gdn_installation = install_gdn_backends(
                    torch, args.gdn_backend, helpers=args.gdn_helpers, options=args.gdn_options)
                if args.verify:
                    # Either first call can specialize a graph. Repeat both A
                    # and B to exercise replay with changed inputs and KV values.
                    for label, selected, expected, chosen_batch in (
                            ("cold_A", noise, eager_action, batch),
                            ("first_B", noise_b, eager_b, verification_batch),
                            ("replay_A", noise, eager_action, batch),
                            ("replay_B", noise_b, eager_b, verification_batch)):
                        print(f"[progress] Selected-backend numerical check {label}", flush=True)
                        calls["tensor"] = selected
                        candidate_state, candidate, probe_timing = inference(chosen_batch)
                        compare_encoded_context(eager_contexts[label[-1]], encoded_context(candidate_state))
                        copied = candidate.detach().float().cpu().numpy().copy()
                        replay_comparisons[label] = action_comparison(expected, copied, pad_np)
                        verification_timings[label] = probe_timing
                    calls["tensor"] = noise
                    for replay in range(2):
                        for label, selected_batch in context_batches.items():
                            print(f"[progress] Selected-backend context check {label}/{replay+1}", flush=True)
                            candidate_state, candidate, _ = inference(selected_batch)
                            reference = context_references[label]
                            context_checks[f"{label}_{replay+1}"] = {
                                "encoded_context": compare_encoded_context(reference["context"], encoded_context(candidate_state)),
                                "action": action_comparison(reference["action"], candidate.detach().float().cpu().numpy().copy(), pad_np),
                            }
                calls["mode"] = args.timing_noise
                warmups = []
                for index in range(args.warmup):
                    state, action, timing = inference()
                    warmups.append(timing)
                    print(f"[warmup {index+1}/{args.warmup}] {timing['total_ms']:.3f} ms", flush=True)
                ids = state.input_ids.detach().cpu().numpy()
                semantic = state.attention_mask.detach().cpu().numpy()
                if ids.shape != (1, 159) or {k: int((semantic == k).sum()) for k in (1, 2, 4)} != {1: 128, 2: 1, 4: 30}:
                    raise RuntimeError("Input token layout differs from 3090 LIBERO S159/128 image/1 proprio/30 text")
                np.save(output / "input_ids.npy", ids)
                runtime.reset_peak_memory()
                records = []
                for index in range(args.iterations):
                    _, action, timing = inference()
                    records.append(timing)
                    print(f"[measure {index+1}/{args.iterations}] {timing['total_ms']:.3f} ms", flush=True)
                memory = runtime.memory()
                last_timed = action.detach().float().cpu().numpy().copy()
                timed_output_validation = action_comparison(last_timed, last_timed, pad_np)
                # Save/compare a shared-noise result separately from native RNG
                # latency measurement. This extra request is never timed above.
                calls["mode"] = "shared"
                calls["tensor"] = noise
                module_profile = None
                if args.profile_modules:
                    from profile_g05_modules import ModuleProfiler
                    with ModuleProfiler(policy, runtime.synchronize) as profiler:
                        with profiler.phase("post_timing_diagnostic"):
                            _, deterministic_action, _ = inference()
                    module_profile = profiler.summary()
                else:
                    _, deterministic_action, _ = inference()
                actual = deterministic_action.detach().float().cpu().numpy().copy()
                trace_profile = None
                if args.trace_dir is not None:
                    if runtime.kind != "npu":
                        raise ValueError("NPU trace profiling requires an NPU device")
                    from torch_npu import profiler as npu_profiler
                    print("[progress] Capturing one post-timing NPU request trace", flush=True)
                    trace_dir = args.trace_dir.resolve()
                    trace_dir.mkdir(parents=True, exist_ok=True)
                    with npu_profiler.profile(
                        activities=[npu_profiler.ProfilerActivity.CPU, npu_profiler.ProfilerActivity.NPU],
                        schedule=npu_profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
                        record_shapes=True, with_stack=False, profile_memory=False,
                        experimental_config=npu_profiler._ExperimentalConfig(profiler_level=npu_profiler.ProfilerLevel.Level1),
                        on_trace_ready=npu_profiler.tensorboard_trace_handler(str(trace_dir)),
                    ) as trace:
                        inference()
                        trace.step()
                    trace_profile = {"directory": str(trace_dir), "requests": 1,
                        "profiler_level": "Level1", "schedule": {"wait": 0, "warmup": 0, "active": 1, "repeat": 1},
                        "scope": "post-timing diagnostic only; profiler overhead excluded from formal samples"}
            finally:
                if context_installation is not None:
                    context_installation.restore()
                if gdn_installation is not None:
                    gdn_installation.restore()
                policy.model.action_expert.forward = original
                policy.model.vlm.forward = original_vlm
                if vision_installation is not None:
                    vision_installation.restore()
                if qknorm_rope_installation is not None:
                    qknorm_rope_installation.restore()
                if rope_installation is not None:
                    rope_installation.restore()
                if norm_installation is not None:
                    norm_installation.restore()
                if time_installation is not None:
                    time_installation.restore()
                if attention_installation is not None:
                    attention_installation.restore()
                if mrope_installation is not None:
                    mrope_installation.restore()
                if triangular_installation is not None:
                    triangular_installation.restore()
            if args.count_workload:
                print("[progress] Counting partial LIBERO workload in one restored-eager diagnostic request (outside timing)", flush=True)
                calls["mode"] = "shared"
                calls["tensor"] = noise
                workload = collect_counted_workload(torch, policy, inference, pad_np)
        expected_calls = args.warmup + args.iterations + 1 + (6 if args.verify else 0) + int(args.count_workload) + 3 * len(context_batches) + int(args.trace_dir is not None)
        if calls["count"] != expected_calls:
            raise RuntimeError("FM did not consume exactly one explicit-noise tensor per inference")
        validation = action_comparison(actual, actual, pad_np)
        np.save(output / "action.npy", actual)
        identity = {"checkpoint_sha256": assets["g05-libero/model.pt"]["sha256"],
                    "sidecars": {k: v["sha256"] for k, v in assets.items() if k != "g05-libero/model.pt"},
                    "input_files_sha256": input_meta["files_sha256"],
                    "noise_tensor_sha256": noise_meta["tensor_sha256"],
                    "input_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
                    "contract": contract, "weights": args.weights, "autocast": "bfloat16"}
        comparison = action_comparison(eager_action, actual, pad_np) if eager_action is not None else None
        external = None
        if args.reference_report:
            reference = json.loads(args.reference_report.read_text(encoding="utf-8"))
            check_reference_identity(reference["comparison_identity"], identity)
            external = action_comparison(np.load(args.reference_report.parent / "action.npy", allow_pickle=False), actual, pad_np)
        dtypes = {}
        for param in policy.parameters():
            entry = dtypes.setdefault(str(param.dtype), {"numel": 0, "bytes": 0})
            entry["numel"] += param.numel()
            entry["bytes"] += param.numel()*param.element_size()
        checks = list(replay_comparisons.values()) + ([comparison] if comparison is not None else [])
        checks.extend(row["action"] for row in context_checks.values())
        passed = all(row["relative_l2"] <= args.max_relative_l2 and row["max_abs_error"] <= args.max_abs_error
                     and row["padded_dimensions_max_abs_candidate"] == 0.0 for row in checks)
        passed = passed and validation["padded_dimensions_max_abs_candidate"] == 0.0
        passed = passed and timed_output_validation["padded_dimensions_max_abs_candidate"] == 0.0
        distinguishable = reference_contrast is None or (reference_contrast["relative_l2"] > args.max_relative_l2 or reference_contrast["max_abs_error"] > args.max_abs_error)
        passed = passed and distinguishable
        if external is not None:
            passed = passed and external["relative_l2"] <= args.max_relative_l2 and external["max_abs_error"] <= args.max_abs_error
        if workload is not None:
            workload["identity"] = identity
            workload["source_revision"] = revision
            workload["artifact"] = "workload.json"
            (output / "workload.json").write_text(json.dumps(workload, indent=2) + "\n", encoding="utf-8")
        return {"status": "success" if passed else "numerical_validation_failed",
            "scope": "prepared device inputs -> Vision/VLM prefill -> FM10 -> normalized [1,32,20] action; no CoT/AR",
            "excluded": ["image loading", "outer image preprocessing", "outer batch H2D", "output D2H", "denormalization", "network", "warmup/compilation", "shared-noise correctness requests"],
            "included_prefill_work": "Official encode_inference/build_proprio_batch CPU work and internal device transfers, matching the original 3090 timer",
            "noise_timing": {"mode": args.timing_noise,
                "included": "original device torch.randn" if args.timing_noise == "native" else "one device clone of fixed explicit noise",
                "correctness": "All verification and saved action.npy use shared noise regardless of timing mode"},
            "source_revision": revision, "asset_revision": ASSET_REVISION, "assets": assets,
            "benchmark_files_sha256": {name: base.sha256_file(Path(__file__).with_name(name))
                for name in ("benchmark_g05_libero.py", "benchmark_g05_full.py", "g05_npu_graphs.py")},
            "source": {"path": str(source), "revision": revision, "tree_state": base.source_tree_state(source),
                "key_files_sha256": {name: base.sha256_file(source / name) for name in (
                    "src/g05/models/g05/qwen35/gated_deltanet.py",
                    "src/g05/models/g05/g05_policy.py", "src/g05/models/g05/g05_policy_qwen35.py",
                    "src/g05/models/g05/helpers/fm_helper.py", "scripts/serve_policy.py")}},
            "config_patches": patches, "contract": contract, "input": input_meta, "noise": noise_meta,
            "comparison_identity": identity, "compilation": compilation,
            "vlm_compilation": vlm_compilation, "gdn_compilation": gdn_installation.metadata,
            "gdn_algorithm": triangular_installation.metadata if triangular_installation is not None else {"name": "original"},
            "context_optimization": context_installation.metadata if context_installation is not None else {"enabled": False},
            "ae_attention": attention_installation.metadata if attention_installation is not None else {"enabled": False, "backend": "eager"},
            "ae_norm": norm_installation.metadata if norm_installation is not None else {"enabled": False},
            "ae_rope": rope_installation.metadata if rope_installation is not None else {"enabled": False},
            "ae_qknorm_rope": qknorm_rope_installation.metadata if qknorm_rope_installation is not None else {"enabled": False},
            "functional_mrope": mrope_installation.metadata if mrope_installation is not None else {"enabled": False},
            "time_conditioning": time_installation.metadata if time_installation is not None else {"enabled": False},
            "static_vision": vision_installation.metadata if vision_installation is not None else {"enabled": False},
            "context_verification": {"cases": context_checks, "reference_contrasts_against_A": context_contrasts,
                "scope": "C changes proprio pose by +0.01; D changes task open -> close; optional E/F independently flip one camera; each is compared to unmodified eager twice with identical noise; no closed-loop accuracy claim"},
            "environment": {"python": platform.python_version(), "torch": torch.__version__, **runtime.info()},
            "load_ms": load_ms, "preparation_ms": preparation_ms, "warmup": warmups,
            "summary_ms": base.summarize_samples(records), "samples": records, "memory": memory,
            "module_profile": module_profile,
            "trace_profile": trace_profile,
            "workload": workload,
            "parameter_dtypes": dtypes, "output_validation": validation,
            "last_timed_output_validation": timed_output_validation,
            "eager_comparison": comparison, "external_comparison": external,
            "replay_verification": {"cases": replay_comparisons, "timings_ms": verification_timings,
                "reference_contrast": reference_contrast, "reference_inputs_distinguishable": distinguishable,
                "scope": "A/B/A/B; A=real sample, B=horizontal image flip plus negated noise, verification only; formal timings use A",
                "coverage": "changed image content, VLM KV and FM action buffers; unchanged token shape; no closed-loop task-accuracy claim"},
            "numerical_thresholds": {"max_relative_l2": args.max_relative_l2, "max_abs_error": args.max_abs_error,
                "meaning": "engineering regression gate, not closed-loop task success"}}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", default=CHECKPOINT_SHA256, help="Pinned official HF LFS content SHA-256")
    parser.add_argument("--input-dir", type=Path, required=True, help="Original case.json, image.png, wrist_image.png")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--weights", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--ae-backend", choices=("eager", "torchair", "torchair_aclgraph", "npugraph_ex", "inductor"), default="eager")
    parser.add_argument("--vlm-backend", choices=("eager", "torchair", "torchair_aclgraph", "npugraph_ex"), default="eager")
    parser.add_argument("--gdn-backend", choices=("eager", "npugraph_ex"), default="eager")
    parser.add_argument("--gdn-algorithm", choices=("original", "triangular", "triton"), default="original",
                        help="Opt-in FP32 triangular solve experiment; requires --verify on NPU")
    parser.add_argument("--ae-attention", choices=("eager", "npu", "sdpa"), default="eager",
                        help="Opt-in AE-only fused attention; requires NPU and full numerical verification")
    parser.add_argument("--ae-norm", choices=("eager", "npu"), default="eager",
                        help="Opt-in FP32 fused AE normalization; requires NPU and full numerical verification")
    parser.add_argument("--ae-rope", choices=("eager", "npu", "matrix"), default="eager",
                        help="Opt-in FP32 fused partial AE RoPE; requires NPU and full numerical verification")
    parser.add_argument("--ae-qknorm-rope", choices=("eager", "ascendc"), default="eager",
                        help="Opt-in Ascend C Q/K norm + partial RoPE + BNSD layout fusion; requires its built OPP/bridge")
    parser.add_argument("--functional-mrope", action="store_true",
                        help="Opt-in FP32 device selection for AE/VLM MRoPE; requires NPU and full numerical verification")
    parser.add_argument("--precompute-time-conditioning", action="store_true",
                        help="Opt-in fixed FM time/AdaLN parameter table; dynamic actions and KV are never cached")
    parser.add_argument("--time-conditioning-lookup", choices=("exact", "request"), default="exact",
                        help="Exact time-value lookup or audited FM request-scoped sequence; requires the precompute flag")
    parser.add_argument("--static-vision", action="store_true",
                        help="Opt-in fixed-geometry two-camera vision ACLGraph; requires NPU and independent camera verification")
    parser.add_argument("--gdn-helpers", type=lambda value: tuple(value.split(",")), default=("chunk", "recurrent"))
    for scope in ("ae", "vlm", "gdn"):
        parser.add_argument(f"--{scope}-options", type=lambda value: validate_graph_options(json.loads(value)), default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=Path, help="Shared float32 [1,32,20] .npy from the reference run")
    parser.add_argument("--timing-noise", choices=("native", "shared"), default="shared",
                        help="Native preserves historical torch.randn latency; shared includes a fixed-noise clone")
    parser.add_argument("--reference-report", type=Path, help="This runner's report.json, with adjacent action.npy")
    parser.add_argument("--verify", action="store_true", help="Compare selected backend with same-device eager")
    parser.add_argument("--profile-modules", action="store_true",
                        help="Synchronize six module boundaries in one separate post-timing diagnostic request")
    parser.add_argument("--skip-unused-action-codec", action="store_true",
                        help="Opt-in removal of unconsumed context-only ActionCodec work; requires --verify and extra changed state/task checks")
    parser.add_argument("--trace-dir", type=Path,
                        help="Capture one NPU CPU/device trace after formal timing; not included in latency statistics")
    parser.add_argument("--count-workload", action="store_true",
                        help="Count partial Linear/Conv/AdaLN FLOPs in one restored-eager request after timing; write workload.json")
    parser.add_argument("--max-relative-l2", type=float, default=0.01)
    parser.add_argument("--max-abs-error", type=float, default=0.05)
    parser.add_argument("--allow-source-revision", action="store_true")
    args = parser.parse_args(argv)
    if args.skip_unused_action_codec and not args.verify:
        parser.error("Context-only ActionCodec optimization requires --verify")
    if (args.ae_attention != "eager" or args.ae_norm != "eager" or args.ae_rope != "eager" or args.ae_qknorm_rope != "eager" or args.functional_mrope or args.precompute_time_conditioning or args.static_vision) and (not args.verify or not args.device.startswith("npu")):
        parser.error("AE attention/time conditioning experiments require --verify and an NPU device")
    if args.ae_qknorm_rope != "eager" and (args.ae_rope != "eager" or args.weights != "bf16"):
        parser.error("Ascend C QK fusion requires BF16 resident weights and replaces --ae-rope")
    if args.time_conditioning_lookup != "exact" and not args.precompute_time_conditioning:
        parser.error("--time-conditioning-lookup request requires --precompute-time-conditioning")
    if args.gdn_algorithm != "original" and (not args.verify or not args.device.startswith("npu")):
        parser.error("GDN algorithm experiments require --verify and an NPU device")
    if args.gdn_algorithm != "original" and args.vlm_backend == "torchair":
        parser.error("GDN inverse experiments require eager or ACLGraph VLM; this GE image lacks their converters")
    if args.warmup < 1 or args.iterations < 1:
        parser.error("warmup and iterations must be positive")
    if len(args.checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.checkpoint_sha256):
        parser.error("checkpoint-sha256 must be 64 lowercase hex characters")
    if args.reference_report and not args.noise:
        parser.error("reference-report requires the exact shared --noise file")
    for scope in ("ae", "vlm", "gdn"):
        options = getattr(args, f"{scope}_options")
        backend = getattr(args, f"{scope}_backend")
        if options and backend not in {"npugraph_ex", "torchair", "torchair_aclgraph"}:
            parser.error(f"--{scope}-options requires a graph backend")
        if options and backend == "torchair" and set(options) - {"frozen_parameter", "softplus_decomposition", "eye_decomposition", "single_stream"}:
            parser.error("Unsupported TorchAir option")
        if options and backend == "torchair_aclgraph" and set(options) - {"frozen_parameter"}:
            parser.error("TorchAir ACL graph only supports frozen_parameter")
        if options and options.get("softplus_decomposition") and not args.verify:
            parser.error("Softplus decomposition requires --verify")
    if args.vlm_backend != "eager" and args.gdn_backend != "eager":
        parser.error("Measure VLM and GDN helper compilation in separate experiments")
    backends = {args.ae_backend, args.vlm_backend, args.gdn_backend}
    if args.static_vision:
        backends.add("torchair_aclgraph")
    if "npugraph_ex" in backends and backends & {"torchair", "torchair_aclgraph"}:
        parser.error("This image cannot mix independent npugraph_ex and TorchAir packages; use torchair_aclgraph with TorchAir GE")
    if set(args.gdn_helpers) - {"chunk", "recurrent", "causal-conv"}:
        parser.error("gdn-helpers must use chunk,recurrent,causal-conv")
    if not all(math.isfinite(value) and value >= 0 for value in (args.max_abs_error, args.max_relative_l2)):
        parser.error("Numerical thresholds must be finite and nonnegative")
    return args


def main():
    args = parse_args()
    report = run(args)
    path = args.output_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(path), "status": report["status"], "summary_ms": report["summary_ms"]}), flush=True)
    if report["status"] != "success":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

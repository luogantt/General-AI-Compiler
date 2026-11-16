#!/usr/bin/env python3
"""Opt-in tensor-only vision graph for the verified two-camera LIBERO geometry.

Adapted from GalaxeaVLA Qwen3.5 vision.py (Qwen/Hugging Face, Apache-2.0;
see upstream THIRD_PARTY_NOTICES.md). Preserve patch/merger FP32 islands, the
four-term learned-position interpolation, RoPE dtype and two independent SDPA
calls in every block. Only geometry indices/weights/RoPE are cached.

No Torch/NPU is loaded on import. Install before graph/profile wrappers; restore
in reverse order. Parameter/device/ordinary in-place mutation invalidates this
installation. Unversioned .data writes bypass PyTorch version tracking and are
unsupported: explicitly restore/reinstall after any weight update. Images and
learned-position outputs are never cached. Independent model verification is
mandatory; the installer itself does not establish numerical correctness.
"""

from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
from types import MethodType


_SOURCE = "g05.models.g05.qwen35.vision"
_GRID = [[1, 16, 16], [1, 16, 16]]
_MARKER = "_g05_static_vision_installation"
_MISSING = object()


def _tensor_stamp(value):
    try:
        version = value._version
    except RuntimeError as exc:
        raise ValueError("Static vision requires versioned model tensors; reinstall outside inference-mode construction") from exc
    return (id(value), tuple(value.shape), str(value.dtype), str(value.device), version)


def _model_stamp(vision):
    return (tuple((name, _tensor_stamp(value)) for name, value in vision.named_parameters()),
            _tensor_stamp(vision.rotary_pos_emb.inv_freq),
            tuple(id(block) for block in vision.blocks), vision.num_grid_per_side,
            vision.spatial_merge_size, vision.patch_size)


def _validate_model(torch, policy, vision, source):
    if type(vision) is not source.Qwen3_5VisionModel:
        raise TypeError("Static vision requires the audited Qwen3_5VisionModel class")
    config = vision.config
    required = {"depth": 24, "hidden_size": 1024, "out_hidden_size": 2048,
                "num_heads": 16, "patch_size": 16, "temporal_patch_size": 2,
                "spatial_merge_size": 2, "in_channels": 3, "num_position_embeddings": 2304}
    for key, expected in required.items():
        if getattr(config, key, None) != expected:
            raise ValueError(f"Static vision requires {key}={expected}")
    if len(vision.blocks) != 24 or vision.num_grid_per_side != 48:
        raise ValueError("Unexpected vision block count/position grid")
    if policy.training or policy.model.training or any(module.training for module in vision.modules()):
        raise ValueError("Static vision requires eval mode throughout the vision tower")
    if vision.gradient_checkpointing:
        raise ValueError("Static vision does not support gradient checkpointing")
    if vision.pos_embed.weight.device.type != "npu":
        raise ValueError("Static vision supports NPU only")
    if vision.pos_embed.weight.dtype != torch.float32 or vision.patch_embed.proj.weight.dtype != torch.float32:
        raise ValueError("Static vision requires the original FP32 position/patch projection islands")
    for block in vision.blocks:
        if type(block) is not source.Qwen3_5VisionBlock or type(block.attn) is not source.Qwen3_5VisionAttention:
            raise TypeError("Unexpected vision block/attention subclass")
        if block.attn.num_heads != 16 or block.attn.head_dim != 64:
            raise ValueError("Unexpected vision attention dimensions")
        if "_attend_spatial" in vars(block.attn):
            raise RuntimeError("Vision spatial attention already has an instance replacement")


def _prepare_geometry(torch, vision):
    """Match upstream CPU linspace/list conversion and original RoPE exactly."""
    device = vision.pos_embed.weight.device
    idx_list, weight_list = [[] for _ in range(4)], [[] for _ in range(4)]
    for _, height, width in _GRID:
        h_idxs = torch.linspace(0, vision.num_grid_per_side - 1, height)
        w_idxs = torch.linspace(0, vision.num_grid_per_side - 1, width)
        h_floor, w_floor = h_idxs.int(), w_idxs.int()
        h_ceil = (h_idxs.int() + 1).clip(max=vision.num_grid_per_side - 1)
        w_ceil = (w_idxs.int() + 1).clip(max=vision.num_grid_per_side - 1)
        dh, dw = h_idxs - h_floor, w_idxs - w_floor
        base_h, base_h_ceil = h_floor * vision.num_grid_per_side, h_ceil * vision.num_grid_per_side
        indices = [(base_h[None].T + w_floor[None]).flatten(),
                   (base_h[None].T + w_ceil[None]).flatten(),
                   (base_h_ceil[None].T + w_floor[None]).flatten(),
                   (base_h_ceil[None].T + w_ceil[None]).flatten()]
        weights = [((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                   ((1 - dh)[None].T * dw[None]).flatten(),
                   (dh[None].T * (1 - dw)[None]).flatten(),
                   (dh[None].T * dw[None]).flatten()]
        for i in range(4):
            idx_list[i].extend(indices[i].tolist())
            weight_list[i].extend(weights[i].tolist())
    indices = torch.tensor(idx_list, dtype=torch.long, device=device)
    weights = torch.tensor(weight_list, dtype=vision.pos_embed.weight.dtype, device=device)
    grid = torch.tensor(_GRID, dtype=torch.long, device=device)
    rotary = vision.rot_pos_emb(grid).reshape(512, -1)
    embeddings = torch.cat((rotary, rotary), dim=-1)
    return {"indices": indices, "weights": weights,
            "cos": embeddings.cos(), "sin": embeddings.sin(),
            "cu_seqlens": torch.tensor([0, 256, 512], dtype=torch.int32, device=device)}


def _make_spatial_attention(torch):
    def attend(self, query_states, key_states, value_states, cu_seqlens):
        # Geometry is checked by the outer forward. Never combine the cameras
        # into one 512-token attention operation or change the SDPA arguments.
        query = query_states.transpose(0, 1).unsqueeze(0)
        key = key_states.transpose(0, 1).unsqueeze(0)
        value = value_states.transpose(0, 1).unsqueeze(0)
        output = torch.cat([
            torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=self.scaling)
            for q, k, v in zip(torch.split(query, (256, 256), dim=2),
                               torch.split(key, (256, 256), dim=2),
                               torch.split(value, (256, 256), dim=2))
        ], dim=2)
        return output.squeeze(0).transpose(0, 1).reshape(512, -1).contiguous()
    return attend


def _interpolate_position(torch, vision, geometry):
    # Lookup stays inside each tensor-core request; do not cache this result.
    positions = vision.pos_embed(geometry["indices"]).to(vision.pos_embed.weight.device)
    positions = positions * geometry["weights"][:, :, None]
    positions = positions[0] + positions[1] + positions[2] + positions[3]
    parts = []
    for part in positions.split((256, 256)):
        # Exactly the upstream repeat(t=1), merge-order permutation and flatten.
        part = part.repeat(1, 1).view(1, 8, 2, 8, 2, -1).permute(0, 1, 3, 2, 4, 5).flatten(0, 4)
        parts.append(part)
    return torch.cat(parts)


def _make_tensor_core(torch, vision, geometry):
    def core(patches):
        with torch.autocast(patches.device.type, enabled=False):
            hidden_states = vision.patch_embed(patches)
            hidden_states = hidden_states + _interpolate_position(torch, vision, geometry)
        hidden_states = hidden_states.reshape(512, -1)
        for block in vision.blocks:
            hidden_states = block(hidden_states, cu_seqlens=geometry["cu_seqlens"],
                                  position_embeddings=(geometry["cos"], geometry["sin"]))
        with torch.autocast(patches.device.type, enabled=False):
            merged = vision.merger(hidden_states)
        return hidden_states, merged
    return core


def _validate_request(torch, policy, vision, patches, grid, num_frames, bsz, stamp, default_dtype):
    if num_frames != 1 or bsz != 1:
        raise ValueError("Static vision requires num_frames=1 and bsz=1")
    if policy.training or policy.model.training or any(module.training for module in vision.modules()):
        raise ValueError("Static vision requires eval mode")
    if vision.gradient_checkpointing or torch.is_grad_enabled():
        raise ValueError("Static vision requires no-grad/inference mode without checkpointing")
    if not torch.is_autocast_enabled("npu") or torch.get_autocast_dtype("npu") != torch.bfloat16:
        raise ValueError("Static vision requires the original NPU BF16 autocast context")
    if patches.device.type != "npu" or patches.device != vision.pos_embed.weight.device:
        raise ValueError("Static vision input device changed")
    if tuple(patches.shape) != (512, 1536) or patches.dtype != torch.float32 or not patches.is_contiguous():
        raise ValueError("Static vision requires contiguous FP32 patches [512,1536]")
    if grid.device != patches.device or grid.dtype != torch.long or tuple(grid.shape) != (2, 3):
        raise ValueError("Static vision requires NPU int64 grid [2,3]")
    # Intentional graph-external read of six geometry integers on every request.
    # This is included in request timing and guards current content, not shape alone.
    if grid.tolist() != _GRID:
        raise ValueError("Static vision requires exact two-camera grid [[1,16,16],[1,16,16]]")
    if torch.get_default_dtype() != default_dtype or _model_stamp(vision) != stamp:
        raise ValueError("Static vision model/geometry state changed; restore and reinstall")


@dataclass
class StaticVisionInstallation:
    metadata: dict
    patches: list
    original: object
    tensor_core: object
    restored: bool = False

    def restore(self):
        if self.restored:
            return
        for owner, name, previous, installed in reversed(self.patches):
            if vars(owner).get(name, _MISSING) is installed:
                if previous is _MISSING:
                    delattr(owner, name)
                else:
                    setattr(owner, name, previous)
        self.restored = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.restore()


def install_static_vision(policy, backend="torchair_aclgraph", *, verify=False):
    if verify is not True:
        raise ValueError("Static vision requires verify=True and independent whole-model verification")
    if backend not in {"eager", "torchair_aclgraph"}:
        raise ValueError("Static vision supports eager or torchair_aclgraph only")
    torch = importlib.import_module("torch")
    source = importlib.import_module(_SOURCE)
    graph_helpers = importlib.import_module("g05_npu_graphs")
    vision = policy.model.vision_tower
    if getattr(vision.forward, _MARKER, False) or "forward" in vars(vision):
        raise RuntimeError("Install static vision once, before graph/profile forward replacements")
    _validate_model(torch, policy, vision, source)
    stamp, default_dtype = _model_stamp(vision), torch.get_default_dtype()
    # Installation normally happens outside the runner's inference context.
    # Compute RoPE in the same NPU BF16 autocast context as original forward;
    # preserve the actual buffer dtype instead of forcing a new FP32 formula.
    with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
        geometry = _prepare_geometry(torch, vision)
    original = vision.forward
    patches = []
    metadata = {"name": "fixed-two-camera-vision", "backend": backend, "default_enabled": False,
                "grid_thw": _GRID, "num_frames": 1, "bsz": 1, "patch_shape": [512, 1536],
                "cu_seqlens": [0, 256, 512], "camera_attention_lengths": [256, 256],
                "output_shapes": [[512, 1024], [128, 2048]], "sdpa_calls_per_block": 2,
                "blocks": 24, "cached": ["interpolation_indices", "interpolation_weights", "RoPE_cos_sin", "cu_seqlens"],
                "learned_position_outputs_cached": False, "input_or_model_outputs_cached": False,
                "learned_position_lookup": "inside tensor core; immutable/version-guarded frozen model",
                "geometry_validation": "exact grid values and frame/batch on every request, outside graph but inside request timer",
                "precision_policy_changed": False, "fallback_allowed": False,
                "autocast": "NPU BF16; same context for cached geometry and every request",
                "verification_requested": True, "runtime_validation": "not performed by installer",
                "requires_independent_npu_and_full_model_validation": True, "calls": 0,
                "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "source_sha256": hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest()}

    def replace(owner, name, installed):
        previous = vars(owner).get(name, _MISSING)
        setattr(owner, name, installed)
        patches.append((owner, name, previous, installed))

    try:
        attend = _make_spatial_attention(torch)
        for block in vision.blocks:
            replace(block.attn, "_attend_spatial", MethodType(attend, block.attn))
        tensor_core = _make_tensor_core(torch, vision, geometry)
        compiled, graph_metadata = graph_helpers._compile_callable(
            torch, tensor_core, backend, "vision patches -> 24 blocks -> merger; fixed geometry",
            {"frozen_parameter": True} if backend != "eager" else {},
        )
        metadata["compilation"] = graph_metadata

        def forward(self, hidden_states, grid_thw, num_frames=1, bsz=1):
            _validate_request(torch, policy, self, hidden_states, grid_thw, num_frames, bsz, stamp, default_dtype)
            result = compiled(hidden_states)
            if tuple(result[0].shape) != (512, 1024) or tuple(result[1].shape) != (128, 2048):
                raise RuntimeError("Static vision output contract changed")
            metadata["calls"] += 1
            return result

        setattr(forward, _MARKER, True)
        replace(vision, "forward", MethodType(forward, vision))
        return StaticVisionInstallation(metadata, patches, original, tensor_core)
    except Exception:
        StaticVisionInstallation(metadata, patches, original, None).restore()
        raise

"""Checkpoint complete Base encoder layers without changing their computation.

Every Tensor argument, including nested masks/kwargs and image features, is an
explicit checkpoint input. The closure holds only the call structure, constants
and the bound layer method, never a hidden differentiable key/value tensor.
"""

import torch
from torch.utils.checkpoint import checkpoint

from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER
from projects.mmdet3d_plugin.bevformer.modules.encoder import BEVFormerLayer


def _pack(value, tensors):
    if isinstance(value, torch.Tensor):
        tensors.append(value)
        return ("tensor", len(tensors) - 1)
    if isinstance(value, dict):
        if any(isinstance(key, torch.Tensor) for key in value):
            raise TypeError("Tensor dictionary keys are not supported by layer checkpointing.")
        return ("dict", tuple((key, _pack(item, tensors)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return ("list" if isinstance(value, list) else "tuple",
                tuple(_pack(item, tensors) for item in value))
    return ("constant", value)


def _unpack(specification, tensors):
    kind, value = specification
    if kind == "tensor":
        return tensors[value]
    if kind == "dict":
        return {key: _unpack(item, tensors) for key, item in value}
    if kind == "list":
        return [_unpack(item, tensors) for item in value]
    if kind == "tuple":
        return tuple(_unpack(item, tensors) for item in value)
    return value


def checkpoint_call(function, args, kwargs):
    tensors = []
    specification = _pack((args, kwargs), tensors)
    # PyTorch 1.9's reentrant checkpoint needs a differentiable input. Calling
    # eagerly in the other case still correctly computes parameter gradients.
    if not any(tensor.requires_grad for tensor in tensors):
        return function(*args, **kwargs)

    def recompute(*inputs):
        positional, named = _unpack(specification, inputs)
        return function(*positional, **named)

    # PyTorch 1.9.1 preserves the forward CUDA autocast mode internally.
    # Keep RNG preservation explicit so dropout is identical on recomputation.
    return checkpoint(recompute, *tensors, preserve_rng_state=True)


@TRANSFORMER_LAYER.register_module()
class CheckpointedBEVFormerLayer(BEVFormerLayer):
    activation_checkpoint = True

    def forward(self, query, key=None, value=None, bev_pos=None, query_pos=None,
                key_pos=None, attn_masks=None, query_key_padding_mask=None,
                key_padding_mask=None, ref_2d=None, ref_3d=None, bev_h=None,
                bev_w=None, reference_points_cam=None, mask=None,
                spatial_shapes=None, level_start_index=None, prev_bev=None,
                **kwargs):
        named = dict(kwargs)
        named.update(
            bev_pos=bev_pos, query_pos=query_pos, key_pos=key_pos,
            attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask,
            key_padding_mask=key_padding_mask, ref_2d=ref_2d, ref_3d=ref_3d,
            bev_h=bev_h, bev_w=bev_w, reference_points_cam=reference_points_cam,
            mask=mask, spatial_shapes=spatial_shapes,
            level_start_index=level_start_index, prev_bev=prev_bev)
        # This bound method belongs to this particular layer invocation. It does
        # not reference an encoder-loop variable that could change before backward.
        original_forward = super().forward
        args = (query, key, value)
        if not self.training or not torch.is_grad_enabled():
            return original_forward(*args, **named)
        return checkpoint_call(original_forward, args, named)

#!/usr/bin/env python3
"""Independent PyTorch references for Div/view/Add/view -> MSDA fusion.

Python creates test data and calls real PyTorch grid_sample for golden outputs.
The compiler under test must discover and lower the original ONNX graph itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh
import torch

from generate_base_ops_models import msda_reference


def generate(directory):
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    rng = np.random.default_rng(50841)
    cases = []
    batch, queries, heads, levels, points, channels = 2, 3, 2, 2, 4, 3
    location_shape = [batch, queries, heads, levels, points, 2]
    spatial_shapes = np.array([[2, 3], [1, 2]], np.int64)

    def make(name, *, spatial=False, first=False, chain=False, full_norm=False,
             special_norm=False, public=None, branch=None, transform=None, reverse_div=False):
        coords_shape = [batch, queries, heads, levels, points // 2, 2, 2] if spatial else location_shape
        reference_shape = [batch, queries, 1, 1, 1, 2, 2] if spatial else [batch, queries, 1, 1, 1, 2]
        inputs = {
            'value': rng.uniform(-1, 1, (batch, 8, heads, channels)).astype(np.float32),
            'offsets': rng.uniform(-1.5, 1.5, location_shape).astype(np.float32),
            'normalizer': np.array([3, 2, 2, 1], np.float32).reshape(1, 1, 1, levels, 1, 2),
            'reference': rng.uniform(0.1, 0.9, reference_shape).astype(np.float32),
            'weights': rng.uniform(0.05, 1, (batch, queries, heads, levels, points)).astype(np.float32),
        }
        inputs['weights'] /= inputs['weights'].sum(axis=(-1, -2), keepdims=True)
        inputs['weights'][0, -1] = 0
        if full_norm:
            inputs['normalizer'] = rng.uniform(0.25, 4, location_shape).astype(np.float32)
            inputs['normalizer'][..., 1] *= -1
        if special_norm:
            # Division by +/-Inf produces finite signed zero. Small negative
            # divisors exercise finite out-of-image samples without dividing by
            # zero or assuming a contract for NaN sampling coordinates.
            inputs['normalizer'][0, 0, 0, :, 0, :] = [[np.inf, -np.inf], [0.125, -0.25]]
        if reverse_div:
            inputs['offsets'] = rng.uniform(2, 8, location_shape).astype(np.float32)
            inputs['normalizer'] = np.array([0.3, 0.6, 0.4, 0.2], np.float32).reshape(1, 1, 1, levels, 1, 2)
        constants = {'spatial_shapes': spatial_shapes}
        nodes = [h.make_node('Div', ['normalizer', 'offsets'] if reverse_div else ['offsets', 'normalizer'],
                             ['divided'], name='coordinates_div')]
        tensors = {key: torch.from_numpy(value) for key, value in inputs.items()}
        with torch.inference_mode():
            divided = tensors['normalizer'] / tensors['offsets'] if reverse_div else tensors['offsets'] / tensors['normalizer']
            current, value = 'divided', divided
            if transform == 'transpose_before':
                nodes.append(h.make_node('Transpose', [current], ['permuted_before'], perm=[0, 1, 3, 2, 4, 5]))
                current, value = 'permuted_before', value.permute(0, 1, 3, 2, 4, 5).contiguous()
            if transform == 'cast_before':
                nodes.extend([h.make_node('Cast', [current], ['half_before'], to=T.FLOAT16),
                              h.make_node('Cast', ['half_before'], ['float_before'], to=T.FLOAT)])
                current, value = 'float_before', value.to(torch.float16).to(torch.float32)
            if chain:
                constants['axis0'] = np.array([0], np.int64)
                nodes.extend([h.make_node('Unsqueeze', [current, 'axis0'], ['expanded']),
                              h.make_node('Squeeze', ['expanded', 'axis0'], ['squeezed'])])
                current, value = 'squeezed', value.unsqueeze(0).squeeze(0)
            if spatial:
                constants['coordinate_shape'] = np.asarray(coords_shape, np.int64)
                nodes.append(h.make_node('Reshape', [current, 'coordinate_shape'], ['coordinate_view']))
                current, value = 'coordinate_view', value.reshape(coords_shape)
            nodes.append(h.make_node('Add', ['reference', current] if first else [current, 'reference'],
                                     ['coordinates'], name='sampling_add'))
            coordinates = tensors['reference'] + value if first else value + tensors['reference']
            current, value = 'coordinates', coordinates
            if spatial:
                constants['location_shape'] = np.asarray(location_shape, np.int64)
                nodes.append(h.make_node('Reshape', [current, 'location_shape'], ['locations_view']))
                current, value = 'locations_view', value.reshape(location_shape)
            if chain:
                nodes.append(h.make_node('Identity', [current], ['identity_locations']))
                current = 'identity_locations'
            if transform == 'transpose_after':
                nodes.append(h.make_node('Transpose', [current], ['permuted_after'], perm=[0, 1, 3, 2, 4, 5]))
                current, value = 'permuted_after', value.permute(0, 1, 3, 2, 4, 5).contiguous()
            if transform == 'cast_after':
                nodes.extend([h.make_node('Cast', [current], ['half_after'], to=T.FLOAT16),
                              h.make_node('Cast', ['half_after'], ['float_after'], to=T.FLOAT)])
                current, value = 'float_after', value.to(torch.float16).to(torch.float32)
            nodes.append(h.make_node('MSDeformAttn', ['value', 'spatial_shapes', current, 'weights'],
                                     ['Y'], name='msda', domain='gac'))
            expected = {'Y': msda_reference(tensors['value'], torch.from_numpy(spatial_shapes),
                                           value, tensors['weights']).numpy()}
            if public == 'locations':
                expected[current] = value.numpy()
            elif public == 'divided':
                expected['divided'] = divided.numpy()
            if branch:
                operand, reference = ('divided', divided) if branch == 'divided' else ('coordinates', coordinates)
                nodes.append(h.make_node('Mul', [operand, operand], ['other'], name='other_consumer'))
                expected['other'] = (reference * reference).numpy()
        assert all(np.isfinite(array).all() for array in expected.values()), name
        graph = h.make_graph(nodes, name,
            [h.make_tensor_value_info(key, T.FLOAT, list(value.shape)) for key, value in inputs.items()],
            [h.make_tensor_value_info(key, T.FLOAT, list(value.shape)) for key, value in expected.items()],
            [nh.from_array(value, key) for key, value in constants.items()])
        model = h.make_model(graph, opset_imports=[h.make_opsetid('', 17), h.make_opsetid('gac', 1)],
                             ir_version=9, producer_name='gac-independent-msda-offsets-tests')
        onnx.checker.check_model(model)
        folder = directory / name
        folder.mkdir(exist_ok=True)
        onnx.save(model, folder / 'model.onnx')
        np.savez(folder / 'inputs.npz', **inputs)
        np.savez(folder / 'expected.npz', **expected)
        cases.append({'name': name, 'expected_fusion': not (public or branch or transform or reverse_div),
                      'reference_first': first, 'coordinate_shape': coords_shape,
                      'locations_shape': location_shape, 'runtime_normalizer_shape': list(inputs['normalizer'].shape),
                      'output_shapes': {key: list(value.shape) for key, value in expected.items()},
                      'transform': transform, 'public': public, 'branch': branch, 'reversed_division': reverse_div})

    make('temporal_reference_right')
    make('temporal_reference_left', first=True)
    make('spatial_rank_seven_right', spatial=True)
    make('spatial_rank_seven_left', spatial=True, first=True)
    make('spatial_single_use_view_chain', spatial=True, chain=True)
    make('temporal_full_shape_normalizer', full_norm=True)
    make('temporal_signed_and_infinite_normalizer', special_norm=True)
    make('public_locations_retained', spatial=True, public='locations')
    make('public_division_retained', public='divided')
    make('division_extra_consumer', branch='divided')
    make('addition_extra_consumer', spatial=True, branch='coordinates')
    make('transpose_before_add_not_absorbed', transform='transpose_before')
    make('transpose_after_add_not_absorbed', transform='transpose_after')
    make('cast_before_add_not_absorbed', transform='cast_before')
    make('cast_after_add_not_absorbed', transform='cast_after')
    make('reversed_division_not_misidentified', reverse_div=True)
    report = {'reference': 'Real PyTorch FP32 Div/Add/views and grid_sample bilinear zeros align_corners=False',
              'versions': {'torch': torch.__version__, 'onnx': onnx.__version__, 'numpy': np.__version__},
              'cases': cases}
    (directory / 'cases.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Generated {len(cases)} independent MSDA-coordinate fusion fixtures in {directory}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    generate(parser.parse_args().output)

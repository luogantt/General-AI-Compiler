#!/usr/bin/env python3
"""Independent ONNX Runtime references for native contiguous-view lowering.

Fixtures validate logical shapes, observable outputs and storage lifetime. They
do not implement the compiler's alias analysis or prescribe allocator offsets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh
import onnxruntime as ort


def generate(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261017)
    cases = []
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    def values(shape):
        return rng.uniform(-1, 1, shape).astype(np.float32)

    def shape(values):
        return np.array(values, dtype=np.int64)

    def save(name, nodes, inputs, constants, outputs, *, max_copies=0,
             required_ops=None, maximum_dispatches=None, constant_root_shape=None):
        graph = h.make_graph(nodes, name,
            [h.make_tensor_value_info(key, h.np_dtype_to_tensor_dtype(value.dtype), list(value.shape))
             for key, value in inputs.items()],
            [h.make_tensor_value_info(key, T.FLOAT, dims) for key, dims in outputs.items()],
            [nh.from_array(value, key) for key, value in constants.items()])
        model = h.make_model(graph, opset_imports=[h.make_opsetid('', 17)], ir_version=9,
                             producer_name='gac-independent-view-lifetime-tests')
        onnx.checker.check_model(model)
        folder = directory / name
        folder.mkdir(exist_ok=True)
        onnx.save(model, folder / 'model.onnx')
        session = ort.InferenceSession(str(folder / 'model.onnx'), sess_options=options,
                                       providers=['CPUExecutionProvider'])
        reference = dict(zip(outputs, session.run(list(outputs), inputs)))
        np.savez(folder / 'inputs.npz', **inputs)
        np.savez(folder / 'expected.npz', **reference)
        cases.append({'name': name, 'max_view_copy_dispatches': max_copies,
                      'required_ops': required_ops or {}, 'maximum_dispatches': maximum_dispatches,
                      'constant_root_shape': constant_root_shape,
                      'onnx_view_nodes': sum(node.op_type in ('Identity', 'Reshape', 'Squeeze', 'Unsqueeze')
                                             for node in nodes),
                      'output_shapes': outputs})

    save('reshape_chain_zero_inferred', [
        h.make_node('Reshape', ['X', 's0'], ['v0']),
        h.make_node('Reshape', ['v0', 's1'], ['v1']),
        h.make_node('Reshape', ['v1', 's2'], ['v2']),
        h.make_node('Add', ['v2', 'B'], ['Y'])],
        {'X': values((2, 3, 4))}, {'s0': shape([0, -1]), 's1': shape([-1, 4]),
         's2': shape([0, 2, -1]), 'B': values((1, 2, 1))}, {'Y': [6, 2, 2]}, maximum_dispatches=1)

    save('input_squeeze_unsqueeze_softmax', [
        h.make_node('Squeeze', ['X', 'axes0'], ['v0']),
        h.make_node('Unsqueeze', ['v0', 'axes1'], ['v1']),
        h.make_node('Reshape', ['v1', 's'], ['v2']),
        h.make_node('Softmax', ['v2'], ['Y'], axis=1)],
        {'X': values((1, 2, 1, 3))}, {'axes0': shape([0, 2]), 'axes1': shape([1]),
         's': shape([2, 3])}, {'Y': [2, 3]}, maximum_dispatches=1)

    save('arena_fork_original_late', [
        h.make_node('MatMul', ['X', 'W'], ['root']),
        h.make_node('Reshape', ['root', 'wide'], ['view']),
        h.make_node('Add', ['view', 'B'], ['biased']),
        h.make_node('Relu', ['biased'], ['active']),
        h.make_node('Mul', ['active', 'active'], ['square']),
        h.make_node('Reshape', ['square', 'original'], ['restored']),
        h.make_node('Add', ['restored', 'root'], ['Y'])],
        {'X': values((4, 8))}, {'W': values((8, 8)), 'wide': shape([2, 16]),
         'original': shape([4, 8]), 'B': values((16,))}, {'Y': [4, 8]}, maximum_dispatches=5)

    save('arena_fork_view_late', [
        h.make_node('MatMul', ['X', 'W0'], ['root']),
        h.make_node('Reshape', ['root', 'wide'], ['late_view']),
        h.make_node('MatMul', ['X', 'W1'], ['other']),
        h.make_node('Relu', ['other'], ['active']),
        h.make_node('Mul', ['active', 'active'], ['square']),
        h.make_node('Reshape', ['square', 'wide'], ['other_view']),
        h.make_node('Add', ['other_view', 'late_view'], ['Y'])],
        {'X': values((4, 8))}, {'W0': values((8, 8)), 'W1': values((8, 8)),
         'wide': shape([2, 16])}, {'Y': [2, 16]}, maximum_dispatches=5)

    save('public_input_view', [h.make_node('Reshape', ['X', 's'], ['public_view'])],
        {'X': values((2, 3))}, {'s': shape([3, 2])}, {'public_view': [3, 2]},
        max_copies=1, required_ops={'gac.reshape': 1}, maximum_dispatches=1)

    save('public_multiple_views', [
        h.make_node('Reshape', ['X', 's0'], ['first_view']),
        h.make_node('Reshape', ['X', 's1'], ['second_view'])],
        {'X': values((2, 3))}, {'s0': shape([3, 2]), 's1': shape([1, 6])},
        {'X': [2, 3], 'first_view': [3, 2], 'second_view': [1, 6]},
        max_copies=2, required_ops={'gac.reshape': 2}, maximum_dispatches=2)

    save('public_arena_view', [
        h.make_node('MatMul', ['X', 'W'], ['original_output']),
        h.make_node('Reshape', ['original_output', 's'], ['reshaped_output'])],
        {'X': values((2, 4))}, {'W': values((4, 6)), 's': shape([3, 4])},
        {'original_output': [2, 6], 'reshaped_output': [3, 4]},
        max_copies=1, required_ops={'gac.reshape': 1}, maximum_dispatches=2)

    save('nonidentity_transpose_materialized', [
        h.make_node('Reshape', ['X', 's0'], ['v0']),
        h.make_node('Transpose', ['v0'], ['permuted'], perm=[0, 2, 1]),
        h.make_node('Reshape', ['permuted', 's1'], ['v1']),
        h.make_node('Relu', ['v1'], ['Y'])],
        {'X': np.array([[1, -2, 3], [4, 5, -6]], np.float32)},
        {'s0': shape([1, 2, 3]), 's1': shape([3, 2])}, {'Y': [3, 2]},
        required_ops={'gac.transpose': 1}, maximum_dispatches=2)

    save('fp16_cast_roundtrip_materialized', [
        h.make_node('Reshape', ['X', 's0'], ['flat']),
        h.make_node('Cast', ['flat'], ['half'], to=T.FLOAT16),
        h.make_node('Cast', ['half'], ['rounded'], to=T.FLOAT),
        h.make_node('Reshape', ['rounded', 's1'], ['restored']),
        h.make_node('Add', ['restored', 'B'], ['Y'])],
        {'X': np.array([[1.0004, 2049, 0.3333], [-1.0004, -2049, -0.3333]], np.float32)},
        {'s0': shape([-1]), 's1': shape([2, 3]), 'B': values((2, 3))}, {'Y': [2, 3]},
        required_ops={'gac.cast': 2}, maximum_dispatches=3)

    save('shared_root_different_softmax_axes', [
        h.make_node('Reshape', ['X', 's0'], ['two_rows']),
        h.make_node('Softmax', ['two_rows'], ['Y0'], axis=1),
        h.make_node('Reshape', ['X', 's1'], ['six_rows']),
        h.make_node('Softmax', ['six_rows'], ['Y1'], axis=1)],
        {'X': values((2, 3, 4))}, {'s0': shape([2, 12]), 's1': shape([6, 4])},
        {'Y0': [2, 12], 'Y1': [6, 4]}, required_ops={'gac.softmax': 2}, maximum_dispatches=2)

    # Small constants would fold before device lowering, never exercising a
    # physical constant root. Just exceed the existing 4M-element fold cap;
    # A runtime bias prevents full constant folding; constant Gather indices
    # select only 12 outputs. Dynamic Gather is outside the native contract.
    constant_shape = [1, 2049, 2048]
    constant = np.arange(np.prod(constant_shape), dtype=np.float32).reshape(constant_shape)
    constant = (constant % 251) * np.float32(0.03125)
    save('constant_root_runtime_add_gather', [
        h.make_node('Squeeze', ['C', 'axes'], ['matrix']),
        h.make_node('Reshape', ['matrix', 's'], ['four_columns']),
        h.make_node('Add', ['four_columns', 'bias'], ['biased']),
        h.make_node('Gather', ['biased', 'indices'], ['Y'], axis=0)],
        {'bias': values((4,))},
        {'C': constant, 'axes': shape([0]), 's': shape([-1, 4]),
         'indices': shape([0, 1024, constant.size // 4 - 1])}, {'Y': [3, 4]},
        required_ops={'gac.gather': 1, 'gac.add': 1}, maximum_dispatches=2, constant_root_shape=constant_shape)

    save('internal_identity_and_reshape', [
        h.make_node('Identity', ['X'], ['identity']),
        h.make_node('Reshape', ['identity', 's'], ['flat']),
        h.make_node('Mul', ['flat', 'B'], ['Y'])],
        {'X': values((2, 3))}, {'s': shape([-1]), 'B': values((6,))}, {'Y': [6]}, maximum_dispatches=1)

    result = {'reference': 'ONNX Runtime CPUExecutionProvider, graph optimizations disabled',
              'versions': {'onnx': onnx.__version__, 'onnxruntime': ort.__version__, 'numpy': np.__version__},
              'cases': cases}
    (directory / 'cases.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'Generated {len(cases)} independent view/lifetime fixtures in {directory}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    generate(parser.parse_args().output)

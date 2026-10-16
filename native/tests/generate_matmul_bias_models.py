#!/usr/bin/env python3
"""Independent ONNX Runtime fixtures for exact-domain MatMul + bias fusion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh
import onnxruntime as ort
from onnx.reference import ReferenceEvaluator


def generate(directory):
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(41873)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    cases = []

    def random(shape):
        return rng.uniform(-0.75, 0.75, shape).astype(np.float32)

    def save(name, x, weight, bias, *, left=False, relu=False, public=False,
             fork=False, fuse=True, output_shape=None, invalid=False, exact=False):
        left_shape = (1, *x.shape) if x.ndim == 1 else x.shape
        right_shape = (*weight.shape, 1) if weight.ndim == 1 else weight.shape
        result_shape = [*np.broadcast_shapes(left_shape[:-2], right_shape[:-2]), left_shape[-2], right_shape[-1]]
        if x.ndim == 1:
            result_shape.pop(-2)
        if weight.ndim == 1:
            result_shape.pop(-1)
        final_shape = output_shape or result_shape
        dtype = T.FLOAT16 if x.dtype == np.float16 else T.FLOAT
        nodes = [h.make_node('MatMul', ['X', 'W'], ['M'], name='product'),
                 h.make_node('Add', ['bias', 'M'] if left else ['M', 'bias'],
                             ['biased' if relu else 'Y'], name='bias_add')]
        if relu:
            nodes.append(h.make_node('Relu', ['biased'], ['Y'], name='activation'))
        outputs = {'Y': final_shape}
        if public:
            outputs['M'] = result_shape
        if fork:
            nodes.append(h.make_node('Mul', ['M', 'M'], ['other'], name='other_consumer'))
            outputs['other'] = result_shape
        graph = h.make_graph(nodes, name, [h.make_tensor_value_info('X', dtype, list(x.shape))],
            [h.make_tensor_value_info(key, dtype, shape) for key, shape in outputs.items()],
            [nh.from_array(weight, 'W'), nh.from_array(bias, 'bias')])
        model = h.make_model(graph, opset_imports=[h.make_opsetid('', 17)], ir_version=9,
                             producer_name='gac-independent-matmul-bias-fusion-tests')
        onnx.checker.check_model(model)
        folder = directory / name
        folder.mkdir(exist_ok=True)
        onnx.save(model, folder / 'model.onnx')
        np.savez(folder / 'inputs.npz', X=x)
        reference = 'ONNX Runtime CPU, graph optimizations disabled'
        if not invalid:
            if x.dtype == np.float16:
                # ORT CPU can retain FP32 intermediates between unsupported
                # FP16 kernels even with graph optimization disabled. The ONNX
                # reference evaluator materializes each declared FP16 tensor.
                reference = 'ONNX ReferenceEvaluator; explicit FP16 operator boundaries'
                actual = dict(zip(outputs, ReferenceEvaluator(model).run(list(outputs), {'X': x})))
            else:
                session = ort.InferenceSession(str(folder / 'model.onnx'), sess_options=options,
                                               providers=['CPUExecutionProvider'])
                actual = dict(zip(outputs, session.run(list(outputs), {'X': x})))
            np.savez(folder / 'expected.npz', **actual)
        cases.append({'name': name, 'valid': not invalid,
                      'reference': reference,
                      'expected_fusion': 'gac.matmul_bias_relu' if relu and fuse else 'gac.matmul_bias' if fuse else None,
                      'output_shapes': outputs, 'exact_finite_comparison': exact,
                      'bias_side': 'left' if left else 'right', 'bias_shape': list(bias.shape),
                      'matmul_shape': result_shape})

    x, weight = random((3, 4)), random((4, 5))
    save('bias_right_scalar', x, weight, np.asarray(0.125, np.float32))
    save('bias_left_channel', x, weight, random((5,)), left=True)
    save('bias_right_matrix', x, weight, random((3, 5)))
    save('bias_left_column', x, weight, random((3, 1)), left=True)
    save('bias_batched_channel', random((2, 3, 4)), weight, random((5,)))
    save('bias_batched_middle_axis', random((2, 3, 4)), weight, random((1, 3, 1)), left=True)
    save('bias_batch_broadcast', random((2, 1, 3, 4)), random((1, 4, 4, 5)), random((1, 4, 1, 5)))
    save('matmul_public_output_retained', x, weight, random((5,)), public=True, fuse=False)
    save('matmul_extra_consumer_retained', x, weight, random((5,)), fork=True, fuse=False)

    # Product rounds at the FP16 MatMul boundary before Add. Fusing the bias
    # into an FP32 accumulator would expose a nonzero residual instead of zero.
    half_x = np.array([[1.0234375]], np.float16)
    half_w = half_x.copy()
    half_bias = np.array([-np.float16(float(half_x[0, 0]) * float(half_w[0, 0]))], np.float16)
    save('fp16_rounding_boundary', half_x, half_w, half_bias, fuse=False, exact=True)
    save('bias_expands_output_domain', x, weight, random((2, 1, 5)),
         output_shape=[2, 3, 5], fuse=False)
    save('bias_right_relu', x, weight, random((5,)), relu=True)
    save('bias_left_relu', x, weight, random((5,)), left=True, relu=True)
    save('bias_after_complete_reduction', np.array([[1e10, -1e10, 1]], np.float32),
         np.ones((3, 1), np.float32), np.array([1], np.float32), exact=True)
    save('nan_inf_semantics', np.array([[1, 2], [np.inf, 1], [np.nan, 0]], np.float32),
         np.array([[1, 0], [-2, 1]], np.float32), np.array([-np.inf, np.inf], np.float32))
    save('zero_inputs_and_bias', np.zeros((2, 4), np.float32), weight,
         np.array([0, -0., 1, -1, 0.25], np.float32), exact=True)
    save('reject_incompatible_bias_shape', x, weight, random((7,)), fuse=False, invalid=True)
    # ONNX MatMul promotes rank-one inputs for multiplication and then removes
    # the inserted matrix axis. In particular, dot products return rank zero.
    save('dot_rank_one_scalar_bias', random((4,)), random((4,)), np.asarray(0.125, np.float32))
    save('vector_matrix_channel_bias', random((4,)), weight, random((5,)))
    save('matrix_vector_scalar_bias_first', x, random((4,)), np.asarray(-0.25, np.float32), left=True)
    save('vector_batched_matrix_bias_first', random((4,)), random((2, 4, 5)), random((5,)), left=True)
    save('batched_matrix_vector_bias', random((2, 3, 4)), random((4,)), random((1, 3)))
    # A one-element vector bias expands scalar dot output into rank one. Equal
    # element counts do not make their logical result domains interchangeable.
    save('dot_vector_bias_expands_rank', random((4,)), random((4,)), random((1,)),
         output_shape=[1], fuse=False)
    result = {'reference': 'ONNX Runtime CPU for FP32; ONNX ReferenceEvaluator for FP16 boundaries; no compiler implementation in Python',
              'versions': {'onnx': onnx.__version__, 'onnxruntime': ort.__version__, 'numpy': np.__version__},
              'cases': cases}
    (directory / 'cases.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'Generated {len(cases)} MatMul/bias fusion fixtures in {directory}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    generate(parser.parse_args().output)

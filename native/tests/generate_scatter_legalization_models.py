#!/usr/bin/env python3
"""Independent ONNX references for bounded ScatterND semantic rewrites."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def generate(output):
    output.mkdir(parents=True, exist_ok=True)
    cases = []

    def save(name, nodes, values, constants, outputs, scatter_count, *,
             remaining, rejection=None, expanded=False):
        infos = lambda items: [helper.make_tensor_value_info(
            key, helper.np_dtype_to_tensor_dtype(value.dtype), list(value.shape))
            for key, value in items.items()]
        graph = helper.make_graph(nodes, name, infos(values), infos(outputs),
                                  [numpy_helper.from_array(value, key)
                                   for key, value in constants.items()])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)],
                                  ir_version=9, producer_name='gac-scatter-legalization-tests')
        folder = output / name
        folder.mkdir(exist_ok=True)
        onnx.save(model, folder / 'model.onnx')
        np.savez(folder / 'inputs.npz', **values)
        if rejection is None:
            onnx.checker.check_model(model)
            expected = ReferenceEvaluator(model).run(list(outputs), values)
            np.savez(folder / 'expected.npz', **dict(zip(outputs, expected)))
        case = dict(name=name, valid=True, unsupported=False, expect_fusion=False,
                    inputs=list(values), outputs=list(outputs),
                    reject_device=rejection is not None, expected_diagnostic=rejection,
                    expect_cse={'scatter_nd': [scatter_count, remaining]})
        if expanded:
            case['expect_optimized_contains'] = ['"gac.expand"(']
        cases.append(case)

    data = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    updates = data * -2 + 17
    full = np.array([(a, b) for a in range(2) for b in range(3)], np.int64)
    for name, indices, shaped_updates, remain in (
        ('full_ordered_identity', full.reshape(2, 3, 2), updates, 0),
        ('full_ordered_reshape', full, updates.reshape(6, 2), 0),
        ('full_negative_indices', full - np.array([2, 3]), updates.reshape(6, 2), 0),
        ('full_permuted_retained', full[::-1].copy(), updates.reshape(6, 2), 1),
        ('partial_retained', full[:3], updates.reshape(6, 2)[:3].copy(), 1),
    ):
        save(name, [helper.make_node('ScatterND', ['X', 'indices', 'U'], ['Y'])],
             {'X': data, 'U': shaped_updates}, {'indices': indices}, {'Y': data},
             1, remaining=remain)
    save('duplicate_indices_rejected', [helper.make_node('ScatterND', ['X', 'indices', 'U'], ['Y'])],
         {'X': data, 'U': updates.reshape(6, 2)[:2].copy()},
         {'indices': np.array([[0, 0], [0, 0]], np.int64)}, {'Y': data},
         1, remaining=1, rejection='duplicate')
    save('reduction_add_retained', [helper.make_node('ScatterND', ['X', 'indices', 'U'], ['Y'], reduction='add')],
         {'X': data, 'U': updates.reshape(6, 2)}, {'indices': full}, {'Y': data},
         1, remaining=1, rejection='reduction')

    data = np.arange(24, dtype=np.float32).reshape(1, 3, 4, 2)
    update = (np.arange(8, dtype=np.float32) * -.3 + 2).reshape(1, 1, 4, 2)
    for variant in ('complete', 'negative', 'reverse_stage_order', 'zero_source_eliminated', 'missing_camera',
                    'overlap_across_stages', 'permuted_update_mapping',
                    'different_update_ssa', 'intermediate_public_output'):
        nodes, constants = [], {}
        values = {'X': data, 'U': update}
        if variant == 'zero_source_eliminated':
            values.pop('X')
            constants['data_shape'] = np.array(data.shape, np.int64)
            nodes.append(helper.make_node('ConstantOfShape', ['data_shape'], ['X'],
                                          value=numpy_helper.from_array(np.array([0], np.float32))))
        previous = 'X'
        cameras = [0, 1] if variant == 'missing_camera' else [0, 1, 2]
        if variant == 'reverse_stage_order':
            cameras.reverse()
        if variant == 'overlap_across_stages':
            cameras = [0, 0, 2]
        for index, camera in enumerate(cameras):
            key = 'index' + str(index)
            indices = np.array([[[[0, camera, q] for q in range(4)]]], np.int64)
            if variant == 'negative':
                indices -= np.array([1, 3, 4])
            if variant == 'permuted_update_mapping':
                indices = indices[:, :, ::-1, :].copy()
            constants[key] = indices
            update_name = 'U'
            if variant == 'different_update_ssa' and index == len(cameras)-1:
                update_name = 'U2'
                values[update_name] = update + 1
            current = 'Y' if index == len(cameras)-1 else 'stage' + str(index)
            nodes.append(helper.make_node('ScatterND', [previous, key, update_name], [current]))
            previous = current
        outputs = {'Y': data}
        if variant == 'intermediate_public_output':
            outputs['stage0'] = data
        optimized = variant in ('complete', 'negative', 'reverse_stage_order', 'zero_source_eliminated')
        save('broadcast_chain_' + variant, nodes, values, constants, outputs,
             len(cameras), remaining=0 if optimized else len(cameras), expanded=optimized)
        if variant == 'zero_source_eliminated':
            cases[-1]['expect_no_constant_shape'] = list(data.shape)
    result = {'cases': cases}
    (output / 'cases.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(f"Generated {len(generate(args.output)['cases'])} ScatterND legalization cases: {args.output}")

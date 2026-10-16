#!/usr/bin/env python3
"""Bounded ONNX Range fixtures; integer references use exact Python arithmetic.

NumPy arange and tolerance-based comparisons can obscure INT64 errors beyond
2**53. References here never pass integers through floating point. The paired
runner additionally checks emitted constant payloads bit-for-bit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def generate(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    cases = []

    def save(name, start, limit, delta, *, dtype=np.int64, tail=None,
             rejected=False, device_rejected=False, diagnostic=None,
             expected=None, declared_size=None, runtime=()):
        dtype = np.dtype(dtype)
        values = {key: np.array(value, dtype=dtype)
                  for key, value in zip(("start", "limit", "delta"),
                                        (start, limit, delta))}
        nodes = [helper.make_node("Range", ["start", "limit", "delta"], ["range_out"])]
        result = None
        if not rejected and not device_rejected:
            result = np.array(list(range(start, limit, delta)), dtype=dtype) \
                if np.issubdtype(dtype, np.integer) else np.array(expected, dtype=dtype)
            if tail is not None:
                values["tail"] = np.array(tail, dtype=dtype)
                nodes.append(helper.make_node("Concat", ["range_out", "tail"], ["Y"], axis=0))
                result = np.concatenate((result, values["tail"]))
            else:
                nodes[0].output[0] = "Y"
            declared_size = result.size
        else:
            nodes[0].output[0] = "Y"
        output_dtype = dtype
        if dtype == np.dtype(np.int16) and result is not None:
            # The existing device ABI excludes i16. Still test exact i16 Range
            # folding by consuming it with a supported i64 constant Cast.
            nodes[-1].output[0] = "before_cast"
            nodes.append(helper.make_node("Cast", ["before_cast"], ["Y"], to=onnx.TensorProto.INT64))
            result = result.astype(np.int64)
            output_dtype = np.dtype(np.int64)
        output_info = helper.make_tensor_value_info("Y", helper.np_dtype_to_tensor_dtype(output_dtype),
                                                     [declared_size])
        input_values = {key: values[key] for key in runtime}
        input_infos = [helper.make_tensor_value_info(key, helper.np_dtype_to_tensor_dtype(dtype), [])
                       for key in runtime]
        graph = helper.make_graph(nodes, name, input_infos, [output_info],
                                  [numpy_helper.from_array(value, key)
                                   for key, value in values.items() if key not in runtime])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)],
                                  ir_version=9, producer_name="gac-range-contract-tests")
        folder = output / name
        folder.mkdir(exist_ok=True)
        onnx.save(model, folder / "model.onnx")
        np.savez(folder / "inputs.npz", **input_values)
        reference_contract = None
        if result is not None:
            onnx.checker.check_model(model)
            evaluated = ReferenceEvaluator(model).run(None, input_values)[0]
            agrees = evaluated.dtype == result.dtype and evaluated.shape == result.shape \
                and evaluated.tobytes() == result.tobytes()
            reference_contract = {"onnx_version": onnx.__version__,
                                  "onnx_reference_exactly_agrees": agrees}
            if not agrees:
                # ReferenceEvaluator and the exact integer formula differ at
                # this 65-bit endpoint distance. ONNX's documented formula and
                # its reference function's float casts also require care at
                # these extremes. This case tests our explicit exact-integer
                # contract, not asserted parity with every ONNX interpreter.
                assert name == "range_i64_cross_extremes", (name, result, evaluated)
                reference_contract.update(
                    authoritative_oracle="Python arbitrary-precision integer range",
                    expected_values=[int(value) for value in result],
                    onnx_reference_values=[int(value) for value in evaluated],
                    reason="INT64 endpoint distance needs 65 bits; exact integer formula and ReferenceEvaluator differ. This fixture validates the explicit exact-integer contract, not ReferenceEvaluator or ORT parity.")
            else:
                # Ordinary regression payloads are the independent ONNX
                # interpreter output, after the exact arithmetic cross-check.
                result = evaluated
            np.savez(folder / "expected.npz", Y=result)
        case = {"name": name, "valid": not rejected, "reject_device": device_rejected,
                "expected_diagnostic": diagnostic, "unsupported": False,
                "expect_fusion": False, "inputs": list(runtime), "outputs": ["Y"],
                "check_constant_payload_exact": result is not None}
        if reference_contract:
            case["reference_contract"] = reference_contract
        if device_rejected:
            case["expect_optimized_contains"] = ['"gac.range"(']
        elif not rejected:
            case["expect_optimized_absent"] = ['"gac.range"(', '"gac.concat"(']
        cases.append(case)

    maximum, minimum = 2**63 - 1, -(2**63)
    save("range_i64_positive", -5, 13, 3)
    save("range_i64_negative", 12, -7, -4)
    save("range_i64_above_double_exactness", 2**53 + 1, 2**53 + 8, 2)
    save("range_i64_below_double_exactness", -(2**53 + 1), -(2**53 + 8), -2)
    save("range_i64_max_no_terminal_overflow", maximum - 1, maximum, 2)
    save("range_i64_min_no_terminal_overflow", minimum + 1, minimum, -2)
    save("range_i64_cross_extremes", minimum, maximum, maximum)
    save("range_i64_minimum_delta", maximum, minimum, minimum)
    save("range_i32_max_no_terminal_overflow", 2**31 - 2, 2**31 - 1, 2, dtype=np.int32)
    save("range_i16_min_no_terminal_overflow", -(2**15) + 1, -(2**15), -2, dtype=np.int16)
    save("range_empty_positive_concat", 5, 2, 1, tail=[-7, 9])
    save("range_empty_negative_concat", 2, 5, -1, tail=[13])
    save("range_empty_equal_concat", maximum, maximum, 1, tail=[maximum])
    save("range_reject_zero_delta", 0, 3, 0, rejected=True,
         diagnostic="delta must not be zero", declared_size=1)
    save("range_exceeds_default_fold_budget", 0, (4 << 20) + 1, 1,
         device_rejected=True, diagnostic="gac.range", declared_size=(4 << 20) + 1)
    save("range_reject_shape_length_overflow", minimum, maximum, 1,
         rejected=True, diagnostic="length", declared_size=None)
    save("range_runtime_start_not_constant", 0, 8, 2, runtime=("start",),
         device_rejected=True, declared_size=4)
    save("range_runtime_parameters_not_constant", 0, 8, 2,
         runtime=("start", "limit", "delta"), device_rejected=True, declared_size=4)
    # Float Range is deliberately not folded by this implementation. Assert
    # both preservation in optimized IR and strict device rejection, including
    # cases whose f64 parameters cannot be rounded to f32 without a change.
    save("range_f32_positive", .5, 2., .25, dtype=np.float32,
         device_rejected=True, declared_size=6)
    save("range_f64_negative", 1., -.5, -.25, dtype=np.float64,
         device_rejected=True, declared_size=6)
    save("range_f64_source_precision", 1. + 2**-40, 1. + 8*2**-40, 2*2**-40,
         dtype=np.float64, device_rejected=True, declared_size=4)
    for key, triple in {"nan_start": (np.nan, 1., .25),
                        "inf_limit": (0., np.inf, .25),
                        "inf_delta": (0., 1., np.inf)}.items():
        save("range_reject_" + key, *triple, dtype=np.float32,
             device_rejected=True, declared_size=None)
    result = {"cases": cases}
    (output / "cases.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(f"Generated {len(generate(args.output)['cases'])} Range fixtures: {args.output}")

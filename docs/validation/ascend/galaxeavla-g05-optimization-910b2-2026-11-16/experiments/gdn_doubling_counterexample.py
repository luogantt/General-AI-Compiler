"""NumPy-only evidence against making FP32 GDN doubling the default.

No torch import, no model/backend installation, no timing claim.
For C32: B=I+L; four updates P=P@P, B=B+P@B give the exact polynomial
(I+L)(I+L^2)(I+L^4)(I+L^8)(I+L^16), but FP32 cancellation is unsafe.
Run: python build/parity/gdn_doubling_counterexample.py
"""
import json
from pathlib import Path
import numpy as np


def compare(lower):
    width = lower.shape[-1]
    eye = np.eye(width, dtype=np.float32)
    original = lower.copy()
    for i in range(1, width):
        row, sub = original[..., i, :i].copy(), original[..., :i, :i].copy()
        original[..., i, :i] = row + (row[..., :, None] * sub).sum(-2, dtype=np.float32)
    original += eye
    power, candidate = lower, eye + lower
    trace = []
    covered = 2
    while covered < width:
        power = power @ power
        candidate = candidate + power @ candidate
        trace.append({"power": covered, "max_abs_power": float(np.max(np.abs(power))),
                      "max_abs_partial_inverse": float(np.max(np.abs(candidate)))})
        covered *= 2
    system = eye.astype(np.float64) - lower.astype(np.float64)
    reference = np.linalg.solve(system, np.broadcast_to(eye, lower.shape).astype(np.float64))
    def error(value):
        delta = value.astype(np.float64) - reference
        return {"max_abs": float(np.max(np.abs(delta))),
                "relative_l2": float(np.linalg.norm(delta.ravel()) / np.linalg.norm(reference.ravel())),
                "inverse_residual_max_abs": float(np.max(np.abs(system @ value.astype(np.float64) - eye)))}
    return {"shape": list(lower.shape), "dtype": str(lower.dtype),
            "system_condition_2": np.linalg.cond(system).tolist(),
            "row_recurrence_vs_fp64": error(original), "doubling_vs_fp64": error(candidate),
            "doubling_intermediates": trace}


def main():
    width = 32
    rng = np.random.default_rng(20261116)
    keys = rng.standard_normal((2, 1, 128)).astype(np.float32) + rng.standard_normal((2, width, 128)).astype(np.float32) * np.float32(.001)
    keys /= np.sqrt((keys * keys).sum(-1, keepdims=True) + np.float32(1e-6))
    cumulative_g = np.cumsum(np.full((2, width), -1e-5, np.float32), axis=-1)
    decay = np.exp(cumulative_g[..., :, None] - cumulative_g[..., None, :])
    lower_gdn = np.tril(-((keys * np.float32(.999)) @ keys.swapaxes(-1, -2)) * decay, -1)
    cases = {"exact_minus_one_lower": compare(-np.tril(np.ones((width, width), np.float32), -1)),
             "normalized_collinear_keys_beta_0.999_g_minus_1e-5": compare(lower_gdn)}
    assert cases["exact_minus_one_lower"]["row_recurrence_vs_fp64"]["max_abs"] == 0
    report = {"scope": "CPU NumPy FP32 inverse counterexamples only; no torch/NPU/GE/model execution or performance measurement",
              "numpy_version": np.__version__, "seed": 20261116,
              "decision": "Reject default FP32 doubling: exact-arithmetic equivalence does not establish floating-point accuracy",
              "c32_operations": {"row_updates": 31, "polynomial_factors": 5, "updates_after_I_plus_L": 4,
                                 "batched_matmuls": 8, "dense_adds": 5,
                                 "row_approx_flops_per_matrix": 21856, "doubling_approx_flops_per_matrix": 529408,
                                 "convention": "1 MAC=2 FLOPs; dense arithmetic includes structural zeros; not kernel latency"},
              "cases": cases}
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "cases": {name: {"row": case["row_recurrence_vs_fp64"]["max_abs"],
                                                             "doubling": case["doubling_vs_fp64"]["max_abs"]}
                                                       for name, case in cases.items()}}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Numerical + latency check for GacGdnDecode against the PyTorch GDN decode path.

The reference is a literal transcription of
transformers.models.qwen3_5.modeling_qwen3_5.torch_recurrent_gated_delta_rule
restricted to batch 1 and sequence length 1, including its rounding: the L2
normalization runs on BF16 tensors, beta is produced by a BF16 sigmoid, and the
recurrence itself runs in FP32.

The operator takes g and beta precomputed, exactly as the model computes them
before the recurrence, so the same rounding boundaries are exercised.

Run in a process where ASCEND_CUSTOM_OPP_PATH already points at the installed
``gacgdn`` vendor.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import torch
import torch.nn.functional as F

HK, HV, DK, DV = 16, 48, 128, 128
GROUP = HV // HK
L2_EPS = 1e-6


def l2norm_bf16(x: torch.Tensor) -> torch.Tensor:
    """The FLA/PyTorch l2norm, evaluated in the input dtype (BF16 here)."""
    inv = torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + L2_EPS)
    return x * inv


def gates(a, b, a_log, dt_bias):
    """The model's pre-recurrence gate computation, in its exact dtypes."""
    g = -a_log.float().exp() * F.softplus(a.float() + dt_bias)
    beta = b.sigmoid().float()
    return g, beta


def reference(q, k, v, g, beta, state):
    """torch_recurrent_gated_delta_rule with batch 1, sequence length 1."""
    qn = l2norm_bf16(q).float() * (1.0 / math.sqrt(DK))
    kn = l2norm_bf16(k).float()
    vf = v.float()

    decay = g.exp()
    s = state.clone().float()
    out = torch.empty(HV, DV, dtype=torch.float32)
    for h in range(HV):
        kh = h // GROUP
        s[h] = s[h] * decay[h]
        kv = (s[h] * kn[kh].unsqueeze(-1)).sum(dim=0)
        delta = (vf[h] - kv) * beta[h]
        s[h] = s[h] + kn[kh].unsqueeze(-1) * delta.unsqueeze(0)
        out[h] = (s[h] * qn[kh].unsqueeze(-1)).sum(dim=0)
    return out.to(torch.bfloat16), s


def make_case(seed: int, scale: float):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = (torch.randn(HK, DK, generator=gen) * scale).to(torch.bfloat16)
    k = (torch.randn(HK, DK, generator=gen) * scale).to(torch.bfloat16)
    v = (torch.randn(HV, DV, generator=gen) * scale).to(torch.bfloat16)
    a = (torch.randn(HV, generator=gen) * 0.5).to(torch.bfloat16)
    b = (torch.randn(HV, generator=gen) * 0.5).to(torch.bfloat16)
    a_log = torch.log(torch.rand(HV, generator=gen) * 16.0)
    dt_bias = torch.rand(HV, generator=gen) + 0.5
    state = torch.randn(HV, DK, DV, generator=gen).to(torch.float32)
    g, beta = gates(a, b, a_log, dt_bias)
    return q, k, v, g, beta, state


def compare(name, got, want):
    diff = (got.float() - want.float()).abs()
    max_abs = diff.max().item()
    l2 = torch.linalg.vector_norm(diff).item()
    base = torch.linalg.vector_norm(want.float()).item()
    rel = l2 / base if base else 0.0
    exact = int((got == want).sum().item())
    total = want.numel()
    print("  %-10s max_abs=%.6e rel_l2=%.6e exact=%d/%d" % (name, max_abs, rel, exact, total))
    return {"max_abs": max_abs, "rel_l2": rel, "exact": exact, "total": total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", required=True)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    torch.ops.load_library(args.bridge)
    if not torch._C._dispatch_has_kernel_for_dispatch_key("gac_qwen38::gdn_decode", "PrivateUse1"):
        raise SystemExit("bridge loaded but the PrivateUse1 kernel is not registered")

    results = {"cases": [], "timing": {}}
    for seed, scale in ((0, 1.0), (1, 4.0), (2, 0.25)):
        case = make_case(seed, scale)
        ref_o, ref_s = reference(*case)
        moved = [t.to(args.device) for t in case]
        with torch.inference_mode():
            got_o, got_s = torch.ops.gac_qwen38.gdn_decode(*moved, L2_EPS)
        print("case seed=%d scale=%.2f" % (seed, scale))
        results["cases"].append({
            "seed": seed, "scale": scale,
            "output": compare("o", got_o.cpu(), ref_o),
            "state": compare("state", got_s.cpu(), ref_s),
            "output_finite": bool(torch.isfinite(got_o.float()).all().item()),
            "state_finite": bool(torch.isfinite(got_s).all().item()),
        })

    # Latency: one call is one GDN layer's decode step.
    case = make_case(0, 1.0)
    moved = [t.to(args.device) for t in case]
    with torch.inference_mode():
        for _ in range(args.warmup):
            torch.ops.gac_qwen38.gdn_decode(*moved, L2_EPS)
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(args.iterations):
            torch.ops.gac_qwen38.gdn_decode(*moved, L2_EPS)
        torch.npu.synchronize()
        elapsed = time.perf_counter() - start
    per_call = elapsed / args.iterations
    print("timing: %.3f ms/call over %d iterations (48 heads)" % (per_call * 1e3, args.iterations))
    results["timing"] = {"op_ms_per_call": per_call * 1e3,
                         "model_equivalent_ms_per_token": per_call * 1e3 * 48,
                         "iterations": args.iterations}
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(results, fh, indent=2)
    worst = max(c["output"]["max_abs"] for c in results["cases"])
    print("worst output max_abs across cases: %.6e" % worst)
    return 0 if worst < 5e-2 else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Create a deterministic production-shape bundle without a UniAD checkpoint.

This utility exists for compiler smoke tests and GPU performance plumbing.  Its
weights are synthetic and cannot be used to evaluate UniAD accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np


from gac.uniad import UniADFFNContract, write_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layer", type=int, default=0, choices=range(6))
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()

    contract = UniADFFNContract(layer=args.layer)
    generator = np.random.default_rng(args.seed)
    weights = {}
    for name, shape in contract.weight_shapes.items():
        if name == "norm_weight":
            value = np.ones(shape, dtype=np.float32)
        else:
            value = generator.normal(0.0, 0.01, shape).astype(np.float32)
        weights[name] = np.ascontiguousarray(value)

    write_bundle(args.output, contract, weights)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(args.output.resolve())
    print("bundle_sha256=" + digest)
    print("warning=synthetic weights; compiler validation only")


if __name__ == "__main__":
    main()

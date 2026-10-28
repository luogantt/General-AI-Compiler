#!/usr/bin/env python3
"""Launch one checked MSDA configuration for Nsight Compute kernel profiling.

Example: ncu --set basic --kernel-name regex:msda_kernel --launch-skip 2
  --launch-count 1 python scripts/profile_msda_kernel.py --library plugin.so
  --inputs capture.npz --config options.json
This does not report whole-model throughput. Run benchmark_msda.py first for
the independent oracle, boundary tests and paired performance comparison.
"""
import argparse
import ctypes
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from bevformer_trt.msda import _fields
    from benchmark_msda import accepted, metrics
    options = _fields(json.loads(args.config.read_text()))
    torch.cuda.set_device(0)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    keys = ('value', 'shapes', 'starts', 'locations', 'weights')
    with np.load(args.inputs, allow_pickle=False) as source:
        arrays = {key: torch.from_numpy(np.array(source[key], copy=True)).cuda().contiguous() for key in keys}
        expected = torch.from_numpy(np.array(source['expected'], copy=True)).cuda()
    value, shapes, starts, locations, weights = (arrays[key] for key in keys)
    b, s, h, c = value.shape
    if locations.ndim != 6:
        raise ValueError('Expected locations [B,Q,H,L,P,2]')
    lb, q, lh, l, p, xy = locations.shape
    if (lb, lh, xy) != (b, h, 2) or weights.shape != locations.shape[:-1]:
        raise ValueError('Inconsistent locations/weights dimensions')
    if shapes.shape != (l, 2) or starts.shape != (l,) or shapes.dtype != torch.int32 or starts.dtype != torch.int32:
        raise ValueError('INT32 spatial metadata required')
    if value.dtype not in (torch.float16, torch.float32) or any(x.dtype != value.dtype for x in (locations, weights)):
        raise ValueError('One supported floating dtype required')
    for (height, width), start in zip(shapes.cpu().tolist(), starts.cpu().tolist()):
        if min(height, width) <= 0 or start < 0 or start + height * width > s:
            raise ValueError('Invalid spatial metadata')
    output = torch.empty((b, q, h, c), dtype=value.dtype, device='cuda')
    library = ctypes.CDLL(str(args.library.resolve()), mode=ctypes.RTLD_GLOBAL)
    forward = library.gac_msda_forward
    forward.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int64] * 7 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
    forward.restype = ctypes.c_int
    for _ in range(3):
        output.fill_(float('nan'))
        code = forward(*[arrays[key].data_ptr() for key in keys], output.data_ptr(), b, s, h, c, l, q, p,
                       int(value.dtype == torch.float16), options['channels_per_thread'], options['block_threads'],
                       options['specialize'], options['prefetch'], stream.cuda_stream)
        if code:
            raise RuntimeError(f'CUDA launch returned {code}')
        stream.synchronize()
    comparison = metrics(output, expected, torch)
    if not accepted(comparison, exact=value.dtype == torch.float16):
        raise AssertionError(comparison)
    print(json.dumps({'options': options, 'captured_reference_comparison': comparison}))


if __name__ == '__main__':
    main()

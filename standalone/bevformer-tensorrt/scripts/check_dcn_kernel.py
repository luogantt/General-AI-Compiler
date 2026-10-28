#!/usr/bin/env python3
"""GPU contract checks for the standalone DCNv2 im2col C ABI.

This checks columns, not GEMM or a TensorRT plugin. FP16 agreement with the
installed NVIDIA plugin is a separate real-input check in benchmark_dcn.py.
Every candidate must match generic columns bit for bit, including FP32.
"""

import argparse
import ctypes as ct
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import traceback

import numpy as np


BLOCKS = (128, 256, 512, 1024)
INVALID_VALUE = 1  # cudaErrorInvalidValue
GUARD_ELEMENTS = 64


@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    channels: int
    height: int
    width: int
    kernel_h: int = 3
    kernel_w: int = 3
    pad_h: int = 1
    pad_w: int = 1
    stride_h: int = 1
    stride_w: int = 1
    dilation_h: int = 1
    dilation_w: int = 1
    deform_groups: int = 1
    oracle: bool = False
    nonfinite_offsets: bool = False

    @property
    def out_h(self):
        return (self.height + 2 * self.pad_h - self.dilation_h * (self.kernel_h - 1) - 1) // self.stride_h + 1

    @property
    def out_w(self):
        return (self.width + 2 * self.pad_w - self.dilation_w * (self.kernel_w - 1) - 1) // self.stride_w + 1

    @property
    def arguments(self):
        return [self.batch, self.channels, self.height, self.width, self.out_h, self.out_w,
                self.kernel_h, self.kernel_w, self.pad_h, self.pad_w, self.stride_h,
                self.stride_w, self.dilation_h, self.dilation_w, self.deform_groups]

    @property
    def column_shape(self):
        return (self.channels * self.kernel_h * self.kernel_w, self.batch, self.out_h, self.out_w)


def cases():
    return [
        Case('group_tail_b2', 2, 6, 5, 7, deform_groups=2, oracle=True),
        Case('channel_tail_b1', 1, 5, 6, 7, oracle=True),
        Case('asymmetric_stride_padding', 2, 6, 8, 9, pad_h=1, pad_w=2,
             stride_h=2, stride_w=1, deform_groups=2),
        Case('dilated_stride', 1, 5, 9, 8, pad_h=2, pad_w=1,
             stride_h=1, stride_w=2, dilation_h=2, dilation_w=1),
        Case('rectangular_fallback', 2, 6, 7, 9, kernel_h=2, kernel_w=3,
             pad_h=1, pad_w=0, stride_h=2, dilation_w=2, deform_groups=2, oracle=True),
        Case('pointwise_fallback', 2, 5, 4, 5, kernel_h=1, kernel_w=1, pad_h=0, pad_w=0),
        Case('single_pixel_group_tail', 2, 2, 1, 1, deform_groups=2, oracle=True),
        Case('nonfinite_coordinates', 1, 8, 5, 3, deform_groups=2, nonfinite_offsets=True),
    ]


def boundary_targets(height, width):
    """Coordinates after applying base+offset, including exact zero-pad edges."""
    ys = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, height - 1.0, height - 0.5, float(height), height + 0.5]
    xs = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, width - 1.0, width - 0.5, float(width), width + 0.5]
    return ([(y, 0.5) for y in ys] + [(0.5, x) for x in xs]
            + [(-1.0, -1.0), (-0.5, -0.5), (height - 0.5, width - 0.5),
               (float(height), float(width)), (-1.5, width + 0.5)])


def make_arrays(case, dtype, seed):
    """Use distinct batch/channel/group data so indexing mistakes remain visible."""
    rng = np.random.default_rng(seed)
    k = case.kernel_h * case.kernel_w
    out_hw = case.out_h * case.out_w
    image = rng.uniform(-1, 1, (case.batch, case.channels, case.height, case.width)).astype(dtype)
    bound = max(case.height, case.width) + 3
    offset = rng.uniform(-bound, bound, (case.batch, 2 * case.deform_groups * k,
                                        case.out_h, case.out_w)).astype(dtype)
    mask = rng.uniform(-0.5, 1.5, (case.batch, case.deform_groups * k,
                                 case.out_h, case.out_w)).astype(dtype)
    targets = boundary_targets(case.height, case.width)
    for b in range(case.batch):
        for group in range(case.deform_groups):
            # Rotate within each group/batch; shared coordinates from the wrong
            # deformable group must not accidentally reproduce the reference.
            rotation = (b * 7 + group * 11) % len(targets)
            group_targets = targets[rotation:] + targets[:rotation]
            for flat, (y, x) in enumerate(group_targets[:k * out_hw]):
                point, spatial = divmod(flat, out_hw)
                kh, kw = divmod(point, case.kernel_w)
                oy, ox = divmod(spatial, case.out_w)
                base_y = oy * case.stride_h - case.pad_h + kh * case.dilation_h
                base_x = ox * case.stride_w - case.pad_w + kw * case.dilation_w
                offset[b, 2 * (group * k + point), oy, ox] = y - base_y
                offset[b, 2 * (group * k + point) + 1, oy, ox] = x - base_x
    flat_mask = mask.reshape(-1)
    flat_mask[::11] = 0
    flat_mask[1::13] = 1
    flat_mask[2::17] = -1
    if case.nonfinite_offsets:
        # Finite masks keep the expected zero-sampling results finite.
        offset.reshape(-1)[-3:] = [np.nan, np.inf, -np.inf]
    return {'input': np.ascontiguousarray(image), 'offset': np.ascontiguousarray(offset),
            'mask': np.ascontiguousarray(mask)}


def fp32_oracle(case, arrays):
    """Independent scalar zero-padded bilinear equation, with double sums.

    This does not call a candidate helper or implement the CUDA coefficient
    tree. FP32 coordinates are rounded as the ABI requires; interpolation is
    evaluated using the triangular basis for the four integer neighbors.
    """
    if any(x.dtype != np.float32 for x in arrays.values()):
        raise ValueError('The independent oracle is for FP32 inputs only')
    image, offset, mask = (arrays[n] for n in ('input', 'offset', 'mask'))
    result = np.zeros(case.column_shape, dtype=np.float64)
    k = case.kernel_h * case.kernel_w
    for b in range(case.batch):
        for c in range(case.channels):
            group = c // (case.channels // case.deform_groups)
            for point in range(k):
                kh, kw = divmod(point, case.kernel_w)
                for oy in range(case.out_h):
                    for ox in range(case.out_w):
                        y = float(np.float32(oy * case.stride_h - case.pad_h + kh * case.dilation_h
                                            + float(offset[b, 2 * (group * k + point), oy, ox])))
                        x = float(np.float32(ox * case.stride_w - case.pad_w + kw * case.dilation_w
                                            + float(offset[b, 2 * (group * k + point) + 1, oy, ox])))
                        sample = 0.0
                        if math.isfinite(y) and math.isfinite(x) and -1 < y < case.height and -1 < x < case.width:
                            for iy in (math.floor(y), math.floor(y) + 1):
                                for ix in (math.floor(x), math.floor(x) + 1):
                                    if 0 <= iy < case.height and 0 <= ix < case.width:
                                        sample += ((1.0 - abs(y - iy)) * (1.0 - abs(x - ix))
                                                   * float(image[b, c, iy, ix]))
                        result[c * k + point, b, oy, ox] = sample * float(mask[b, group * k + point, oy, ox])
    return result


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(array):
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def write_report(stream, report):
    stream.seek(0)
    json.dump(report, stream, indent=2, allow_nan=False)
    stream.write('\n')
    stream.truncate()
    stream.flush()


def invalid_arguments(case):
    changes = [('empty_batch', 0, 0), ('negative_channels', 1, -1),
               ('nondivisible_groups', 14, 4), ('wrong_output_height', 4, case.out_h + 1),
               ('empty_kernel', 6, 0), ('negative_padding', 8, -1),
               ('zero_stride', 10, 0), ('zero_dilation', 12, 0),
               ('dimension_overflow', 0, 1 << 62), ('element_count_overflow', 1, (1 << 31) - 2)]
    for name, index, value in changes:
        arguments = case.arguments
        arguments[index] = value
        yield name, arguments, None, None, None, None
    yield 'unsupported_dtype', case.arguments, 7, None, None, None
    yield 'unsupported_block', case.arguments, None, 64, None, None
    yield 'unsupported_specialization', case.arguments, None, None, 5, None
    for index, name in enumerate(('input', 'offset', 'mask', 'columns')):
        yield f'null_{name}', case.arguments, None, None, None, (index, 'null')
        yield f'misaligned_{name}', case.arguments, None, None, None, (index, 'misaligned')


def run(args):
    if args.report.exists():
        raise FileExistsError(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {'schema': 'gac.dcn.kernel_checks.v1', 'status': 'running',
              'scope': 'Standalone im2col columns, no GEMM/TensorRT or performance claim',
              'started_utc': datetime.now(timezone.utc).isoformat(), 'seed': args.seed,
              'library': str(args.library.resolve()), 'script_sha256': sha(__file__),
              'blocks': args.blocks, 'cases': [], 'checks': [], 'oracle_checks': [], 'invalid_argument_checks': []}
    with args.report.open('x', encoding='utf-8') as archive:
        write_report(archive, report)
        try:
            report['library_sha256'] = sha(args.library)
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is required for the kernel checks')
            torch.cuda.set_device(args.device)
            stream = torch.cuda.Stream(device=args.device)
            torch.cuda.set_stream(stream)
            report['environment'] = {'torch': torch.__version__, 'cuda': torch.version.cuda,
                                     'device_index': args.device, 'device_name': torch.cuda.get_device_name(args.device),
                                     'compute_capability': list(torch.cuda.get_device_capability(args.device))}
            library = ct.CDLL(str(args.library.resolve()))
            fn = library.dcn_im2col
            fn.argtypes = [ct.c_void_p] * 4 + [ct.c_int64] * 15 + [ct.c_int] * 3 + [ct.c_void_p]
            fn.restype = ct.c_int
            for case_index, case in enumerate(cases()):
                for dtype_code, dtype_name, np_dtype, torch_dtype, bit_dtype in (
                        (0, 'fp32', np.float32, torch.float32, torch.int32),
                        (1, 'fp16', np.float16, torch.float16, torch.int16)):
                    fixture_seed = args.seed + case_index * 101
                    arrays = make_arrays(case, np_dtype, fixture_seed)
                    record = {'case': case.name, 'dtype': dtype_name, 'seed': fixture_seed,
                              'arguments': case.arguments, 'column_shape': list(case.column_shape),
                              'inputs': {n: {'shape': list(a.shape), 'sha256': array_digest(a),
                                             'nonfinite_count': int(np.count_nonzero(~np.isfinite(a)))}
                                         for n, a in arrays.items()}}
                    report['cases'].append(record)
                    device = {n: torch.from_numpy(a).to(device=f'cuda:{args.device}') for n, a in arrays.items()}
                    count = math.prod(case.column_shape)
                    storage = torch.empty(count + 2 * GUARD_ELEMENTS, dtype=torch_dtype, device=f'cuda:{args.device}')
                    columns = storage[GUARD_ELEMENTS:GUARD_ELEMENTS + count].view(case.column_shape)
                    pointers = [device[n].data_ptr() for n in ('input', 'offset', 'mask')] + [columns.data_ptr()]

                    def poison():
                        storage.fill_(-123.5)
                        columns.fill_(float('nan'))

                    def guards_ok():
                        return bool((storage[:GUARD_ELEMENTS] == -123.5).all()) and bool((storage[-GUARD_ELEMENTS:] == -123.5).all())

                    def launch(block, specialize):
                        return fn(*pointers, *case.arguments, dtype_code, block, specialize, stream.cuda_stream)

                    poison()
                    baseline_status = launch(128, 0)
                    if baseline_status:
                        raise RuntimeError(f'{case.name}/{dtype_name}: generic launch failed: {baseline_status}')
                    stream.synchronize()
                    if not guards_ok() or not bool(torch.isfinite(columns).all()):
                        raise AssertionError(f'{case.name}/{dtype_name}: generic poison/canary check failed')
                    baseline = columns.clone()
                    if dtype_code == 0 and case.oracle:
                        oracle = fp32_oracle(case, arrays)
                        actual = baseline.cpu().numpy().astype(np.float64)
                        diff = np.abs(actual - oracle)
                        entry = {'case': case.name, 'dtype': dtype_name, 'finite': bool(np.isfinite(oracle).all()),
                                 'max_abs': float(diff.max()), 'rtol': 2e-5, 'atol': 2e-6,
                                 'allclose': bool(np.allclose(actual, oracle, rtol=2e-5, atol=2e-6))}
                        report['oracle_checks'].append(entry)
                        if not entry['finite'] or not entry['allclose']:
                            raise AssertionError(f'Independent FP32 column oracle mismatch: {entry}')

                    for specialize in range(5):
                        for block in args.blocks:
                            entry = {'case': case.name, 'dtype': dtype_name, 'block': block, 'specialize': specialize}
                            report['current_check'] = entry
                            poison()
                            before = storage.view(bit_dtype).clone()
                            status = launch(block, specialize)
                            entry['cuda_status'] = status
                            stream.synchronize()
                            entry['guard_intact'] = guards_ok()
                            if specialize == 2 and (case.kernel_h, case.kernel_w) != (3, 3):
                                entry['expected_rejection'] = True
                                entry['storage_unchanged'] = bool(torch.equal(storage.view(bit_dtype), before))
                                entry['passed'] = status == INVALID_VALUE and entry['storage_unchanged'] and entry['guard_intact']
                            else:
                                entry['finite'] = bool(torch.isfinite(columns).all())
                                entry['bitwise_equal'] = bool(torch.equal(columns.view(bit_dtype), baseline.view(bit_dtype)))
                                entry['passed'] = status == 0 and entry['guard_intact'] and entry['finite'] and entry['bitwise_equal']
                                if entry['finite']:
                                    entry['max_abs'] = float((columns.float() - baseline.float()).abs().max())
                            report['checks'].append(entry)
                            if not entry['passed']:
                                raise AssertionError(f'DCN kernel check failed: {entry}')

                    if case_index == 0:
                        for name, shape, dtype_override, block_override, spec_override, pointer_override in invalid_arguments(case):
                            report['current_check'] = {'case': name, 'dtype': dtype_name, 'expected_rejection': True}
                            poison()
                            before = storage.view(bit_dtype).clone()
                            ptrs = pointers.copy()
                            if pointer_override:
                                i, action = pointer_override
                                ptrs[i] = 0 if action == 'null' else ptrs[i] + 1
                            status = fn(*ptrs, *shape, dtype_code if dtype_override is None else dtype_override,
                                        128 if block_override is None else block_override,
                                        0 if spec_override is None else spec_override, stream.cuda_stream)
                            stream.synchronize()
                            entry = {'case': name, 'dtype': dtype_name, 'cuda_status': status,
                                     'storage_unchanged': bool(torch.equal(storage.view(bit_dtype), before))}
                            entry['passed'] = status == INVALID_VALUE and entry['storage_unchanged']
                            report['invalid_argument_checks'].append(entry)
                            if not entry['passed']:
                                raise AssertionError(f'DCN invalid-argument check failed: {entry}')
                    report.pop('current_check', None)
                    write_report(archive, report)
                    print(json.dumps({'case': case.name, 'dtype': dtype_name, 'status': 'passed'}), flush=True)
            report['status'] = 'passed'
            report['counts'] = {name: len(report[name]) for name in ('cases', 'checks', 'oracle_checks', 'invalid_argument_checks')}
        except Exception as error:
            report['status'] = 'failed'
            report['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
            raise
        finally:
            report['finished_utc'] = datetime.now(timezone.utc).isoformat()
            write_report(archive, report)
    print(json.dumps({'status': report['status'], 'counts': report['counts'], 'report': str(args.report.resolve())}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--report', '--output', dest='report', type=Path, required=True,
                        help='New JSON archive; existing files are rejected')
    parser.add_argument('--seed', type=int, default=20261021)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--blocks', type=int, nargs='+', choices=BLOCKS, default=list(BLOCKS))
    args = parser.parse_args()
    if args.seed < 0 or args.device < 0 or len(set(args.blocks)) != len(args.blocks):
        parser.error('seed/device must be nonnegative and block sizes must be unique')
    run(args)


if __name__ == '__main__':
    main()

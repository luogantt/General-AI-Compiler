#!/usr/bin/env python3
"""Check and tune GAC MSDA against NVIDIA's TensorRT plugin on the same inputs.

This measures a single operator, never whole-model FPS. Optional captured NPZ
inputs must contain value, shapes, starts, locations, weights. Random fixtures
are explicitly labelled. Timing alternates baseline/candidate order.
The numerical reference is always NVIDIA's plugin, even when a GAC configuration
is selected as the performance baseline.
"""
import argparse
import ctypes
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import subprocess


OPTION_VALUES = {
    'channels_per_thread': (1, 2, 4),
    'block_threads': (128, 256, 512, 768),
    'specialize': (0, 1, 2, 3),
    'prefetch': (0, 1, 2),
}


def options(cpt=1, block=256, specialize=1, prefetch=0):
    return dict(channels_per_thread=cpt, block_threads=block,
                specialize=specialize, prefetch=prefetch)


def validate_options(value, source='configuration'):
    """Reject coercion, missing/extra fields and unsupported values before CUDA init."""
    if not isinstance(value, dict) or set(value) != set(OPTION_VALUES):
        raise ValueError(f'{source}: expected exactly {list(OPTION_VALUES)}')
    for name, allowed in OPTION_VALUES.items():
        if type(value[name]) is not int or value[name] not in allowed:
            raise ValueError(f'{source}: {name} must be an integer in {allowed}, got {value[name]!r}')
    return {name: value[name] for name in OPTION_VALUES}


def read_options(path, *, multiple=False):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'{path}: duplicate JSON field {key}')
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=unique_object)
    if not multiple:
        return validate_options(value, str(path))
    if not isinstance(value, list) or not value:
        raise ValueError(f'{path}: --configs must contain a nonempty JSON list')
    return [validate_options(item, f'{path}[{index}]') for index, item in enumerate(value)]


def candidate_options(args):
    if args.configs:
        return read_options(args.configs, multiple=True)
    if args.config:
        return [read_options(args.config)]
    grid = ((1, 2, 4), (128, 256, 768), (1,), (0, 1)) if args.quick else (
        (1, 2, 4), (128, 256, 512, 768), (0, 1), (0, 1))
    return [options(*values) for values in itertools.product(*grid)]


def config_sha256(config):
    canonical = json.dumps(config, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(canonical).hexdigest()


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--inputs', type=Path)
    p.add_argument('--dtype', choices=('fp16', 'fp32'), default='fp16')
    p.add_argument('--repeat', type=int, default=20)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--checks-only', action='store_true')
    p.add_argument('--quick', action='store_true')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--config', type=Path, help='Retest one options JSON from a previous search')
    group.add_argument('--configs', type=Path, help='Search only this nonempty JSON list of options objects')
    p.add_argument('--baseline-config', type=Path,
                   help='Time against this GAC options JSON; NVIDIA remains the numerical reference')
    return p


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def device_state():
    r = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,temperature.gpu,clocks.sm,clocks.mem,memory.used,utilization.gpu,power.draw',
                        '--format=csv,noheader'], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else r.stderr.strip()


def builtin(arrays, trt, torch, logger, stream, custom=False, options=None):
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    keys = ('value', 'shapes', 'starts', 'locations', 'weights')
    inputs = []
    for key in keys:
        a = arrays[key]
        dtype = trt.int32 if a.dtype == torch.int32 else (trt.float16 if a.dtype == torch.float16 else trt.float32)
        inputs.append(network.add_input(key, dtype, tuple(a.shape)))
    name, version = ('GACMSDeformAttn', '1') if custom else ('MultiscaleDeformableAttnPlugin_TRT', '2')
    creator = trt.get_plugin_registry().get_creator(name, version, '')
    if creator is None:
        raise RuntimeError(f'Missing plugin {name} v{version}')
    import numpy as np
    backing = {k: np.array([v], np.int32) for k, v in (options or {}).items()}
    fields = trt.PluginFieldCollection([trt.PluginField(k, v, trt.PluginFieldType.INT32) for k, v in backing.items()])
    plugin = creator.create_plugin('msda', fields, trt.TensorRTPhase.BUILD)
    layer = network.add_plugin_v3(inputs, [], plugin)
    layer.get_output(0).name = 'output'
    network.mark_output(layer.get_output(0))
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 20)
    config.max_aux_streams = 0
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError('Single-MSDA engine build failed')
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    for key in keys:
        expected_type = trt.int32 if key in ('shapes', 'starts') else inputs[0].dtype
        if engine.get_tensor_dtype(key) != expected_type:
            raise RuntimeError(f'Unexpected {key} binding dtype: {engine.get_tensor_dtype(key)}')
    if engine.get_tensor_dtype('output') != inputs[0].dtype:
        raise RuntimeError(f'Unexpected output binding dtype: {engine.get_tensor_dtype("output")}')
    context = engine.create_execution_context()
    output = torch.empty(tuple(engine.get_tensor_shape('output')), device='cuda', dtype=arrays['value'].dtype)
    for key in keys:
        if not context.set_tensor_address(key, arrays[key].data_ptr()):
            raise RuntimeError(f'Cannot bind {key}')
    context.set_tensor_address('output', output.data_ptr())
    def execute():
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError('TensorRT enqueue failed')
    return execute, output, dict(runtime=runtime, engine=engine, context=context, plugin=plugin,
                                 engine_sha256=hashlib.sha256(bytes(plan)).hexdigest())


def fixture(torch, b=2, h=2, c=5, levels=((3, 5), (2, 2), (1, 1)), q=7, p=3):
    shapes = torch.tensor(levels, dtype=torch.int32, device='cuda')
    sizes = [a * z for a, z in levels]
    starts = torch.tensor([sum(sizes[:i]) for i in range(len(sizes))], dtype=torch.int32, device='cuda')
    result = dict(value=torch.randn((b, sum(sizes), h, c), device='cuda'), shapes=shapes, starts=starts,
                locations=torch.rand((b, q, h, len(levels), p, 2), device='cuda') * 1.5 - .25,
                weights=torch.softmax(torch.randn((b, q, h, len(levels) * p), device='cuda'), -1).reshape(b, q, h, len(levels), p))
    # Include exact image corners and fully masked points, not only random positions.
    boundary=torch.tensor(((0,0),(1,1),(0,1),(1,0),(-1,-1),(2,2),(.5,.5)),device='cuda')
    flat=result['locations'].reshape(-1,2)
    count=min(len(flat),len(boundary));flat[:count]=boundary[:count]
    return result


def oracle(arrays):
    """Small CPU reference, including TensorRT 10.13's observed half FMA order."""
    import numpy as np
    a = {k: v.cpu().numpy() for k, v in arrays.items()}
    value, loc, weights = a['value'], a['locations'], a['weights']
    bsz, _, heads, channels = value.shape
    _, queries, _, levels, points, _ = loc.shape
    dt = value.dtype.type
    def integer(x):
        rounded=dt(x)
        if value.dtype == np.float16 and float(rounded) > x:
            rounded=np.nextafter(rounded,dt(-np.inf))
        return rounded
    def hfma(x, y, z):
        return np.asarray(np.asarray(x, dtype=np.float64) * np.asarray(y, dtype=np.float64)
                          + np.asarray(z, dtype=np.float64), dtype=np.float16)
    result = np.zeros((bsz, queries, heads, channels), dtype=value.dtype)
    for b, q, h in itertools.product(range(bsz), range(queries), range(heads)):
        acc = np.zeros(channels, dtype=value.dtype)
        for l, p in itertools.product(range(levels), range(points)):
            height, width = map(int, a['shapes'][l]); start = int(a['starts'][l])
            if value.dtype == np.float16:
                y = hfma(loc[b,q,h,l,p,1], integer(height), dt(-.5))
                x = hfma(loc[b,q,h,l,p,0], integer(width), dt(-.5))
            else:
                y = dt(dt(loc[b,q,h,l,p,1] * integer(height)) - dt(.5))
                x = dt(dt(loc[b,q,h,l,p,0] * integer(width)) - dt(.5))
            if not (-1 < y < integer(height) and -1 < x < integer(width)):
                continue
            yl, xl = int(np.floor(y)), int(np.floor(x))
            ly, lx = dt(y - integer(yl)), dt(x - integer(xl)); hy, hx = dt(1-ly), dt(1-lx)
            vs, raw, coefficients = [], [], []
            for yy, xx, wy, wx in ((yl,xl,hy,hx),(yl,xl+1,hy,lx),(yl+1,xl,ly,hx),(yl+1,xl+1,ly,lx)):
                v = value[b,start+yy*width+xx,h] if 0 <= yy < height and 0 <= xx < width else np.zeros(channels,dtype=value.dtype)
                raw.append(v); coefficients.append(dt(wy*wx))
                vs.append(np.asarray(dt(wy*wx) * v, dtype=value.dtype))
            if value.dtype == np.float16:
                sampled = np.asarray(hfma(coefficients[0],raw[0],vs[1]) + hfma(coefficients[2],raw[2],vs[3]),dtype=value.dtype)
                acc = hfma(sampled,weights[b,q,h,l,p],acc)
            else:
                sampled = ((vs[0]+vs[1])+vs[2])+vs[3]
                acc = np.asarray(acc + np.asarray(sampled*weights[b,q,h,l,p],dtype=value.dtype),dtype=value.dtype)
        result[b,q,h] = acc
    return result


def metrics(actual, expected, torch):
    a, b = actual.float(), expected.float()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    exact=actual.dtype==expected.dtype and torch.equal(actual.contiguous().view(torch.uint8),expected.contiguous().view(torch.uint8))
    return dict(finite=finite, exact=bool(exact),
                max_abs=float((a-b).abs().max()) if finite else None,
                relative_rmse=float(torch.linalg.vector_norm(a-b) / torch.linalg.vector_norm(b).clamp_min(1e-30)) if finite else None,
                allclose=bool(torch.allclose(a,b,rtol=1e-5,atol=2e-6)))


def accepted(comparison, *, exact=False):
    """Same-signed infinities can compare equal, but never pass validation."""
    return comparison['finite'] and comparison['allclose'] and (not exact or comparison['exact'])


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.report.exists():
        raise FileExistsError(args.report)
    if args.repeat < 2 or args.warmup < 0:
        raise ValueError('repeat >=2 and warmup >=0 required')
    configs = candidate_options(args)
    baseline_config = read_options(args.baseline_config) if args.baseline_config else None
    import numpy as np
    import torch
    import tensorrt as trt
    torch.manual_seed(20261021)
    torch.cuda.set_device(0)
    lib = ctypes.CDLL(str(args.library.resolve()), mode=ctypes.RTLD_GLOBAL)
    forward = lib.gac_msda_forward
    forward.argtypes = [ctypes.c_void_p]*6 + [ctypes.c_int64]*7 + [ctypes.c_int]*5 + [ctypes.c_void_p]
    forward.restype = ctypes.c_int
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, '')
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    report = dict(schema='gac.msda.benchmark.v1',script_sha256=sha(__file__),
                  scope='Single-MSDA CUDA Graph replay; not whole-model FPS', library_sha256=sha(args.library),
                  environment=dict(torch=torch.__version__,tensorrt=trt.__version__,cuda=torch.version.cuda,
                                   device=torch.cuda.get_device_name(),before=device_state()),
                  checks=[], oracle_checks=[], candidates=[], dtype=args.dtype, warmup=args.warmup,repeat=args.repeat)
    report['requested_candidates'] = configs.copy()
    report['candidate_config_source'] = (
        dict(path=str((args.configs or args.config).resolve()), sha256=sha(args.configs or args.config))
        if args.configs or args.config else dict(default_grid='quick18' if args.quick else 'full48'))
    report['performance_baseline'] = dict(
        type='gac_cuda_kernel' if baseline_config else 'nvidia_tensorrt_plugin',
        options=baseline_config,
        options_sha256=config_sha256(baseline_config) if baseline_config else None,
        config_source=(dict(path=str(args.baseline_config.resolve()), sha256=sha(args.baseline_config))
                       if args.baseline_config else None),
        library_sha256=sha(args.library) if baseline_config else None)
    report['numerical_reference'] = dict(type='nvidia_tensorrt_plugin',
                                         name='MultiscaleDeformableAttnPlugin_TRT', version='2')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def candidate(a, options, out=None):
        b, s, h, c = a['value'].shape
        _, q, _, l, points, _ = a['locations'].shape
        if out is None:
            out = torch.empty((b,q,h,c),device='cuda',dtype=a['value'].dtype)
        def execute():
            code = forward(*[a[k].data_ptr() for k in ('value','shapes','starts','locations','weights')],out.data_ptr(),
                           b,s,h,c,l,q,points,int(out.dtype==torch.float16),options['channels_per_thread'],
                           options['block_threads'],options['specialize'],options['prefetch'],stream.cuda_stream)
            if code:
                raise RuntimeError(f'gac_msda_forward returned CUDA error {code}: {options}')
        return execute,out
    for case in (dict(c=5),dict(b=1,h=1,c=1,q=3,p=1),dict(b=1,h=8,c=32,levels=((7,9),(3,5),(2,2),(1,1)),q=11,p=8)):
        for dtype in (torch.float32,torch.float16):
            a=fixture(torch,**case)
            a={k:v.to(dtype) if v.is_floating_point() else v for k,v in a.items()}
            baseline, expected, keep = builtin(a,trt,torch,logger,stream)
            baseline();stream.synchronize()
            ref=torch.from_numpy(oracle(a)).cuda()
            oracle_check=metrics(expected,ref,torch)
            report['oracle_checks'].append(dict(shape=list(a['value'].shape),dtype=str(dtype),comparison=oracle_check))
            if not accepted(oracle_check):
                diagnostic=args.report.with_suffix('.oracle-failure.npz')
                np.savez(diagnostic, **{k:v.cpu().numpy() for k,v in a.items()},
                         expected=expected.cpu().numpy(), oracle=ref.cpu().numpy())
                report.update(status='oracle_failed',oracle_comparison=oracle_check,diagnostic=str(diagnostic))
                save()
                raise AssertionError(f'Independent oracle disagrees with NVIDIA: {case}, {dtype}, {oracle_check}; '
                                     f'output={expected.flatten()[:16].tolist()}; reference={ref.flatten()[:16].tolist()}')
            # Three shapes exercise C32/L4/P8 specialization and its generic/odd-C fallbacks.
            for cpt, block, spec, pre in itertools.product((1,2,4),(128,768),(0,1,3),(0,1,2)):
                opt=options(cpt,block,spec,pre); run,out=candidate(a,opt);run();stream.synchronize()
                check=metrics(out,expected,torch)
                row=dict(shape=list(a['value'].shape),dtype=str(dtype),options=opt,comparison=check)
                report['checks'].append(row)
                if not accepted(check,exact=dtype==torch.float16):
                    save();raise AssertionError(row)
            for opt in (options(2), options(4,128,3,0), options(4,128,3,2)):
                custom,out,custom_keep=builtin(a,trt,torch,logger,stream,True,opt)
                custom();stream.synchronize()
                check=metrics(out,expected,torch)
                row=dict(kind='serialized_plugin',shape=list(a['value'].shape),dtype=str(dtype),
                         options=opt,comparison=check,engine_sha256=custom_keep['engine_sha256'])
                report['checks'].append(row)
                if not accepted(check,exact=dtype==torch.float16):
                    save();raise AssertionError(f'Custom plugin mismatch: {row}')
                del custom_keep,custom,out
            del keep,baseline,a,ref,expected
    save()
    if args.checks_only:
        report['status']='checks_passed';save();return
    dtype=torch.float16 if args.dtype=='fp16' else torch.float32
    captured_expected=None
    if args.inputs:
        with np.load(args.inputs,allow_pickle=False) as f:
            a={k:torch.from_numpy(np.array(f[k],copy=True)).cuda() for k in ('value','shapes','starts','locations','weights')}
            if 'expected' in f and a['value'].dtype == dtype:
                captured_expected=torch.from_numpy(np.array(f['expected'],copy=True)).cuda()
        report['inputs']=dict(path=str(args.inputs),sha256=sha(args.inputs),scope='Captured tensors; see capture manifest')
        manifest=args.inputs.with_name('capture.report.json')
        if manifest.is_file():
            report['inputs']['manifest']=json.loads(manifest.read_text())
        report['inputs']['floating_cast']=a['value'].dtype != dtype
    else:
        a=fixture(torch,b=6,h=8,c=32,levels=((116,200),(58,100),(29,50),(15,25)),q=40000,p=8)
        report['inputs']=dict(scope='Synthetic random tensors at BEVFormer spatial MSDA shape; not real model activations')
    a={k:v.to(dtype).contiguous() if v.is_floating_point() else v.to(torch.int32).contiguous() for k,v in a.items()}
    b,s,h,c=a['value'].shape
    locshape=tuple(a['locations'].shape)
    if len(locshape)!=6 or locshape[0]!=b or locshape[2]!=h or locshape[-1]!=2:
        raise ValueError('Invalid sampling_locations dimensions')
    l=locshape[3]
    if tuple(a['weights'].shape)!=locshape[:-1] or tuple(a['shapes'].shape)!=(l,2) or tuple(a['starts'].shape)!=(l,):
        raise ValueError('Inconsistent MSDA tensor dimensions')
    for (height,width),start in zip(a['shapes'].cpu().tolist(),a['starts'].cpu().tolist()):
        if min(height,width)<=0 or start<0 or start+height*width>s:
            raise ValueError('Invalid level metadata')
    if any(not bool(torch.isfinite(v).all()) for v in a.values() if v.is_floating_point()):
        raise ValueError('Nonfinite captured inputs are not eligible for performance tuning')
    report['shapes']={k:list(v.shape) for k,v in a.items()}
    official,official_output,official_keep=builtin(a,trt,torch,logger,stream)
    official();stream.synchronize()
    gold=official_output.clone()
    report['numerical_reference']['engine_sha256']=official_keep['engine_sha256']
    if captured_expected is not None:
        captured_check=metrics(gold,captured_expected,torch)
        report['captured_reference_comparison']=captured_check
        if not accepted(captured_check,exact=True):
            save();raise AssertionError(f'Single-MSDA baseline differs from captured official output: {captured_check}')
        del captured_expected
    # A faster GAC baseline changes only the timing denominator, never the oracle.
    if baseline_config:
        baseline,baseline_output=candidate(a,baseline_config)
        baseline();stream.synchronize()
        baseline_check=metrics(baseline_output,gold,torch)
        report['performance_baseline']['comparison_to_nvidia']=baseline_check
        if not accepted(baseline_check,exact=dtype==torch.float16):
            report['status']='baseline_validation_failed';save()
            raise AssertionError(f'GAC timing baseline differs from NVIDIA gold: {baseline_check}')
    else:
        baseline,baseline_output=official,official_output
        report['performance_baseline']['engine_sha256']=official_keep['engine_sha256']
    output=torch.empty_like(gold)
    # Deterministic shuffled order avoids always testing one template last/hottest.
    np.random.default_rng(20261021).shuffle(configs)
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    def time_one(fn):
        begin.record(stream);fn();end.record(stream);end.synchronize()
        return begin.elapsed_time(end)
    for _ in range(3):baseline()
    stream.synchronize()
    base_graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(base_graph,stream=stream): baseline()
    baseline_gold=baseline_output.clone()
    baseline_output.fill_(float('nan'));base_graph.replay();stream.synchronize()
    baseline_replay_check=metrics(baseline_output,baseline_gold,torch)
    report['performance_baseline']['graph_comparison']=baseline_replay_check
    if not accepted(baseline_replay_check,exact=True):
        report['status']='baseline_graph_validation_failed';save()
        raise AssertionError('Captured baseline graph did not reproduce its output')
    for index,opt in enumerate(configs):
        run,out=candidate(a,opt,output)
        run();stream.synchronize();check=metrics(out,gold,torch)
        row=dict(options=opt,comparison=check,state_before=device_state())
        report['candidates'].append(row)
        if not accepted(check,exact=dtype==torch.float16):
            row['eligible']=False;save();continue
        for _ in range(args.warmup):baseline();run()
        stream.synchronize();bt,ct=[],[]
        candidate_graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(candidate_graph,stream=stream): run()
        out.fill_(float('nan'));candidate_graph.replay();stream.synchronize()
        replay_check=metrics(out,gold,torch)
        row['graph_comparison']=replay_check
        if not accepted(replay_check,exact=dtype==torch.float16):
            row['eligible']=False;save();continue
        for _ in range(args.warmup):base_graph.replay();candidate_graph.replay()
        stream.synchronize()
        for i in range(args.repeat):
            if i%2:ct.append(time_one(candidate_graph.replay));bt.append(time_one(base_graph.replay))
            else:bt.append(time_one(base_graph.replay));ct.append(time_one(candidate_graph.replay))
        row.update(eligible=True,baseline_ms=bt,candidate_ms=ct,baseline_median_ms=statistics.median(bt),
                   candidate_median_ms=statistics.median(ct),paired_median_speedup=statistics.median([x/y for x,y in zip(bt,ct)]),
                   state_after=device_state())
        print(index,opt,row['candidate_median_ms'],row['paired_median_speedup'],flush=True);save()
    eligible=[r for r in report['candidates'] if r.get('eligible')]
    if not eligible:raise RuntimeError('No numerically valid candidate')
    best=max(eligible,key=lambda r:r['paired_median_speedup'])
    report.update(status='measured',selection=best['options'],selection_observed_speedup=best['paired_median_speedup'],
                  selection_note='Search result; independent rerun and whole-model validation required',environment_after=device_state())
    save()


if __name__=='__main__':
    main()

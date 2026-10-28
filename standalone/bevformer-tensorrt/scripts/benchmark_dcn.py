#!/usr/bin/env python3
"""Measure DCNv2 im2col candidates separately from GEMM, using captured inputs.

This is a standalone operator experiment, not a TensorRT replacement plugin or
whole-model speed claim. The installed NVIDIA plugin is the output reference.
"""
import argparse
import ctypes as ct
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bevformer_trt import engine as engine_tools
from benchmark_msda import device_state, sha


def checked(status, operation):
    if status:
        raise RuntimeError(f'{operation} failed with status {status}')


def attributes(record):
    a = record['selected_node']['attributes']
    expected = {'stride', 'padding', 'dilation', 'group', 'deformable_group'}
    if set(a) != expected:
        raise ValueError(f'Expected DCN attributes {expected}, got {set(a)}')
    for name in ('stride', 'padding', 'dilation'):
        if not isinstance(a[name], list) or len(a[name]) != 2:
            raise ValueError(f'{name} must be [width, height]')
        if any(type(v) is not int or v < (0 if name == 'padding' else 1) for v in a[name]):
            raise ValueError(f'Invalid {name}')
    for name in ('group', 'deformable_group'):
        if type(a[name]) is not int or a[name] <= 0:
            raise ValueError(f'Invalid {name}')
    return a


def build_official(arrays, attrs, torch, trt, stream, custom=None):
    import numpy as np
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, '')
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    keys = ['input', 'offset', 'mask', 'weight'] + (['bias'] if 'bias' in arrays else [])
    tensors = [network.add_input(k, trt.float16, tuple(arrays[k].shape)) for k in keys]
    backing = {k: np.array(v, np.int32).reshape(-1) for k, v in {**attrs,**(custom or {})}.items()}
    fields = trt.PluginFieldCollection([
        trt.PluginField(k, v, trt.PluginFieldType.INT32) for k, v in backing.items()])
    name, version = ('GACModulatedDeformConv2d','1') if custom else ('ModulatedDeformConv2d','2')
    creator = trt.get_plugin_registry().get_creator(name, version, '')
    if creator is None:
        raise RuntimeError('NVIDIA ModulatedDeformConv2d v2 unavailable')
    plugin = creator.create_plugin('dcn', fields, trt.TensorRTPhase.BUILD)
    if plugin is None:
        raise RuntimeError('Cannot create official DCN plugin')
    layer = network.add_plugin_v3(tensors, [], plugin)
    layer.get_output(0).name = 'output'
    network.mark_output(layer.get_output(0))
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.max_aux_streams = 0
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError('Official single-DCN engine build failed')
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    output = torch.empty(tuple(engine.get_tensor_shape('output')), dtype=torch.float16, device='cuda')
    for k in keys + ['output']:
        if engine.get_tensor_dtype(k) != trt.float16 or engine.get_tensor_format(k) != trt.TensorFormat.LINEAR:
            raise ValueError(f'Unexpected binding format: {k}')
        if not context.set_tensor_address(k, (output if k == 'output' else arrays[k]).data_ptr()):
            raise RuntimeError(f'Cannot bind {k}')
    def execute():
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError('Official DCN enqueue failed')
    return execute, output, (runtime, engine, context, plugin, logger)


class Hgemm:
    """cuBLAS legacy HGEMM, matching the official DCNv2 wrapper's contract."""
    def __init__(self, stream):
        self.lib = ct.CDLL('libcublas.so.12')
        self.handle = ct.c_void_p()
        self.lib.cublasCreate_v2.argtypes = [ct.POINTER(ct.c_void_p)]
        self.lib.cublasSetStream_v2.argtypes = [ct.c_void_p, ct.c_void_p]
        self.lib.cublasDestroy_v2.argtypes = [ct.c_void_p]
        self.lib.cublasSetMathMode.argtypes = [ct.c_void_p, ct.c_int]
        self.lib.cublasHgemm.argtypes = [ct.c_void_p] + [ct.c_int] * 5 + [ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_int]
        checked(self.lib.cublasCreate_v2(ct.byref(self.handle)), 'cublasCreate')
        checked(self.lib.cublasSetStream_v2(self.handle, stream.cuda_stream), 'cublasSetStream')
        self.alpha, self.beta = ct.c_uint16(0x3c00), ct.c_uint16(0)

    def run(self, columns, weight, output, groups):
        m, k = weight.shape[0] // groups, weight[0].numel()
        n = output.shape[-1] * output.shape[-2]
        for g in range(groups):
            checked(self.lib.cublasHgemm(self.handle, 0, 0, n, m, k,
                ct.byref(self.alpha), columns.data_ptr() + g*k*n*2, n,
                weight.data_ptr() + g*m*k*2, k, ct.byref(self.beta),
                output.data_ptr() + g*m*n*2, n), 'cublasHgemm')

    def close(self):
        checked(self.lib.cublasDestroy_v2(self.handle), 'cublasDestroy')


def comparison(output, reference, torch):
    if not bool(torch.isfinite(output).all()) or not bool(torch.isfinite(reference).all()):
        raise ValueError('Nonfinite DCN output')
    diff = (output.float() - reference.float()).abs()
    return {'bitwise_equal': bool(torch.equal(output.view(torch.int16), reference.view(torch.int16))),
            'max_abs': float(diff.max()), 'mean_abs': float(diff.mean()),
            'allclose_rtol_1e-3_atol_1e-3': bool(torch.allclose(output, reference, rtol=1e-3, atol=1e-3))}


def timed(fn, stream, torch, launches):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record(stream)
    for _ in range(launches):
        fn()
    end.record(stream)
    end.synchronize()
    return start.elapsed_time(end) / launches


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--repeat', type=int, default=40)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--launches', type=int, default=20)
    p.add_argument('--cuda-graph', action='store_true', help='Capture repeated single-op calls to exclude Python launch gaps; fails on capture errors')
    p.add_argument('--specializations', type=int, nargs='+', choices=range(5), default=[0,1])
    p.add_argument('--plugin-library', type=Path, help='Also validate and time the integrated TensorRT DCN plugin (block128/spec4)')
    args = p.parse_args()
    if args.repeat < 2 or args.warmup < 1 or args.launches < 1:
        p.error('repeat>=2, warmup>=1, launches>=1 required')
    if args.report.exists():
        raise FileExistsError(args.report)
    record = json.loads(args.metadata.read_text())
    attrs = attributes(record)
    if record.get('archive_sha256') != sha(args.inputs):
        raise ValueError('Input archive does not match capture metadata SHA-256')
    import numpy as np
    engine_tools.load_cuda_math_libraries()
    if args.plugin_library:
        engine_tools.load_plugin_libraries([args.plugin_library])
    import torch
    import tensorrt as trt
    torch.cuda.set_device(0)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    with np.load(args.inputs, allow_pickle=False) as saved:
        cpu_arrays = {k: np.ascontiguousarray(saved[k]) for k in saved.files}
    from capture_dcn import validate_arrays
    validate_arrays(cpu_arrays, np, record['selected_node'])
    arrays = {k: torch.from_numpy(v).cuda() for k,v in cpu_arrays.items()}
    if set(arrays) not in ({'input','offset','mask','weight','expected'}, {'input','offset','mask','weight','bias','expected'}):
        raise ValueError('Unexpected capture keys')
    if any(a.dtype != torch.float16 or not bool(torch.isfinite(a).all()) for a in arrays.values()):
        raise ValueError('This probe requires finite FP16 captured tensors')
    b, c, h, w = arrays['input'].shape
    oc, ci, kh, kw = arrays['weight'].shape
    ho, wo = arrays['expected'].shape[-2:]
    if b != 1 or ci*attrs['group'] != c or oc % attrs['group']:
        raise ValueError('Probe currently requires batch=1 with valid convolution groups')
    if tuple(arrays['offset'].shape) != (b, 2*attrs['deformable_group']*kh*kw, ho, wo):
        raise ValueError('Offset shape mismatch')
    if tuple(arrays['mask'].shape) != (b, attrs['deformable_group']*kh*kw, ho, wo):
        raise ValueError('Mask shape mismatch')
    if tuple(arrays['expected'].shape) != (b, oc, ho, wo):
        raise ValueError('Expected shape mismatch')
    official, reference, owners = build_official(arrays, attrs, torch, trt, stream)
    reference.fill_(float('nan')); official(); stream.synchronize()
    capture_comparison = comparison(reference, arrays['expected'], torch)
    if not capture_comparison['bitwise_equal']:
        raise ValueError(f'Single-plugin reference differs from captured output: {capture_comparison}')
    ref = reference.clone()
    lib = ct.CDLL(str(args.library.resolve()))
    fn = lib.dcn_im2col
    fn.restype = ct.c_int
    columns = torch.empty((c*kh*kw, ho*wo), dtype=torch.float16, device='cuda')
    output = torch.empty_like(ref)
    shape_args = [b,c,h,w,ho,wo,kh,kw,attrs['padding'][1],attrs['padding'][0],
                  attrs['stride'][1],attrs['stride'][0],attrs['dilation'][1],attrs['dilation'][0],attrs['deformable_group']]
    # The ABI has 15 int64 shape arguments. Validate from the installed header.
    fn.argtypes = [ct.c_void_p]*4 + [ct.c_int64]*15 + [ct.c_int]*3 + [ct.c_void_p]
    def im2col(block, specialize):
        checked(fn(arrays['input'].data_ptr(), arrays['offset'].data_ptr(), arrays['mask'].data_ptr(),
                   columns.data_ptr(), *shape_args, 1, block, specialize, stream.cuda_stream), 'dcn_im2col')
    gemm = Hgemm(stream)
    def multiply():
        gemm.run(columns, arrays['weight'], output, attrs['group'])
        if 'bias' in arrays:
            output.add_(arrays['bias'].view(1,-1,1,1))
    def complete(block, specialize):
        im2col(block, specialize); multiply()
    im2col(512,0); stream.synchronize()
    baseline_columns = columns.clone()
    complete(512,0); stream.synchronize()
    baseline_comparison = comparison(output, ref, torch)
    results = []
    before = device_state()
    for spec in args.specializations:
        for block in (128, 256, 512, 1024):
            columns.fill_(float('nan')); output.fill_(float('nan'))
            complete(block,spec); stream.synchronize()
            col_equal = bool(torch.equal(columns.view(torch.int16), baseline_columns.view(torch.int16)))
            numeric = comparison(output, ref, torch)
            if not col_equal:
                raise ValueError(f'Candidate changes im2col values: {block}, {spec}')
            if not numeric['bitwise_equal']:
                result = {'block':block,'specialize':spec,'columns_bitwise_equal':col_equal,
                          'vs_official':numeric,'status':'numerical_mismatch_not_timed'}
                results.append(result)
                print(json.dumps(result),flush=True)
                continue
            for _ in range(args.warmup): official(); complete(block,spec)
            stream.synchronize()
            times = {'official_ms': [], 'complete_ms': [], 'im2col_ms': [], 'gemm_ms': []}
            funcs = {'official_ms': official, 'complete_ms': lambda:complete(block,spec),
                     'im2col_ms': lambda:im2col(block,spec), 'gemm_ms': multiply}
            graphs = {}
            if args.cuda_graph:
                for key, action in funcs.items():
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        for _ in range(args.launches): action()
                    graphs[key] = graph
                stream.synchronize()
                for key, graph in graphs.items():
                    target = reference if key == 'official_ms' else (columns if key == 'im2col_ms' else output)
                    target.fill_(float('nan'))
                    graph.replay()
                    stream.synchronize()
                    expected = baseline_columns if key == 'im2col_ms' else ref
                    if not comparison(target,expected,torch)['bitwise_equal']:
                        raise ValueError(f'CUDA Graph did not reproduce the checked output: {key}')
            for repeat in range(args.repeat):
                keys = list(funcs) if repeat%2 == 0 else list(reversed(funcs))
                for key in keys:
                    value = (timed(graphs[key].replay,stream,torch,1)/args.launches if args.cuda_graph
                             else timed(funcs[key],stream,torch,args.launches))
                    times[key].append(value)
            medians = {key: statistics.median(value) for key,value in times.items()}
            result = {'block':block,'specialize':spec,'columns_bitwise_equal':col_equal,
                      'vs_official':numeric, 'median':medians, 'samples':times}
            results.append(result)
            print(json.dumps({k:v for k,v in result.items() if k != 'samples'}),flush=True)
    plugin_result = None
    if args.plugin_library:
        custom, plugin_output, plugin_owners = build_official(arrays,attrs,torch,trt,stream,
                                                           {'block_threads':128,'specialize':4})
        plugin_output.fill_(float('nan'));custom();stream.synchronize()
        check = comparison(plugin_output,ref,torch)
        if not check['bitwise_equal']:
            raise ValueError(f'Integrated plugin changed numerical results: {check}')
        for _ in range(args.warmup): official();custom()
        stream.synchronize()
        graphs = {}
        for name, action in [('official',official),('custom',custom)]:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g,stream=stream):
                for _ in range(args.launches): action()
            target = reference if name == 'official' else plugin_output
            target.fill_(float('nan'));g.replay();stream.synchronize()
            if not comparison(target,ref,torch)['bitwise_equal']:
                raise ValueError(f'Integrated {name} Graph replay validation failed')
            graphs[name] = g
        times = {'official':[],'custom':[]}
        for repeat in range(args.repeat):
            for name in (('official','custom') if repeat%2 == 0 else ('custom','official')):
                times[name].append(timed(graphs[name].replay,stream,torch,1)/args.launches)
        plugin_result = {'vs_official':check,'config':{'block_threads':128,'specialize':4},
                         'library_sha256':sha(args.plugin_library),'cuda_graph':True,
                         'samples':times,'median_ms':{k:statistics.median(v) for k,v in times.items()}}
        print('INTEGRATED_PLUGIN '+json.dumps(plugin_result['median_ms']),flush=True)
    gemm.close()
    passed = all(row['vs_official']['bitwise_equal'] for row in results)
    report = {'schema':'gac.dcn.operator_probe.v1', 'status':'passed' if passed else 'numerical_mismatch',
              'scope':'Standalone real-input DCNv2; not whole-model latency. Repeated CUDA Graph replay when selected, otherwise event loops include launch gaps. Separately measured components are not an additive attribution.',
              'inputs_sha256':sha(args.inputs),'metadata_sha256':sha(args.metadata),'library_sha256':sha(args.library),
              'attributes':attrs,'shapes':{k:list(v.shape) for k,v in arrays.items()},
              'capture_vs_official':capture_comparison,'generic_vs_official':baseline_comparison,
              'device_before':before,'device_after':device_state(),'repeat':args.repeat,
              'launches_per_sample':args.launches,'cuda_graph':args.cuda_graph,'results':results,
              'integrated_plugin':plugin_result}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with args.report.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    if not passed:
        raise RuntimeError('One or more DCN candidates failed bitwise comparison; diagnostic report saved')


if __name__ == '__main__':
    main()

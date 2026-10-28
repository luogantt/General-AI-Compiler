#!/usr/bin/env python3
"""Validate two live TensorRT contexts on different streams and different inputs."""
import argparse
import json
from pathlib import Path

from benchmark_dcn import attributes, build_official, comparison, engine_tools, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    record = json.loads(args.metadata.read_text())
    if record['archive_sha256'] != sha(args.inputs):
        raise ValueError('Capture metadata/archive mismatch')
    attrs = attributes(record)
    import numpy as np
    from capture_dcn import validate_arrays
    with np.load(args.inputs,allow_pickle=False) as saved:
        cpu = {k:np.ascontiguousarray(saved[k]) for k in saved.files}
    validate_arrays(cpu,np,record['selected_node'])
    engine_tools.load_cuda_math_libraries()
    engine_tools.load_plugin_libraries([args.library])
    import torch
    import tensorrt as trt
    first = {k:torch.from_numpy(v).cuda() for k,v in cpu.items()}
    second = {k:v.clone() for k,v in first.items()}
    second['input'].mul_(-0.25)  # Distinct values detect cross-context state/buffer leaks.
    torch.cuda.synchronize()
    s1,s2 = torch.cuda.Stream(),torch.cuda.Stream()
    official1,reference1,owner1 = build_official(first,attrs,torch,trt,s1)
    official2,reference2,owner2 = build_official(second,attrs,torch,trt,s2)
    official1();official2();s1.synchronize();s2.synchronize()
    execute1,out1,owners = build_official(first,attrs,torch,trt,s1,
                                         {'block_threads':128,'specialize':4})
    engine = owners[1]
    context2 = engine.create_execution_context()
    if context2 is None:
        raise RuntimeError('Second custom execution context allocation failed')
    out2 = torch.empty_like(out1)
    for key in ('input','offset','mask','weight','output'):
        if not context2.set_tensor_address(key,(out2 if key=='output' else second[key]).data_ptr()):
            raise RuntimeError(f'Cannot bind context2/{key}')
    checks=[]
    for iteration in range(10):
        with torch.cuda.stream(s1): out1.fill_(float('nan'))
        with torch.cuda.stream(s2): out2.fill_(float('nan'))
        if iteration%2: execute1()
        if not context2.execute_async_v3(s2.cuda_stream):
            raise RuntimeError('Second context enqueue failed')
        if not iteration%2: execute1()
        s1.synchronize();s2.synchronize()
        row = {'iteration':iteration,'context1':comparison(out1,reference1,torch),
               'context2':comparison(out2,reference2,torch)}
        if not row['context1']['bitwise_equal'] or not row['context2']['bitwise_equal']:
            raise ValueError(f'Multi-context numerical mismatch: {row}')
        checks.append(row)
    result={'schema':'gac.dcn.multi_context.v1','status':'passed','contexts':2,'streams':2,
            'input_variants':'captured and input multiplied by -0.25; weights/offset/mask unchanged',
            'reference':'NVIDIA DCNv2 v2 independently evaluated for each input',
            'library_sha256':sha(args.library),'inputs_sha256':sha(args.inputs),'checks':checks}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with args.report.open('x') as f: json.dump(result,f,indent=2,allow_nan=False)
    print('TWO_CONTEXTS_BITWISE_PASSED')


if __name__=='__main__':
    main()

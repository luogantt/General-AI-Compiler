"""Isolated FP32 partial-RoPE candidates; not whole-model performance."""
import hashlib
import json
import time
import traceback
from pathlib import Path

import torch
import torch_npu
import torchair
from torchair.configs.compiler_config import CompilerConfig

torch.set_num_threads(1)
torch.npu.set_device(0)
torch.manual_seed(42)
torch._dynamo.config.suppress_errors = False

def original(x, c, s):
    rot, tail = x[..., :64], x[..., 64:]
    half = torch.cat((-rot[..., 32:], rot[..., :32]), -1)
    return torch.cat((rot*c + half*s, tail), -1)

def sliced(x, c, s):
    return torch.cat((torch_npu.npu_rotary_mul(x[..., :64], c, s), x[..., 64:]), -1)

def full_matrix(x, c, s, rotation):
    c = torch.cat((c, torch.ones_like(c).repeat(1, 1, 1, 3)), -1)
    s = torch.cat((s, torch.zeros_like(s).repeat(1, 1, 1, 3)), -1)
    return torch_npu.npu_rotary_mul(x, c, s, 'half', rotation)

def main():
    result = {'scope':'FP32 partial RoPE64/head256 microbenchmark only',
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'torch':torch.__version__, 'torch_npu':torch_npu.__version__, 'cases':[]}
    with torch.inference_mode():
        rotation = torch.zeros(256, 256, device='npu', dtype=torch.float32)
        idx = torch.arange(32, device='npu')
        rotation[idx,idx+32] = 1
        rotation[idx+32,idx] = -1
        for heads in (8, 2):
            # Actual upstream values have been cast from BF16 before RoPE.
            x = torch.randn(1, heads, 32, 256, device='npu', dtype=torch.bfloat16).float()
            angle = torch.randn(1,1,32,32,device='npu').repeat(1,1,1,2)
            c, s = angle.cos(), angle.sin()
            for name, fn, args in [('original',original,(x,c,s)), ('sliced',sliced,(x,c,s)),
                                    ('full_matrix',full_matrix,(x,c,s,rotation))]:
                case={'heads':heads,'variant':name}
                try:
                    refs = [original(sign*x,c,s).cpu().clone() for sign in (1,-1)]
                    eager = fn(*args)
                    case['eager_max_abs'] = float((eager.cpu()-refs[0]).abs().max())
                    config = CompilerConfig()
                    config.experimental_config.frozen_parameter=True
                    compiled = torch.compile(fn, backend=torchair.get_npu_backend(compiler_config=config), fullgraph=True, dynamic=False)
                    t=time.perf_counter(); checks=[]
                    for i in (0,1,0,1):
                        vals=((-1 if i else 1)*x,*args[1:])
                        actual=compiled(*vals).cpu().clone()
                        checks.append(float((actual-refs[i]).abs().max()))
                    case['compile_replay_ms']=(time.perf_counter()-t)*1000
                    case['replay_max_abs']=checks
                    for _ in range(5): compiled(*args)
                    torch.npu.synchronize(); t=time.perf_counter()
                    for _ in range(30): compiled(*args)
                    torch.npu.synchronize()
                    case['compiled_mean_ms']=(time.perf_counter()-t)*1000/30
                    case['passed']=max(checks)<1e-4
                except Exception as e:
                    case.update(passed=False,error=str(e),traceback=traceback.format_exc())
                result['cases'].append(case)
                print(json.dumps({k:v for k,v in case.items() if k!='traceback'}),flush=True)
                torch._dynamo.reset()
    Path('/root/work/probe-ae-rotary-npu.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__': main()

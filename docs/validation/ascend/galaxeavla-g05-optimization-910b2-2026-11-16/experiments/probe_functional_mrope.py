"""Real checkpoint configuration, tiny rotary modules, no model weights."""
import json
import hashlib
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace as NS
import torch
import torch_npu
import torchair
from torchair.configs.compiler_config import CompilerConfig
from omegaconf import OmegaConf
from g05.models.g05.qwen35.modules import Qwen3_5TextRotaryEmbedding

sys.path.insert(0, '/root/work')
import g05_functional_mrope as helper

def main():
    report={'scope':'real configuration and rotary modules only; no model weights',
            'cases':[], 'status':'failure',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'helper_sha256':hashlib.sha256(Path(helper.__file__).read_bytes()).hexdigest()}
    try:
        torch.set_num_threads(1)
        torch.npu.set_device(0)
        cfg=OmegaConf.load('/root/work/models/galaxeavla-g05/g05-libero/.hydra/config.yaml')
        arch=cfg.model.model_arch
        rotary={n:Qwen3_5TextRotaryEmbedding(getattr(arch,n),device='npu').eval() for n in ('vlm','action_expert')}
        report['original_section_types']={n:type(r.mrope_section).__name__ for n,r in rotary.items()}
        report['original_sections']={n:list(r.mrope_section) for n,r in rotary.items()}
        mixtures={n:NS(training=False,layers=[NS(self_attn=NS(rotary_emb=r))],layer_types=['full_attention']) for n,r in rotary.items()}
        policy=NS(training=False,model=NS(training=False,**mixtures))
        refs={}; inputs={}
        x=torch.zeros((),device='npu',dtype=torch.float32)
        with torch.inference_mode():
            for n,r in rotary.items():
                length=159 if n=='vlm' else 32
                t=torch.arange(length,device='npu').reshape(1,length)
                a=torch.stack((t,t*3+2,t*2-5))
                b=torch.stack((t*2+1,t.flip(-1),t*5+7))
                inputs[n]=(a,b)
                refs[n]=[tuple(z.cpu().clone() for z in r(x,p)) for p in (a,b)]
            handle=helper.install_functional_mrope(policy,verify=True)
            try:
                for n,r in rotary.items():
                    for backend in ('eager','torchair','torchair_aclgraph'):
                        fn=r.forward
                        if backend!='eager':
                            config=CompilerConfig();config.experimental_config.frozen_parameter=True
                            if backend=='torchair_aclgraph':config.mode='npugraph_ex'
                            fn=torch.compile(fn,backend=torchair.get_npu_backend(compiler_config=config),fullgraph=True,dynamic=False)
                        for index in (0,1,0,1):
                            actual=tuple(z.cpu().clone() for z in fn(x,inputs[n][index]))
                            err=max(float((a-b).abs().max()) for a,b in zip(actual,refs[n][index]))
                            row={'module':n,'backend':backend,'case':index,'max_abs':err}
                            report['cases'].append(row)
                            assert err<=1e-6, row
                        print('PASS',n,backend,flush=True)
                report['metadata']=handle.metadata
            finally:
                handle.restore()
            for n,r in rotary.items():
                for index,p in enumerate(inputs[n]):
                    assert all(torch.equal(a.cpu(),b) for a,b in zip(r(x,p),refs[n][index]))
            report['restored_exact']=True
        report['status']='success'
    except Exception as exc:
        report.update(error=str(exc),traceback=traceback.format_exc())
    Path('/root/work/probe-functional-mrope.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'checks':len(report['cases']),'error':report.get('error')}),flush=True)
    if report['status']!='success':raise SystemExit(1)

if __name__=='__main__':main()

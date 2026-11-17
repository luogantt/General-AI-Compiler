import json,traceback,time
from pathlib import Path
import torch,torch_npu,torchair
from torchair import ge
from torchair.configs.compiler_config import CompilerConfig
torch.set_num_threads(1)
torch.npu.set_device(0)
lib=torch.library.Library('gac_direct_probe','DEF')
lib.define('qk(Tensor x, Tensor w, Tensor c, Tensor s, int stride, float eps) -> Tensor')
@torch.library.register_fake('gac_direct_probe::qk')
def fake(x,w,c,s,stride,eps):return torch.empty((1,8 if stride==512 else 2,32,256),device=x.device,dtype=x.dtype)
@torchair.register_fx_node_ge_converter(torch.ops.gac_direct_probe.qk.default)
def convert(x,w,c,s,stride,eps,meta_outputs=None):
 return ge.custom_op('GacQkNormRope',inputs={'x':x,'weight':w,'cos':c,'sin':s},attrs={'head_stride':ge.attr.Int(stride),'epsilon':ge.attr.Float(eps)},outputs=['y'])
cfg=CompilerConfig()
cfg.experimental_config.frozen_parameter=True
out={'status':'failure','scope':'GE-only initial kernel check, no eager bridge or full model'}
try:
 with torch.inference_mode():
  x=torch.randn(1,32,4096,device='npu').bfloat16()
  w=torch.linspace(-.9,.8,256,device='npu')
  angle=torch.randn(1,1,32,32,device='npu').repeat(1,1,1,2)
  c,s=angle.cos(),angle.sin()
  fn=torch.compile(lambda x,w,c,s:torch.ops.gac_direct_probe.qk(x,w,c,s,512,1e-6),backend=torchair.get_npu_backend(compiler_config=cfg),fullgraph=True,dynamic=False)
  raw=x.view(1,32,8,512)[...,:256].float()
  norm=(raw*torch.rsqrt(raw.square().mean(-1,keepdim=True)+1e-6)*(1+w)).bfloat16().transpose(1,2).float()
  r=norm[...,:64]; ref=torch.cat((r*c+torch.cat((-r[...,32:],r[...,:32]),-1)*s,norm[...,64:]),-1).bfloat16().float().cpu()
  act=fn(x,w,c,s).float().cpu()
  err=act-ref
  out.update(status='completed',max_abs=float(err.abs().max()),relative_l2=float(err.norm()/ref.norm()),finite=bool(torch.isfinite(act).all()))
except Exception as e:out.update(error=str(e),traceback=traceback.format_exc())
Path('/root/work/qknorm-ge-direct.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out),flush=True)

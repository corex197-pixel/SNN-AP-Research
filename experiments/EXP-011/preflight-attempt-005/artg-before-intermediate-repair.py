"""Independent ARTG implementations. G0 keeps the original baseline neuron."""
import torch
from models.submodules.layers import ATLIF,zif_backward

# Only enabled for designated diagnostic batches; stores scalar summaries, never activations.
DIAGNOSTICS={'enabled':False,'scale':1.,'rows':[]}
class Eq13(torch.autograd.Function):
 @staticmethod
 def forward(ctx,mem,theta,sp):
  s=(mem>=theta).float();updates=(sp*zif_backward(mem-theta,theta)*s).sum(0).mean().item();ctx.save_for_backward(mem,theta);return s*theta,updates
 @staticmethod
 def backward(ctx,q,dummy):
  mem,theta=ctx.saved_tensors;g=(1-((mem-theta)/theta).abs()).clamp_min(0);return q*g,-(q*g).mean().reshape_as(theta),None
class Artg(torch.autograd.Function):
 @staticmethod
 def forward(ctx,mem,theta,sp,name):
  s=(mem>=theta).float();updates=(sp*zif_backward(mem-theta,theta)*s).sum(0).mean().item();ctx.save_for_backward(mem,theta,s);ctx.name=name;ctx.set_materialize_grads(False);return s*theta,s,updates
 @staticmethod
 def backward(ctx,a,b,dummy):
  mem,theta,s=ctx.saved_tensors
  if a is None:a=torch.zeros_like(mem)
  if b is None:b=torch.zeros_like(mem)
  compute_dtype=torch.promote_types(mem.dtype,theta.dtype);a=a.to(compute_dtype);b=b.to(compute_dtype)
  r=mem/theta;g=(1-((mem-theta)/theta).abs()).clamp_min(0);v1=a*s;v2=-a*r*g;v3=-b*r*g/theta
  if DIAGNOSTICS['enabled']:
   scale=DIAGNOSTICS['scale'];DIAGNOSTICS['rows'].append({'module':ctx.name,'direct_mean':(v1/scale).mean().item(),'state_mean':(v2/scale).mean().item(),'reset_mean':(v3/scale).mean().item(),'direct_abs_mean':(v1/scale).abs().mean().item(),'state_abs_mean':(v2/scale).abs().mean().item(),'reset_abs_mean':(v3/scale).abs().mean().item()})
  # Algebraically R37; combine reset and amplitude upstream before multiplying g,
  # matching the original FP32 evaluation order for the non-threshold path.
  return (a+b/theta)*g,(v1+v2+v3).mean().reshape_as(theta),None,None
class ArtgATLIF(ATLIF):
 def forward(self,x):
  mem=0.;outputs=[]
  for t in range(self.T):
   mem=mem*self.tau+x[t,...];spike,s,updates=Artg.apply(mem,self.thresh,self.sp,self.artg_name);mem=(1.-s)*mem;outputs.append(spike);self.update_value+=updates/self.T
  x=torch.stack(outputs,dim=0);self.r=(x/self.thresh.data).mean().item();self.s=(x.mean(1)/self.thresh.data).sum().item();self.act_value=x.reshape(x.size(0),-1).mean(1).sum();return x

def configure(model,group):
 for n,m in model.named_modules():
  if isinstance(m,ATLIF):
   if group=='G1':m.act=Eq13.apply
   if group=='M4':m.__class__=ArtgATLIF;m.artg_name=n
 return model

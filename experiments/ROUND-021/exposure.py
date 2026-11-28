"""R93-R96 explicit valid-input spike exposure without batch-sized unfold."""
import json,sys
from pathlib import Path
import torch,numpy as np
from torch.nn import functional as F
from spikingjelly.activation_based import functional
from utils.utils import SOPMonitor

def batch_counts(x,kernel,stride,padding,dilation):
 # Sum T and B first; unfold only [1,C,H,W], never [T*B,C,H,W].
 summed=x.sum((0,1),dtype=torch.float64).unsqueeze(0)
 return F.unfold(summed,kernel_size=kernel,dilation=dilation,padding=padding,stride=stride).sum(-1).view(x.shape[2],*kernel)
def dense_counts(h,w,kernel,stride,padding,dilation,device):
 x=torch.ones((1,1,h,w),dtype=torch.float64,device=device)
 return F.unfold(x,kernel_size=kernel,dilation=dilation,padding=padding,stride=stride).sum(-1).view(*kernel)
def unit_tests():
 # Independent explicit nested enumeration, stride=2 padding=1 dilation=2, two unequal batches.
 x=torch.tensor([[[[[1.,0.,1.,0.],[0.,1.,0.,1.],[1.,1.,0.,0.]]],[[[0.,1.,0.,1.],[1.,0.,1.,0.],[0.,0.,1.,1.]]]]])
 x=x.reshape(1,2,1,3,4);k=(2,2);stride=(2,2);padding=(1,1);dilation=(2,2);c=batch_counts(x,k,stride,padding,dilation);ref=torch.zeros_like(c);H,W=3,4;oh=(H+2-2*(2-1)-1)//2+1;ow=(W+2-2*(2-1)-1)//2+1
 for n in range(2):
  for u in range(oh):
   for v in range(ow):
    for a in range(2):
     for b in range(2):
      i=u*2-1+a*2;j=v*2-1+b*2
      if 0<=i<H and 0<=j<W:ref[0,a,b]+=x[0,n,0,i,j]
 assert torch.equal(c,ref);dense=dense_counts(H,W,k,stride,padding,dilation,x.device);assert torch.equal(batch_counts(torch.ones_like(x),k,stride,padding,dilation)[0],dense*2);assert batch_counts(torch.zeros_like(x),k,stride,padding,dilation).sum()==0
 mask=torch.tensor([[[[1.,0.],[1.,1.]]]],dtype=torch.float64);actual=F.conv2d(x.flatten(0,1).double(),mask,stride=stride,padding=padding,dilation=dilation).sum();reconstructed=(c*mask.sum(0)).sum();assert actual==reconstructed
 m2=mask.clone();m2[0,0,1,1]=0;deleted=actual-F.conv2d(x.flatten(0,1).double(),m2,stride=stride,padding=padding,dilation=dilation).sum();assert deleted==c[0,1,1]
 assert torch.equal(batch_counts(x[:,:1],k,stride,padding,dilation)+batch_counts(x[:,1:],k,stride,padding,dilation),c)
 return {'status':'PASSED','explicit_integer_enumeration':True,'stride_padding_dilation':True,'all_ones_p1':True,'all_zero_p0':True,'R94':True,'R95_single_connection':True,'unequal_batch_weighting':True}
def measure(m,data,ids_label,epoch,K,guard):
 m.eval();mon=SOPMonitor(m);mods=dict(m.named_modules());names=[n for n in mon.monitored_layers if K.get(n,0)>0];counts={};dense={};ones={};elems={};sop={n:0. for n in names};shapes={};hooks=[];first={};check=[];seen=0;outputs=[]
 for n in names:
  z=mods[n];assert z.groups==1
  def h(mod,args,n=n):
   x=args[0].detach();assert x.ndim==5 and bool(((x==0)|(x==1)).all());k=tuple(mod.kernel_size);q=batch_counts(x,k,mod.stride,mod.padding,mod.dilation);counts[n]=counts.get(n,torch.zeros_like(q))+q;d=dense_counts(x.shape[3],x.shape[4],k,mod.stride,mod.padding,mod.dilation,x.device);dense[n]=dense.get(n,torch.zeros_like(d))+d*x.shape[0]*x.shape[1];ones[n]=ones.get(n,0)+int(x.sum().item());elems[n]=elems.get(n,0)+x.numel();shapes[n]=list(x.shape);assert len(mods[n].sparse_mask().shape)==4
  hooks.append(z.register_forward_pre_hook(h))
 try:
  with torch.no_grad():
   for idx,(x,y,sid) in enumerate(data):
    guard(100);x=x.cuda(non_blocking=True);mon.clear_recorded_data();out=m(x);assert torch.isfinite(out).all();seen+=len(sid)
    for n in names:sop[n]+=sum(float(z.item()) for z in mon[n])
    if idx==0:outputs.append(out.detach().clone());first['x']=x.detach().clone()
    functional.reset_net(m)
 finally:
  mon.remove_hooks()
  for h in hooks:h.remove()
 if outputs:
  with torch.no_grad():z=m(first['x']);assert torch.equal(z,outputs[0]);functional.reset_net(m)
 rows=[];audit=[]
 for n in names:
  mod=mods[n];e=counts[n]/(seen*4);den=dense[n].sum()*mod.in_channels;p=counts[n].sum()/den;assert torch.isfinite(p) and 0<=p<=1 and den>0;mask=mod.sparse_mask().detach();ef=mask.sum(0,dtype=torch.float64);reconstructed=(4*(e*ef).sum()).item();observed=sop[n]/seen;err=abs(reconstructed-observed)/(1+abs(observed));assert err<=1e-5;vec=e.cpu().numpy().flatten();rows.append({'dataset':ids_label,'epoch':epoch,'layer':n,'sample_count':seen,'shape_last_batch':json.dumps(shapes[n]),'kernel':str(mod.kernel_size),'stride':str(mod.stride),'padding':str(mod.padding),'dilation':str(mod.dilation),'input_FR':ones[n]/elems[n],'p':p.item(),'e_mean':float(vec.mean()),'e_std':float(vec.std()),'e_p10':float(np.quantile(vec,.1)),'e_p50':float(np.quantile(vec,.5)),'e_p90':float(np.quantile(vec,.9)),'K':K[n],'threshold':mod.threshold.item(),'density':mask.mean().item(),'SOP_sample_weighted':observed,'SOP_R94':reconstructed,'normalized_error':err});audit.append({'dataset':ids_label,'epoch':epoch,'layer':n,'relative_error':err,'binary_verified':True})
 return rows,{'status':'PASSED','checks':audit,'samples':seen,'capture_output_equal':True,'denominator':'entire sample count times T'}

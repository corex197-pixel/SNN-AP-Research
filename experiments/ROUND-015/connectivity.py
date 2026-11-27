"""EXP-016D CPU synthetic mechanism; all results are synthetic, not performance."""
import os,sys,json,csv,time,resource
from pathlib import Path
import numpy as np,torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'vendor/UniSparse'))
from models.submodules.layers import SparseConv2d
from utils.utils import SOPMonitor
O=ROOT/'experiments/EXP-016';torch.set_num_threads(1);t0=time.monotonic();c0=resource.getrusage(resource.RUSAGE_SELF)
def enum(x,m,stride,pad,dil,groups):
 T,B,C,H,W=x.shape;CO,CI,KH,KW=m.shape;sh,sw=stride;ph,pw=pad;dh,dw=dil;OH=(H+2*ph-dh*(KH-1)-1)//sh+1;OW=(W+2*pw-dw*(KW-1)-1)//sw+1;f=np.zeros((C,H,W),dtype=np.int64)
 for j in range(CO):
  g=j//(CO//groups)
  for ch in range(CI):
   c=g*CI+ch
   for ky in range(KH):
    for kx in range(KW):
     if not m[j,ch,ky,kx]:continue
     for oy in range(OH):
      for ox in range(OW):
       h=oy*sh-ph+ky*dh;w=ox*sw-pw+kx*dw
       if 0<=h<H and 0<=w<W:f[c,h,w]+=1
 return f,float((x*f[None,None]).sum()/B)
def scenario(name,x,m,stride=(1,1),pad=(0,0),dil=(1,1),groups=1):
 f,c=enum(x,m,stride,pad,dil,groups);count=float(torch.nn.functional.conv2d(torch.tensor(x.reshape(-1,*x.shape[2:]),dtype=torch.float64),torch.tensor(m,dtype=torch.float64),stride=stride,padding=pad,dilation=dil,groups=groups).sum().item()/x.shape[1]);assert c==count
 r=x.mean((0,1,3,4));d=m.sum((0,2,3)) if groups==1 else None;orig=float(m.shape[0]*m.shape[2]*m.shape[3]*r.sum()) if groups==1 else None;approx=float((d*r).sum()) if groups==1 else None;scaled=approx*x.shape[0]*x.shape[3]*x.shape[4] if approx is not None else None
 mon=None
 if groups==1:
  class Net(nn.Module):
   def __init__(self):super().__init__();self.conv=SparseConv2d(x.shape[2],m.shape[0],m.shape[2:],stride,pad,dil);self.skip=[];self.conv.weight.data.copy_(torch.tensor(m,dtype=torch.float32))
   def forward(self,z):return self.conv(z)
  model=Net();monitor=SOPMonitor(model);model(torch.tensor(x,dtype=torch.float32));mon=sum(v.item() for v in monitor.records)/x.shape[1];monitor.remove_hooks();assert mon==c
 return {'case':name,'shape':list(x.shape),'mask_shape':list(m.shape),'stride':list(stride),'padding':list(pad),'dilation':list(dil),'groups':groups,'density':float(m.mean()),'FR_per_channel':r.tolist(),'nominal_proxy':orig,'effective_fanout':d.tolist() if d is not None else None,'effective_proxy':approx,'proxy_scaled':scaled,'proxy_vs_exact_error':scaled-c if scaled is not None else None,'exact_sops_per_sample':c,'independent_conv_sops':count,'author_sops':mon,'spatial_fanout':f.tolist()}
x=np.ones((1,1,2,1,1),dtype=np.float32);mask=np.zeros((9,2,1,1),dtype=np.float32);mask[:,0]=1;mask[0,1]=1
r=scenario('9_vs_1',x,mask);delta=[]
for ch in (0,1):
 y=x.copy();y[0,0,ch,0,0]=0;delta.append(r['exact_sops_per_sample']-scenario('removed',y,mask)['exact_sops_per_sample'])
assert delta==[9,1] and r['nominal_proxy']==18 and r['effective_fanout']==[9,1]
results=[r];base=np.array([[[[[1,0,1],[0,1,0],[1,0,1]],[[0,1,0],[1,0,1],[0,1,0]]]]],dtype=np.float32)
for label,m in [('dense',np.ones((2,2,3,3))),('uniform_sparse',np.tile(np.array([[1,0,1],[0,1,0],[1,0,1]]),(2,2,1,1))),('unequal',np.concatenate([np.ones((2,1,3,3)),np.zeros((2,1,3,3))],axis=1)),('one_channel_zero',np.concatenate([np.ones((2,1,3,3)),np.zeros((2,1,3,3))],axis=1)),('all_zero',np.zeros((2,2,3,3)))]:results.append(scenario(label,base,m,pad=(1,1)))
for label,shape,k,stride,pad,dil,groups in [('1x1',(2,3,2,3,5),1,(1,1),(0,0),(1,1),1),('stride2',(2,3,2,4,5),3,(2,2),(1,1),(1,1),1),('dilation2',(1,3,2,5,7),3,(1,1),(2,2),(2,2),1),('grouped_formula_only',(1,2,2,4,5),3,(1,1),(1,1),(1,1),2)]:
 rng=np.random.default_rng(1234);xx=rng.integers(0,2,shape).astype('float32');mm=rng.integers(0,2,(2,shape[2]//groups,k,k)).astype('float32');results.append(scenario(label,xx,mm,stride,pad,dil,groups))
# Incomplete batches: author averages batch sums / configured batch size; exact weights actual N.
batch_counts=[8,3];batch_sizes=[4,2];author=sum(batch_counts)/len(batch_counts)/4;weighted=sum(batch_counts)/sum(batch_sizes)
(O/'connectivity_cost_tests.json').write_text(json.dumps({'kind':'UNIT_TEST','cases':results,'single_spike_costs_9_vs_1':delta,'R60_enum_equals_conv_and_author_all_applicable':True,'R61_single_spike_verified':True,'dense_proxy_exact_only_under_no_boundary_and_appropriate_scaling':True,'note':'R59 excludes spatial boundaries; grouped case independent formula only'},indent=2)+'\n');(O/'sop_aggregation_audit.json').write_text(json.dumps({'kind':'UNIT_TEST','batch_sums':batch_counts,'actual_batch_sizes':batch_sizes,'configured_batch_size':4,'author':author,'sample_weighted':weighted,'difference':author-weighted,'nonbinary':'author sums raw values, so binary-event interpretation does not apply generally'},indent=2)+'\n')
with (O/'connectivity_cost_samples.csv').open('w',newline='') as f:
 fields=['case','density','nominal_proxy','effective_proxy','proxy_scaled','exact_sops_per_sample','proxy_vs_exact_error','author_sops'];w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows([{k:r[k] for k in fields} for r in results])
r=resource.getrusage(resource.RUSAGE_SELF);(O/'cpu_connectivity_resources.json').write_text(json.dumps({'wall_seconds':time.monotonic()-t0,'cpu_seconds':r.ru_utime+r.ru_stime-c0.ru_utime-c0.ru_stime,'gpu_seconds':0},indent=2)+'\n');print(json.dumps({'status':'PASSED','cases':len(results),'single_spike_costs':delta,'author_aggregation':author,'weighted':weighted}))

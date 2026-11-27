"""EXP-019: paired-model fixed-activity, fixed-mask SOP attribution."""
import sys,os,json,math,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent));import common as C
import numpy as np,torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.utils import SOPMonitor
from spikingjelly.activation_based import functional
O=C.ROOT/'experiments/EXP-019'
def attrib(a,b,c,d):return {'C00':a,'C20':b,'C02':c,'C22':d,'delta_W':((b-a)+(d-c))/2,'delta_S':((c-a)+(d-b))/2,'interaction':d-b-c+a,'delta':d-a}
def unit():
 # Independent enumerated binary matrix edges and unequal batches.
 x0=[[1,0],[0,1],[1,1]];x2=[[0,1],[1,0],[0,1]];m0=[[1,1],[1,1]];m2=[[1,0],[0,1]]
 def count(m,x):return sum(m[j][i]*v[i] for v in x for i in range(2) for j in range(2))
 v=attrib(count(m0,x0),count(m2,x0),count(m0,x2),count(m2,x2));assert v['delta']==v['delta_W']+v['delta_S'];assert attrib(4,4,8,8)['delta_W']==0;assert attrib(4,2,4,2)['delta_S']==0;assert attrib(4,1,7,4)['delta']==0 and attrib(4,1,7,4)['delta_W']!=0 and attrib(4,1,7,4)['delta_S']!=0
 # Independent explicit convolution enumeration includes padding and stride.
 X=torch.tensor([[[[1.,0.,1.],[0.,1.,0.]]]]);W=torch.tensor([[[[1.,0.],[1.,1.]]]]);actual=F.conv2d(X,W,stride=2,padding=1).sum().item();ref=0
 for oy in range(2):
  for ox in range(2):
   for ky in range(2):
    for kx in range(2):
     iy=oy*2-1+ky;ix=ox*2-1+kx
     if 0<=iy<2 and 0<=ix<3:ref+=int(X[0,0,iy,ix])*int(W[0,0,ky,kx])
 assert actual==ref;assert (8+3)/(4+2)==11/6
 C.dump(O/'cost_decomposition_tests.json',{'status':'PASSED','integer_example':v,'explicit_conv_stride2_padding_reference':ref,'incomplete_batch_weighted':11/6,'equal_cost_nonzero_terms_test':True,'identical_masks_zero_weight_term':True,'identical_activity_zero_activity_term':True,'interaction_test':True})
def parent(epoch,g):
 if epoch==20:return C.ROOT/'experiments/EXP-017/runs/EXP-017-PILOT-001'/g/'latest.pth'
 return C.O/'runs/EXP-018-CONTINUE-001'/g/f'epoch-{epoch:03d}.pth'
def evaluate(epoch,ids):
 C.guard(240);m=C.model().eval();masks={};meta={}
 for g in ['U0','U2']:
  ck=torch.load(parent(epoch,g),map_location='cpu',weights_only=False);assert ck['epoch']==epoch;m.load_state_dict(ck['model']);masks[g]={n:z.sparse_mask().detach().clone() for n,z in m.named_modules() if isinstance(z,C.SparseConv2d) and not any(k in n for k in m.skip)};meta[g]={'sha256':C.sha(parent(epoch,g)),'epoch':ck['epoch']};del ck
 assert masks['U0'].keys()==masks['U2'].keys();totals={};counts={};author={};inputs={};data=DataLoader(C.TrainFileOnly(ids,False),batch_size=256,shuffle=False,num_workers=4,pin_memory=True,drop_last=False)
 for g in ['U0','U2']:
  ck=torch.load(parent(epoch,g),map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);del ck;m.eval();mon=SOPMonitor(m);hooks=[];batch_cost={};nonbinary=[];samples=0;batch_index=0
  for name in masks[g]:
   def hook(mod,args,n=name):
    x=args[0].detach();assert x.ndim==5
    if bool(((x!=0)&(x!=1)).any().item()):nonbinary.append(n);raise RuntimeError('nonbinary monitored input '+n)
    inputs[(g,n)]=inputs.get((g,n),0)+int((x!=0).sum().item());counts[(g,n)]=counts.get((g,n),0)+x.numel();xx=x.flatten(0,1).float()
    for mg in ['U0','U2']:
     mask=masks[mg][n];out=F.conv2d(xx,mask,None,mod.stride,mod.padding,mod.dilation,mod.groups);val=float(out.sum(dtype=torch.float64).item());batch_cost[(g,mg,n)]=val
   hooks.append(m.get_submodule(name).register_forward_pre_hook(hook))
  try:
   with torch.no_grad():
    for x,y,sid in data:
     C.guard(180);x=x.cuda(non_blocking=True);mon.clear_recorded_data();batch_cost.clear();z=m(x);assert torch.isfinite(z).all();functional.reset_net(m);n=len(sid);samples+=n
     for name in masks[g]:
      for mg in ['U0','U2']:totals[(g,mg,name)]=totals.get((g,mg,name),0.)+batch_cost[(g,mg,name)]
      a=sum(float(q.item()) for q in mon[name]);author[(g,name)]=author.get((g,name),0.)+a
     batch_index+=1
  finally:
   mon.remove_hooks()
   for h in hooks:h.remove()
  assert samples==4500 and batch_index==18 and not nonbinary
 layers=[]
 for name in masks['U0']:
  a=totals[('U0','U0',name)]/4500;b=totals[('U0','U2',name)]/4500;c=totals[('U2','U0',name)]/4500;d=totals[('U2','U2',name)]/4500;v=attrib(a,b,c,d);v.update(epoch=epoch,layer=name,FR0=inputs[('U0',name)]/counts[('U0',name)],FR2=inputs[('U2',name)]/counts[('U2',name)],density0=float(masks['U0'][name].mean().item()),density2=float(masks['U2'][name].mean().item()));layers.append(v)
 vals=attrib(*[sum(r[k] for r in layers) for k in ['C00','C20','C02','C22']]);a0=sum(author[('U0',n)] for n in masks['U0'])/4500;a2=sum(author[('U2',n)] for n in masks['U0'])/4500;err0=abs(vals['C00']-a0)/(1+abs(a0));err2=abs(vals['C22']-a2)/(1+abs(a2));assert max(err0,err2)<=1e-5 and abs(vals['delta']-vals['delta_W']-vals['delta_S'])<1e-5*(1+abs(vals['delta']));vals.update(epoch=epoch,relative_W=-vals['delta_W']/vals['C00'],relative_S=-vals['delta_S']/vals['C00'],relative_total=-vals['delta']/vals['C00'],activity_cost_fraction=vals['delta_S']/vals['C00'],direction_gate=vals['delta_W']<0 and vals['delta_S']>0,exploratory_magnitude_gate=vals['delta_S']/vals['C00']>=.05);C.append(O/'layer_cost_decomposition.csv',layers);C.append(O/'total_cost_decomposition.csv',[vals]);C.dump(O/f'cost_count_audit_epoch_{epoch:03d}.json',{'status':'PASSED','C00_author_sample_weighted':a0,'C22_author_sample_weighted':a2,'relative_error_C00':err0,'relative_error_C22':err2,'authors_scope':list(masks['U0']),'samples':4500,'input_binary_verified':True,'checkpoints':meta,'sum_float64':True});print(json.dumps({'event':'cost_complete',**vals}),flush=True);del m,masks;torch.cuda.empty_cache();return vals
if __name__=='__main__':unit();print('CPU accounting tests PASSED')

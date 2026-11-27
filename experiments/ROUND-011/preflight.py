import sys,json,time,traceback,hashlib,copy,os,math
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
from timm.optim import create_optimizer_v2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/EXP-004/controlled'));sys.path.insert(0,str(ROOT/'experiments/ROUND-011'))
from core import model_create,ATLIF,isolated_eval,transforms_pair,DATA,threshold_update,seed_all
from models.submodules.layers import Surrogate,zif_backward
from utils.utils import SOPMonitor
from artg import Eq13,Artg,configure,DIAGNOSTICS
from spikingjelly.activation_based import functional
OUT=ROOT/'experiments/EXP-011/preflight';OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();cpu=time.process_time();result={'status':'RUNNING','start_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'tests':{}};torch.set_num_threads(2)
M=ROOT/'experiments/EXP-004/runs/EXP-004-RUN-003/checkpoint_best_validation.pth';S=ROOT/'experiments/EXP-004/splits/indices.json';I=ROOT/'experiments/EXP-007/calibration_indices.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def error(x,y):
 x=torch.as_tensor(x).detach().double();y=torch.as_tensor(y).detach().double();return ((x-y).abs()/(1+y.abs())).max().item()
def manual():
 results=[]
 for dtype in [torch.float64,torch.float32]:
  tol=1e-10 if dtype==torch.float64 else 1e-5;r=torch.tensor([-.5,0.,.5,.9,1.,1.1,1.5,2.,3.],dtype=dtype);a=torch.tensor([-.9,.3,.7,-.2,.5,-.4,.2,.8,-.1],dtype=dtype);b=torch.tensor([.4,-.8,.6,.3,-.5,.1,-.7,.9,.2],dtype=dtype);theta0=torch.tensor(1.3,dtype=dtype);x0=r*theta0;g=(1-((x0-theta0)/theta0).abs()).clamp_min(0);s=(x0>=theta0).float();rr=x0/theta0
  for group in ['G0','G1']:
   x=x0.clone().requires_grad_();theta=theta0.clone().requires_grad_();o,_=(Surrogate if group=='G0' else Eq13).apply(x,theta,.002);q=a.to(o.dtype);dx,dt=torch.autograd.grad(o,(x,theta),q);ex=q.to(dtype)*g;et=-(q.to(dtype)*g*(g if group=='G0' else 1)).mean();err=max(error(dx,ex),error(dt,et));results.append({'dtype':str(dtype),'case':group,'normalized_error':err,'tolerance':tol});assert err<tol,(dtype,group,err)
  for case,A,B in [('amplitude',a,None),('reset',None,b),('both',a,b)]:
   x=x0.clone().requires_grad_();theta=theta0.clone().requires_grad_();o,sout,_=Artg.apply(x,theta,.002,'test');outs=[];weights=[]
   if A is not None:outs.append(o);weights.append(A.to(o.dtype))
   if B is not None:outs.append(sout);weights.append(B.to(sout.dtype))
   dx,dt=torch.autograd.grad(outs,(x,theta),weights);aa=torch.zeros_like(x0) if A is None else A.to(o.dtype).to(dtype);bb=torch.zeros_like(x0) if B is None else B.to(sout.dtype).to(dtype);gg=(1-(rr-1).abs()).clamp_min(0);ex=aa*gg+bb*gg/theta0;et=(aa*(s.to(dtype)-rr*gg)-bb*rr*gg/theta0).mean();err=max(error(dx,ex),error(dt,et));results.append({'dtype':str(dtype),'case':'M4-'+case,'normalized_error':err,'tolerance':tol});assert err<tol,(dtype,case,err)
 # Mixed precision: large fp32 upstream must not overflow a shared fp32 threshold gradient.
 x=torch.tensor([2.],dtype=torch.float16,requires_grad=True);theta=torch.tensor(1.3,dtype=torch.float32,requires_grad=True);o,spike,_=Artg.apply(x,theta,.002,'mixed');q=torch.tensor([100000.],dtype=torch.float32);_,grad_theta=torch.autograd.grad(o,(x,theta),q);r=(x.detach()/theta.detach()).float();g=(1-((x.detach()-theta.detach())/theta.detach()).abs()).clamp_min(0).float();expected=(q*(spike.detach()-r*g)).mean();err=error(grad_theta,expected);assert torch.isfinite(grad_theta) and err<1e-5,('mixed_dtype',err,grad_theta)
 results.append({'dtype':'float16 membrane / float32 threshold and upstream','case':'M4-amplitude-large-upstream','normalized_error':err,'tolerance':1e-5,'threshold_gradient':grad_theta.item(),'expected':expected.item()})
 save('gradient_unit_tests.json',{'cases':results,'status':'PASSED','note':'Surrogate estimates, not finite differences of hard thresholds'})

def local():
 tests=[]
 for dtype in [torch.float64,torch.float32]:
  tol=1e-10 if dtype==torch.float64 else 1e-5;gen=torch.Generator().manual_seed(911);x0=torch.randn(8,9,generator=gen,dtype=dtype)*.8+.8;aa=torch.randn(8,9,generator=gen,dtype=dtype);hh=torch.randn(8,9,generator=gen,dtype=dtype)*.3;theta0=torch.tensor(1.3,dtype=dtype);common_forward=None
  for group in ['G0','G1','M4']:
   outputs=[]
   for is_reference in [False,True]:
    x=x0.clone().requires_grad_();theta=theta0.clone().requires_grad_();u=torch.zeros_like(x[0]);trace=[];loss=0.;stats=[]
    for t in range(8):
     mem=.9*u+x[t];binary=(mem>=theta).float();actual_dtype=(binary*theta).dtype
     if not is_reference:
      if group=='M4':o,s,v=Artg.apply(mem,theta,.002,'unit');u=(1-s)*mem
      else:o,v=(Surrogate if group=='G0' else Eq13).apply(mem,theta,.002);s=o/theta.detach();u=(1-s)*mem
     else:
      # Explicit STE graph: hard forward, local derivatives and original float32 spike casts.
      r=mem.detach()/theta.detach();g=(1-(r-1).abs()).clamp_min(0);g_orig=(1-((mem.detach()-theta.detach())/theta.detach()).abs()).clamp_min(0);dm=mem-mem.detach();dt=theta-theta.detach();N=mem.numel();v=(.002*zif_backward(mem-theta,theta)*binary).sum(0).mean().item()
      if group=='M4':
       o=(binary*theta.detach()+dm*g+dt*(binary.to(dtype)-r*g)/N).to(actual_dtype);s=(binary+dm*g/theta.detach()-dt*r*g/theta.detach()/N).to(binary.dtype);u=(1-s)*mem
      else:
       factor=g_orig*g_orig if group=='G0' else g_orig;o=(binary*theta.detach()+dm*g_orig-dt*factor/N).to(actual_dtype);s=o/theta.detach();u=(1-s)*mem
     trace.append((mem.detach().clone(),o.detach().clone(),(o.detach()>0),u.detach().clone()));stats.append(v);loss=loss+(o*aa[t]).sum()+(u*hh[t]).sum()
    dx,dt=torch.autograd.grad(loss,(x,theta));outputs.append({'trace':trace,'stats':stats,'dx':dx,'dt':dt})
   actual,ref=outputs
   for t in range(8):
    for j in range(4):
     if not torch.equal(actual['trace'][t][j],ref['trace'][t][j]):raise RuntimeError('explicit reference forward mismatch '+group)
   err_dx=error(actual['dx'],ref['dx']);err_dt=error(actual['dt'],ref['dt']);assert max(err_dx,err_dt)<tol,(group,dtype,err_dx,err_dt)
   if common_forward is None:common_forward=actual
   else:
    for t in range(8):
     for j in range(4):
      if not torch.equal(actual['trace'][t][j],common_forward['trace'][t][j]):raise RuntimeError('multistep forward mismatch '+group)
    if not np.allclose(common_forward['stats'],actual['stats'],atol=0,rtol=0):raise RuntimeError('manual updates mismatch')
   tests.append({'dtype':str(dtype),'group':group,'input_gradient_normalized_error':err_dx,'threshold_gradient_normalized_error':err_dt,'tolerance':tol})
 save('forward_equivalence.json',{'status':'PASSED','T':8,'forward_spikes_outputs_membranes_reset_and_manual_update_exact':True,'cases':tests,'reference':'independent explicit STE graph with per-time mean normalization and original float32 output casts; not hard finite differences'})

def main():
 for p,expected in [(M,'c068f54379cb5d29bcfa5fe2d9de676cd3cc153433ada1ed5aa1c66355d0b32a'),(S,'69dccfe992236766aea575a0a66495802ca889ca329d61f51e8de3c1ab7f3165'),(I,'b849b74ef19cb4a09df756c54871122f3b02aa047c704d0436feab15a38bde52')]:assert sha(p)==expected
 assert torch.cuda.device_count()==1 and '4090' in torch.cuda.get_device_name(0)
 manual();local();ck=torch.load(M,map_location='cpu',weights_only=False);spl=json.loads(S.read_text());ids=json.loads(I.read_text())['indices'];assert len(ids)==500 and set(ids)<=set(spl['train']) and not set(ids)&set(spl['validation']) and not set(ids)&set(spl['diagnostic'])
 ds=CIFAR10(DATA,train=True,download=False,transform=transforms_pair()[1]);x=torch.stack([ds[i][0] for i in ids[:4]]).cuda();y=torch.tensor([ds[i][1] for i in ids[:4]],device='cuda');reference=None;records={};layerrows=[]
 for group in ['G0','G1','M4']:
  seed_all(2030);m=configure(model_create().cuda(),group);m.load_state_dict(ck['model']);m.train()
  for z in m.modules():
   if isinstance(z,ATLIF):z.thresh.data.mul_(1.25)
  hsh=hashlib.sha256()
  for n,z in m.state_dict().items():hsh.update(n.encode());hsh.update(z.detach().cpu().numpy().tobytes())
  init=hsh.hexdigest();mods={n:z for n,z in m.named_modules() if isinstance(z,ATLIF)};assert len(mods)==19;spikes={};hs=[]
  for n,z in mods.items():hs.append(z.register_forward_hook(lambda z,a,o,n=n:spikes.update({n:(o.detach()>0).cpu()})))
  mon=SOPMonitor(m);opt=create_optimizer_v2(m,opt='sgd',lr=.005,weight_decay=5e-4,momentum=.9,nesterov=True);optids={id(p) for g in opt.param_groups for p in g['params']};assert all(id(z.thresh) in optids for z in mods.values())
  opt.zero_grad(set_to_none=True);DIAGNOSTICS.update(enabled=(group=='M4'),scale=1.,rows=[]);z=m(x);loss=F.cross_entropy(z.mean(0),y);loss.backward();DIAGNOSTICS['enabled']=False;sop=sum(float(a.item()) for a in mon.records);mon.remove_hooks()
  for h in hs:h.remove()
  grads={n:p.grad.detach().cpu().clone() for n,p in m.named_parameters() if p.grad is not None};updates={n:v.update_value for n,v in mods.items()};threshold_before={n:v.thresh.detach().clone() for n,v in mods.items()};diag=list(DIAGNOSTICS['rows']);opt.step();afteropt={n:v.thresh.detach().clone() for n,v in mods.items()};threshold_update(m,.005);aftermanual={n:v.thresh.detach().clone() for n,v in mods.items()}
  if any(not torch.isfinite(v).all() or v.item()<=0 for v in aftermanual.values()):raise RuntimeError('invalid threshold preflight')
  for n,v in mods.items():
   row={'group':group,'module':n,'threshold_initial':threshold_before[n].item(),'task_gradient':grads[n+'.thresh'].item(),'manual_update_value':updates[n],'optimizer_threshold_delta':(afteropt[n]-threshold_before[n]).item(),'manual_threshold_delta':(aftermanual[n]-afteropt[n]).item()}
   if group=='M4':
    zz=[d for d in diag if d['module']==n]
    for k in ['direct_mean','state_mean','reset_mean','direct_abs_mean','state_abs_mean','reset_abs_mean']:row[k]=sum(v[k] for v in zz)
   layerrows.append(row)
  if reference is None:reference={'init':init,'z':z.detach().cpu(),'spikes':spikes,'sop':sop,'grads':grads,'updates':updates}
  else:
   assert init==reference['init'];ld=error(z.detach().cpu(),reference['z']);assert ld<=1e-5;assert all(torch.equal(spikes[n],reference['spikes'][n]) for n in spikes);assert sop==reference['sop'];gd=max(error(v,reference['grads'][n]) for n,v in grads.items() if not n.endswith('thresh'));ud=max(abs(updates[n]-reference['updates'][n]) for n in updates);records[group]={'logits_normalized_error':ld,'non_threshold_gradient_max_normalized_error':gd,'manual_update_max_absolute_difference':ud};assert gd<=1e-5 and ud<=1e-6,(group,gd,ud)
  assert all(torch.isfinite(v).all() for v in grads.values());records.setdefault(group,{}).update(initial_hash=init,sop=sop,loss=loss.item(),all_threshold_gradients_finite=True,threshold_count=19,thresholds_in_optimizer=True,peak_allocated_bytes=torch.cuda.max_memory_allocated())
  # Evaluation isolation test after nonzero activity.
  state={n:t.detach().clone() for n,t in m.state_dict().items()};attrs={n:(v.update_value,v.r,v.s) for n,v in mods.items()};mode=m.training
  with isolated_eval(m):m(x)
  assert all(torch.equal(state[n],v) for n,v in m.state_dict().items()) and attrs=={n:(v.update_value,v.r,v.s) for n,v in mods.items()} and m.training==mode
  # Controlled AMP step uses the exact original scaler/threshold-update order.
  mm=configure(model_create().cuda(),group);mm.load_state_dict(ck['model']);mm.train()
  for v in mm.modules():
   if isinstance(v,ATLIF):v.thresh.data.mul_(1.25)
  oo=create_optimizer_v2(mm,opt='sgd',lr=.005,weight_decay=5e-4,momentum=.9,nesterov=True);sc=torch.cuda.amp.GradScaler(enabled=True);oo.zero_grad(set_to_none=True)
  with torch.cuda.amp.autocast():ll=F.cross_entropy(mm(x).mean(0),y)
  if not torch.isfinite(ll):raise RuntimeError('AMP nonfinite loss')
  old=sc.get_scale();sc.scale(ll).backward();sc.step(oo);threshold_update(mm,.005);sc.update();skipped=sc.get_scale()<old
  assert all(torch.isfinite(v.thresh).all() and v.thresh.item()>0 for v in mm.modules() if isinstance(v,ATLIF));records[group]['AMP_step_ok']=True;records[group]['AMP_skipped']=skipped;records[group]['eval_state_isolation']=True
  del mm,oo,sc,m,opt;functional.reset_net if False else None;torch.cuda.empty_cache()
 save('real_network_gradient_check.json',{'status':'PASSED','groups':records,'layers':layerrows,'common_input_samples':ids[:4],'C0_weights_checksum':sha(M),'fp32_non_threshold_gradient_tolerance':1e-5})
 result['status']='PASSED';result['tests']={'manual':'PASSED','multistep':'PASSED','real_model':'PASSED'}
try:main()
except BaseException as e:result.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
finally:result.update(wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,end_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat());save('status.json',result);print(json.dumps(result,indent=2),flush=True)
if result['status']!='PASSED':sys.exit(1)

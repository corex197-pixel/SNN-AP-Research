import os,sys,time,json,traceback,datetime,copy,math
from pathlib import Path
import torch
from torch import nn
from common import *
OUT=ROOT/'experiments/EXP-004/round004/PREFLIGHT-001'
OUT.mkdir(parents=True,exist_ok=False); start=time.monotonic();result={'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'RUNNING'}
try:
 if torch.cuda.device_count()!=1 or not torch.cuda.is_available():raise RuntimeError('expected exactly one visible GPU')
 torch.cuda.set_device(0);torch.set_num_threads(4);seed_all();device=torch.device('cuda:0');free,total=torch.cuda.mem_get_info();result.update(visible_devices=torch.cuda.device_count(),device=torch.cuda.get_device_name(0),free_bytes=free,total_bytes=total,split=validate_split())
 if free<8*1024**3:raise RuntimeError('GPU free memory below 8 GiB')
 train,val=loaders();model=model_create().to(device);opt=opt_create(model);sched=sched_create(opt);scaler=torch.cuda.amp.GradScaler();model.train();it=iter(train);ts=[];torch.cuda.reset_peak_memory_stats()
 for i in range(20):
  if time.monotonic()-start>510:raise TimeoutError('preflight nearing 10 minute budget')
  x,y=next(it);x=x.to(device);y=y.to(device);torch.cuda.synchronize();t=time.monotonic();loss,z,skipped=train_step(model,opt,scaler,x,y);torch.cuda.synchronize();t=time.monotonic()-t
  if skipped:raise RuntimeError('AMP skip occurred in preflight throughput run')
  if i>=10:ts.append(t)
 result['training_steps']={'warmup':10,'timed':len(ts),'mean_seconds':sum(ts)/len(ts),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'amp_enabled':scaler.is_enabled()}
 metrics,layer=validate(model,val,device,max_batches=2);result['validation']={'count':metrics['count'],'layers':len(layer),'finite':all(math.isfinite(metrics[k]) for k in ('loss','accuracy','fr_global'))}
 # checkpoint-equivalent A/C: restored model and RNG; C receives isolated evaluation before identical train step.
 ck=OUT/'temporary-preflight.pth';atomic_save(ck,pack(model,opt,sched,scaler,0,.42,{},'preflight'));x=x[:4];y=y[:4]
 def do_step(with_eval):
  restore(ck,model,opt,sched,scaler);model.train()
  if with_eval:
   with isolated_eval(model):model(x)
  train_step(model,opt,scaler,x,y);torch.cuda.synchronize()
  return {'model':{n:v.detach().cpu().clone() for n,v in model.state_dict().items()},'optimizer':copy.deepcopy(opt.state_dict()),'scaler':copy.deepcopy(scaler.state_dict()),'rng':torch.cuda.get_rng_state().clone()}
 def compare(a,b):
  if torch.is_tensor(a):return (a.detach().cpu().float()-b.detach().cpu().float()).abs().max().item() if a.numel() else 0.
  if isinstance(a,dict):
   if a.keys()!=b.keys():return float('inf')
   return max([compare(a[k],b[k]) for k in a] or [0.])
  if isinstance(a,(list,tuple)):return max([compare(x,y) for x,y in zip(a,b)] or [0.]) if len(a)==len(b) else float('inf')
  return 0. if a==b else float('inf')
 A=do_step(False);C=do_step(True);result['isolation_resume']={'model_diff':compare(A['model'],C['model']),'optimizer_diff':compare(A['optimizer'],C['optimizer']),'scaler_diff':compare(A['scaler'],C['scaler']),'cuda_rng_equal':torch.equal(A['rng'],C['rng']),'includes_bn_and_threshold':True,'best_score_restored':restore(ck,model,opt,sched,scaler)['best_validation_accuracy']}
 if any(result['isolation_resume'][k]>1e-5 for k in ('model_diff','optimizer_diff','scaler_diff')) or not result['isolation_resume']['cuda_rng_equal'] or result['isolation_resume']['best_score_restored']!=.42:raise RuntimeError('GPU A/C or checkpoint test failed')
 # Isolated tiny model; explicitly inject non-finite grad after a normal training step.
 from models.submodules.layers import ATLIF
 class Tiny(nn.Module):
  def __init__(self):super().__init__();self.fc=nn.Linear(4,4);self.n=ATLIF(T=4,tau=.9,sp=.002);self.head=nn.Linear(4,2)
  def forward(self,x):return self.head(self.n(self.fc(x).unsqueeze(0).repeat(4,1,1)))
 tiny=Tiny().to(device);o=torch.optim.SGD(tiny.parameters(),lr=.1,momentum=.9);sc=torch.cuda.amp.GradScaler();xx=torch.ones(8,4,device=device);yy=torch.zeros(8,dtype=torch.long,device=device)
 train_step(tiny,o,sc,xx,yy);o.zero_grad(set_to_none=True)
 with torch.cuda.amp.autocast():loss=nn.functional.cross_entropy(tiny(xx).mean(0),yy)
 v=tiny.n.update_value;before_theta=tiny.n.thresh.item();before={n:p.detach().clone() for n,p in tiny.named_parameters()};before_opt=copy.deepcopy(o.state_dict());sc.scale(loss).backward();tiny.fc.weight.grad.fill_(float('inf'));scale_before=sc.get_scale();sc.step(o)
 params_unchanged=all(torch.equal(before[n],p) for n,p in tiny.named_parameters());optimizer_diff=compare(before_opt,o.state_dict());threshold_update(tiny,.1);sc.update();after=tiny.n.thresh.item();delta=after-before_theta
 result['amp_overflow']={'update_v':v,'lr':.1,'theta_before':before_theta,'theta_after':after,'delta_theta':delta,'expected_delta':.1*v,'absolute_error':abs(delta-.1*v),'weights_and_threshold_unchanged_before_manual_update':params_unchanged,'optimizer_state_diff':optimizer_diff,'scaler_before':scale_before,'scaler_after':sc.get_scale(),'forward_statistics_finite':math.isfinite(v)}
 if not(params_unchanged and optimizer_diff==0 and math.isfinite(v) and sc.get_scale()<scale_before and abs(delta-.1*v)<1e-5):raise RuntimeError('AMP overflow preservation test failed')
 ck.unlink();result['status']='COMPLETED'
except BaseException as e:result.update(status='FAILED',error=repr(e),traceback=traceback.format_exc())
finally:
 result.update(end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.monotonic()-start)
 (OUT/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if result['status']!='COMPLETED':sys.exit(1)

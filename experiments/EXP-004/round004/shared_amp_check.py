"""Single-GPU, nonzero-activity AMP-overflow preflight; no formal model."""
import os,time,json,datetime,traceback,sys,copy,math
from pathlib import Path
import torch
from torch import nn
from common import *
out=ROOT/'experiments/EXP-004/round004/PREFLIGHT-003';out.mkdir(parents=True,exist_ok=False)
t0=time.monotonic();r={'status':'RUNNING','start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}
try:
 if torch.cuda.device_count()!=1 or not torch.cuda.is_available():raise RuntimeError('expected exactly one visible GPU')
 torch.cuda.set_device(0);torch.set_num_threads(2);seed_all(42);free,total=torch.cuda.mem_get_info(0)
 r.update(device_count=1,device_name=torch.cuda.get_device_name(0),free_bytes=free,total_bytes=total,split=validate_split())
 if free<10*1024**3:raise RuntimeError('less than 10 GiB free on selected shared GPU')
 class Tiny(nn.Module):
  def __init__(self):super().__init__();self.fc=nn.Linear(4,4);self.n=ATLIF(T=4,tau=.9,sp=.002);self.head=nn.Linear(4,2)
  def forward(self,x):return self.head(self.n(self.fc(x).unsqueeze(0).repeat(4,1,1)))
 tiny=Tiny().cuda();o=torch.optim.SGD(tiny.parameters(),lr=.1,momentum=.9);sc=torch.cuda.amp.GradScaler();xx=torch.ones(8,4,device='cuda');yy=torch.zeros(8,dtype=torch.long,device='cuda')
 loss0,_,skip0=train_step(tiny,o,sc,xx,yy)
 with torch.no_grad():tiny.fc.weight.zero_();tiny.fc.bias.fill_(1.05)
 o.zero_grad(set_to_none=True)
 with torch.cuda.amp.autocast():loss=nn.functional.cross_entropy(tiny(xx).mean(0),yy)
 v=tiny.n.update_value;before_theta=tiny.n.thresh.item();before={n:p.detach().clone() for n,p in tiny.named_parameters()};before_opt=copy.deepcopy(o.state_dict());sc.scale(loss).backward();tiny.fc.weight.grad.fill_(float('inf'));scale_before=sc.get_scale();sc.step(o)
 unchanged=all(torch.equal(before[n],p) for n,p in tiny.named_parameters())
 def equal(a,b):
  if torch.is_tensor(a):return bool(torch.equal(a,b))
  if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
  if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
  return a==b
 opt_unchanged=equal(before_opt,o.state_dict());threshold_update(tiny,.1);sc.update();after=tiny.n.thresh.item();delta=after-before_theta
 r.update(normal_step_loss=float(loss0),normal_step_skipped=bool(skip0),amp_overflow={'v':v,'lr':.1,'theta_before':before_theta,'theta_after':after,'delta_theta':delta,'expected_delta':.1*v,'absolute_error':abs(delta-.1*v),'optimizer_parameters_unchanged':unchanged,'optimizer_state_unchanged':opt_unchanged,'scaler_before':scale_before,'scaler_after':sc.get_scale(),'forward_statistics_finite':math.isfinite(v)})
 if not(v>0 and math.isfinite(v) and not skip0 and unchanged and opt_unchanged and sc.get_scale()<scale_before and abs(delta-.1*v)<1e-5):raise RuntimeError('AMP nonzero update preflight failed')
 r['status']='COMPLETED'
except BaseException as e:r.update(status='FAILED',error=repr(e),traceback=traceback.format_exc())
finally:
 r.update(end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.monotonic()-t0);(out/'metrics.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2),flush=True)
if r['status']!='COMPLETED':sys.exit(1)

"""Shared-threshold sum-equivalent instrumentation for EXP-012/013."""
import sys,os,json,csv,time,hashlib,resource,random,datetime
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from torchvision.datasets import CIFAR10
from timm.optim import create_optimizer_v2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/EXP-004/controlled'));sys.path.insert(0,str(ROOT/'experiments/ROUND-011'))
from core import model_create,ATLIF,DATA,transforms_pair,seed_all,threshold_update
from artg import configure,DIAGNOSTICS
from utils.utils import SOPMonitor
from spikingjelly.activation_based import functional
O=ROOT/'experiments/ROUND-012';D=ROOT/'experiments/EXP-012';E=ROOT/'experiments/EXP-013';CFG=json.loads((O/'config.json').read_text());M=ROOT/'experiments/EXP-004/runs/EXP-004-RUN-003/checkpoint_best_validation.pth';S=ROOT/'experiments/EXP-004/splits/indices.json';I=ROOT/'experiments/EXP-007/calibration_indices.json';START=time.monotonic();CPU0=resource.getrusage(resource.RUSAGE_SELF);PRIOR_WALL=0.;PRIOR_CPU=0.
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def cpu():
 r=resource.getrusage(resource.RUSAGE_SELF);c=resource.getrusage(resource.RUSAGE_CHILDREN);return r.ru_utime+r.ru_stime-CPU0.ru_utime-CPU0.ru_stime+c.ru_utime+c.ru_stime+PRIOR_CPU
def guard(reserve=0):
 if time.monotonic()-START+PRIOR_WALL>3450-reserve:raise TimeoutError('GPU wall budget guard')
 if cpu()>5200-reserve:raise TimeoutError('CPU cumulative budget guard')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check_files():
 for p,k in [(M,'checkpoint_sha256'),(S,'split_sha256'),(I,'calibration_sha256')]:
  if sha(p)!=CFG[k]:raise RuntimeError('locked file checksum mismatch '+str(p))
def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def append(p,rows):
 if not rows:return
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
  if f.tell()==0:w.writeheader()
  w.writerows(rows);f.flush();os.fsync(f.fileno())
def safenum(x):
 v=float(x);return v if np.isfinite(v) else str(v)
def modules(m):return {n:z for n,z in m.named_modules() if isinstance(z,ATLIF)}
def model(group,ck,alpha=1.25):
 g={'D0':'G0','G0-S':'G0','G1-S':'G1','M4-S':'M4','C0':'G0'}[group];m=configure(model_create().cuda(),g);m.load_state_dict(ck['model']);
 for z in modules(m).values():z.thresh.data.mul_(alpha);z.sp=.002;z.update_value=0.
 return m
def optimizer(m):return create_optimizer_v2(m,opt='sgd',lr=.005,weight_decay=.0005,momentum=.9,nesterov=True)
def opt_decay(opt,mods):
 out={}
 for name,m in mods.items():
  matches=[g['weight_decay'] for g in opt.param_groups if any(p is m.thresh for p in g['params'])]
  if len(matches)!=1:raise RuntimeError('threshold missing or duplicated in optimizer')
  out[name]=matches[0]
 return out
class ShapeTracker:
 def __init__(self,m):
  self.shapes={};self.hooks=[]
  for name,z in modules(m).items():
   def hook(mod,args,n=name):
    x=args[0]
    if x.ndim!=5 or x.shape[0]!=8 or n in self.shapes:raise RuntimeError('multiple invocation or unexpected temporal layout '+n)
    self.shapes[n]={'shape':list(x.shape),'N':int(x[0].numel()),'batch':int(x.shape[1])}
   self.hooks.append(z.register_forward_pre_hook(hook))
 def reset(self):self.shapes={}
 def validate(self,m,batch):
  if set(self.shapes)!=set(modules(m)) or any(v['batch']!=batch for v in self.shapes.values()):raise RuntimeError('threshold shape/call mismatch')
 def remove(self):
  for h in self.hooks:h.remove()
def finite_grads(opt):
 vals=[torch.isfinite(p.grad).all() for g in opt.param_groups for p in g['params'] if p.grad is not None]
 return bool(torch.stack(vals).all().item()) if vals else False
def scale_thresholds(mods,shapes):
 for name,m in mods.items():
  if m.thresh.grad is None:raise RuntimeError('missing shared threshold gradient '+name)
  m.thresh.grad.mul_(shapes[name]['N'])
 for name,m in mods.items():
  if not torch.isfinite(m.thresh.grad).all().item():raise FloatingPointError('nonfinite introduced by post-unscale gradient scaling: '+name+' N='+str(shapes[name]['N']))
def ulps(x):
 return torch.nextafter(x,torch.full_like(x,float('inf')))-x, x-torch.nextafter(x,torch.full_like(x,float('-inf')))
def clean(m):
 functional.reset_net(m)
 for z in modules(m).values():z.act_value=0.;z.update_value=0.
def validate_thresholds(m):
 for name,z in modules(m).items():
  if not torch.isfinite(z.thresh).all() or z.thresh.item()<=0:raise RuntimeError('nonpositive or nonfinite threshold '+name)
def digest(m):
 h=hashlib.sha256()
 for n,v in m.state_dict().items():h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
class Indexed(Dataset):
 def __init__(self,ds,ids):self.ds=ds;self.ids=ids
 def __len__(self):return len(self.ids)
 def __getitem__(self,i):
  x,y=self.ds[self.ids[i]];return x,y,self.ids[i]
def loader(ids,training):
 ds=CIFAR10(DATA,train=True,download=False,transform=transforms_pair()[0 if training else 1]);return DataLoader(Indexed(ds,ids),batch_size=128,shuffle=False,num_workers=4,pin_memory=True)
def step(m,opt,scaler,tracker,x,y,group,epoch,batch,mode='sum_equivalent',diagnostic_components=False):
 """Original AMP/threshold-update order, with explicit post-unscale finite guard."""
 guard();mods=modules(m);tracker.reset();opt.zero_grad(set_to_none=True);DIAGNOSTICS.update(enabled=diagnostic_components,scale=scaler.get_scale(),rows=[]);before=torch.stack([z.thresh.detach().clone() for z in mods.values()]);
 with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):logits=m(x).mean(0).float();loss=F.cross_entropy(logits,y)
 if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
 tracker.validate(m,len(y));scale_before=scaler.get_scale();scaler.scale(loss).backward();DIAGNOSTICS['enabled']=False;scaler.unscale_(opt);raw_finite=finite_grads(opt);means={n:safenum(z.thresh.grad.item()) if z.thresh.grad is not None else 'missing' for n,z in mods.items()};momentum_before={n:safenum(opt.state[z.thresh]['momentum_buffer'].item()) if 'momentum_buffer' in opt.state[z.thresh] else None for n,z in mods.items()};manual={n:z.update_value for n,z in mods.items()};fr={n:z.r for n,z in mods.items()};component=list(DIAGNOSTICS['rows']);
 if raw_finite and mode=='sum_equivalent':scale_thresholds(mods,tracker.shapes)
 scaled={n:safenum(z.thresh.grad.item()) if z.thresh.grad is not None else 'missing' for n,z in mods.items()};scaler.step(opt);afteropt=torch.stack([z.thresh.detach().clone() for z in mods.values()]);threshold_update(m,opt.param_groups[0]['lr']);aftermanual=torch.stack([z.thresh.detach().clone() for z in mods.values()]);scaler.update();skipped=scaler.get_scale()<scale_before
 if skipped==raw_finite:raise RuntimeError('AMP skip and gradient-finiteness disagree')
 validate_thresholds(m);up,down=ulps(before);lr=opt.param_groups[0]['lr'];rows=[];decay=opt_decay(opt,mods)
 for i,(name,z) in enumerate(mods.items()):
  old=before[i].item();opt_delta=(afteropt[i]-before[i]).item();manual_delta=(aftermanual[i]-afteropt[i]).item();sg=scaled[name];nominal=-lr*sg if isinstance(sg,float) else sg;mb=momentum_before[name];ma=safenum(opt.state[z.thresh]['momentum_buffer'].item()) if 'momentum_buffer' in opt.state[z.thresh] else None;rr={'group':group,'epoch':epoch,'batch':batch,'module':name,'N':tracker.shapes[name]['N'],'batch_actual':len(y),'threshold_before':old,'mean_task_gradient':means[name],'scaled_task_gradient':sg,'learning_rate':lr,'momentum_before':mb,'momentum_after':ma,'weight_decay':decay[name],'ulp_up':up[i].item(),'ulp_down':down[i].item(),'nominal_sgd_delta':nominal,'optimizer_delta':opt_delta,'manual_update_value':manual[name],'manual_delta':manual_delta,'threshold_after':aftermanual[i].item(),'firing_rate':fr[name],'AMP_skipped':skipped,'raw_gradients_finite':raw_finite}
  for k in ['direct_mean','state_mean','reset_mean','direct_abs_mean','state_abs_mean','reset_abs_mean']:
   vv=[z[k] for z in component if z['module']==name];rr[k]=safenum(sum(vv)) if vv else '';rr[k+'_sum_equivalent']=safenum(sum(vv)*tracker.shapes[name]['N']) if vv and raw_finite and mode=='sum_equivalent' else ''
  rows.append(rr)
 if not skipped and any(abs(row['optimizer_delta'])>=row['threshold_before'] for row in rows):raise RuntimeError('obvious threshold update explosion')
 n=len(y);accuracy=int((logits.argmax(1)==y).sum().item());clean(m);return {'loss':loss.item(),'correct':accuracy,'samples':n,'skipped':skipped,'raw_finite':raw_finite,'rows':rows,'shapes':tracker.shapes.copy()}

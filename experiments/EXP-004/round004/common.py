"""ROUND-004 helpers. No original baseline files modified."""
import sys,os,json,hashlib,copy,time,csv,math,random
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/EXP-004/controlled'))
from core import ATLIF,model_create,opt_create,sched_create,seed_all,loaders,isolated_eval,threshold_update,regularize_spike,functional,rng_get,rng_set,SPLIT,DATA
from torchvision.datasets import CIFAR10
EXPECTED_SPLIT='69dccfe992236766aea575a0a66495802ca889ca329d61f51e8de3c1ab7f3165'
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def validate_split():
 if sha(SPLIT)!=EXPECTED_SPLIT:raise RuntimeError('split SHA mismatch')
 d=json.loads(SPLIT.read_text());sets=[set(d[k]) for k in ('train','validation','diagnostic')];q=d['diagnostic_256']
 if [len(s) for s in sets]!=[45000,4500,500] or len(q)!=256 or len(set(q))!=256 or not set(q)<=sets[2] or sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2] or set.union(*sets)!=set(range(50000)):raise RuntimeError('invalid split')
 for name,expected in d['metadata']['train_batch_sha256'].items():
  if sha(Path(DATA)/'cifar-10-batches-py'/name)!=expected:raise RuntimeError('CIFAR file SHA mismatch: '+name)
 dataset=CIFAR10(DATA,train=True,download=False)
 for k in ('train','validation','diagnostic'):
  counts=np.bincount(np.array(dataset.targets)[d[k]],minlength=10)
  if len(set(counts))!=1:raise RuntimeError('unbalanced split')
 return {'split_sha256':EXPECTED_SPLIT,'counts':[len(s) for s in sets],'diagnostic_256':len(q),'dataset_integrity':dataset._check_integrity()}
class Activity:
 def __init__(self,model):
  self.values={};self.handles=[]
  for n,m in model.named_modules():
   if isinstance(m,ATLIF):self.handles.append(m.register_forward_hook(self.hook(n)))
 def hook(self,n):
  def f(m,args,out):
   count=(out.detach()>0).sum(dtype=torch.int64);states=out.numel();rate=count.to(torch.float64)/states
   if n not in self.values:self.values[n]=[count,states,rate,1]
   else:
    a=self.values[n];a[0]=a[0]+count;a[1]+=states;a[2]=a[2]+rate;a[3]+=1
  return f
 def rows(self):
  result={}
  for n,(count,states,rates,batches) in self.values.items():
   count=count.item();result[n]={'spike_count':count,'neuron_time_states':states,'fr':count/states,'author_batch_mean_fr':rates.item()/batches,'batches':batches}
  return result
 def close(self):
  for h in self.handles:h.remove()
def global_fr(rows):return sum(x['spike_count'] for x in rows.values())/sum(x['neuron_time_states'] for x in rows.values())
def train_step(model,opt,scaler,x,y):
 opt.zero_grad(set_to_none=True)
 with torch.cuda.amp.autocast(enabled=True):
  logits=model(x).mean(0);loss=torch.nn.functional.cross_entropy(logits,y)+0.*regularize_spike(model)
 if not torch.isfinite(loss):raise RuntimeError('non-finite loss')
 old=scaler.get_scale();scaler.scale(loss).backward();scaler.step(opt);threshold_update(model,opt.param_groups[0]['lr']);scaler.update();functional.reset_net(model)
 if any(not torch.isfinite(m.thresh).all() or (m.thresh<=0).any() for m in model.modules() if isinstance(m,ATLIF)):raise RuntimeError('non-positive or non-finite threshold')
 return loss.detach(),logits.detach(),scaler.get_scale()<old
def validate(model,loader,device,max_batches=None):
 count=0;correct=0;loss=0.;stats=Activity(model)
 try:
  with isolated_eval(model):
   for i,(x,y) in enumerate(loader):
    x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True);logits=model(x).mean(0)
    loss+=torch.nn.functional.cross_entropy(logits,y,reduction='sum').item();correct+=(logits.argmax(1)==y).sum().item();count+=len(y);functional.reset_net(model)
    if max_batches and i+1>=max_batches:break
  rows=stats.rows()
 finally:stats.close()
 return {'loss':loss/count,'accuracy':correct/count,'count':count,'fr_global':global_fr(rows),'fr_author':sum(v['author_batch_mean_fr'] for v in rows.values())/len(rows)},rows
def pack(model,opt,sched,scaler,epoch,best,config,code_sha):
 return {'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'best_validation_accuracy':best,'rng':rng_get(),'atlif_state':{n:{k:(getattr(m,k).detach().clone() if torch.is_tensor(getattr(m,k)) else copy.deepcopy(getattr(m,k))) for k in ('update_value','r','s','act_value')} for n,m in model.named_modules() if isinstance(m,ATLIF)},'config':config,'split_sha256':EXPECTED_SPLIT,'code_sha':code_sha}
def atomic_save(path,obj):
 tmp=path.with_suffix(path.suffix+'.tmp')
 try:
  with open(tmp,'wb') as f:torch.save(obj,f);f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if tmp.exists():tmp.unlink()
def restore(path,model,opt,sched,scaler):
 ck=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler']);scaler.load_state_dict(ck['scaler'])
 for n,attrs in ck['atlif_state'].items():
  for k,v in attrs.items():setattr(dict(model.named_modules())[n],k,v.to(next(model.parameters()).device) if torch.is_tensor(v) else v)
 rng_set(ck['rng']);return ck

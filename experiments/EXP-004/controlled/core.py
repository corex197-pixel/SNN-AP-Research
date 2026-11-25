"""Controlled baseline adapters; original ATLIF algorithm is unchanged."""
import copy,contextlib,random,sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets,transforms
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'Activity-Pruning-SNN-main'))
from models.submodules.layers import ATLIF
from models.resnet_cifar import ResNet20
from utils.utils import threshold_update,regularize_spike
from spikingjelly.activation_based import functional
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from utils.augment import Cutout

DATA='/home/hym/dataset/cifar10'
SPLIT=ROOT/'experiments/EXP-004/splits/indices.json'
def rng_get():
 return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(torch.cuda.current_device()) if torch.cuda.is_initialized() else None}
def rng_set(r):
 random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'])
 if r['cuda'] is not None:torch.cuda.set_rng_state(r['cuda'],torch.cuda.current_device())
@contextlib.contextmanager
def isolated_eval(model):
 modes={m:m.training for m in model.modules()}; rng=rng_get()
 buffers={n:b.detach().clone() for n,b in model.named_buffers()}
 params={n:p.detach().clone() for n,p in model.named_parameters()}
 attrs={m:{k:getattr(m,k) for k in ('update_value','r','s','act_value')} for m in model.modules() if isinstance(m,ATLIF)}
 memories={m:copy.deepcopy(m._memories) for m in model.modules() if hasattr(m,'_memories')}
 try:
  model.eval()
  with torch.no_grad(): yield
 finally:
  with torch.no_grad():
   for n,p in model.named_parameters():
    if not torch.equal(p,params[n]):p.copy_(params[n])
   for n,b in model.named_buffers():b.copy_(buffers[n])
  for m,a in attrs.items():
   for k,v in a.items():setattr(m,k,v)
  for m,v in memories.items():m._memories=v
  for m,v in modes.items():m.training=v
  rng_set(rng)

def model_create():return ResNet20(T=8,activation=ATLIF,activation_kwargs={'T':8,'sp':.002,'tau':.9})
def opt_create(model):return create_optimizer_v2(model,opt='sgd',lr=.1,weight_decay=5e-4)
def sched_create(opt):return create_scheduler_v2(opt,sched='cosine',num_epochs=200,cooldown_epochs=10,min_lr=1e-6,warmup_lr=0.,warmup_epochs=0)[0]
def seed_all(seed=42):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def transforms_pair():
 norm=transforms.Normalize((.4914,.4822,.4465),(.2023,.1994,.2010))
 return transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),norm,Cutout(n_holes=1,length=16)]),transforms.Compose([transforms.ToTensor(),norm])
def datasets_split():
 data=json.loads(SPLIT.read_text()); train_t,eval_t=transforms_pair()
 a=datasets.CIFAR10(DATA,train=True,download=False,transform=train_t)
 b=datasets.CIFAR10(DATA,train=True,download=False,transform=eval_t)
 return Subset(a,data['train']),Subset(b,data['validation']),Subset(b,data['diagnostic'])
def loaders(batch_size=128,workers=4):
 a,b,_=datasets_split()
 return DataLoader(a,batch_size=batch_size,num_workers=workers,pin_memory=True,shuffle=False),DataLoader(b,batch_size=batch_size,num_workers=workers,pin_memory=True,shuffle=False)
def step(model,opt,scaler,x,y):
 opt.zero_grad(set_to_none=True)
 with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
  output=model(x);loss=nn.functional.cross_entropy(output.mean(0),y)+0.*regularize_spike(model)
 if not torch.isfinite(loss):raise RuntimeError('non-finite loss')
 old_scale=scaler.get_scale()
 if scaler.is_enabled():scaler.scale(loss).backward();scaler.step(opt)
 else:loss.backward();opt.step()
 # Preserves original baseline behaviour, including AMP overflow case.
 threshold_update(model,opt.param_groups[0]['lr'])
 if scaler.is_enabled():scaler.update()
 functional.reset_net(model)
 vals=[m.thresh.detach() for m in model.modules() if isinstance(m,ATLIF)]
 if any(not torch.isfinite(v).all() or (v<=0).any() for v in vals):raise RuntimeError('threshold nonfinite or nonpositive')
 return float(loss.detach()),bool(scaler.is_enabled() and scaler.get_scale()<old_scale)
def evaluate(model,loader,device,max_batches=None):
 loss=0.;correct=0;count=0
 with isolated_eval(model):
  for i,(x,y) in enumerate(loader):
   x=x.to(device);y=y.to(device);logits=model(x).mean(0)
   loss+=nn.functional.cross_entropy(logits,y,reduction='sum').item();correct+=(logits.argmax(1)==y).sum().item();count+=y.numel();functional.reset_net(model)
   if max_batches is not None and i+1>=max_batches:break
 return {'loss':loss/count,'top1':correct/count,'count':count}
def save(path,model,opt,sched,scaler,epoch,best):
 obj={'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'best_validation_accuracy':best,'rng':rng_get(),'atlif_state':{n:{k:copy.deepcopy(getattr(m,k)) if k!='act_value' else getattr(m,k).detach().clone() if torch.is_tensor(getattr(m,k)) else getattr(m,k) for k in ('update_value','r','s','act_value')} for n,m in model.named_modules() if isinstance(m,ATLIF)}}
 torch.save(obj,path)
def restore(path,model,opt,sched,scaler,device):
 ck=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler']);scaler.load_state_dict(ck['scaler'])
 for n,attrs in ck['atlif_state'].items():
  for k,v in attrs.items():setattr(dict(model.named_modules())[n],k,v)
 rng_set(ck['rng']);return ck['epoch'],ck['best_validation_accuracy']

"""Fixed EXP-029 models, FP32 KD formulas and sample controls."""
import os,sys,time,json,hashlib,csv,random,importlib.util,math,subprocess,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT);sys.path[:0]=[str(ROOT/'experiments/ROUND-019'),str(ROOT/'experiments/ROUND-022')]
import core as K
import numpy as np,torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset
from spikingjelly.activation_based import functional
from models.submodules.layers import SparseConv2d,SparseNeuron,SparseBN
spec=importlib.util.spec_from_file_location('compact_esc',ROOT/'experiments/ROUND-022/core.py');esc=importlib.util.module_from_spec(spec);spec.loader.exec_module(esc)
E=ROOT/'experiments/EXP-029';CFG=json.loads((E/'config.json').read_text());OLD=ROOT/'experiments/EXP-026';NAMES=CFG['groups'];DGRAD=K.S['D_grad'];DHC=CFG['D_grad_sha256'];SPL=ROOT/'experiments/EXP-004/splits/indices.json';B0=OLD/'models/B0.pth';P0=OLD/'models/75-R1.pth';CODE=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
def sha(p):return K.sha(p)
def idhash(x):return hashlib.sha256(json.dumps(list(x),separators=(',',':')).encode()).hexdigest()
def dump(n,x):
 p=E/n;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(q,p)
def write(n,rows):
 if not rows:return
 with (E/n).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def statehash(m):
 h=hashlib.sha256()
 for n,v in m.state_dict().items():h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def snapshot(m):return {n:v.detach().cpu().clone() for n,v in m.state_dict().items()}
def init_runtime():
 assert torch.cuda.device_count()==1 and '4090' in torch.cuda.get_device_name();torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;random.seed(12450);np.random.seed(12450);torch.manual_seed(12450);torch.cuda.manual_seed(12450)
def model_teacher():
 assert sha(B0)==CFG['teacher_sha256'];m=K.model();c=torch.load(B0,map_location='cpu',weights_only=False);assert c['group']=='B0';m.load_state_dict(c['state_dict']);m.eval();return m
def model_p0(teacher=None):
 assert sha(P0)==CFG['p0_sha256'];c=torch.load(P0,map_location='cpu',weights_only=False);assert c['group']=='75-R1';own=teacher is None;teacher=K.model() if own else teacher;s=json.loads((OLD/'selected_neurons.json').read_text())['ratios']['75']['R1'];m,keep=esc.compact(teacher,s['deleted']);assert keep.cpu().tolist()==c['retained']==s['retained'];m.load_state_dict(c['state_dict']);m.eval()
 for param in m.parameters():param.requires_grad_(False)
 if own:del teacher
 return m,keep,s

def setup_student(m):
 for n,p in m.named_parameters():p.requires_grad_(n in ('fc2.weight','fc2.bias'))
 m.eval();m.fc2.train();opt=torch.optim.SGD([p for p in m.parameters() if p.requires_grad],lr=.01,momentum=.9,weight_decay=0);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=5,eta_min=0);sc=torch.amp.GradScaler('cuda');return opt,sch,sc
def forward(m,x,capture=False):
 cache={};h=None
 if capture:h=m.activation_fc1.register_forward_hook(lambda mod,args,out:cache.update(spikes=out.detach().clone()))
 functional.reset_net(m)
 try:z=m(x)
 finally:
  functional.reset_net(m)
  if h:h.remove()
 return z,cache.get('spikes')
def eval_loader(ids):return DataLoader(K.TrainFileOnly(ids,False),batch_size=64,shuffle=False,num_workers=4,pin_memory=True,drop_last=False)
def training_loader(dataset,ids,epoch):
 perm=torch.randperm(len(ids),generator=torch.Generator().manual_seed(12450+epoch)).tolist();seq=[ids[i] for i in perm[:44800]];dropped=[ids[i] for i in perm[44800:]];dl=DataLoader(Subset(dataset,perm[:44800]),batch_size=256,shuffle=False,num_workers=4,pin_memory=True,drop_last=True,generator=torch.Generator().manual_seed(99912450+epoch));return dl,seq,dropped
def weights(zt,zp,y,rho=.5):
 zt=zt.float();zp=zp.float();ix=y.view(1,-1,1).expand(zt.shape[0],-1,1);true=zt.gather(2,ix).squeeze(2);comp=zt.clone().scatter(2,ix,-float('inf')).argmax(2);mt=true-zt.gather(2,comp.unsqueeze(2)).squeeze(2);mp=zp.gather(2,ix).squeeze(2)-zp.gather(2,comp.unsqueeze(2)).squeeze(2);d=(mt>0).float()*(mt-mp).clamp_min(0);den=d.sum(0,keepdim=True);safe=torch.where(den>0,den,torch.ones_like(den));w=(1-rho)+rho*zt.shape[0]*d/safe;w=torch.where(den>0,w,torch.ones_like(w));return d.detach(),w.detach(),comp.detach(),mt.detach(),mp.detach()
def losses(group,zs,y,zt=None,zp=None):
 # CE has precisely the prior accepted training AMP context. KL is explicitly FP32.
 ce=F.cross_entropy(zs.mean(0),y)
 if group=='K0':return ce,ce,torch.zeros((),device=zs.device),None,None,None
 with torch.autocast('cuda',enabled=False):
  zsf=zs.float();ztf=zt.detach().float();T=4.;pt=F.softmax(ztf/T,dim=-1);lt=F.log_softmax(ztf/T,dim=-1);ls=F.log_softmax(zsf/T,dim=-1);kl=(pt*(lt-ls)).sum(-1)
  if group=='K1':
   a=ztf.mean(0)/T;b=zsf.mean(0)/T;p=F.softmax(a,dim=-1);kd=(p*(F.log_softmax(a,dim=-1)-F.log_softmax(b,dim=-1))).sum(-1).mean()*16;w=d=None
  elif group=='K2':kd=kl.mean()*16;w=d=None
  else:
   d,w,_,_,_=weights(ztf,zp.detach(),y,.5);actual=w.roll(shifts=-1,dims=0) if group=='K3' else w;kd=(actual*kl).mean()*16
 return ce+.5*kd,ce,kd,d,w,kl.detach()
def init_ledger():
 p=E/'resource_ledger.json'
 if not p.exists():dump('resource_ledger.json',{'gpu_process_wall_seconds':0.,'cpu_accounted_seconds':120.,'cpu_initial_reserve_seconds':120.,'stages':[]})
def budgets():
 init_ledger();return json.loads((E/'resource_ledger.json').read_text())
def atomic_torch(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');torch.save(obj,tmp);os.replace(tmp,p)

manual_calls={'count':0}
def _forbid_manual_penalty(*args,**kwargs):
 manual_calls['count']+=1
 raise RuntimeError('manual sparsity penalty is forbidden')
SparseConv2d.penalize=_forbid_manual_penalty
SparseBN.penalize=_forbid_manual_penalty

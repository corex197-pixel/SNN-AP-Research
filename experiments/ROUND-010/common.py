"""ROUND-010 isolated no-gradient model, BN calibration and SOP evaluation."""
import os,sys,time,json,csv,hashlib,copy,math,resource,datetime
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset
from torchvision.datasets import CIFAR10
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/EXP-004/controlled'))
from core import model_create,transforms_pair,DATA,ATLIF
from utils.utils import SOPMonitor
from spikingjelly.activation_based import functional
CFG=json.loads((ROOT/'experiments/ROUND-010/config.json').read_text());S=ROOT/'experiments/EXP-004/splits/indices.json';I=ROOT/'experiments/EXP-007/calibration_indices.json'
PATHS={'C0':ROOT/'experiments/EXP-004/runs/EXP-004-RUN-003/checkpoint_best_validation.pth','C1':ROOT/'experiments/EXP-008/runs/EXP-008-F0-RUN-001/epoch-005.pth','C2':ROOT/'experiments/EXP-008/runs/EXP-008-F2-RUN-001/epoch-005.pth'}
OUT=ROOT/'experiments/ROUND-010'; start=time.monotonic();resource0=resource.getrusage(resource.RUSAGE_SELF);prior_wall=prior_cpu=0.;counts={'forwards':0,'samples':0};latest=''
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cpu():
 r=resource.getrusage(resource.RUSAGE_SELF);ch=resource.getrusage(resource.RUSAGE_CHILDREN);return r.ru_utime+r.ru_stime-resource0.ru_utime-resource0.ru_stime+ch.ru_utime+ch.ru_stime+prior_cpu
def guard(reserve=0):
 if time.monotonic()-start+prior_wall>1710-reserve:raise TimeoutError('GPU wall safety budget')
 if cpu()>2550-reserve:raise TimeoutError('CPU safety budget')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def append(p,rows):
 if not rows:return
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
  if f.tell()==0:w.writeheader()
  w.writerows(rows);f.flush();os.fsync(f.fileno())
def clean(model):
 functional.reset_net(model)
 for m in model.modules():
  if isinstance(m,ATLIF):m.update_value=0.;m.act_value=0.;m.r=0.;m.s=0.
def checksums():
 for k,p in PATHS.items():
  if sha(p)!=CFG['checkpoint_sha256'][k]:raise RuntimeError('checkpoint checksum mismatch '+k)
 for p,key in [(S,'split_sha256'),(I,'calibration_sha256')]:
  if sha(p)!=CFG[key]:raise RuntimeError('dataset index checksum mismatch')
def load():
 global CK,REF,SP,IDS
 checksums();CK={n:torch.load(p,map_location='cpu',weights_only=False) for n,p in PATHS.items()};REF={n:t.clone() for n,t in CK['C0']['model'].items() if n.endswith('thresh')};SP=json.loads(S.read_text());IDS=json.loads(I.read_text())['indices'];assert len(REF)==19 and len(IDS)==500 and len(set(IDS))==500 and set(IDS)<=set(SP['train']) and not set(IDS)&set(SP['validation']) and not set(IDS)&set(SP['diagnostic'])
 for g,ck in CK.items():
  assert ck['epoch']==(193 if g=='C0' else 5)
  for n,t in ck['model'].items():
   if not torch.isfinite(t).all():raise RuntimeError('nonfinite model '+g+' '+n)
  for n,t in REF.items():
   if not torch.allclose(ck['model'][n],t*(1. if g=='C0' else 1.25),rtol=1e-6,atol=1e-7):raise RuntimeError('unexpected stored threshold '+g+' '+n)
def model(g,alpha):
 m=model_create().cuda();m.load_state_dict(CK[g]['model']);m.eval()
 for p in m.parameters():p.requires_grad_(False)
 for n,z in m.named_modules():
  if isinstance(z,ATLIF):
   z.thresh.data.copy_(REF[n+'.thresh'].to(z.thresh.device)*alpha);z.sp=0.;z.update_value=0.
   if not torch.isfinite(z.thresh).all() or z.thresh.item()<=0:raise RuntimeError('threshold nonpositive')
 return m
def bnmap(m):return {n:z for n,z in m.named_modules() if isinstance(z,nn.modules.batchnorm._BatchNorm)}
def batches(ids):
 ds=CIFAR10(DATA,train=True,download=False,transform=transforms_pair()[1]);return list(DataLoader(Subset(ds,ids),batch_size=128,shuffle=False,num_workers=4,pin_memory=True))
def forward(m,x):
 global latest
 guard();counts['forwards']+=1;counts['samples']+=x.size(0);latest=now();return m(x)
class Moments:
 def __init__(self):self.n=0;self.mean=None;self.m2=None;self.shapes=[]
 def add(self,x):
  if x.ndim==5: x=x.permute(2,0,1,3,4)
  elif x.ndim==4:x=x.permute(1,0,2,3)
  else:raise RuntimeError('unexpected BN input layout '+str(x.shape))
  self.shapes.append(tuple(x.shape));x=x.reshape(x.shape[0],-1).double();v,mu=torch.var_mean(x,dim=1,correction=0);n=x.shape[1];self.merge(n,mu,v*n)
 def merge(self,n,mu,m2):
  if self.n==0:self.n=n;self.mean=mu;self.m2=m2;return
  total=self.n+n;d=mu-self.mean;self.m2=self.m2+m2+d*d*(self.n*n/total);self.mean=self.mean+d*(n/total);self.n=total
 def result(self):
  if not self.n:raise RuntimeError('empty moments')
  return self.mean,self.m2/self.n

def adaptive(m,cal):
 bn=bnmap(m);before={n:(v.momentum,v.training) for n,v in bn.items()}
 for z in bn.values():z.reset_running_stats();z.momentum=None;z.train()
 try:
  with torch.no_grad():
   for x,_ in cal:forward(m,x.cuda());clean(m)
 finally:
  for n,z in bn.items():z.eval();z.momentum=before[n][0]
 return {'batches':len(cal),'last_batch':len(cal[-1][1]),'method':'PyTorch cumulative equal-batch momentum; running_var uses unbiased per-batch estimate'}

def sequential(m,cal,order):
 bn=bnmap(m)
 for name in order:
  mom=Moments();h=bn[name].register_forward_pre_hook(lambda z,args:mom.add(args[0]))
  try:
   with torch.no_grad():
    for x,_ in cal:forward(m,x.cuda());clean(m)
  finally:h.remove()
  mean,var=mom.result()
  if not torch.isfinite(mean).all() or not torch.isfinite(var).all() or (var<0).any():raise RuntimeError('invalid BN moments')
  bn[name].running_mean.copy_(mean.to(bn[name].running_mean));bn[name].running_var.copy_(var.to(bn[name].running_var));bn[name].num_batches_tracked.fill_(0)
 return {'BN_count':len(order),'batches_per_BN':len(cal),'last_batch':len(cal[-1][1]),'method':'float64 Chan merge, population variance; sequential eval BN'}
def bnstate(m):
 return {n:{'mean':z.running_mean.detach().cpu().tolist(),'var':z.running_var.detach().cpu().tolist(),'tracked':int(z.num_batches_tracked.item())} for n,z in bnmap(m).items()}
def setbn(m,state):
 for n,z in bnmap(m).items():
  a=state[n];z.running_mean.copy_(torch.tensor(a['mean'],device=z.running_mean.device,dtype=z.running_mean.dtype));z.running_var.copy_(torch.tensor(a['var'],device=z.running_var.device,dtype=z.running_var.dtype));z.num_batches_tracked.fill_(a['tracked'])

def evaluate(m,cal,ids,tag,detail=False,bn_training=False,bn_diag=False):
 m.eval();bn=bnmap(m)
 if bn_training:
  for z in bn.values():z.train()
 mon=SOPMonitor(m);mods={n:z for n,z in m.named_modules() if isinstance(z,ATLIF)};spikes={};states={};rates={};bcount={};hooks=[];ins={};outs={};bn_before=bnstate(m) if bn_diag else None
 for name,z in mods.items():
  def hook(mod,a,o,n=name):
   v=(o.detach()>0).sum(dtype=torch.int64);spikes[n]=spikes.get(n,0)+v;states[n]=states.get(n,0)+o.numel();rates[n]=rates.get(n,0)+v.double()/o.numel();bcount[n]=bcount.get(n,0)+1
  hooks.append(z.register_forward_hook(hook))
 if bn_diag:
  for name,z in bn.items():
   ins[name]=Moments();outs[name]=Moments()
   def hook(mod,a,o,n=name):ins[n].add(a[0]);outs[n].add(o)
   hooks.append(z.register_forward_hook(hook))
 t=time.monotonic();per=[];br=[];sop=0.;count=0
 try:
  with torch.no_grad():
   for j,(x,y) in enumerate(cal):
    x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);mon.clear_recorded_data();z=forward(m,x).mean(0).float();loss=F.cross_entropy(z,y,reduction='none');pred=z.argmax(1);n=len(y);total=sum(float(r.item()) for r in mon.records);sop+=total
    if not torch.isfinite(z).all() or not torch.isfinite(loss).all():raise RuntimeError('nonfinite output')
    if detail:
     br.append({'tag':tag,'batch':j,'sample_count':n,'sop_count':total})
     for ix,l,p,y0 in zip(ids[count:count+n],loss.cpu().tolist(),pred.cpu().tolist(),y.cpu().tolist()):per.append({'tag':tag,'sample_id':int(ix),'loss':l,'prediction':p,'correct':int(p==y0)})
    else:
     for l,p,y0 in zip(loss.cpu().tolist(),pred.cpu().tolist(),y.cpu().tolist()):per.append({'loss':l,'correct':int(p==y0)})
    count+=n;clean(m)
 finally:
  mon.remove_hooks()
  for h in hooks:h.remove()
 lay={n:{'FR':int(spikes[n].item())/states[n],'spikes':int(spikes[n].item()),'states':states[n],'author_FR':float((rates[n]/bcount[n]).item()),'threshold':float(z.thresh.item())} for n,z in mods.items()}
 out={'tag':tag,'accuracy':float(np.mean([r['correct'] for r in per])),'loss':float(np.mean([r['loss'] for r in per])),'SOPs':sop/count,'FR_global':sum(v['spikes'] for v in lay.values())/sum(v['states'] for v in lay.values()),'FR_author':float(np.mean([v['author_FR'] for v in lay.values()])),'n':count,'seconds':time.monotonic()-t,'parameter_count':sum(p.numel() for p in m.parameters())}
 diags=[]
 if bn_diag:
  after=bnstate(m)
  for name,z in bn.items():
   im,iv=ins[name].result();om,ov=outs[name].result()
   for c in range(len(im)):
    diags.append({'tag':tag,'BN':name,'channel':c,'input_mean':im[c].item(),'input_var_population':iv[c].item(),'output_mean':om[c].item(),'output_var_population':ov[c].item(),'running_mean_before':bn_before[name]['mean'][c],'running_var_before':bn_before[name]['var'][c],'running_mean_after':after[name]['mean'][c],'running_var_after':after[name]['var'][c],'input_count':ins[name].n})
 return out,lay,per,br,diags

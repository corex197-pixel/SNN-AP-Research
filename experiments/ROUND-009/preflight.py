import os,sys,json,time,hashlib,traceback,datetime
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader,Subset
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/EXP-004/controlled'));sys.path.insert(0,str(ROOT/'experiments/ROUND-009'))
from core import model_create,transforms_pair,DATA,ATLIF
from proxy import Proxy,fanout
from utils.utils import SOPMonitor
M=ROOT/'experiments/EXP-004/runs/EXP-004-RUN-003/checkpoint_best_validation.pth';S=ROOT/'experiments/EXP-004/splits/indices.json';I=ROOT/'experiments/EXP-007/calibration_indices.json';OUT=ROOT/'experiments/EXP-008/runs/PREFLIGHT-001';OUT.mkdir(parents=True,exist_ok=False)
start=time.monotonic();proc=time.process_time();result={'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':{},'status':'RUNNING'}
def save(): (OUT/'smoke.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
def main():
 torch.set_num_threads(2)
 if torch.cuda.device_count()!=1 or '4090' not in torch.cuda.get_device_name(0):raise RuntimeError('exactly one RTX 4090 required')
 for p,expected in [(M,'c068f54379cb5d29bcfa5fe2d9de676cd3cc153433ada1ed5aa1c66355d0b32a'),(S,'69dccfe992236766aea575a0a66495802ca889ca329d61f51e8de3c1ab7f3165'),(I,'b849b74ef19cb4a09df756c54871122f3b02aa047c704d0436feab15a38bde52')]:
  if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise RuntimeError('checksum mismatch '+str(p))
 # Independently enumerate connections including padding, stride, groups, dilation and borders.
 cases=[]
 for ci,co,g,k,stride,pad,dil,h,w in [(2,3,1,3,1,1,1,5,4),(4,6,2,3,2,1,1,6,5),(2,2,2,2,2,0,1,5,6),(2,4,1,2,1,1,2,5,6)]:
  conv=nn.Conv2d(ci,co,k,stride,pad,dilation=dil,groups=g,bias=False);actual=fanout(conv,(ci,h,w),torch.device('cpu'))[0];expect=torch.zeros_like(actual);oh=(h+2*pad-dil*(k-1)-1)//stride+1;ow=(w+2*pad-dil*(k-1)-1)//stride+1
  for ic in range(ci):
   for iy in range(h):
    for ix in range(w):
     for oy in range(oh):
      for ox in range(ow):
       for ky in range(k):
        for kx in range(k):
         if iy==oy*stride-pad+ky*dil and ix==ox*stride-pad+kx*dil:expect[ic,iy,ix]+=co//g
  diff=(actual-expect).abs();cases.append({'case':[ci,co,g,k,stride,pad,dil,h,w],'max_absolute_error':diff.max().item()})
  if diff.max().item()!=0:raise RuntimeError('synthetic fanout mismatch')
 result['checks']['synthetic_fanout']=cases;save()
 split=json.loads(S.read_text());ids=json.loads(I.read_text())['indices'];assert len(ids)==500 and set(ids)<=set(split['train']) and not(set(ids)&set(split['validation'])) and not(set(ids)&set(split['diagnostic']))
 data=CIFAR10(DATA,train=True,download=False,transform=transforms_pair()[1]);loader=DataLoader(Subset(data,ids),batch_size=128,shuffle=False,num_workers=0)
 model=model_create().cuda().eval();model.load_state_dict(torch.load(M,map_location='cpu',weights_only=False)['model']);mods=[m for m in model.modules() if isinstance(m,ATLIF)];base=[m.thresh.detach().clone() for m in mods]
 for p in model.parameters():p.requires_grad_(True)
 for m in mods:m.thresh.requires_grad_(False);m.sp=0
 p=Proxy(model);mon=SOPMonitor(model);examples=[];calcounts={}
 try:
  for alpha in [1.,1.25]:
   for m,b in zip(mods,base):m.thresh.data.copy_(b*alpha)
   total=0;count=0
   with torch.no_grad():
    for i,(x,y) in enumerate(loader):
     x=x.cuda();p.clear();mon.clear_recorded_data();z=model(x);v=float(p.value().item());orig=sum(t.item() for t in mon.records)/len(y);err=abs(v-orig);rel=err/max(abs(orig),1);examples.append({'alpha':alpha,'batch':len(y),'proxy':v,'original':orig,'absolute_error':err,'relative_error':rel});total+=orig*len(y);count+=len(y)
     if rel>1e-5:raise RuntimeError('proxy relative error exceeds 1e-5')
   calcounts[str(alpha)]=total/count
  expected=set(mon.monitored_layers)
  if set(p.mappings)!=expected:raise RuntimeError('incomplete convolution mapping')
  mapping=dict(p.mappings);(OUT/'sop_mapping.json').write_text(json.dumps(mapping,indent=2)+'\n')
  result['checks']['counter_cases']=examples;result['checks']['calibration_sops']=calcounts;result['checks']['C0_locked']=54945117.504;result['checks']['C0_absolute_difference']=calcounts['1.0']-54945117.504;result['checks']['C0_relative_difference']=abs(calcounts['1.0']-54945117.504)/54945117.504
  if result['checks']['C0_relative_difference']>1e-5:raise RuntimeError('calibration baseline mismatch')
  x,y=next(iter(loader));x=x[:4].cuda();mon.remove_hooks();p.clear()
  with torch.no_grad():zp=model(x).detach().clone()
  p.remove();
  with torch.no_grad():zn=model(x).detach().clone()
  logitdiff=(zp-zn).abs().max().item();result['checks']['instrumentation_max_logit_difference']=logitdiff
  if logitdiff!=0:raise RuntimeError('proxy changes logits')
  p=Proxy(model);model.train();p.clear();model.zero_grad(set_to_none=True);model(x);v=p.value();hinge=F.relu(v/54945117.504-0.0).square();hinge.backward();grad=[z.grad.detach().abs().sum().item() for n,z in model.named_parameters() if z.requires_grad and z.grad is not None];nonzero=sum(g>0 for g in grad)
  if not nonzero or not all(torch.isfinite(torch.tensor(grad))):raise RuntimeError('SOP gradient absent/nonfinite')
  dummy=torch.tensor(0.7,requires_grad=True);F.relu(dummy-0.8).square().backward()
  result['checks']['gradient']={'over_budget_nonzero_parameter_tensors':nonzero,'over_budget_abs_grad_sum':sum(grad),'under_budget_hinge_gradient':dummy.grad.item(),'threshold_grad_none':all(m.thresh.grad is None for m in mods)}
  if dummy.grad.item()!=0 or not all(m.thresh.grad is None for m in mods):raise RuntimeError('hinge or threshold gradient incorrect')
  result['checks']['residual_shortcut_count']=[n for n in mapping if '.shortcut.' in n]
  if len(result['checks']['residual_shortcut_count'])!=2:raise RuntimeError('shortcut count mismatch')
  result['status']='PASSED'
 finally:
  p.remove()
try:main()
except BaseException as e:result['status']='FAILED';result['error']=repr(e);result['traceback']=traceback.format_exc();traceback.print_exc()
finally:result['end_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();result['wall_seconds']=time.monotonic()-start;result['cpu_seconds']=time.process_time()-proc;save();print(json.dumps(result,indent=2),flush=True)
if result['status']!='PASSED':sys.exit(1)

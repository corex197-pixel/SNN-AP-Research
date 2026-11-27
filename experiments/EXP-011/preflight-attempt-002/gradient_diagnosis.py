import sys,json,hashlib,time
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
R=Path('/home/hym/keyan1');sys.path.insert(0,str(R/'experiments/EXP-004/controlled'));sys.path.insert(0,str(R/'experiments/ROUND-011'))
from core import model_create,ATLIF,transforms_pair,DATA,seed_all
from artg import configure
from utils.utils import SOPMonitor
M=R/'experiments/EXP-004/runs/EXP-004-RUN-003/checkpoint_best_validation.pth';I=R/'experiments/EXP-007/calibration_indices.json';O=R/'experiments/EXP-011/preflight-attempt-002';start=time.monotonic();cpu=time.process_time();torch.set_num_threads(2);ck=torch.load(M,map_location='cpu',weights_only=False);ids=json.loads(I.read_text())['indices'][:4];ds=CIFAR10(DATA,train=True,download=False,transform=transforms_pair()[1]);x=torch.stack([ds[i][0] for i in ids]).cuda();y=torch.tensor([ds[i][1] for i in ids]).cuda();data={}
for group in ['G0','G1','M4']:
 seed_all(2030);m=configure(model_create().cuda(),group);m.load_state_dict(ck['model']);m.train()
 for n,p in m.named_modules():
  if isinstance(p,ATLIF):p.thresh.data.mul_(1.25)
 z=m(x);loss=F.cross_entropy(z.mean(0),y);loss.backward();data[group]={n:p.grad.detach().cpu().clone() for n,p in m.named_parameters() if p.grad is not None and not n.endswith('thresh')};del m
base=data['G0'];results={}
for group in ['G1','M4']:
 diffs=[];normsq=refnormsq=0
 for name,a in base.items():
  b=data[group][name];d=(a-b).double();da=d.abs();normal=da/(1+a.double().abs());normsq+=float((d*d).sum());refnormsq+=float((a.double()*a.double()).sum());diffs.append({'name':name,'max_normalized':normal.max().item(),'max_abs':da.max().item(),'diff_l2':d.norm().item(),'ref_l2':a.double().norm().item(),'rel_l2':d.norm().item()/(1e-30+a.double().norm().item()),'ref_max_abs':a.abs().max().item()})
 results[group]={'global_relative_l2':(normsq/refnormsq)**.5,'parameters_over_1e-5':sum(d['max_normalized']>1e-5 for d in diffs),'worst_by_normalized_error':sorted(diffs,key=lambda z:z['max_normalized'],reverse=True)[:10],'all_parameters':diffs}
result={'status':'COMPLETED','fixed_calibration_ids':ids,'comparisons':results,'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.monotonic()-start};(O/'gradient_diagnosis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:{z:v for z,v in r.items() if z!='all_parameters'} for k,r in results.items()},indent=2))

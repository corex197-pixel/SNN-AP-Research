"""CPU-only round-trip of retained epoch-20 models and optimizer/scheduler/scaler states."""
import os,sys,json,hashlib,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];O=ROOT/'experiments/EXP-017';sys.path.insert(0,str(ROOT/'vendor/UniSparse'))
import torch
from models.cifarnet import cifarnet
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
torch.set_num_threads(2)
def equal(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
result={}
for g in ['U0','U1','U2']:
 path=O/'runs/EXP-017-PILOT-001'/g/'latest.pth';ck=torch.load(path,map_location='cpu',weights_only=False);m=cifarnet(T=4,base_channels=256,num_classes=10);opt=create_optimizer_v2(m,opt='sgd',lr=.1);sch=create_scheduler_v2(opt,sched='cosine',num_epochs=300,cooldown_epochs=0,min_lr=0.,warmup_lr=0.,warmup_epochs=0)[0];sc=torch.amp.GradScaler('cpu');m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sch.load_state_dict(ck['scheduler']);sc.load_state_dict(ck['scaler']);v={'epoch':ck['epoch'],'model_equal':equal(ck['model'],m.state_dict()),'optimizer_equal_including_momentum':equal(ck['optimizer'],opt.state_dict()),'scheduler_equal':equal(ck['scheduler'],sch.state_dict()),'scaler_equal':equal(ck['scaler'],sc.state_dict()),'rng_components_present':set(ck['rng'])=={'python','numpy','torch','cuda'},'sampling_next_epoch':ck['sampling']['next_epoch'],'gpu_used':False};assert v['epoch']==20 and v['sampling_next_epoch']==21 and all(v[x] for x in ['model_equal','optimizer_equal_including_momentum','scheduler_equal','scaler_equal','rng_components_present']);result[g]=v;del ck,m,opt,sch,sc;gc.collect()
(O/'checkpoint_recovery_audit.json').write_text(json.dumps({'status':'PASSED','groups':result,'scope':'CPU load and state equality; no extra training, no claim of cross-environment bitwise continuation'},indent=2)+'\n');print(json.dumps(result))

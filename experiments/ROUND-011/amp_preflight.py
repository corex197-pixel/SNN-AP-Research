import sys,json,time,torch
from pathlib import Path
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
from timm.optim import create_optimizer_v2
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'experiments/EXP-004/controlled'));sys.path.insert(0,str(R/'experiments/ROUND-011'))
from core import model_create,ATLIF,DATA,transforms_pair,threshold_update,seed_all
from artg import configure
start=time.monotonic();cpu=time.process_time();torch.set_num_threads(2);ck=torch.load(R/'experiments/EXP-004/runs/EXP-004-RUN-003/checkpoint_best_validation.pth',map_location='cpu',weights_only=False);ids=json.loads((R/'experiments/EXP-007/calibration_indices.json').read_text())['indices'][:4];ds=CIFAR10(DATA,train=True,download=False,transform=transforms_pair()[1]);x=torch.stack([ds[i][0] for i in ids]).cuda();y=torch.tensor([ds[i][1] for i in ids],device='cuda');results={}
for group in ['G0','G1','M4']:
 seed_all(2030);m=configure(model_create().cuda(),group);m.load_state_dict(ck['model']);m.train()
 for z in m.modules():
  if isinstance(z,ATLIF):z.thresh.data.mul_(1.25)
 opt=create_optimizer_v2(m,opt='sgd',lr=.005,weight_decay=5e-4,momentum=.9,nesterov=True);sc=torch.cuda.amp.GradScaler(init_scale=1.,enabled=True);opt.zero_grad(set_to_none=True);before={n:p.detach().clone() for n,p in m.named_parameters()}
 with torch.cuda.amp.autocast():loss=F.cross_entropy(m(x).mean(0),y)
 sc.scale(loss).backward();old=sc.get_scale();sc.step(opt);threshold_update(m,.005);sc.update();skip=sc.get_scale()<old;changed=any(not torch.equal(before[n],p) for n,p in m.named_parameters() if not n.endswith('thresh'));valid=all(torch.isfinite(z.thresh).all() and z.thresh.item()>0 for z in m.modules() if isinstance(z,ATLIF));results[group]={'successful_optimizer_step':not skip,'non_threshold_parameters_changed':changed,'thresholds_positive_finite':valid,'loss':loss.item(),'initial_scale_for_preflight_only':1.};assert not skip and changed and valid
 del m,opt,sc;torch.cuda.empty_cache()
out={'status':'PASSED','groups':results,'wall_seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,'note':'disposable preflight models; training uses baseline default GradScaler'};(R/'experiments/EXP-011/preflight/amp_success.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

import sys,json,time,math,torch
from pathlib import Path
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'experiments/EXP-004/controlled'));sys.path.insert(0,str(R/'experiments/ROUND-011'))
from core import model_create,ATLIF,transforms_pair,DATA,seed_all
from artg import Artg,configure
start=time.monotonic();cpu=time.process_time();torch.set_num_threads(2);cases=[]
for kind,av,bv in [('amplitude',100000.,0.),('reset',0.,-100000.),('both',100000.,-100000.)]:
 x=torch.tensor([2.],device='cuda',dtype=torch.float16,requires_grad=True);th=torch.tensor(1.3,device='cuda',dtype=torch.float32,requires_grad=True);o,s,_=Artg.apply(x,th,.002,'mixed');a=torch.tensor([av],device='cuda');b=torch.tensor([bv],device='cuda');_,gt=torch.autograd.grad((o,s),(x,th),(a,b));r=(x.detach()/th.detach()).float();g=(1-((x.detach()-th.detach())/th.detach()).abs()).clamp_min(0).float();expected=(a*(s.detach()-r*g)-b*r*g/th.detach()).mean();err=((gt-expected).abs()/(1+expected.abs())).item();cases.append({'case':kind,'threshold_grad':gt.item(),'expected':expected.item(),'normalized_error':err,'finite':torch.isfinite(gt).item()});assert torch.isfinite(gt) and err<1e-5
ck=torch.load(R/'experiments/EXP-004/runs/EXP-004-RUN-003/checkpoint_best_validation.pth',map_location='cpu',weights_only=False);ids=json.loads((R/'experiments/EXP-007/calibration_indices.json').read_text())['indices'][:4];ds=CIFAR10(DATA,train=True,download=False,transform=transforms_pair()[1]);x=torch.stack([ds[i][0] for i in ids]).cuda();y=torch.tensor([ds[i][1] for i in ids],device='cuda');groups={}
for group in ['G0','G1','M4']:
 seed_all(2030);m=configure(model_create().cuda(),group);m.load_state_dict(ck['model']);m.train()
 for z in m.modules():
  if isinstance(z,ATLIF):z.thresh.data.mul_(1.25)
 with torch.cuda.amp.autocast():z=m(x);loss=F.cross_entropy(z.mean(0),y)
 loss.backward();groups[group]={'logits':z.detach().cpu(),'grads':{n:p.grad.detach().cpu() for n,p in m.named_parameters() if p.grad is not None},'all_threshold_grads_finite':all(torch.isfinite(p.grad).all().item() for n,p in m.named_parameters() if n.endswith('thresh'))};del m
r=groups['G0'];compare={}
for group in ['G1','M4']:
 gg=groups[group];d=[];total_diff=total_ref=0.
 for n,a in r['grads'].items():
  if n.endswith('thresh'):continue
  b=gg['grads'][n];diff=(a-b).double();d.append(float((diff.abs()/(1+a.double().abs())).max()));total_diff+=float((diff*diff).sum());total_ref+=float((a.double()*a.double()).sum())
 compare[group]={'AMP_logit_max_absolute_difference':float((gg['logits']-r['logits']).abs().max()),'AMP_non_threshold_grad_max_normalized_difference':max(d),'AMP_non_threshold_grad_global_relative_l2':math.sqrt(total_diff/total_ref),'threshold_grads_finite':gg['all_threshold_grads_finite']};assert compare[group]['AMP_logit_max_absolute_difference']==0 and compare[group]['threshold_grads_finite']
out={'status':'PASSED','mixed_dtype_large_upstream_cases':cases,'real_network_unscaled_AMP_comparison':compare,'note':'FP32 preflight remains primary required tolerance; AMP comparisons measured separately','wall_seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu};(R/'experiments/EXP-011/preflight/mixed_precision.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

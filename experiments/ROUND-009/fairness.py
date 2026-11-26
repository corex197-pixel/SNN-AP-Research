import sys,json,time,copy,datetime,hashlib
from pathlib import Path
import torch
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/EXP-004/controlled'))
from core import model_create,ATLIF,DATA,transforms_pair
M=ROOT/'experiments/EXP-004/runs/EXP-004-RUN-003/checkpoint_best_validation.pth';OUT=ROOT/'experiments/EXP-008/runs/PREFLIGHT-002';OUT.mkdir(exist_ok=False)
torch.set_num_threads(2);start=time.monotonic();cpu=time.process_time();ck=torch.load(M,map_location='cpu',weights_only=False)
def make(sp,alpha):
 m=model_create().cuda();m.load_state_dict(ck['model'])
 for n in m.modules():
  if isinstance(n,ATLIF):n.sp=sp;n.thresh.data.mul_(alpha);n.thresh.requires_grad_(False)
 return m
A=make(.002,1.25);B=make(0.,1.25);A.eval();B.eval();data=CIFAR10(DATA,train=True,download=False,transform=transforms_pair()[1]);ids=json.loads((ROOT/'experiments/EXP-007/calibration_indices.json').read_text())['indices'][:2];x=torch.stack([data[i][0] for i in ids]).cuda();y=torch.tensor([data[i][1] for i in ids],device='cuda')
za=A(x).mean(0);zb=B(x).mean(0);F.cross_entropy(za,y).backward();F.cross_entropy(zb,y).backward();logitdiff=(za-zb).abs().max().item();diffs={n:(p.grad-dict(B.named_parameters())[n].grad).abs().max().item() for n,p in A.named_parameters() if p.grad is not None};gmax=max(diffs.values());
teacher=make(0.,1.).eval();student=make(0.,1.25).train();teacher_state={n:p.detach().clone() for n,p in teacher.state_dict().items()};thresholds={n:p.detach().clone() for n,p in student.named_parameters() if n.endswith('thresh')}
for p in teacher.parameters():p.requires_grad_(False)
opt=torch.optim.SGD((p for p in student.parameters() if p.requires_grad),lr=.005,momentum=.9,nesterov=True);opt.zero_grad(set_to_none=True)
with torch.no_grad():zt=teacher(x).mean(0)
zs=student(x).mean(0);loss=F.cross_entropy(zs,y)+.5*F.kl_div(F.log_softmax(zs/3,-1),F.softmax(zt/3,-1),reduction='batchmean')*9;loss.backward();opt.step()
teacher_unchanged=all(torch.equal(teacher_state[n],p) for n,p in teacher.state_dict().items());thresholds_unchanged=all(torch.equal(thresholds[n],p) for n,p in student.named_parameters() if n in thresholds);teacher_no_grad=all(p.grad is None for p in teacher.parameters());result={'status':'PASSED' if logitdiff==0 and gmax==0 and teacher_unchanged and thresholds_unchanged and teacher_no_grad else 'FAILED','sp_zero_vs_baseline_logits_max_abs':logitdiff,'sp_zero_vs_baseline_non_threshold_grad_max_abs':gmax,'teacher_unchanged_after_student_step':teacher_unchanged,'teacher_no_grad':teacher_no_grad,'student_thresholds_unchanged_after_step':thresholds_unchanged,'gradient_tensor_count':len(diffs),'wall_seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(OUT/'smoke.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if result['status']!='PASSED':raise SystemExit(1)

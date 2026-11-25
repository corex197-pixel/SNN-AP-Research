"""CPU deterministic checkpoint/resume training-equivalence smoke test."""
import copy,sys,tempfile,json
from pathlib import Path
import torch
from torch import nn
from core import ATLIF,seed_all,step,save,restore,sched_create
class Net(nn.Module):
 def __init__(self):super().__init__();self.a=nn.Linear(4,4);self.n=ATLIF(T=4,sp=.002,tau=.9);self.o=nn.Linear(4,2)
 def forward(self,x):return self.o(self.n(self.a(x).unsqueeze(0).repeat(4,1,1)))
seed_all(42);torch.set_num_threads(2);x=torch.randn(3,4);y=torch.tensor([0,1,0]);m=Net().train();opt=torch.optim.SGD(m.parameters(),lr=.1,momentum=.9);sched=sched_create(opt);scaler=torch.cuda.amp.GradScaler(enabled=False)
before={k:v.clone() for k,v in m.state_dict().items()};step(m,opt,scaler,x,y);sched.step(1);after={k:v.clone() for k,v in m.state_dict().items()};weight_changed=any(not torch.equal(before[k],after[k]) for k in before if 'weight' in k);finite_positive=bool(torch.isfinite(m.n.thresh).all() and m.n.thresh.item()>0)
with tempfile.TemporaryDirectory() as d:
 path=Path(d)/'ck.pth';save(path,m,opt,sched,scaler,0,.73)
 step(m,opt,scaler,x,y); expected={k:v.clone() for k,v in m.state_dict().items()};expected_opt=copy.deepcopy(opt.state_dict())
 n=Net().train();op2=torch.optim.SGD(n.parameters(),lr=.1,momentum=.9);sc2=sched_create(op2);amp2=torch.cuda.amp.GradScaler(enabled=False);epoch,best=restore(path,n,op2,sc2,amp2,'cpu');step(n,op2,amp2,x,y)
 diff=max((expected[k]-v).abs().max().item() for k,v in n.state_dict().items());odiff=max((expected_opt['state'][i]['momentum_buffer']-op2.state_dict()['state'][i]['momentum_buffer']).abs().max().item() for i in expected_opt['state'])
result={'model_max_abs_diff':diff,'optimizer_momentum_max_abs_diff':odiff,'restored_epoch':epoch,'restored_best_validation':best,'weights_updated':weight_changed,'threshold_finite_positive':finite_positive,'theta_before':before['n.thresh'].item(),'theta_after':after['n.thresh'].item(),'pass':diff<=1e-5 and odiff<=1e-5 and epoch==0 and best==.73 and weight_changed and finite_positive}
print(json.dumps(result,indent=2));sys.exit(0 if result['pass'] else 1)

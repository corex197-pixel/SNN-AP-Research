import sys,json,torch,time
from pathlib import Path
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
R=Path('/home/hym/keyan1');sys.path.insert(0,str(R/'experiments/EXP-004/controlled'));sys.path.insert(0,str(R/'experiments/ROUND-009'))
from core import model_create,ATLIF,DATA,transforms_pair
from proxy import Proxy
torch.set_num_threads(2);start=time.monotonic();cpu=time.process_time();m=model_create();m.load_state_dict(torch.load(R/'experiments/EXP-004/runs/EXP-004-RUN-003/checkpoint_best_validation.pth',map_location='cpu',weights_only=False)['model']);m.eval()
for z in m.modules():
 if isinstance(z,ATLIF):z.thresh.requires_grad_(False);z.sp=0
ids=json.loads((R/'experiments/EXP-007/calibration_indices.json').read_text())['indices'][:4];ds=CIFAR10(DATA,train=True,download=False,transform=transforms_pair()[1]);x=torch.stack([ds[i][0] for i in ids]);p=Proxy(m);m(x);c=p.value();ratio=c/54945117.504;loss=10*F.relu(ratio-.8).square();loss.backward();g=[q.grad.abs().sum().item() for q in m.parameters() if q.requires_grad and q.grad is not None];result={'ratio':ratio.item(),'target':.8,'loss':loss.item(),'nonzero_gradient_tensors':sum(x>0 for x in g),'finite':all(torch.isfinite(torch.tensor(g))),'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.monotonic()-start,'performed_after_training_start':True};out=R/'experiments/EXP-008/runs/PREFLIGHT-001/actual_hinge_cpu_check.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));assert result['ratio']>.8 and result['nonzero_gradient_tensors']>0 and result['finite']

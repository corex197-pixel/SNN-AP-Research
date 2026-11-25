import json,copy,tempfile,time,sys,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from core import ATLIF,isolated_eval,seed_all,threshold_update,save,restore,sched_create,transforms_pair,datasets_split,SPLIT
from torchvision.transforms import RandomCrop,RandomHorizontalFlip
from torchvision.datasets import CIFAR10
from core import DATA
seed_all(42);torch.set_num_threads(2)
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.fc=nn.Linear(4,4);self.bn=nn.BatchNorm1d(4);self.n=ATLIF(T=4,tau=.9,sp=.002);self.out=nn.Linear(4,2)
 def forward(self,x):
  z=self.bn(self.fc(x));z=z.unsqueeze(0).repeat(4,1,1);return self.out(self.n(z))
base=Net();base.n.update_value=.123
X=torch.tensor([[1.,2.,3.,4.],[2.,1.,4.,3.],[0.,1.,2.,3.],[4.,3.,2.,1.]]);y=torch.tensor([0,1,0,1]);v=X*1.3
out={};models={};optimizers={}
for label in 'ABC':
 model=copy.deepcopy(base);model.train();opt=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9);before=model.n.update_value
 if label=='B':
  model.eval()
  with torch.no_grad():pred=model(v)
 elif label=='C':
  with isolated_eval(model):pred=model(v)
  predC=pred.clone()
 after=model.n.update_value
 model.train();opt.zero_grad();loss=nn.functional.cross_entropy(model(X).mean(0),y);loss.backward();opt.step();threshold_update(model,.1)
 models[label]=model;optimizers[label]=opt
 out[label]={'update_before':before,'update_after_evaluation':after,'theta':model.n.thresh.item()}
 if label=='B':predB=pred.clone()
def compare(x,y):
 return max((x[k]-y[k]).abs().max().item() for k in x if torch.is_tensor(x[k]))
def optdiff(a,b):
 x=a.state_dict();y=b.state_dict(); vals=[]
 for k in x['state']:
  for key in x['state'][k]:
   v=x['state'][k][key];w=y['state'][k][key];vals.append((v-w).abs().max().item() if torch.is_tensor(v) else abs(v-w))
 return max(vals,default=0.)
out['D_theta_B']=abs(out['B']['theta']-out['A']['theta']);out['D_theta_C']=abs(out['C']['theta']-out['A']['theta'])
out['weights_diff_B']=max((p-q).abs().max().item() for (n,p),(n2,q) in zip(models['A'].named_parameters(),models['B'].named_parameters()) if 'thresh' not in n)
out['weights_diff_C']=max((p-q).abs().max().item() for (n,p),(n2,q) in zip(models['A'].named_parameters(),models['C'].named_parameters()) if 'thresh' not in n)
out['bn_diff_B']=compare(dict(models['A'].named_buffers()),dict(models['B'].named_buffers()));out['bn_diff_C']=compare(dict(models['A'].named_buffers()),dict(models['C'].named_buffers()));out['optimizer_diff_B']=optdiff(optimizers['A'],optimizers['B']);out['optimizer_diff_C']=optdiff(optimizers['A'],optimizers['C']);out['evaluation_output_max_diff']=(predB-predC).abs().max().item()
model=copy.deepcopy(base);model.train();model.bn.eval();before_modes=[m.training for m in model.modules()];before_rng=torch.get_rng_state().clone();before_up=model.n.update_value
try:
 with isolated_eval(model): model(v);torch.rand(5);raise RuntimeError('intentional')
except RuntimeError:pass
out['exception_restored']=([m.training for m in model.modules()]==before_modes and torch.equal(before_rng,torch.get_rng_state()) and model.n.update_value==before_up)
# Checkpoint restores best score, optimizer, scheduler, scaler, nonzero ATLIF state and RNG.
opt=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9);sched=sched_create(opt);scaler=torch.cuda.amp.GradScaler(enabled=False)
with tempfile.TemporaryDirectory() as t:
 path=Path(t)/'state.pth';save(path,model,opt,sched,scaler,4,.73);expected=torch.rand(3);model.n.update_value=9.;torch.rand(3);epoch,best=restore(path,model,opt,sched,scaler,'cpu');actual=torch.rand(3)
 out['checkpoint']={'epoch':epoch,'best':best,'update_value':model.n.update_value,'rng_equal':torch.equal(expected,actual)}
train,val,diag=datasets_split();raw=CIFAR10(DATA,train=True,download=False);T=set(train.indices);V=set(val.indices);D=set(diag.indices);parts=json.loads(SPLIT.read_text());out['split']={'train':len(T),'validation':len(V),'diagnostic':len(D),'diagnostic_256':len(parts['diagnostic_256']),'disjoint':not(T&V or T&D or V&D),'union':len(T|V|D),'per_class':parts['metadata']['per_class'],'eval_deterministic':torch.equal(val[0][0],val[0][0]),'training_transform':str(train.dataset.transform),'validation_transform':str(val.dataset.transform),'test_dataset_constructed':False}
out['pass']=out['D_theta_C']<=1e-5 and out['weights_diff_C']<=1e-5 and out['bn_diff_C']<=1e-5 and out['optimizer_diff_C']<=1e-5 and out['evaluation_output_max_diff']<=1e-5 and out['exception_restored'] and epoch==4 and best==.73 and model.n.update_value==before_up and out['checkpoint']['rng_equal'] and out['split']['disjoint'] and out['split']['union']==50000 and out['split']['eval_deterministic']
print(json.dumps(out,indent=2));sys.exit(0 if out['pass'] else 1)

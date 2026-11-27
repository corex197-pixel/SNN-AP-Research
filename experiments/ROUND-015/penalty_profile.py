"""EXP-016 one extra fixed-config timing step; wrappers only instrument original local forward/backward."""
import os,sys,time,json,resource,traceback,contextlib,datetime
from pathlib import Path
start=time.monotonic();ROOT=Path(__file__).resolve().parents[2];O=ROOT/'experiments/EXP-016';R=O/'runs/EXP-016-PROFILE-001';R.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(ROOT/'vendor/UniSparse'));sys.path.insert(0,str(ROOT/'experiments/ROUND-014'))
import numpy as np,torch,cupy as cp
from torch.nn import functional as F
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional
from timm.optim import create_optimizer_v2
from models.cifarnet import cifarnet
from models.submodules.layers import SparseNeuron
from utils.utils import calc_penalty,penalize,clear_penalty
from controlled_data import TrainFileOnly
import random
cpu0=resource.getrusage(resource.RUSAGE_SELF);torch.set_num_threads(2);random.seed(12450);np.random.seed(12450);torch.manual_seed(12450);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
s={'status':'RUNNING','run_id':R.name,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES')}
def sync():torch.cuda.synchronize()
try:
 assert torch.cuda.device_count()==1 and '4090' in torch.cuda.get_device_name(0);torch.cuda.set_per_process_memory_fraction(.88)
 with (O/'cupy_show_config.txt').open('w') as f:
  with contextlib.redirect_stdout(f):cp.show_config()
 ids=json.loads((ROOT/'experiments/EXP-004/splits/indices.json').read_text())['train'];d=TrainFileOnly(ids[:256],True);x,y,_=next(iter(DataLoader(d,batch_size=256,num_workers=0)));x=x.cuda();y=y.cuda();m=cifarnet(T=4,base_channels=256,num_classes=10).cuda().train();opt=create_optimizer_v2(m,opt='sgd',lr=.1);scaler=torch.amp.GradScaler('cuda');opt.zero_grad(set_to_none=True)
 with torch.autocast('cuda'):z=m(x);loss=F.cross_entropy(z.mean(0),y)
 scaler.scale(loss).backward();functional.reset_net(m);scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
 timing={'local_forward_seconds':0.,'penalty_backward_seconds':0.,'local_calls':0};originals=[]
 for mod in m.modules():
  if isinstance(mod,SparseNeuron):
   old=mod.local_forward
   def wrap(old=old):
    sync();t=time.monotonic();value=old();sync();timing['local_forward_seconds']+=time.monotonic()-t;timing['local_calls']+=1;return value
   mod.local_forward=wrap;originals.append((mod,old))
 original_backward=torch.Tensor.backward
 def backward(tensor,*args,**kwargs):
  sync();t=time.monotonic();result=original_backward(tensor,*args,**kwargs);sync();timing['penalty_backward_seconds']+=time.monotonic()-t;return result
 try:
  torch.Tensor.backward=backward;sync();t=time.monotonic();calc_penalty(m);sync();timing['calc_penalty_total_seconds']=time.monotonic()-t
 finally:
  torch.Tensor.backward=original_backward
  for mod,old in originals:mod.local_forward=old
 t=time.monotonic();penalize(m,.001,.001);clear_penalty(m);sync();timing['manual_clear_seconds']=time.monotonic()-t;timing['other_calc_overhead_seconds']=timing['calc_penalty_total_seconds']-timing['local_forward_seconds']-timing['penalty_backward_seconds'];timing['sync_instrumented']=True;timing['synthetic_lambda']=.01;timing['not_a_paper_hyperparameter']=True;timing['batch_size']=256;timing['extra_training_steps']=1;timing['finite']=all(torch.isfinite(p).all().item() for p in m.parameters());assert timing['finite'];(O/'penalty_timing_profile.json').write_text(json.dumps(timing,indent=2)+'\n');s.update(status='COMPLETED',timing=timing)
except BaseException as e:s.update(status='PARTIAL',error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
finally:
 r=resource.getrusage(resource.RUSAGE_SELF);s.update(end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.monotonic()-start,cpu_seconds=r.ru_utime+r.ru_stime-cpu0.ru_utime-cpu0.ru_stime,peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0);(R/'status.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s),flush=True)
if s['status']!='COMPLETED':sys.exit(1)

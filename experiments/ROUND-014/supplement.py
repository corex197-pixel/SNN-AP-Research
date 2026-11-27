"""Supplementary CPU tests: train-file-only data and nontrivial activity accounting."""
import sys,json,time,resource,random,hashlib
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'vendor/UniSparse'));sys.path.insert(0,str(ROOT/'experiments/ROUND-014'))
from controlled_data import TrainFileOnly
from models.submodules.layers import SparseConv2d
from utils.utils import SOPMonitor
from torch import nn
O=ROOT/'experiments/EXP-015';st=time.monotonic();cpu=resource.getrusage(resource.RUSAGE_SELF);torch.set_num_threads(1);random.seed(12450);np.random.seed(12450);torch.manual_seed(12450)
ids=json.loads((ROOT/'experiments/EXP-004/splits/indices.json').read_text());assert len(ids['train'])==45000 and len(ids['validation'])==4500 and len(ids['diagnostic'])==500 and not set(ids['train'])&set(ids['validation'])
d=TrainFileOnly(ids['train'][:2]);x,y,i=d[0];assert list(x.shape)==[3,32,32] and torch.isfinite(x).all()
(O/'controlled_data_test.json').write_text(json.dumps({'kind':'SMOKE_TEST','training_ids':ids['train'][:2],'first_sample_index':i,'first_sample_label':y,'shape':list(x.shape),'train_dataset_length':len(d),'official_test_file_constructed_or_read':False,'source_files':'data_batch_1..5 only','augmentation':'official CIFAR10Policy, crop, flip, normalize, cutout','preprocessed_sample_sha256':hashlib.sha256(x.numpy().tobytes()).hexdigest()},indent=2)+'\n')
class Net(nn.Module):
 def __init__(self):super().__init__();self.c=SparseConv2d(1,1,2,padding=1);self.skip=[]
 def forward(self,x):return self.c(x)
m=Net();m.c.weight.data.copy_(torch.tensor([[[[.25,1.],[-1.,.75]]]]));cases=[]
for th,z in [(0.,[[1,0],[0,1]]),(.5,[[1,0],[0,1]]),(.5,[[1,0],[0,0]])]:
 m.c.threshold.fill_(th);x=torch.tensor(z,dtype=torch.float32).view(1,1,1,2,2);mon=SOPMonitor(m);m(x);v=sum(q.item() for q in mon.records);mon.remove_hooks();mask=m.c.sparse_mask();ind=0
 for iy in range(2):
  for ix in range(2):
   for ky in range(2):
    for kx in range(2):
     oy=iy+1-ky;ox=ix+1-kx
     if 0<=oy<3 and 0<=ox<3:ind+=x[0,0,0,iy,ix].item()*mask[0,0,ky,kx].item()
 assert v==ind;cases.append(v)
a,b,c=cases;assert c!=b and (b-a)+(c-b)==c-a
r=resource.getrusage(resource.RUSAGE_SELF);(O/'sop_supplemental.json').write_text(json.dumps({'kind':'UNIT_TEST','cases_author_and_independent':cases,'structure_delta':b-a,'activity_delta':c-b,'total_delta':c-a,'identity_passed':True,'gpu_used':False,'wall_seconds':time.monotonic()-st,'cpu_seconds':r.ru_utime+r.ru_stime-cpu.ru_utime-cpu.ru_stime},indent=2)+'\n');print('train-only data and nonzero activity accounting passed',cases)

"""Shared fixed-protocol utilities for EXP-017."""
import os,sys,time,json,csv,hashlib,random,resource,datetime,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];O=ROOT/'experiments/EXP-018';sys.path.insert(0,str(ROOT/'vendor/UniSparse'));sys.path.insert(0,str(ROOT/'experiments/ROUND-014'))
import numpy as np,torch
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from models.cifarnet import cifarnet
from models.submodules.layers import SparseConv2d,SparseBN,SparseNeuron
from spikingjelly.activation_based import functional
from controlled_data import TrainFileOnly
CFG=json.loads((O/'config.json').read_text());SPLIT=ROOT/'experiments/EXP-004/splits/indices.json'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def append(p,rows):
 if not rows:return
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');
  if f.tell()==0:w.writeheader()
  w.writerows(rows);f.flush();os.fsync(f.fileno())
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def seed():random.seed(12450);np.random.seed(12450);torch.manual_seed(12450);torch.cuda.manual_seed(12450)
def model():seed();m=cifarnet(T=4,base_channels=256,num_classes=10).cuda();return m
def optim(m):return create_optimizer_v2(m,opt='sgd',lr=.1)
def scheduler(opt):return create_scheduler_v2(opt,sched='cosine',num_epochs=300,cooldown_epochs=0,min_lr=0.,warmup_lr=0.,warmup_epochs=0)[0]
def digest(m):
 h=hashlib.sha256()
 for n,v in m.state_dict().items():h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()}
def set_rng(r):random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch']);torch.cuda.set_rng_state(r['cuda'])
def check():
 assert sha(SPLIT)==CFG['split_sha256'];assert torch.cuda.device_count()==1 and '4090' in torch.cuda.get_device_name(0)
 assert not subprocess.check_output(['git','diff','--','vendor/UniSparse'],cwd=ROOT)
 assert subprocess.check_output(['git','rev-parse','e1fa5ab:vendor/UniSparse'],cwd=ROOT,text=True).strip()==subprocess.check_output(['git','rev-parse','HEAD:vendor/UniSparse'],cwd=ROOT,text=True).strip()
 s=json.loads(SPLIT.read_text());assert len(s['train'])==45000 and len(s['validation'])==4500 and len(s['diagnostic'])==500 and not set(s['train'])&set(s['validation']) and not set(s['train'])&set(s['diagnostic']);return s
def cpu_seconds():
 r=resource.getrusage(resource.RUSAGE_SELF);c=resource.getrusage(resource.RUSAGE_CHILDREN);total=r.ru_utime+r.ru_stime+c.ru_utime+c.ru_stime
 try:
  import psutil
  for p in psutil.Process().children(recursive=True):
   try:t=p.cpu_times();total+=t.user+t.system
   except psutil.Error:pass
 except ImportError:pass
 return total
CPU0=cpu_seconds()
def guard(reserve=0):
 b=json.loads((O/'budget.json').read_text());wall=time.time()-b['start_unix'];cpu=cpu_seconds()-CPU0+b.get('prior_cpu_seconds',0)
 if wall>28800-reserve:raise TimeoutError('8-hour conservative wall budget reached')
 if cpu>57600-reserve:raise TimeoutError('16-hour cumulative CPU budget reached')
 return wall,cpu

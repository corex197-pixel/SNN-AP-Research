"""EXP-018 paired recovery test and EXP-019 epoch20 admission."""
import os,sys,time,json,traceback,hashlib,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent));import common as C
import torch,numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset
from spikingjelly.activation_based import functional
from utils.utils import calc_penalty,penalize,clear_penalty
import cost,cupy
import audit_functions as A
O=C.O;D=C.ROOT/'experiments/EXP-019';torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;EXPECTED={'U0':'1594cbb3f65c1acf91efcbe9c0a842c9ad08fbb57dbfac2ee9333afa951439c2','U1':'bbf3a5dc7c0035bbaf8c08a18f2143bbcccc920fba6e2516b86b69624f327f81','U2':'c4d5d9e882282c1e18ebae3134ab6a1bcbc661acb24035087d1c44f62cbc464c'};LAM={'U0':0.,'U2':8.834108737251828e-08};start=time.time();status={'status':'RUNNING','run_id':'EXP-018-PREFLIGHT-001','start_utc':C.now(),'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES')}
def parent(g):return C.ROOT/f'experiments/EXP-017/runs/EXP-017-PILOT-001/{g}/latest.pth'
def load(g,cp=None):
 c=torch.load(cp or parent(g),map_location='cpu',weights_only=False);assert c['epoch']==20 and c['group']==g and c['lambda_weight']==LAM[g] and c['lambda_neuron']==LAM[g];m=C.model().train();m.load_state_dict(c['model']);o=C.optim(m);sch=C.scheduler(o);sc=torch.amp.GradScaler('cuda');o.load_state_dict(c['optimizer']);sch.load_state_dict(c['scheduler']);sc.load_state_dict(c['scaler']);C.set_rng(c['rng']);return c,m,o,sch,sc
def hash_optimizer(o):
 h=hashlib.sha256();sd=o.state_dict()
 for k,v in sorted(sd['state'].items()):
  h.update(str(k).encode())
  for x,y in sorted(v.items()):
   h.update(x.encode());h.update(y.detach().cpu().numpy().tobytes() if torch.is_tensor(y) else repr(y).encode())
 h.update(repr(sd['param_groups']).encode());return h.hexdigest()
def update(m,o,sch,sc,x,y,g):
 lr=o.param_groups[0]['lr'];o.zero_grad(set_to_none=True)
 with torch.autocast('cuda'):z=m(x);loss=F.cross_entropy(z.mean(0),y)
 assert torch.isfinite(loss);sc.scale(loss).backward();functional.reset_net(m);old=sc.get_scale();sc.step(o);sc.update();o.zero_grad(set_to_none=True)
 if LAM[g]:calc_penalty(m);penalize(m,LAM[g]*lr,LAM[g]*lr);clear_penalty(m)
 assert sc.get_scale()>=old and all(torch.isfinite(p).all().item() for p in m.parameters());return {'model_sha256':C.digest(m),'optimizer_sha256':hash_optimizer(o),'scheduler':sch.state_dict(),'scaler':sc.state_dict(),'loss':float(loss.item())}
def run():
 s=C.check();assert cupy.__version__=='13.5.1';torch.cuda.set_per_process_memory_fraction(.90);manifest=[]
 for g in EXPECTED:
  p=parent(g);h=C.sha(p);assert h==EXPECTED[g];manifest.append({'group':g,'path':str(p),'sha256':h,'bytes':p.stat().st_size,'read_only':True})
 C.dump(O/'parent_checkpoint_manifest.json',manifest);cost.unit();ds=C.TrainFileOnly(s['train'],True);gen=torch.Generator().manual_seed(12471);ids=torch.randperm(len(ds),generator=gen).tolist()[:256];wg=torch.Generator().manual_seed(99912471);loader=DataLoader(Subset(ds,ids),batch_size=256,num_workers=4,pin_memory=True,generator=wg);x,y,sid=next(iter(loader));x=x.cuda();y=y.cuda();tests=[]
 for g in ['U0','U2']:
  C.guard(300);c,m,o,sch,sc=load(g);A.layer_audit(m,g,20);assert abs(o.param_groups[0]['lr']-float(list(__import__('csv').DictReader((C.ROOT/'experiments/EXP-017/learning_rate_schedule.csv').open()))[20]['learning_rate']))<1e-12;buf={n:v.detach().clone() for n,v in m.named_buffers()};m.eval()
  with torch.no_grad():z=m(x[:4]);assert torch.isfinite(z).all()
  functional.reset_net(m);assert all(torch.equal(buf[n],v) for n,v in m.named_buffers());m.train();C.set_rng(c['rng']);a=update(m,o,sch,sc,x,y,g);del m,o,sch,sc;torch.cuda.empty_cache();c,m,o,sch,sc=load(g);tmp=O/'runs/EXP-018-PREFLIGHT-001'/f'{g}-temporary.pth';tmp.parent.mkdir(parents=True,exist_ok=True);torch.save({'model':m.state_dict(),'optimizer':o.state_dict(),'scheduler':sch.state_dict(),'scaler':sc.state_dict(),'rng':C.rng(),'epoch':20,'group':g,'lambda_weight':LAM[g],'lambda_neuron':LAM[g]},tmp);del m,o,sch,sc;torch.cuda.empty_cache();c,m,o,sch,sc=load(g,tmp);b=update(m,o,sch,sc,x,y,g);same=a==b;tests.append({'group':g,'direct_vs_saved_restored':same,'direct':a,'restored':b,'lr_epoch21':o.param_groups[0]['lr'],'sample_sha256':hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest(),'eval_buffers_unchanged':True});assert same;del m,o,sch,sc,c;torch.cuda.empty_cache()
 C.dump(O/'resume_preflight_tests.json',{'status':'PASSED','groups':tests,'sampling_epoch21_seed':12471,'worker_seed':99912471,'formal_checkpoints_unmodified':True});C.dump(O/'recovery_audit.json',{'status':'PASSED','scope':'single same real train batch direct vs serialized/restored state; original ordering; no cross-environment guarantee','groups':tests});del x,y,ds;torch.cuda.empty_cache();v=cost.evaluate(20,s['validation']);old=list(__import__('csv').DictReader((C.ROOT/'experiments/EXP-017/epoch_metrics.csv').open()));reference={g:float(next(r['SOP_sample_weighted'] for r in old if r['group']==g and int(r['epoch'])==20)) for g in ['U0','U2']};err={'U0':abs(v['C00']-reference['U0'])/(1+abs(reference['U0'])),'U2':abs(v['C22']-reference['U2'])/(1+abs(reference['U2']))};assert max(err.values())<=1e-5;C.dump(D/'epoch_020_training_metric_crosscheck.json',{'status':'PASSED','relative_errors':err,'previous_metrics':reference});return 'PASSED'
try:status['status']=run()
except BaseException as e:status.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
finally:
 status.update(end_utc=C.now(),wall_seconds=time.time()-start,cpu_seconds=C.cpu_seconds()-C.CPU0,peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0);C.dump(O/'preflight_status.json',status);b=json.loads((O/'budget.json').read_text());b['prior_cpu_seconds']=status['cpu_seconds'];C.dump(O/'budget.json',b);print(json.dumps(status),flush=True)
if status['status']!='PASSED':sys.exit(1)

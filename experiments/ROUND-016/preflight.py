"""R68 real model and isolated preflight; no official test data."""
import os,sys,time,json,csv,traceback,subprocess,hashlib,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent));import common as C
import torch,numpy as np
from torch import nn
from torch.nn import functional as F
from utils.utils import calc_penalty,penalize,clear_penalty,CriterionWarpper
from spikingjelly.activation_based import functional
import cupy
O=C.O;start=time.time();torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
status={'status':'RUNNING','run_id':'EXP-017-PREFLIGHT-001','start_utc':C.now(),'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=C.ROOT,text=True).strip(),'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES')}
def run():
 s=C.check();assert cupy.__version__=='13.5.1';torch.cuda.set_per_process_memory_fraction(.90);m=C.model().train();initial=C.digest(m);opt=C.optim(m);sched=C.scheduler(opt);schedule=[]
 for e in range(300):
  schedule.append({'epoch':e+1,'learning_rate':float(opt.param_groups[0]['lr']),'update_opportunities':175});sched.step(e+1)
 C.append(O/'learning_rate_schedule.csv',schedule);S=sum(r['learning_rate']*175 for r in schedule);S20=sum(r['learning_rate']*175 for r in schedule[:20]);S1=schedule[0]['learning_rate']*175
 # Capture K from original local forward for two different actual batches; batch-independent fixed shapes.
 ks=[]
 for b in [2,3]:
  C.guard(60);x=torch.randn(b,3,32,32,device='cuda');z=m(x);assert list(z.shape)==[4,b,10] and torch.isfinite(z).all();y=torch.arange(b,device='cuda')%10;assert torch.allclose(CriterionWarpper(nn.CrossEntropyLoss())(z,y),F.cross_entropy(z.mean(0),y));functional.reset_net(m);calc_penalty(m);ks.append({n:float(v.penalty.item()) for n,v in m.named_modules() if isinstance(v,C.SparseConv2d)});clear_penalty(m)
 assert ks[0]==ks[1];K=ks[0];ref=[]
 for n,v in m.named_modules():
  if isinstance(v,C.SparseConv2d):
   w=v.weight.detach().abs().flatten();q=float(w.median().item());k=K[n];assert torch.equal(v.sparse_weight(),v.weight) and torch.equal(v.sparse_mask(),(v.weight!=0).float());r={'layer':n,'num_weights':w.numel(),'K':k,'median_abs_weight':q,'included':bool(k>0 and q>0),'lambda_ref_layer':q/(k*S) if k>0 and q>0 else None};ref.append(r)
 vals=[r['lambda_ref_layer'] for r in ref if r['included']];assert vals and all(np.isfinite(vals));lam=float(np.median(vals));assert lam>0 and np.isfinite(lam);C.append(O/'lambda_reference.csv',ref)
 pred=[]
 for r in ref:
  for group,l in [('U0',0.),('U1',lam),('U2',3*lam)]:
   for epoch,ss in [(1,S1),(20,S20),(300,S)]:
    val=l*r['K']*ss;pred.append({'group':group,'layer':r['layer'],'epoch':epoch,'predicted_threshold':val,'threshold_to_initial_median':val/r['median_abs_weight'] if r['median_abs_weight']>0 else None})
 C.append(O/'lambda_threshold_prediction.csv',pred);C.dump(O/'lambda_derivation.json',{'S300':S,'S20':S20,'S1':S1,'lambda_ref':lam,'K_actual_batch2_and3':ks,'derivation':'median across penalized convolution layers of q/(K*S300)','fp32_accumulation_not_exact':True,'source':'UNKNOWN official paper lambda'})
 # Three separate model instantiations start from same state. Disposable updates use original order.
 init=[];test_groups=[];del m,opt,sched,z,x
 for g,l in [('U0',0.),('U1',lam),('U2',3*lam)]:
  C.guard(60);m=C.model().train();h=C.digest(m);init.append({'group':g,'initial_state_sha256':h});assert h==initial;opt=C.optim(m);sched=C.scheduler(opt);scaler=torch.amp.GradScaler('cuda');x=torch.randn(4,3,32,32,device='cuda');y=torch.tensor([0,1,2,3],device='cuda');opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda'):z=m(x);loss=F.cross_entropy(z.mean(0),y)
  assert torch.isfinite(loss);scaler.scale(loss).backward();functional.reset_net(m);sc0=scaler.get_scale();scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True);before={n:float(v.threshold.item()) for n,v in m.named_modules() if isinstance(v,C.SparseConv2d)}
  if l:calc_penalty(m);penalize(m,l*.1,l*.1);clear_penalty(m)
  after={n:float(v.threshold.item()) for n,v in m.named_modules() if isinstance(v,C.SparseConv2d)};err=max(abs(after[n]-(before[n]+.1*l*K[n])) for n in K);assert all(np.isfinite(v) for v in after.values()) and err<1e-7 and (not l or any(after[n]!=before[n] for n in K));sched.step(1);assert abs(opt.param_groups[0]['lr']-schedule[1]['learning_rate'])<1e-12
  if g=='U2':
   # Checkpoint full state on CPU and restore into a separate model, scheduler and optimizer.
   cp=O/'runs/EXP-017-PREFLIGHT-001/recovery_test.pth';cp.parent.mkdir(parents=True,exist_ok=True);torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'scaler':scaler.state_dict(),'rng':C.rng(),'epoch':1},cp);m2=C.model().train();o2=C.optim(m2);sch2=C.scheduler(o2);sc2=torch.amp.GradScaler('cuda');ck=torch.load(cp,map_location='cpu',weights_only=False);m2.load_state_dict(ck['model']);o2.load_state_dict(ck['optimizer']);sch2.load_state_dict(ck['scheduler']);sc2.load_state_dict(ck['scaler']);assert C.digest(m)==C.digest(m2) and opt.param_groups[0]['lr']==o2.param_groups[0]['lr'] and sched.state_dict()==sch2.state_dict() and scaler.state_dict()==sc2.state_dict();del m2,o2,sch2,sc2,ck
  # Eval BN/sparse buffers and state do not change; isolate training cache and reset memories.
  buf={n:v.detach().clone() for n,v in m.named_buffers()};m.eval()
  with torch.no_grad():z=m(x);assert torch.isfinite(z).all()
  functional.reset_net(m);assert all(torch.equal(buf[n],v) for n,v in m.named_buffers());test_groups.append({'group':g,'lambda':l,'initial_hash':h,'threshold_formula_max_abs_error':err,'AMP_skipped':scaler.get_scale()<sc0,'eval_buffers_unchanged':True});del m,opt,sched,scaler,z,x,loss;torch.cuda.empty_cache()
 C.dump(O/'initialization_manifest.json',{'seed':12450,'groups':init,'identical':True,'model_parameters':36720484});C.dump(O/'selection_lock.json',{'locked_utc':C.now(),'lambda_weight_neuron':{'U0':0.,'U1':lam,'U2':3*lam},'seed':12450,'initialization_sha256':initial,'planned_epochs':20,'scheduler_epochs':300,'selection':'fixed epoch20; see TASK.md','source':'R68; official value UNKNOWN','locked_before_training':True});C.dump(O/'preflight_tests.json',{'status':'PASSED','groups':test_groups,'K_fixed_shapes_batch2_and3':True,'checkpoint_model_optimizer_scheduler_scaler_restoration':True,'eval_buffers_unchanged':True,'CuPy':cupy.__version__,'validation_and_official_test_accessed':False});return 'PASSED'
try:status['status']=run()
except BaseException as e:status.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
finally:
 status.update(end_utc=C.now(),wall_seconds=time.time()-start,cpu_seconds=C.cpu_seconds()-C.CPU0,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0);C.dump(O/'preflight_status.json',status);b=json.loads((O/'budget.json').read_text());b['prior_cpu_seconds']=status['cpu_seconds'];C.dump(O/'budget.json',b);print(json.dumps(status),flush=True)
if status['status']!='PASSED':sys.exit(1)

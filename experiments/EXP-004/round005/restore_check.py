import os,sys,time,json,datetime,traceback,copy,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'round004'))
import torch
from common import *
OUT=ROOT/'experiments/EXP-004/round005/RECOVERY-CHECK-001';OUT.mkdir(parents=True,exist_ok=False)
start=time.monotonic();result={'status':'RUNNING','start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}
parent=ROOT/'experiments/EXP-004/runs/EXP-004-RUN-002';latest=parent/'checkpoint_latest.pth';best=parent/'checkpoint_best_validation.pth'
def compare(a,b):
 if torch.is_tensor(a):return (a.detach().cpu().double()-b.detach().cpu().double()).abs().max().item() if a.numel() else 0.
 if isinstance(a,dict):return max([compare(a[k],b[k]) for k in a] or [0.]) if a.keys()==b.keys() else float('inf')
 if isinstance(a,(list,tuple)):return max([compare(x,y) for x,y in zip(a,b)] or [0.]) if len(a)==len(b) else float('inf')
 if isinstance(a,(int,float)):return abs(a-b)
 return 0. if a==b else float('inf')
try:
 if torch.cuda.device_count()!=1:raise RuntimeError('must expose exactly one GPU')
 torch.cuda.set_device(0);torch.set_num_threads(4);seed_all();free,total=torch.cuda.mem_get_info();result.update(device=torch.cuda.get_device_name(0),visible_devices=1,initial_free_bytes=free,split=validate_split())
 if free<10*1024**3:raise RuntimeError('less than 10 GiB free')
 hashes=[sha(latest),sha(best)];assert hashes==['a82200f20b07fe328b285f2b191d8df7c62811fada5c00bd5e94f2164f4f4b02','bdc36b9f9ed2c8d2d8b3517f5146b7d6913d2eb2db44141ad60a6978f3ba4eba']
 model=model_create().cuda();opt=opt_create(model);sched=sched_create(opt);scaler=torch.cuda.amp.GradScaler();train,val=loaders()
 ck=restore(latest,model,opt,sched,scaler);assert ck['epoch']==36 and ck['best_validation_accuracy']==.814 and ck['scheduler']['t_initial']==200 and ck['code_sha']=='365da5f0c371b2f9f010f44b8327a0e489184a17'
 lr=opt.param_groups[0]['lr'];assert abs(lr-.09179045016473283)<1e-12
 def two_batches():
  it=iter(train);a,b=next(it);c,d=next(it);del it;return (a[:8].cuda(),b[:8].cuda(),c[:8].cuda(),d[:8].cuda())
 a=two_batches();restore(latest,model,opt,sched,scaler);b=two_batches();result['input_max_diff']=max(compare(x,y) for x,y in zip(a,b));assert result['input_max_diff']==0
 restore(latest,model,opt,sched,scaler);initA=copy.deepcopy(pack(model,opt,sched,scaler,36,.814,ck['config'],ck['code_sha']));model.train();train_step(model,opt,scaler,a[0],a[1]);train_step(model,opt,scaler,a[2],a[3]);torch.cuda.synchronize();A=copy.deepcopy(pack(model,opt,sched,scaler,38,.814,ck['config'],ck['code_sha']))
 restore(latest,model,opt,sched,scaler);initB=copy.deepcopy(pack(model,opt,sched,scaler,36,.814,ck['config'],ck['code_sha']));model.train()
 with isolated_eval(model):model(b[0])
 train_step(model,opt,scaler,b[0],b[1]);tmp=OUT/'temporary-recovery.pth';atomic_save(tmp,pack(model,opt,sched,scaler,37,.814,ck['config'],ck['code_sha']));restore(tmp,model,opt,sched,scaler);train_step(model,opt,scaler,b[2],b[3]);torch.cuda.synchronize();B=copy.deepcopy(pack(model,opt,sched,scaler,38,.814,ck['config'],ck['code_sha']));tmp.unlink()
 result['initial_diff']={k:compare(initA[k],initB[k]) for k in ('model','optimizer','scheduler','scaler','atlif_state')};result['post_step_diff']={k:compare(A[k],B[k]) for k in ('model','optimizer','scheduler','scaler','atlif_state')};result['parameter_change_max']=compare(initA['model'],A['model']);result['threshold_change_max']=max(compare(v,A['model'][k]) for k,v in initA['model'].items() if k.endswith('.thresh'));result['starting_lr']=lr;result['parent_epoch']=ck['epoch']+1;result['parent_hashes_unchanged']=hashes==[sha(latest),sha(best)];result['worker_policy']='4 workers, persistent_workers=False; workers respawn per epoch from restored torch CPU RNG; fixed input comparison reproduced; interrupted unobserved trajectory cannot be proven bitwise identical'
 if max(*result['initial_diff'].values(),*result['post_step_diff'].values())>1e-5 or result['parameter_change_max']<=0 or result['threshold_change_max']<=0 or not result['parent_hashes_unchanged']:raise RuntimeError('restoration test mismatch')
 result['status']='COMPLETED'
except BaseException as e:result.update(status='FAILED',error=repr(e),traceback=traceback.format_exc())
finally:
 result.update(end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.monotonic()-start);(OUT/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if result['status']!='COMPLETED':sys.exit(1)

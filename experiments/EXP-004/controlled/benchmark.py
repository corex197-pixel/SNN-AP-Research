import argparse,time,json,statistics,subprocess,datetime,sys,traceback
from pathlib import Path
import torch
from core import ROOT,seed_all,model_create,opt_create,loaders,step,evaluate
p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--device',type=int,required=True);a=p.parse_args()
if a.run_id not in ('EXP-004-BENCH-01','EXP-004-BENCH-01-RUN-002'):sys.exit('unapproved benchmark id')
out=ROOT/'experiments/EXP-004/bench'/a.run_id;out.mkdir(exist_ok=False,parents=True)
started=time.monotonic();cpu=time.process_time();begin=datetime.datetime.now(datetime.timezone.utc).isoformat();result={'run_id':a.run_id,'start_utc':begin,'device_index':a.device,'warmup_target':10,'measurement_target':50,'batch_size':128,'T':8,'amp':True,'status':'RUNNING','train_batch_seconds':[],'data_load_seconds':[],'amp_skipped_steps':0}
try:
 torch.cuda.set_device(a.device);device=torch.device('cuda',a.device);torch.set_num_threads(4);seed_all();result['gpu_name']=torch.cuda.get_device_name(device);result['initial_free_bytes'],result['total_bytes']=torch.cuda.mem_get_info(device)
 result['nvidia_smi_before']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True)
 if result['initial_free_bytes']<20*1024**3:raise RuntimeError('less than 20 GiB free before measurement; stop')
 model=model_create().to(device).train();opt=opt_create(model);scaler=torch.cuda.amp.GradScaler();train,val=loaders();it=iter(train);torch.cuda.reset_peak_memory_stats(device)
 result['optimizer_groups']=[{k:v for k,v in group.items() if k!='params'} for group in opt.param_groups]
 for i in range(60):
  if time.monotonic()-started>550:raise TimeoutError('GPU benchmark time limit nearing 10 minutes')
  t=time.perf_counter();x,y=next(it);load=time.perf_counter()-t
  torch.cuda.synchronize(device);t=time.perf_counter();x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True);loss,skipped=step(model,opt,scaler,x,y);torch.cuda.synchronize(device);elapsed=time.perf_counter()-t
  result['amp_skipped_steps']+=int(skipped)
  if i>=10:result['train_batch_seconds'].append(elapsed);result['data_load_seconds'].append(load)
  print(f'batch {i+1}/60 loss={loss:.6f} seconds={elapsed:.3f}',flush=True)
 torch.cuda.synchronize(device);t=time.perf_counter();validation=evaluate(model,val,device,max_batches=3);torch.cuda.synchronize(device);result['validation_3_batches_seconds']=time.perf_counter()-t;result['validation_count']=validation['count']
 ts=result['train_batch_seconds'];ls=result['data_load_seconds'];result.update(train_mean_seconds=statistics.mean(ts),train_median_seconds=statistics.median(ts),train_p90_seconds=sorted(ts)[int(.9*(len(ts)-1))],data_load_mean_seconds=statistics.mean(ls),samples_per_second=128/statistics.mean(ts),end_to_end_samples_per_second=128/(statistics.mean(ts)+statistics.mean(ls)),peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),estimate_200_epochs_seconds=200*(352*(statistics.mean(ts)+statistics.mean(ls))+36*result['validation_3_batches_seconds']/3),estimate_method='352 batches training + 36 validation batches/epoch; mean step+loader wait; validation three-batch timing includes setup and isolation; excludes full checkpoint/logging overhead',status='COMPLETED')
except BaseException as e:
 result.update(status='FAILED',error=repr(e),traceback=traceback.format_exc())
 if torch.cuda.is_initialized():
  result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
finally:
 result.update(end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu)
 (out/'metrics.json').write_text(json.dumps(result,indent=2,default=str)+'\n');print(json.dumps(result,indent=2,default=str),flush=True)
if result['status']!='COMPLETED':sys.exit(1)

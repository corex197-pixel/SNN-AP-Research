import os,sys,json,time,datetime,csv,subprocess,traceback,argparse,hashlib,math
from pathlib import Path
import yaml,torch
from common import *
p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--deadline-utc',required=True);p.add_argument('--approved-gpu-uuid',required=True);p.add_argument('--code-sha',required=True);args=p.parse_args()
if args.run_id not in ('EXP-004-RUN-001','EXP-004-RUN-002'):sys.exit('run ID not authorized')
if os.environ.get('CUDA_VISIBLE_DEVICES')!=args.approved_gpu_uuid or torch.cuda.device_count()!=1:sys.exit('GPU isolation failed')
if not torch.cuda.is_available():sys.exit('CUDA unavailable')
deadline=datetime.datetime.fromisoformat(args.deadline_utc);assert deadline.tzinfo
stop=deadline.timestamp()-120
if time.time()>=stop:sys.exit('budget exhausted')
out=ROOT/'experiments/EXP-004/runs'/args.run_id;out.mkdir(parents=True,exist_ok=False)
cfg=yaml.safe_load((ROOT/'experiments/EXP-004/config.yaml').read_text());cfg['min_lr']=float(cfg['min_lr']);cfg['weight_decay']=float(cfg['weight_decay']);expected={'dataset':'CIFAR10','model':'ResNet20','seed':42,'T':8,'eta':.002,'eta2':0.,'batch_size':128,'accumulation_steps':1,'epochs':200,'optimizer':'sgd','lr':.1,'lr_scheduler':'cosine','min_lr':1e-6,'weight_decay':.0005,'amp':True,'TET':False,'data_path':'/home/hym/dataset/cifar10','num_workers':4,'label_smoothing':0,'warmup_lr':0.,'warmup_epochs':0,'save_latest':True}
if any(cfg.get(k)!=v for k,v in expected.items()):sys.exit('configuration mismatch')
code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
if code!=args.code_sha or subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():sys.exit('code SHA mismatch or tracked files modified')
seed_all(42);torch.set_num_threads(4);torch.cuda.set_device(0);model=model_create().cuda();opt=opt_create(model);sched=sched_create(opt);scaler=torch.cuda.amp.GradScaler();train,val=loaders();validate_split()
metadata={'run_id':args.run_id,'command':sys.argv,'code_sha':code,'config':cfg,'split_sha256':EXPECTED_SPLIT,'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deadline_utc':args.deadline_utc,'gpu_uuid':args.approved_gpu_uuid,'gpu_name':torch.cuda.get_device_name(0),'scheduler':str(sched.__dict__),'optimizer_groups':[{k:v for k,v in g.items() if k!='params'} for g in opt.param_groups],'training_policy':'single seed 42, eval isolation, original AMP threshold update','metric_definitions':'FR_global = total binary spikes / total neuron-time states; FR_author = mean across layers of unweighted batch-mean layer FR, reported separately; no test set'}
(out/'run_metadata.json').write_text(json.dumps(metadata,indent=2,default=str)+'\n')
fields=['epoch','learning_rate','train_loss','train_accuracy','validation_loss','validation_accuracy','best_validation_accuracy','training_sample_count','validation_sample_count','amp_skipped_steps','optimizer_steps','epoch_wall_seconds','peak_allocated_bytes','peak_reserved_bytes','train_fr_global','validation_fr_global','train_fr_author','validation_fr_author','latest_checkpoint_path','latest_checkpoint_sha256','best_checkpoint_path','best_checkpoint_sha256']
layerfields=['epoch','module_name','phase','threshold','firing_rate','author_batch_mean_fr','neuron_state_count','spike_count']
def write_csv(path,fields,rows):
 with path.open('a',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n')
  if f.tell()==0:w.writeheader()
  w.writerows(rows);f.flush();os.fsync(f.fileno())
status={'run_id':args.run_id,'status':'RUNNING','completed_epochs':0,'code_sha':code,'start_utc':metadata['start_utc'],'deadline_utc':args.deadline_utc};best=-1.;consecutive=0;bestsha='';torch.cuda.reset_peak_memory_stats()
def putstatus():
 tmp=out/'status.json.tmp';tmp.write_text(json.dumps(status,indent=2)+'\n');os.replace(tmp,out/'status.json')
putstatus()
try:
 for epoch in range(1,201):
  if time.time()>=stop:raise TimeoutError('budget safety margin reached before epoch')
  if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=code or sha(SPLIT)!=EXPECTED_SPLIT:raise RuntimeError('code version or split changed')
  model.train();ep=time.monotonic();low_memory_since=None;slow_since=None;recent_steps=[];lr=opt.param_groups[0]['lr'];n=0;correct=0;ls=0.;skips=0;steps=0;activity=Activity(model)
  try:
   for x,y in train:
    if time.time()>=stop:raise TimeoutError('budget safety margin reached during epoch')
    batch_start=time.monotonic();x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);loss,logits,skipped=train_step(model,opt,scaler,x,y);n+=len(y);ls+=loss.item()*len(y);correct+=(logits.argmax(1)==y).sum().item();steps+=1;skips+=int(skipped);consecutive=consecutive+1 if skipped else 0
    recent_steps.append(time.monotonic()-batch_start)
    if len(recent_steps)>30:recent_steps.pop(0)
    now=time.monotonic();free_bytes,_=torch.cuda.mem_get_info()
    low_memory_since=(low_memory_since or now) if free_bytes<1024**3 else None
    slow_since=(slow_since or now) if len(recent_steps)==30 and sum(recent_steps)/len(recent_steps)>0.95 else None
    if low_memory_since and now-low_memory_since>120:raise RuntimeError('shared GPU free memory below 1 GiB continuously for 120s')
    if slow_since and now-slow_since>300:raise RuntimeError('shared GPU step time above 0.95 seconds continuously for 300s')
    if consecutive>=10:raise RuntimeError('10 consecutive AMP skipped steps')
   tr=activity.rows()
  finally:activity.close()
  if n!=45000:raise RuntimeError('unexpected training sample count')
  sched.step(epoch);vm,vr=validate(model,val,'cuda')
  if vm['count']!=4500:raise RuntimeError('unexpected validation sample count')
  improved=vm['accuracy']>best
  if improved:best=vm['accuracy']
  ck=pack(model,opt,sched,scaler,epoch-1,best,cfg,code);latest=out/'checkpoint_latest.pth';atomic_save(latest,ck);latestsha=sha(latest)
  if improved:
   atomic_save(out/'checkpoint_best_validation.pth',ck);bestsha=sha(out/'checkpoint_best_validation.pth')
  row={'epoch':epoch,'learning_rate':lr,'train_loss':ls/n,'train_accuracy':correct/n,'validation_loss':vm['loss'],'validation_accuracy':vm['accuracy'],'best_validation_accuracy':best,'training_sample_count':n,'validation_sample_count':vm['count'],'amp_skipped_steps':skips,'optimizer_steps':steps,'epoch_wall_seconds':time.monotonic()-ep,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'train_fr_global':global_fr(tr),'validation_fr_global':vm['fr_global'],'train_fr_author':sum(v['author_batch_mean_fr'] for v in tr.values())/len(tr),'validation_fr_author':vm['fr_author'],'latest_checkpoint_path':str(latest),'latest_checkpoint_sha256':latestsha,'best_checkpoint_path':str(out/'checkpoint_best_validation.pth'),'best_checkpoint_sha256':bestsha}
  write_csv(out/'epochs.csv',fields,[row]);thresholds={n:m.thresh.item() for n,m in model.named_modules() if isinstance(m,ATLIF)}
  write_csv(out/'layers.csv',layerfields,[{'epoch':epoch,'module_name':name,'phase':phase,'threshold':thresholds[name],'firing_rate':v['fr'],'author_batch_mean_fr':v['author_batch_mean_fr'],'neuron_state_count':v['neuron_time_states'],'spike_count':v['spike_count']} for phase,d in [('train',tr),('validation',vr)] for name,v in d.items()])
  status.update(completed_epochs=epoch,last_epoch=row,best_validation_accuracy=best,best_epoch=(epoch if improved else status.get('best_epoch')));putstatus();print(json.dumps(row),flush=True)
  if skips/steps>.05:raise RuntimeError('AMP skipped steps exceed 5% in complete epoch')
 status['status']='COMPLETED'
except BaseException as e:
 status.update(status='PARTIAL' if status['completed_epochs'] else 'FAILED',error=repr(e),traceback=traceback.format_exc())
finally:
 status['end_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();putstatus()
if status['status']!='COMPLETED':sys.exit(1)

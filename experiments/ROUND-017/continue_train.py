"""EXP-018 staged continuation with EXP-019 paired cost accounting."""
import os,sys,time,json,csv,gzip,traceback,hashlib,subprocess,gc,signal
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent));import common as C
import torch,numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset
from spikingjelly.activation_based import functional
from utils.utils import calc_penalty,penalize,clear_penalty
import audit_functions as A,cost
O=C.O;D=C.ROOT/'experiments/EXP-019';RUN=O/'runs/EXP-018-CONTINUE-001';RUN.mkdir(parents=True,exist_ok=False);CODE=subprocess.check_output(['git','rev-parse','HEAD'],cwd=C.ROOT,text=True).strip();K=json.loads((C.ROOT/'experiments/EXP-017/lambda_derivation.json').read_text())['K_actual_batch2_and3'][0];SCHEDULE=list(csv.DictReader((C.ROOT/'experiments/EXP-017/learning_rate_schedule.csv').open()));torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.cuda.set_per_process_memory_fraction(.90);status={'status':'RUNNING','run_id':RUN.name,'start_utc':C.now(),'code_sha':CODE,'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),'completed':{'U0':20,'U2':20},'stage':None,'group':None,'epoch':None};gates={};parents={}
def dump():
 wall,cpu=C.guard();status.update(last_utc=C.now(),wall_seconds=wall,cpu_seconds=cpu);C.dump(O/'status.json',status);C.dump(RUN/'status.json',status)
def abort(signum,frame):raise TimeoutError('budget watchdog signal '+str(signum))
signal.signal(signal.SIGTERM,abort);signal.signal(signal.SIGINT,abort)
def save(m,o,sch,sc,epoch,g,lam,metrics):
 d=RUN/g;d.mkdir(parents=True,exist_ok=True);p=d/'latest.pth';tmp=d/'latest.pth.tmp';torch.save({'model':m.state_dict(),'optimizer':o.state_dict(),'scheduler':sch.state_dict(),'scaler':sc.state_dict(),'epoch':epoch,'group':g,'lambda_weight':lam,'lambda_neuron':lam,'rng':C.rng(),'sampling':{'method':'epoch-specific torch randperm and DataLoader worker seed','seed_base':12450,'next_epoch':epoch+1},'config':C.CFG,'code_sha':CODE,'parent_run_id':'EXP-017-PILOT-001','metrics':metrics},tmp);os.replace(tmp,p);return p
def link(src,dst):
 dst=Path(dst);tmp=dst.with_suffix('.pth.tmp');
 if tmp.exists():tmp.unlink()
 os.link(src,tmp);os.replace(tmp,dst)
def restore(g):
 f=(RUN/g/'latest.pth') if (RUN/g/'latest.pth').exists() else cost.parent(20,g);c=torch.load(f,map_location='cpu',weights_only=False);m=C.model().train();m.load_state_dict(c['model']);o=C.optim(m);sch=C.scheduler(o);sc=torch.amp.GradScaler('cuda');o.load_state_dict(c['optimizer']);sch.load_state_dict(c['scheduler']);sc.load_state_dict(c['scaler']);C.set_rng(c['rng']);assert c['group']==g and c['lambda_weight']==C.CFG['groups'][g] and c['lambda_neuron']==C.CFG['groups'][g];assert abs(o.param_groups[0]['lr']-float(SCHEDULE[c['epoch']]['learning_rate']))<1e-12;return c,m,o,sch,sc
def record_shard(g,e):
 rows=[]
 for kind in ['validation_per_sample','validation_batch_sops']:
  p=O/kind/g/f'epoch-{e:03d}.csv.gz'
  with gzip.open(p,'rt') as f:n=sum(1 for _ in f)-1
  assert n==(4500 if kind=='validation_per_sample' else 18);rows.append({'group':g,'epoch':e,'kind':kind,'path':str(p.relative_to(C.ROOT)),'format':'gzip CSV','rows':n,'sha256':C.sha(p),'bytes':p.stat().st_size})
 C.append(O/'validation_shards_manifest.csv',rows)
def marker(g,e,metrics):
 p=RUN/g/f'epoch-{e:03d}.pth';c=torch.load(p,map_location='cpu',weights_only=False);assert c['epoch']==e and c['group']==g and all(k in c for k in ['model','optimizer','scheduler','scaler','rng','sampling']);row={'group':g,'epoch':e,'path':str(p),'sha256':C.sha(p),'bytes':p.stat().st_size,'parent_run_id':'EXP-017-PILOT-001','code_sha':c['code_sha'],'load_verified':True};C.append(O/'checkpoint_manifest.csv',[row]);del c

def run_group(g,end,train,val_loader,train_ids):
 c,m,opt,sch,scaler=restore(g);start=int(c['epoch'])+1;lam=C.CFG['groups'][g];assert start<=end;status.update(group=g,epoch=start);successful=int(c['metrics']['cumulative_successful_steps']);skipped=int(c['metrics']['cumulative_amp_skips']);manual=int(c['metrics']['cumulative_manual_updates']);zero_streak={n:0 for n in K if K[n]>0};d=RUN/g;d.mkdir(parents=True,exist_ok=True);oldrows=list(csv.DictReader((C.ROOT/'experiments/EXP-017/epoch_metrics.csv').open()));oldbest=max(float(x['validation_accuracy']) for x in oldrows if x['group']==g);newrows=list(csv.DictReader((O/'epoch_metrics.csv').open())) if (O/'epoch_metrics.csv').exists() else [];best=max([oldbest]+[float(x['validation_accuracy']) for x in newrows if x['group']==g]);bestpath=d/'best_validation.pth'
 if not bestpath.exists():link(C.ROOT/f'experiments/EXP-017/runs/EXP-017-PILOT-001/{g}/best_validation.pth',bestpath)
 for epoch in range(start,end+1):
  C.guard(reserve=300);status.update(group=g,epoch=epoch);dump();t0=time.monotonic();lr=opt.param_groups[0]['lr'];gen=torch.Generator().manual_seed(12450+epoch);perm=torch.randperm(len(train),generator=gen).tolist();used=perm[:175*256];dropped=[train_ids[i] for i in perm[175*256:]];assert len(dropped)==200;wg=torch.Generator().manual_seed(99912450+epoch);loader=DataLoader(Subset(train,used),batch_size=256,shuffle=False,num_workers=4,pin_memory=True,drop_last=True,generator=wg);order_hash=hashlib.sha256(json.dumps([train_ids[i] for i in used],separators=(',',':')).encode()).hexdigest();C.append(O/'training_sample_audit.csv',[{'group':g,'epoch':epoch,'sampler_seed':12450+epoch,'worker_seed':99912450+epoch,'sample_order_sha256':order_hash,'dropped_count':200,'dropped_ids':json.dumps(dropped)}]);m.train();count=correct=loss_sum=0;es=ek=em=0
  for j,(x,y,sid) in enumerate(loader):
   if j%10==0:C.guard(reserve=240)
   x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda'):z=m(x);loss=F.cross_entropy(z.mean(0),y)
   if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
   scaler.scale(loss).backward();functional.reset_net(m);old=scaler.get_scale();scaler.step(opt);scaler.update();skip=scaler.get_scale()<old;opt.zero_grad(set_to_none=True)
   if lam:calc_penalty(m);penalize(m,lam*lr,lam*lr);clear_penalty(m);em+=1
   if j%20==0:
    for name,v in m.named_modules():
     if isinstance(v,C.SparseConv2d) and not torch.isfinite(v.threshold).all():raise FloatingPointError('nonfinite threshold '+name)
    wall,cpu=C.guard(reserve=240);C.dump(O/'heartbeat.json',{'utc':C.now(),'group':g,'epoch':epoch,'batch':j,'wall_seconds':wall,'cpu_seconds':cpu,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
   n=len(y);loss_sum+=loss.item()*n;correct+=(z.mean(0).argmax(1)==y).sum().item();count+=n;ek+=int(skip);es+=int(not skip)
  assert count==44800 and j==174;successful+=es;skipped+=ek;manual+=em;train_seconds=time.monotonic()-t0;v0=time.monotonic();v=A.evaluate(m,val_loader,g,epoch);validation_seconds=time.monotonic()-v0;record_shard(g,epoch);layer=A.layer_audit(m,g,epoch)
  for n,z in m.named_modules():
   if isinstance(z,C.SparseConv2d) and K[n]>0:
    zero_streak[n]=zero_streak[n]+1 if z.density()==0 else 0
    if zero_streak[n]>=2:raise RuntimeError('penalized layer zero density persisted two epochs '+n)
  sch.step(epoch);row={'group':g,'epoch':epoch,'learning_rate':lr,'train_loss':loss_sum/count,'train_accuracy':correct/count,'train_samples':count,**v,**layer,'epoch_successful_steps':es,'epoch_amp_skips':ek,'epoch_manual_updates':em,'cumulative_successful_steps':successful,'cumulative_amp_skips':skipped,'cumulative_manual_updates':manual,'train_seconds':train_seconds,'validation_seconds':validation_seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()};t=time.monotonic();ck=save(m,opt,sch,scaler,epoch,g,lam,row);row['checkpoint_seconds']=time.monotonic()-t
  if v['validation_accuracy']>best:best=v['validation_accuracy'];link(ck,bestpath)
  if epoch==end:link(ck,d/f'epoch-{epoch:03d}.pth')
  C.append(O/'epoch_metrics.csv',[row]);expected=lam*sum(float(x['learning_rate'])*175 for x in SCHEDULE[:epoch]);layers=[x for x in csv.DictReader((O/'layer_sparsity.csv').open()) if x['group']==g and int(x['epoch'])==epoch];C.append(O/'threshold_prediction_audit.csv',[{'group':g,'epoch':epoch,'layer':x['layer'],'observed':x['threshold'],'predicted':expected*K[x['layer']],'difference':float(x['threshold'])-expected*K[x['layer']]} for x in layers]);wall,cpu=C.guard(180);C.append(O/'resource_usage.csv',[{'group':g,'epoch':epoch,'wall_seconds':wall,'cpu_seconds':cpu,'train_seconds':train_seconds,'validation_seconds':validation_seconds,'checkpoint_seconds':row['checkpoint_seconds'],'peak_allocated_bytes':row['peak_allocated_bytes'],'peak_reserved_bytes':row['peak_reserved_bytes']}]);status['completed'][g]=epoch;dump();print(json.dumps({'event':'epoch_complete',**row,'wall_seconds':wall,'cpu_seconds':cpu}),flush=True)
 marker(g,end,row);del c,m,opt,sch,scaler;gc.collect();torch.cuda.empty_cache();return row

def stage_sync(stage):
 # Commits happen between stages; code SHA was captured at process launch and never inferred again.
 try:
  subprocess.run(['git','add','experiments/EXP-018','experiments/EXP-019'],cwd=C.ROOT,check=True,timeout=180);r=subprocess.run(['git','diff','--cached','--quiet'],cwd=C.ROOT)
  if r.returncode==1:subprocess.run(['git','commit','-m',f'results(EXP-018): record paired epoch {stage} stage'],cwd=C.ROOT,check=True,timeout=180,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
  q=subprocess.run(['git','push'],cwd=C.ROOT,check=True,timeout=180,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);return {'status':'PASSED','output':q.stdout[-2000:]}
 except Exception as e:return {'status':'FAILED','error':repr(e)}
def run():
 s=C.check();assert json.loads((O/'resume_preflight_tests.json').read_text())['status']=='PASSED';assert (D/'cost_count_audit_epoch_020.json').exists();train=C.TrainFileOnly(s['train'],True);val=C.TrainFileOnly(s['validation'],False);vl=DataLoader(val,batch_size=256,shuffle=False,num_workers=4,pin_memory=True,drop_last=False)
 for stage in [60,100,200,300]:
  C.guard(reserve=400);status['stage']=stage;results={}
  for g in ['U0','U2']:results[g]=run_group(g,stage,train,vl,s['train'])
  r0,r2=results['U0'],results['U2'];cond={'accuracy_drop_le_5pp':r0['validation_accuracy']-r2['validation_accuracy']<=.05,'sop_savings_ge_5pct':1-r2['SOP_sample_weighted']/r0['SOP_sample_weighted']>=.05,'all_penalized_layers_density_ge_1pct':r2['min_penalized_density']>=.01,'both_finished_and_checkpoints_verified':status['completed']['U0']==stage==status['completed']['U2'],'finite_and_data_valid':True};cond['passed']=all(cond.values());gates[str(stage)]={'conditions':cond,'U0':r0,'U2':r2,'sop_savings':1-r2['SOP_sample_weighted']/r0['SOP_sample_weighted'],'accuracy_delta':r2['validation_accuracy']-r0['validation_accuracy']};C.dump(O/'stage_gate_results.json',gates);v=cost.evaluate(stage,s['validation']);expected=[r0['SOP_sample_weighted'],r2['SOP_sample_weighted']];err=[abs(v[k]-x)/(1+abs(x)) for k,x in zip(['C00','C22'],expected)];assert max(err)<=1e-5;C.dump(D/f'epoch_{stage:03d}_training_metric_crosscheck.json',{'status':'PASSED','relative_errors':err,'epoch':stage});status['last_stage_gate']=cond;dump();sync=stage_sync(stage);C.append(O/'stage_sync.csv',[{'epoch':stage,'status':sync['status'],'detail':str(sync.get('error',sync.get('output','')))}]);print(json.dumps({'event':'stage_complete','stage':stage,'conditions':cond,'sync':sync}),flush=True)
  if not cond['passed'] and stage<300:status['stop_reason']='stage gate failed at epoch '+str(stage);return 'PARTIAL'
 return 'COMPLETED'
try:status['status']=run()
except BaseException as e:status.update(status='PARTIAL',error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
finally:
 b=json.loads((O/'budget.json').read_text());status.update(end_utc=C.now(),wall_seconds=time.time()-b['start_unix'],cpu_seconds=C.cpu_seconds()-C.CPU0+b.get('prior_cpu_seconds',0));C.dump(O/'status.json',status);C.dump(RUN/'status.json',status);print(json.dumps(status),flush=True)
if status['status']!='COMPLETED':sys.exit(1)

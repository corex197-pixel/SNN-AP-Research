"""EXP-017 fixed 20-epoch pilots, original UniSparse updates and isolated validation."""
import sys,os,time,json,traceback,hashlib,random,datetime,subprocess,gc,signal
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent));import common as C
import torch,numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset
from utils.utils import calc_penalty,penalize,clear_penalty,SOPMonitor
from spikingjelly.activation_based import functional
O=C.O;start=time.time();torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.cuda.set_per_process_memory_fraction(.90);CODE=subprocess.check_output(['git','rev-parse','HEAD'],cwd=C.ROOT,text=True).strip();LOCK=json.loads((O/'selection_lock.json').read_text());K=json.loads((O/'lambda_derivation.json').read_text())['K_actual_batch2_and3'][0];RUN=O/'runs/EXP-017-PILOT-001';RUN.mkdir(parents=True,exist_ok=False);status={'status':'RUNNING','run_id':RUN.name,'start_utc':C.now(),'code_sha':CODE,'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),'completed':{},'current_group':None,'current_epoch':None}
def dump():
 wall,cpu=C.guard();status.update(last_utc=C.now(),wall_seconds_including_preflight=wall,cpu_seconds_including_preflight=cpu);C.dump(O/'status.json',status);C.dump(RUN/'status.json',status)
def stop_signal(signum,frame):raise TimeoutError('external budget watchdog signal '+str(signum))
signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
def save_ck(m,opt,sch,scaler,epoch,group,lam,metrics):
 d=RUN/group;d.mkdir(parents=True,exist_ok=True);path=d/'latest.pth';temp=d/'latest.pth.tmp';obj={'model':m.state_dict(),'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'group':group,'lambda_weight':lam,'lambda_neuron':lam,'rng':C.rng(),'sampling':{'method':'epoch-specific torch randperm and DataLoader worker seed','seed_base':12450,'next_epoch':epoch+1},'config':C.CFG,'code_sha':CODE,'metrics':metrics};torch.save(obj,temp);os.replace(temp,path);return path
def link_ck(src,name):
 dst=src.parent/name;tmp=dst.with_suffix('.pth.tmp');
 if tmp.exists():tmp.unlink()
 os.link(src,tmp);os.replace(tmp,dst);return dst
def layer_audit(m,group,epoch):
 rows=[];bn=[];fan=[]
 mods=dict(m.named_modules())
 for n,z in mods.items():
  if isinstance(z,C.SparseConv2d):
   w=z.sparse_weight().detach();count=int((w!=0).sum().item());rows.append({'group':group,'epoch':epoch,'layer':n,'penalized':bool(K[n]>0),'threshold':z.threshold.item(),'nonzero':count,'total':w.numel(),'density':count/w.numel()});assert torch.isfinite(z.threshold).item() and torch.isfinite(w).all().item()
  if isinstance(z,C.SparseBN):bn.append({'group':group,'epoch':epoch,'layer':n,'gamma_mean':z.weight.mean().item(),'gamma_std':z.weight.std(unbiased=False).item(),'beta_mean':z.bias.mean().item(),'beta_std':z.bias.std(unbiased=False).item()})
 for neuron,consumer in [('static_conv.activation','conv1.conv'),('conv1.activation','conv2'),('conv3.activation','conv4.conv'),('conv4.activation','conv5')]:
  v=mods[consumer];d=v.sparse_mask().sum(dim=(0,2,3)).detach().float().cpu().numpy();mean=float(d.mean());fan.append({'group':group,'epoch':epoch,'neuron':neuron,'consumer':consumer,'nominal':v.out_channels*v.kernel_size[0]*v.kernel_size[1],'mean':mean,'std':float(d.std()),'CV':float(d.std()/mean) if mean else 'NA_zero_mean','min':float(d.min()),'p10':float(np.quantile(d,.1)),'p50':float(np.quantile(d,.5)),'p90':float(np.quantile(d,.9)),'max':float(d.max()),'zero_fraction':float((d==0).mean()),'channel_fanout_json':json.dumps(d.tolist())})
 C.append(O/'layer_sparsity.csv',rows);C.append(O/'threshold_trajectory.csv',[{'group':r['group'],'epoch':r['epoch'],'layer':r['layer'],'threshold':r['threshold'],'penalized':r['penalized']} for r in rows]);C.append(O/'bn_statistics.csv',bn)
 if epoch in [0,1,5,10,20]:C.append(O/'effective_fanout_statistics.csv',fan)
 total=sum(r['total'] for r in rows);nonzero=sum(r['nonzero'] for r in rows);pen=[r for r in rows if r['penalized']];return {'all_sparse_weight_sparsity':1-nonzero/total,'penalized_weight_sparsity':1-sum(r['nonzero'] for r in pen)/sum(r['total'] for r in pen),'min_penalized_density':min(r['density'] for r in pen)}
def evaluate(m,data,group,epoch):
 pre_rng=C.rng();pre_modes={z:z.training for z in m.modules()};pre_buffers={n:v.detach().clone() for n,v in m.named_buffers()};saved_cache={z:z.pre_norm_input for z in m.modules() if hasattr(z,'pre_norm_input')};m.eval();mon=SOPMonitor(m);spikes={};states={};batch_rates={};batches={};first_numel={};hooks=[];nonbinary={}
 for n,z in m.named_modules():
  if isinstance(z,C.SparseNeuron):
   def hook(mod,inp,out,n=n):
    a=(out!=0).sum(dtype=torch.int64);spikes[n]=spikes.get(n,0)+a;states[n]=states.get(n,0)+out.numel();batch_rates[n]=batch_rates.get(n,0)+a.double()/out.numel();batches[n]=batches.get(n,0)+1;first_numel.setdefault(n,out.numel())
   hooks.append(z.register_forward_hook(hook))
 # Check monitored layer inputs are binary on the first validation batch.
 for name in mon.monitored_layers:
  def hook(mod,args,name=name):
   if name not in nonbinary:
    a=args[0];nonbinary[name]=bool(((a!=0)&(a!=1)).any().item())
  hooks.append(m.get_submodule(name).register_forward_pre_hook(hook))
 total_sop=0.;batch_sops=[];samples=[];count=0;loss_sum=0.;correct=0
 try:
  with torch.no_grad():
   for batch,(x,y,sid) in enumerate(data):
    C.guard(reserve=70);x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);mon.clear_recorded_data();z=m(x).mean(0);loss=F.cross_entropy(z,y,reduction='none');assert torch.isfinite(loss).all();pred=z.argmax(1);so=sum(float(v.item()) for v in mon.records);total_sop+=so;count+=len(y);loss_sum+=float(loss.sum().item());correct+=int((pred==y).sum().item());batch_sops.append({'group':group,'epoch':epoch,'batch':batch,'samples':len(y),'sop_count':so});samples.extend({'group':group,'epoch':epoch,'sample_id':i,'label':t,'prediction':p,'correct':int(p==t),'loss':l} for i,t,p,l in zip(sid.tolist(),y.cpu().tolist(),pred.cpu().tolist(),loss.cpu().tolist()));functional.reset_net(m)
 finally:
  mon.remove_hooks()
  for h in hooks:h.remove()
  functional.reset_net(m)
  for n,v in m.named_buffers():
   if not torch.equal(v,pre_buffers[n]):raise RuntimeError('eval mutated buffer '+n)
  for z,val in saved_cache.items():z.pre_norm_input=val
  for z,mode in pre_modes.items():z.training=mode
  C.set_rng(pre_rng)
 assert count==4500;C.append(O/'validation_batch_sops.csv',batch_sops);C.append(O/'validation_per_sample.csv',samples);weighted=sum(int(v.item()) for v in spikes.values())/sum(states.values());author=sum(float((batch_rates[n]/batches[n]).item())*first_numel[n] for n in spikes)/sum(first_numel.values());return {'validation_accuracy':correct/count,'validation_loss':loss_sum/count,'FR_sample_weighted':weighted,'FR_author':author,'SOP_author':sum(r['sop_count'] for r in batch_sops)/len(batch_sops)/256,'SOP_sample_weighted':total_sop/count,'nonbinary_monitored_inputs':json.dumps([k for k,v in nonbinary.items() if v]),'validation_samples':count}
def run():
 s=C.check();assert json.loads((O/'preflight_tests.json').read_text())['status']=='PASSED';train=C.TrainFileOnly(s['train'],True);val=C.TrainFileOnly(s['validation'],False);val_loader=DataLoader(val,batch_size=256,shuffle=False,num_workers=4,pin_memory=True,drop_last=False);summ=[]
 for group in ['U0','U1','U2']:
  C.guard(reserve=100);lam=LOCK['lambda_weight_neuron'][group];status['current_group']=group;m=C.model().train();assert C.digest(m)==LOCK['initialization_sha256'];opt=C.optim(m);sch=C.scheduler(opt);scaler=torch.amp.GradScaler('cuda');torch.cuda.reset_peak_memory_stats();successful=skipped=manual=0;best=-1.;zero_streak={n:0 for n in K if K[n]>0};layer_audit(m,group,0)
  for epoch in range(1,21):
   C.guard(reserve=100);status['current_epoch']=epoch;dump();t0=time.monotonic();lr=opt.param_groups[0]['lr'];gen=torch.Generator().manual_seed(12450+epoch);perm=torch.randperm(len(train),generator=gen).tolist();used=perm[:175*256];dropped=[s['train'][i] for i in perm[175*256:]];assert len(dropped)==200;worker_gen=torch.Generator().manual_seed(99912450+epoch);loader=DataLoader(Subset(train,used),batch_size=256,shuffle=False,num_workers=4,pin_memory=True,drop_last=True,generator=worker_gen);hash_ids=hashlib.sha256(json.dumps([s['train'][i] for i in used],separators=(',',':')).encode()).hexdigest();C.append(O/'training_sample_audit.csv',[{'group':group,'epoch':epoch,'sampler_seed':12450+epoch,'worker_seed':99912450+epoch,'sample_order_sha256':hash_ids,'dropped_count':len(dropped),'dropped_ids':json.dumps(dropped)}]);m.train();count=correct=loss_sum=0;epoch_success=epoch_skip=epoch_manual=0
   for j,(x,y,sid) in enumerate(loader):
    if j%10==0:C.guard(reserve=75)
    x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda'):z=m(x);loss=F.cross_entropy(z.mean(0),y)
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
    scaler.scale(loss).backward();functional.reset_net(m);old=scaler.get_scale();scaler.step(opt);scaler.update();skip=scaler.get_scale()<old;opt.zero_grad(set_to_none=True)
    if lam:
     calc_penalty(m);penalize(m,lam*lr,lam*lr);clear_penalty(m);epoch_manual+=1
    if j%20==0:
     for name,v in m.named_modules():
      if isinstance(v,C.SparseConv2d) and not torch.isfinite(v.threshold).all():raise FloatingPointError('nonfinite threshold '+name)
     wall,cpu=C.guard(reserve=75);C.dump(O/'heartbeat.json',{'utc':C.now(),'group':group,'epoch':epoch,'batch':j,'wall_seconds':wall,'cpu_seconds':cpu,'gpu_peak_allocated':torch.cuda.max_memory_allocated()})
    n=len(y);loss_sum+=loss.item()*n;correct+=(z.mean(0).argmax(1)==y).sum().item();count+=n;epoch_skip+=int(skip);epoch_success+=int(not skip)
   assert count==44800 and j==174;successful+=epoch_success;skipped+=epoch_skip;manual+=epoch_manual;train_seconds=time.monotonic()-t0;v0=time.monotonic();valmetrics=evaluate(m,val_loader,group,epoch);validation_seconds=time.monotonic()-v0;stats=layer_audit(m,group,epoch)
   for n,z in m.named_modules():
    if isinstance(z,C.SparseConv2d) and K[n]>0:
     zero_streak[n]=zero_streak[n]+1 if z.density()==0 else 0
     if zero_streak[n]>=2:raise RuntimeError('penalized layer zero density persisted two epochs '+n)
   sch.step(epoch);row={'group':group,'epoch':epoch,'learning_rate':lr,'train_loss':loss_sum/count,'train_accuracy':correct/count,'train_samples':count,**valmetrics,**stats,'epoch_successful_steps':epoch_success,'epoch_amp_skips':epoch_skip,'epoch_manual_updates':epoch_manual,'cumulative_successful_steps':successful,'cumulative_amp_skips':skipped,'cumulative_manual_updates':manual,'train_seconds':train_seconds,'validation_seconds':validation_seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()};t=time.monotonic();ck=save_ck(m,opt,sch,scaler,epoch,group,lam,row);checkpoint_seconds=time.monotonic()-t;row['checkpoint_seconds']=checkpoint_seconds
   if valmetrics['validation_accuracy']>best:best=valmetrics['validation_accuracy'];link_ck(ck,'best_validation.pth')
   if epoch in [5,10,20]:link_ck(ck,f'epoch-{epoch:03d}.pth')
   C.append(O/'epoch_metrics.csv',[row]);wall,cpu=C.guard(reserve=50);C.append(O/'resource_usage.csv',[{'group':group,'epoch':epoch,'wall_seconds':wall,'cpu_seconds':cpu,'train_seconds':train_seconds,'validation_seconds':validation_seconds,'checkpoint_seconds':checkpoint_seconds,'allocated_peak_bytes':row['peak_allocated_bytes'],'reserved_peak_bytes':row['peak_reserved_bytes']}]);status['completed'][group]=epoch;dump();print(json.dumps({'event':'epoch_complete',**row,'wall_seconds':wall,'cpu_seconds':cpu}),flush=True)
  # Independently reload the final checkpoint, verify training state and record every retained artifact.
  check=torch.load(ck,map_location='cpu',weights_only=False);h=hashlib.sha256()
  for n,v in check['model'].items():h.update(n.encode());h.update(v.detach().contiguous().numpy().tobytes())
  assert h.hexdigest()==C.digest(m) and check['epoch']==20 and check['scheduler']==sch.state_dict() and check['scaler']==scaler.state_dict() and check['optimizer']['param_groups'][0]['lr']==opt.param_groups[0]['lr']
  assert all(k in check['rng'] for k in ['python','numpy','torch','cuda']);manifest_path=O/'checkpoint_manifest.json';manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else []
  for name in ['latest.pth','best_validation.pth','epoch-005.pth','epoch-010.pth','epoch-020.pth']:
   f=ck.parent/name;manifest.append({'group':group,'path':str(f),'bytes':f.stat().st_size,'sha256':C.sha(f),'final_checkpoint_reload_verified':name in ['latest.pth','epoch-020.pth']})
  C.dump(manifest_path,manifest);del check;row['checkpoint_reload_verified']=True
  summ.append(row);C.append(O/'summary.csv',[row]);del m,opt,sch,scaler;gc.collect();torch.cuda.empty_cache()
 ref=summ[0];by={r['group']:r for r in summ};conditions={}
 for g in ['U1','U2']:
  r=by[g];conditions[g]={'penalized_sparsity_ge_1pct':r['penalized_weight_sparsity']>=.01,'sop_savings_ge_5pct':1-r['SOP_sample_weighted']/ref['SOP_sample_weighted']>=.05,'accuracy_drop_le_5pp':ref['validation_accuracy']-r['validation_accuracy']<=.05,'each_penalized_layer_density_ge_1pct':r['min_penalized_density']>=.01,'finite_and_checkpoint':bool(r['checkpoint_reload_verified'])};conditions[g]['passed']=all(conditions[g].values())
 valid=[by[g] for g in ['U1','U2'] if conditions[g]['passed']];chosen=sorted(valid,key=lambda r:(r['SOP_sample_weighted'],-r['validation_accuracy'],r['group']))[0]['group'] if valid else None;C.dump(O/'pilot_selection.json',{'conditions':conditions,'recommended':chosen,'reason':'pre-registered SOP then accuracy then U1 tie-break; single seed exploratory','comparison_epoch':20});return 'COMPLETED'
try:status['status']=run()
except BaseException as e:status.update(status='PARTIAL',error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
finally:
 try:
  b=json.loads((O/'budget.json').read_text());status.update(end_utc=C.now(),wall_seconds_including_preflight=time.time()-b['start_unix'],cpu_seconds_including_preflight=C.cpu_seconds()-C.CPU0+b.get('prior_cpu_seconds',0));C.dump(O/'status.json',status);C.dump(RUN/'status.json',status);print(json.dumps(status),flush=True)
 except BaseException:traceback.print_exc()
if status['status']!='COMPLETED':sys.exit(1)

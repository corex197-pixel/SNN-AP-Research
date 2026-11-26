"""ROUND-009 fixed-protocol, one GPU. Results and checkpoints are append-only."""
import os,sys,time,json,csv,hashlib,random,datetime,traceback,subprocess,resource,copy
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset,Dataset
from torchvision.datasets import CIFAR10
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/EXP-004/controlled'));sys.path.insert(0,str(ROOT/'experiments/ROUND-009'))
from core import model_create,transforms_pair,DATA,ATLIF,seed_all
from proxy import Proxy
from utils.utils import SOPMonitor
from timm.optim import create_optimizer_v2
from spikingjelly.activation_based import functional
M=ROOT/'experiments/EXP-004/runs/EXP-004-RUN-003/checkpoint_best_validation.pth';S=ROOT/'experiments/EXP-004/splits/indices.json';OUT=ROOT/'experiments/EXP-008';C0=54945117.504
SHA='c068f54379cb5d29bcfa5fe2d9de676cd3cc153433ada1ed5aa1c66355d0b32a';SSHA='69dccfe992236766aea575a0a66495802ca889ca329d61f51e8de3c1ab7f3165';start=time.monotonic();cpu0=resource.getrusage(resource.RUSAGE_SELF);pre=json.loads((OUT/'runs/PREFLIGHT-001/smoke.json').read_text());pre2=json.loads((OUT/'runs/PREFLIGHT-002/smoke.json').read_text());pre['wall_seconds']+=pre2['wall_seconds'];pre['cpu_seconds']+=pre2['cpu_seconds'];GPU_LIMIT=4400.;CPU_LIMIT=5200.
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def append(path,rows):
 if not rows:return
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('a',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));
  if f.tell()==0:w.writeheader()
  w.writerows(rows);f.flush();os.fsync(f.fileno())
def cpusec():
 me=resource.getrusage(resource.RUSAGE_SELF);ch=resource.getrusage(resource.RUSAGE_CHILDREN)
 return me.ru_utime+me.ru_stime-cpu0.ru_utime-cpu0.ru_stime+ch.ru_utime+ch.ru_stime+pre['cpu_seconds']
def guard():
 if time.monotonic()-start+pre['wall_seconds']>GPU_LIMIT:raise TimeoutError('GPU budget guard')
 if cpusec()>CPU_LIMIT:raise TimeoutError('aggregate CPU budget guard')
def digest_state(model):
 h=hashlib.sha256()
 for n,t in model.state_dict().items():h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def filehash(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class Indexed(Dataset):
 def __init__(self,ds,ids):self.ds=ds;self.ids=ids
 def __len__(self):return len(self.ids)
 def __getitem__(self,i):
  x,y=self.ds[self.ids[i]];return x,y,self.ids[i]
def loader(ids,training):
 return DataLoader(Indexed(CIFAR10(DATA,train=True,download=False,transform=transforms_pair()[0 if training else 1]),ids),batch_size=128,shuffle=False,num_workers=4,pin_memory=True)
def newmodel(alpha,freeze_all=False):
 model=model_create().cuda();model.load_state_dict(ck['model'])
 for m in model.modules():
  if isinstance(m,ATLIF):m.thresh.data.mul_(alpha);m.thresh.requires_grad_(False);m.sp=0.;m.update_value=0.
 if freeze_all:
  for p in model.parameters():p.requires_grad_(False)
 return model
def clean(model):
 functional.reset_net(model)
 for m in model.modules():
  if isinstance(m,ATLIF):m.act_value=0.;m.update_value=0.
def evaluate(model,tag,ids):
 guard();model.eval();mon=SOPMonitor(model);layers={};hs=[]
 for n,m in model.named_modules():
  if isinstance(m,ATLIF):
   def hook(m,a,o,n=n):
    r=layers.setdefault(n,{'spikes':0,'states':0,'batch_fr_sum':0.,'batches':0,'threshold':float(m.thresh.item())});v=int((o>0).sum().item());r['spikes']+=v;r['states']+=o.numel();r['batch_fr_sum']+=v/o.numel();r['batches']+=1
   hs.append(m.register_forward_hook(hook))
 data=[];batches=[];count=0;start_eval=time.monotonic()
 try:
  with torch.no_grad():
   for i,(x,y,idsbatch) in enumerate(loader(ids,False)):
    guard();x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);mon.clear_recorded_data();z=model(x).mean(0).float();loss=F.cross_entropy(z,y,reduction='none');pred=z.argmax(1)
    if not torch.isfinite(loss).all():raise RuntimeError('nonfinite eval')
    so=sum(float(a.item()) for a in mon.records);count+=len(y)
    batches.append({'group':tag,'batch':i,'samples':len(y),'sop_count':so})
    for sid,l,pr,target in zip(idsbatch.tolist(),loss.tolist(),pred.tolist(),y.tolist()):data.append({'group':tag,'sample_id':sid,'loss':l,'prediction':pr,'correct':int(pr==target)})
    clean(model)
 finally:
  mon.remove_hooks()
  for h in hs:h.remove()
 append(OUT/'validation_per_sample.csv',data);append(OUT/'validation_batch_sops.csv',batches)
 for n,v in layers.items():v['FR']=v['spikes']/v['states'];v['author_FR']=v['batch_fr_sum']/v['batches']
 append(OUT/'layer_metrics.csv',[{'group':tag,'module':n,**v} for n,v in layers.items()])
 return {'group':tag,'accuracy':np.mean([r['correct'] for r in data]).item(),'loss':np.mean([r['loss'] for r in data]).item(),'SOPs':sum(r['sop_count'] for r in batches)/count,'FR_global':sum(v['spikes'] for v in layers.values())/sum(v['states'] for v in layers.values()),'FR_author':np.mean([v['author_FR'] for v in layers.values()]).item(),'parameter_count':sum(p.numel() for p in model.parameters()),'eval_seconds':time.monotonic()-start_eval,'n':count},data,layers

def run():
 global ck
 if pre['status']!='PASSED' or pre2['status']!='PASSED':raise RuntimeError('preflight failed')
 if torch.cuda.device_count()!=1 or '4090' not in torch.cuda.get_device_name(0):raise RuntimeError('invalid visible device')
 torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 if filehash(M)!=SHA or filehash(S)!=SSHA:raise RuntimeError('data version mismatch')
 ck=torch.load(M,map_location='cpu',weights_only=False);spl=json.loads(S.read_text());train_ids=spl['train'];val_ids=spl['validation']
 assert len(train_ids)==45000 and len(val_ids)==4500 and not set(train_ids)&set(spl['diagnostic'])
 torch.cuda.reset_peak_memory_stats();torch.cuda.set_device(0)
 model=newmodel(1.,True);B0,per0,l0=evaluate(model,'B0',val_ids);del model
 model=newmodel(1.25,True);P0,perp,lp=evaluate(model,'P0',val_ids);del model
 summaries=[B0,P0];per={'B0':per0,'P0':perp};initial_hash=None;first_batches=None;paths=[]
 for group in ['F0','F1','F2','M2']:
  guard();seed_all(2030);torch.cuda.reset_peak_memory_stats();group_start=time.monotonic();run=OUT/'runs'/('EXP-008-'+group+'-RUN-001');run.mkdir(parents=True,exist_ok=False)
  model=newmodel(1.25);model.train();rng_before_teacher=torch.get_rng_state();cuda_before_teacher=torch.cuda.get_rng_state();teacher=newmodel(1.,True).eval() if group in ('F2','M2') else None;torch.set_rng_state(rng_before_teacher);torch.cuda.set_rng_state(cuda_before_teacher)
  init=digest_state(model);initial_hash=initial_hash or init
  if init!=initial_hash:raise RuntimeError('student initialization mismatch')
  teacher_initial=digest_state(teacher) if teacher else None
  opt=create_optimizer_v2(model,opt='sgd',lr=.005,weight_decay=5e-4,momentum=.9,nesterov=True);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=5,eta_min=.0001);scaler=torch.cuda.amp.GradScaler(enabled=True);proxy=Proxy(model);fixed={n:p.detach().clone() for n,p in model.named_parameters() if n.endswith('thresh')}
  successful=skips=0;fingerprints=[]
  dump(run/'metadata.json',{'group':group,'started_utc':now(),'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'initial_model_sha256':init,'teacher_model_sha256':teacher_initial,'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),'optimizer_param_groups':[{'weight_decay':g['weight_decay'],'parameter_count':len(g['params'])} for g in opt.param_groups]})
  try:
   for epoch in range(1,6):
    train=loader(train_ids,True);total=correct=n=nb=violate=0;ce_sum=kd_sum=sop_sum=proxy_sum=0.;tic=time.monotonic();lr=opt.param_groups[0]['lr']
    for batch,(x,y,sids) in enumerate(train):
     guard()
     if batch<3 and epoch==1:fingerprints.append({'ids':sids.tolist(),'images_sha256':hashlib.sha256(x.numpy().tobytes()).hexdigest()})
     x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True);proxy.clear()
     if teacher:
      rng=torch.cuda.get_rng_state();cpu_rng=torch.get_rng_state()
      with torch.no_grad(),torch.cuda.amp.autocast():zt=teacher(x).mean(0).float()
      if not torch.equal(rng,torch.cuda.get_rng_state()) or not torch.equal(cpu_rng,torch.get_rng_state()):raise RuntimeError('teacher consumed random state')
      clean(teacher)
     with torch.cuda.amp.autocast():
      z=model(x).mean(0).float();ce=F.cross_entropy(z,y);c=proxy.value();sop=F.relu(c/C0-.8).square();kd=F.kl_div(F.log_softmax(z/3,dim=-1),F.softmax(zt/3,dim=-1),reduction='batchmean')*9 if teacher else torch.zeros((),device=z.device);loss=ce+(0.5*kd if teacher else 0)+(10*sop if group in ('F1','M2') else 0)
     if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
     before=scaler.get_scale();scaler.scale(loss).backward();scaler.step(opt);scaler.update();skipped=scaler.get_scale()<before;skips+=int(skipped);successful+=int(not skipped)
     bs=len(y);n+=bs;nb+=1;total+=float(loss.item())*bs;ce_sum+=float(ce.item())*bs;kd_sum+=float(kd.item())*bs;sop_sum+=float(sop.item())*bs;proxy_sum+=float(c.item())*bs;violate+=int(c.item()/C0>.8);correct+=int((z.argmax(1)==y).sum().item());proxy.clear();clean(model)
     if batch%40==0:
      dump(OUT/'heartbeat.json',{'utc':now(),'group':group,'epoch':epoch,'batch':batch,'gpu_elapsed_seconds':time.monotonic()-start+pre['wall_seconds'],'cpu_seconds':cpusec()})
      print(json.dumps({'group':group,'epoch':epoch,'batch':batch,'loss':float(loss.item()),'wall_s':time.monotonic()-start,'cpu_s':cpusec()}),flush=True)
    if n!=45000:raise RuntimeError('unexpected sample count')
    if first_batches is None:first_batches=copy.deepcopy(fingerprints);dump(OUT/'first_batch_fingerprints.json',{'reference':first_batches})
    elif epoch==1 and fingerprints!=first_batches:raise RuntimeError('train ordering/augmentation mismatch')
    for name,p in model.named_parameters():
     if name in fixed and not torch.equal(p,fixed[name]):raise RuntimeError('threshold changed')
    if teacher and digest_state(teacher)!=teacher_initial:raise RuntimeError('teacher changed')
    sched.step();row={'group':group,'epoch':epoch,'samples':n,'batches':nb,'train_accuracy':correct/n,'train_CE':ce_sum/n,'train_KD':kd_sum/n if teacher else '', 'train_SOP_loss':sop_sum/n,'train_C_proxy':proxy_sum/n,'train_total_loss':total/n,'budget_violation_batch_fraction':violate/nb,'learning_rate':lr,'successful_steps_cumulative':successful,'amp_skips_cumulative':skips,'epoch_seconds':time.monotonic()-tic};append(OUT/'epoch_metrics.csv',[row]);append(run/'epoch_metrics.csv',[row])
    checkpoint=run/f'epoch-{epoch:03d}.pth';tmp=checkpoint.with_suffix('.pth.tmp');torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'group':group,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()},tmp);os.replace(tmp,checkpoint)
    dump(run/'status.json',{'status':'RUNNING','epoch':epoch,'checkpoint':str(checkpoint),'sha256':filehash(checkpoint),'successful_steps':successful,'skips':skips,'fingerprints':fingerprints,'utc':now()})
    print(json.dumps({'event':'epoch_complete',**row}),flush=True)
   proxy.remove();result,vals,layers=evaluate(model,group,val_ids);result.update({'epochs':5,'successful_steps':successful,'amp_skips':skips,'train_and_eval_seconds':time.monotonic()-group_start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()});summaries.append(result);per[group]=vals
   path=run/'epoch-005.pth';paths.append({'group':group,'path':str(path),'bytes':path.stat().st_size,'sha256':filehash(path)});dump(run/'status.json',{'status':'COMPLETED','epoch':5,'checkpoint':str(path),'sha256':filehash(path),'successful_steps':successful,'skips':skips,'fingerprints':fingerprints,'utc':now()})
  finally:
   proxy.remove();del model,teacher,opt,sched,scaler;torch.cuda.empty_cache()
 for x in summaries:x['sop_savings']=1-x['SOPs']/B0['SOPs']
 keys=['group','accuracy','loss','SOPs','sop_savings','FR_global','FR_author','parameter_count','eval_seconds','n','epochs','successful_steps','amp_skips','train_and_eval_seconds','peak_allocated_bytes','peak_reserved_bytes'];append(OUT/'summary.csv',[{k:r.get(k,'') for k in keys} for r in summaries]);dump(OUT/'checkpoint_manifest.json',paths)
 rng=np.random.default_rng(2027);ix=rng.integers(0,len(val_ids),size=(2000,len(val_ids)));comparisons={}
 for a,b in [('M2','F1'),('F2','F0'),('F1','F0'),('F0','P0'),('M2','B0')]:
  if [r['sample_id'] for r in per[a]]!=[r['sample_id'] for r in per[b]]:raise RuntimeError('pair alignment')
  comparisons[a+'_vs_'+b]={}
  for metric in ['correct','loss']:
   d=np.array([x[metric]-y[metric] for x,y in zip(per[a],per[b])]);comparisons[a+'_vs_'+b][metric]={'delta':float(d.mean()),'ci95':[float(v) for v in np.quantile(d[ix].mean(1),[.025,.975])]}
 dic={v['group']:v for v in summaries};b0,p0,f0,f1,m2=[dic[k] for k in ['B0','P0','F0','F1','M2']];rebound0=(f0['SOPs']-p0['SOPs'])/b0['SOPs'];rebound1=(f1['SOPs']-p0['SOPs'])/b0['SOPs'];match=abs(m2['sop_savings']-f1['sop_savings'])<=.02;h7=rebound0>=.02 and (f0['SOPs']-f1['SOPs'])/b0['SOPs']>=.02 and f0['accuracy']-f1['accuracy']<=.01;h8=match and m2['accuracy']-f1['accuracy']>=.005
 gates={'M2_savings_at_least_20pct':m2['sop_savings']>=.2,'M2_acc_drop_at_most_1pp':b0['accuracy']-m2['accuracy']<=.01,'M2_F1_budget_match':match,'M2_acc_gain_at_least_0.5pp':m2['accuracy']-f1['accuracy']>=.005,'parameter_count_unchanged':all(r['parameter_count']==b0['parameter_count'] for r in summaries),'preflight_pass':True}
 dump(OUT/'metrics.json',{'status':'COMPLETED','summary':summaries,'paired_bootstrap':comparisons,'H007_gate':h7,'H008_gate':h8,'rebound_F0':rebound0,'rebound_F1':rebound1,'screening_gate':gates,'screening_signal':all(gates.values()),'checkpoint_manifest':paths,'wall_seconds_including_preflight':time.monotonic()-start+pre['wall_seconds'],'cpu_seconds_including_preflight':cpusec()})
 dump(OUT/'status.json',{'execution_status':'COMPLETED','H-007':'UNTESTED','H-008':'UNTESTED','end_utc':now()})
if __name__=='__main__':
 try:run()
 except BaseException as e:
  print(traceback.format_exc(),flush=True);dump(OUT/'status.json',{'execution_status':'PARTIAL','error':repr(e),'traceback':traceback.format_exc(),'end_utc':now(),'wall_seconds_including_preflight':time.monotonic()-start+pre['wall_seconds'],'cpu_seconds_including_preflight':cpusec(),'H-007':'UNTESTED','H-008':'UNTESTED'});raise

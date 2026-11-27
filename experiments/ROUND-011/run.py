"""EXP-011: three controlled short fine-tunes, then calibration-locked evaluation."""
import os,sys,json,csv,time,random,resource,hashlib,subprocess,datetime,traceback,copy
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset,Subset
from torchvision.datasets import CIFAR10
from timm.optim import create_optimizer_v2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/EXP-004/controlled'));sys.path.insert(0,str(ROOT/'experiments/ROUND-011'))
from core import model_create,ATLIF,DATA,transforms_pair,seed_all,threshold_update
from artg import configure,DIAGNOSTICS
from utils.utils import SOPMonitor
from spikingjelly.activation_based import functional
O=ROOT/'experiments/EXP-011';CONF=json.loads((O/'config.json').read_text());M=ROOT/'experiments/EXP-004/runs/EXP-004-RUN-003/checkpoint_best_validation.pth';S=ROOT/'experiments/EXP-004/splits/indices.json';I=ROOT/'experiments/EXP-007/calibration_indices.json';MAP=json.loads((ROOT/'experiments/EXP-008/runs/PREFLIGHT-001/sop_mapping.json').read_text());ck=None;start=time.monotonic();usage0=resource.getrusage(resource.RUSAGE_SELF)
prepaths=[O/'preflight-attempt-001/status.json',O/'preflight-attempt-002/status.json',O/'preflight-attempt-003/status.json',O/'preflight/status.json',O/'preflight/amp_success.json',O/'preflight-attempt-002/gradient_diagnosis.json'];pre=[json.loads(p.read_text()) for p in prepaths];prior_wall=sum(x.get('wall_seconds',0) for x in pre);prior_cpu=sum(x.get('cpu_seconds',0) for x in pre);code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(q,p)
def append(p,rows):
 if not rows:return
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');
  if f.tell()==0:w.writeheader()
  w.writerows(rows);f.flush();os.fsync(f.fileno())
def cputime():
 r=resource.getrusage(resource.RUSAGE_SELF);c=resource.getrusage(resource.RUSAGE_CHILDREN);return r.ru_utime+r.ru_stime-usage0.ru_utime-usage0.ru_stime+c.ru_utime+c.ru_stime+prior_cpu
def guard(reserve=0):
 if time.monotonic()-start+prior_wall>3450-reserve:raise TimeoutError('GPU time budget guard')
 if cputime()>5200-reserve:raise TimeoutError('CPU budget guard')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def hashes():
 for path,key in [(M,'checkpoint_sha256'),(S,'split_sha256'),(I,'calibration_sha256')]:
  if sha(path)!=CONF[key]:raise RuntimeError('locked checksum mismatch')
class Indexed(Dataset):
 def __init__(self,ds,ids):self.ds=ds;self.ids=ids
 def __len__(self):return len(self.ids)
 def __getitem__(self,i):
  x,y=self.ds[self.ids[i]];return x,y,self.ids[i]
def loader(ids,training):
 ds=CIFAR10(DATA,train=True,download=False,transform=transforms_pair()[0 if training else 1]);return DataLoader(Indexed(ds,ids),batch_size=128,shuffle=False,num_workers=4,pin_memory=True)
def clean(m):
 functional.reset_net(m)
 for z in m.modules():
  if isinstance(z,ATLIF):z.act_value=0.;z.update_value=0.
def newmodel(group,source,alpha):
 m=configure(model_create().cuda(),group if group in ('G0','G1','M4') else 'G0');m.load_state_dict(source['model'])
 for z in m.modules():
  if isinstance(z,ATLIF):z.thresh.data.mul_(alpha);z.sp=.002;z.update_value=0.
 return m
def digest(m):
 h=hashlib.sha256()
 for n,p in m.state_dict().items():h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def validate_thresholds(m):
 for z in m.modules():
  if isinstance(z,ATLIF) and (not torch.isfinite(z.thresh).all() or z.thresh.item()<=0):raise RuntimeError('nonfinite or nonpositive threshold')
def safe_number(x):return x if np.isfinite(x) else str(x)
def train_group(group,ids,baseline_init,first_batches):
 guard(reserve=180);seed_all(2030);torch.cuda.reset_peak_memory_stats();group_start=time.monotonic();run=O/'runs'/('EXP-011-'+group+'-RUN-001');run.mkdir(parents=True,exist_ok=False);m=newmodel(group,ck,1.25);m.train();mods={n:z for n,z in m.named_modules() if isinstance(z,ATLIF)};initial=digest(m)
 if baseline_init and initial!=baseline_init:raise RuntimeError('initial state mismatch')
 opt=create_optimizer_v2(m,opt='sgd',lr=.005,weight_decay=5e-4,momentum=.9,nesterov=True);groups=[{'weight_decay':p['weight_decay'],'count':len(p['params'])} for p in opt.param_groups];ids_opt={id(p) for g in opt.param_groups for p in g['params']};assert all(id(z.thresh) in ids_opt for z in mods.values());sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=5,eta_min=.0001);scaler=torch.cuda.amp.GradScaler(enabled=True);success=skips=0;fp=[]
 dump(run/'metadata.json',{'group':group,'initial_sha256':initial,'code_sha':code,'started_utc':now(),'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'optimizer_groups':groups,'diagnostic_batch_index':8,'amp_manual_update_policy':'original controlled baseline, including skipped AMP optimizer step'})
 for epoch in range(1,6):
  counts=right=nb=0;loss_sum=0.;tic=time.monotonic();lr=opt.param_groups[0]['lr'];data=loader(ids,True)
  for batch,(x,y,sid) in enumerate(data):
   guard(reserve=180)
   if epoch==1 and batch<3:fp.append({'ids':sid.tolist(),'images_sha256':hashlib.sha256(x.numpy().tobytes()).hexdigest()})
   x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True);diagnostic=(batch==8);mon=SOPMonitor(m) if diagnostic else None
   if diagnostic and set(mon.monitored_layers)!=set(MAP):raise RuntimeError('SOP mapping mismatch')
   if diagnostic:DIAGNOSTICS.update(enabled=(group=='M4'),scale=scaler.get_scale(),rows=[])
   with torch.cuda.amp.autocast():z=m(x).mean(0).float();loss=F.cross_entropy(z,y)
   if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
   old=scaler.get_scale();scaler.scale(loss).backward();DIAGNOSTICS['enabled']=False;scaler.unscale_(opt)
   if diagnostic:
    sop={n:0. for n in mods}
    for conv,indices in mon.name_records_index.items():sop[MAP[conv]]+=sum(float(mon.records[i].item()) for i in indices)/len(y)
    mon.remove_hooks();before={n:z.thresh.detach().clone() for n,z in mods.items()};manual={n:z.update_value for n,z in mods.items()};grad={n:z.thresh.grad.detach().clone() if z.thresh.grad is not None else None for n,z in mods.items()};rate={n:z.r for n,z in mods.items()};diag=list(DIAGNOSTICS['rows'])
   scaler.step(opt)
   if diagnostic:afteropt={n:z.thresh.detach().clone() for n,z in mods.items()}
   threshold_update(m,lr)
   if diagnostic:aftermanual={n:z.thresh.detach().clone() for n,z in mods.items()}
   scaler.update();skipped=scaler.get_scale()<old;skips+=int(skipped);success+=int(not skipped);validate_thresholds(m)
   if diagnostic:
    rows=[];up=[]
    for n,z0 in mods.items():
     v=grad[n];v=float(v.item()) if v is not None else float('nan');components=[d for d in diag if d['module']==n];row={'group':group,'epoch':epoch,'batch':batch,'module':n,'threshold_before':before[n].item(),'FR':rate[n],'SOP_contribution_per_sample':sop[n],'task_threshold_gradient':safe_number(v),'gradient_zero':v==0,'gradient_nonfinite_count':int(not np.isfinite(v)),'AMP_skipped':skipped}
     for k in ['direct_mean','state_mean','reset_mean','direct_abs_mean','state_abs_mean','reset_abs_mean']:row[k]=safe_number(sum(d[k] for d in components)) if group=='M4' else ''
     rows.append(row);up.append({'group':group,'epoch':epoch,'batch':batch,'module':n,'threshold_before':before[n].item(),'task_gradient':safe_number(v),'manual_update_value':manual[n],'optimizer_delta':(afteropt[n]-before[n]).item(),'manual_delta':(aftermanual[n]-afteropt[n]).item(),'threshold_after':aftermanual[n].item(),'AMP_skipped':skipped,'learning_rate':lr})
    append(O/'threshold_gradient_statistics.csv',rows);append(O/'threshold_updates.csv',up)
   n=len(y);counts+=n;nb+=1;right+=int((z.argmax(1)==y).sum().item());loss_sum+=loss.item()*n;clean(m)
   if batch%40==0:
    heartbeat={'utc':now(),'group':group,'epoch':epoch,'batch':batch,'wall_seconds_including_preflight':time.monotonic()-start+prior_wall,'cpu_seconds_including_preflight':cputime()};dump(O/'heartbeat.json',heartbeat);print(json.dumps(heartbeat),flush=True)
  if counts!=45000:raise RuntimeError('unexpected training samples')
  if first_batches is None:first_batches=copy.deepcopy(fp);dump(O/'first_batch_fingerprints.json',{'reference':fp})
  elif epoch==1 and fp!=first_batches:raise RuntimeError('training sample or augmentation mismatch')
  sched.step();row={'group':group,'epoch':epoch,'samples':counts,'batches':nb,'train_accuracy':right/counts,'train_loss':loss_sum/counts,'learning_rate':lr,'successful_steps_cumulative':success,'amp_skips_cumulative':skips,'epoch_seconds':time.monotonic()-tic};append(O/'epoch_metrics.csv',[row]);append(run/'epoch_metrics.csv',[row]);cp=run/f'epoch-{epoch:03d}.pth';tmp=cp.with_suffix('.pth.tmp');torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'group':group,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'code_sha':code},tmp);os.replace(tmp,cp);dump(run/'status.json',{'status':'RUNNING','epoch':epoch,'checkpoint':str(cp),'checkpoint_sha256':sha(cp),'amp_skips':skips,'successful_steps':success,'fingerprints':fp,'utc':now()});print(json.dumps({'event':'epoch_complete',**row}),flush=True)
 cp=run/'epoch-005.pth';out={'group':group,'path':str(cp),'bytes':cp.stat().st_size,'sha256':sha(cp),'completed_epochs':5,'successful_steps':success,'amp_skips':skips,'wall_seconds':time.monotonic()-group_start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'initial_sha256':initial};dump(run/'status.json',{'status':'COMPLETED','epoch':5,'checkpoint':str(cp),'checkpoint_sha256':out['sha256'],'amp_skips':skips,'successful_steps':success,'fingerprints':fp,'utc':now()});del m,opt,scaler,sched;torch.cuda.empty_cache();return out,initial,first_batches

def evaluate(group,source,alpha,ids,training_data,tag,detail=False):
 guard();m=newmodel(group,source,alpha);m.eval();mods={n:z for n,z in m.named_modules() if isinstance(z,ATLIF)};mon=SOPMonitor(m);spikes={};states={};rates={};bn={};hooks=[]
 for n,z in mods.items():
  def f(mod,args,out,n=n):
   v=(out.detach()>0).sum(dtype=torch.int64);spikes[n]=spikes.get(n,0)+v;states[n]=states.get(n,0)+out.numel();rates[n]=rates.get(n,0)+v.double()/out.numel();bn[n]=bn.get(n,0)+1
  hooks.append(z.register_forward_hook(f))
 samples=[];batches=[];count=0;sop=0.;tic=time.monotonic()
 try:
  with torch.no_grad():
   for j,(x,y,sid) in enumerate(training_data):
    guard();x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);mon.clear_recorded_data();z=m(x).mean(0).float();loss=F.cross_entropy(z,y,reduction='none');pred=z.argmax(1);n=len(y);sp=sum(float(x.item()) for x in mon.records);sop+=sp
    if not torch.isfinite(loss).all():raise RuntimeError('nonfinite evaluation')
    for i,l,p,t in zip(sid.tolist(),loss.cpu().tolist(),pred.cpu().tolist(),y.cpu().tolist()):samples.append({'tag':tag,'sample_id':i,'loss':l,'prediction':p,'correct':int(p==t)})
    batches.append({'tag':tag,'batch':j,'samples':n,'sop_count':sp});count+=n;clean(m)
 finally:
  mon.remove_hooks()
  for h in hooks:h.remove()
 assert count==len(ids)
 layer={n:{'FR':int(spikes[n].item())/states[n],'spikes':int(spikes[n].item()),'states':states[n],'author_FR':float((rates[n]/bn[n]).item()),'threshold':v.thresh.item()} for n,v in mods.items()};out={'tag':tag,'accuracy':float(np.mean([r['correct'] for r in samples])),'loss':float(np.mean([r['loss'] for r in samples])),'SOPs':sop/count,'FR_global':sum(x['spikes'] for x in layer.values())/sum(x['states'] for x in layer.values()),'FR_author':float(np.mean([x['author_FR'] for x in layer.values()])),'parameter_count':sum(x.numel() for x in m.parameters()),'samples':count,'seconds':time.monotonic()-tic}
 if detail:
  append(O/'validation_per_sample.csv',samples);append(O/'validation_batch_sops.csv',batches);append(O/'layer_metrics.csv',[{'tag':tag,'module':n,**v} for n,v in layer.items()])
 del m;return out,samples

def run():
 global ck
 if pre[3]['status']!='PASSED' or pre[4]['status']!='PASSED':raise RuntimeError('required preflight failure')
 if torch.cuda.device_count()!=1 or '4090' not in torch.cuda.get_device_name(0):raise RuntimeError('not exactly one RTX4090')
 torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;hashes();ck=torch.load(M,map_location='cpu',weights_only=False);split=json.loads(S.read_text());calids=json.loads(I.read_text())['indices'];assert len(split['train'])==45000 and len(split['validation'])==4500 and len(calids)==500 and set(calids)<=set(split['train']) and not set(calids)&set(split['validation']) and not set(calids)&set(split['diagnostic']);manifest=[];init=None;fp=None
 for group in CONF['groups']:
  r,init,fp=train_group(group,split['train'],init,fp);manifest.append(r);dump(O/'checkpoint_manifest.json',manifest)
 sources={r['group']:torch.load(r['path'],map_location='cpu',weights_only=False) for r in manifest};sources['C0']=ck
 # Perform candidate calibration exclusively on fixed training-origin images.
 cal=list(loader(calids,False));baseline,_=evaluate('C0',ck,1.,calids,cal,'C0-cal');C0=baseline['SOPs'];dump(O/'calibration_baseline.json',{'C0_cal':C0,'reference_previous':54945117.504,'difference':C0-54945117.504});cache={}
 def candidate(group,alpha):
  key=(group,round(alpha,10))
  if key not in cache:
   guard(reserve=120);r,_=evaluate(group,sources[group],alpha,calids,cal,group+'-cal');row={'group':group,'alpha':alpha,'accuracy':r['accuracy'],'loss':r['loss'],'SOPs':r['SOPs'],'FR_global':r['FR_global'],'FR_author':r['FR_author'],'savings_vs_C0_cal':1-r['SOPs']/C0,'seconds':r['seconds']};cache[key]=row;append(O/'candidate_history.csv',[row]);print(json.dumps({'event':'candidate',**row}),flush=True)
  return cache[key]
 selections={}
 for group in ['C0','G0','G1','M4']:
  for a in CONF['alphas']:candidate(group,a)
  for target in CONF['targets']:
   for _ in range(3):
    rr=[v for (g,a),v in cache.items() if g==group];hits=[r for r in rr if target<=r['savings_vs_C0_cal']<=target+.02]
    if hits:break
    ordered=sorted(rr,key=lambda r:r['alpha']);cross=[]
    for a,b in zip(ordered,ordered[1:]):
     if (a['savings_vs_C0_cal']-target)*(b['savings_vs_C0_cal']-target)<=0:cross.append((abs(a['savings_vs_C0_cal']-target)+abs(b['savings_vs_C0_cal']-target),a['alpha'],b['alpha']))
    if not cross:break
    _,a,b=min(cross);candidate(group,(a+b)/2)
  allrows=[v for (g,a),v in cache.items() if g==group];selections[group]={}
  for target in CONF['targets']:
   hits=[r for r in allrows if target<=r['savings_vs_C0_cal']<=target+.02];best=min(hits or allrows,key=lambda r:(abs(r['savings_vs_C0_cal']-target),r['alpha']));selections[group][str(target)]={'alpha':best['alpha'],'calibration_savings':best['savings_vs_C0_cal'],'matched_window':bool(hits)}
 dump(O/'selected_configs.json',selections);dump(O/'selection_lock.json',{'locked_utc':now(),'code_sha':code,'selection_inputs':'only 500 calibration samples and SOPs','validation_touched_before_lock':False})
 val=loader(split['validation'],False);summ=[];per={}
 def valrun(group,alpha,tag):
  r,s=evaluate(group,sources[group],alpha,split['validation'],val,tag,True);r.update(group=group,alpha=alpha);summ.append(r);per[tag]=s;print(json.dumps({'event':'validation',**r}),flush=True)
 valrun('C0',1.,'B0')
 for g in CONF['groups']:valrun(g,1.,g+'-direct')
 for g in ['C0','G0','G1','M4']:
  for t0 in ['0.2','0.3']:valrun(g,selections[g][t0]['alpha'],g+'-'+t0)
 base=summ[0]['SOPs']
 for row in summ:row['savings_vs_B0']=1-row['SOPs']/base
 append(O/'summary.csv',summ);by={x['tag']:x for x in summ};rng=np.random.default_rng(2027);ix=rng.integers(0,len(split['validation']),(2000,len(split['validation'])));comparisons={}
 for t0 in ['0.2','0.3']:
  for a,b in [('M4','G0'),('M4','G1'),('M4','C0'),('G1','G0')]:
   A=a+'-'+t0;B=b+'-'+t0;assert [r['sample_id'] for r in per[A]]==[r['sample_id'] for r in per[B]];out={}
   for metric in ['correct','loss']:
    d=np.array([x[metric]-y[metric] for x,y in zip(per[A],per[B])]);out[metric]={'delta':float(d.mean()),'ci95':[float(x) for x in np.quantile(d[ix].mean(1),[.025,.975])]}
   out['budget_difference']=abs(by[A]['savings_vs_B0']-by[B]['savings_vs_B0']);out['matched_budget']=out['budget_difference']<=.02;comparisons[A+'_vs_'+B]=out
 M4=by['M4-0.2'];B0=by['B0'];opponents=[by[x+'-0.2'] for x in ['G0','G1','C0']];gate={'savings_at_least_20pct':M4['savings_vs_B0']>=.2,'accuracy_loss_at_most_1pp':B0['accuracy']-M4['accuracy']<=.01,'all_budget_matched':all(abs(M4['savings_vs_B0']-x['savings_vs_B0'])<=.02 for x in opponents),'accuracy_gain_at_least_half_pp_vs_all':all(M4['accuracy']-x['accuracy']>=.005 for x in opponents),'same_parameter_count':all(M4['parameter_count']==x['parameter_count'] for x in [B0]+opponents),'preflight_pass':True};hashes();res={'status':'COMPLETED','code_sha':code,'calibration_baseline_SOPs':C0,'summary':summ,'comparisons':comparisons,'H011_conditions':gate,'H011_screening_signal':all(gate.values()),'checkpoint_manifest':manifest,'wall_seconds_including_preflight':time.monotonic()-start+prior_wall,'cpu_seconds_including_preflight':cputime()};dump(O/'metrics.json',res);return res
if __name__=='__main__':
 status={'execution_status':'RUNNING','code_sha':code,'start_utc':now(),'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),'H-011':'UNTESTED'}
 try:r=run();status.update(execution_status=r['status'])
 except BaseException as e:status.update(execution_status='PARTIAL',error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
 finally:status.update(end_utc=now(),wall_seconds_including_preflight=time.monotonic()-start+prior_wall,cpu_seconds_including_preflight=cputime());dump(O/'status.json',status);print(json.dumps(status,indent=2),flush=True)
 if status['execution_status']!='COMPLETED':sys.exit(1)

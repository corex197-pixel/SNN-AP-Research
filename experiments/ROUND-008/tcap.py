"""EXP-007 preregistered calibration-only threshold search; no training/test set."""
import os,sys,csv,json,time,datetime,hashlib,traceback,subprocess,math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset
from torchvision.datasets import CIFAR10
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/EXP-004/controlled'))
from core import model_create,isolated_eval,transforms_pair,DATA,ATLIF
from utils.utils import SOPMonitor
from spikingjelly.activation_based import functional
MODEL=ROOT/'experiments/EXP-004/runs/EXP-004-RUN-003/checkpoint_best_validation.pth';SPLIT=ROOT/'experiments/EXP-004/splits/indices.json';IDS=ROOT/'experiments/EXP-007/calibration_indices.json';OUT=ROOT/'experiments/EXP-007/runs/EXP-007-RUN-002'
MSHA='c068f54379cb5d29bcfa5fe2d9de676cd3cc153433ada1ed5aa1c66355d0b32a';SSHA='69dccfe992236766aea575a0a66495802ca889ca329d61f51e8de3c1ab7f3165';ISHA='b849b74ef19cb4a09df756c54871122f3b02aa047c704d0436feab15a38bde52'
levels=[1.,1.1,1.25,1.5,2.];uniforms=[1.,1.05,1.1,1.15,1.2,1.25,1.3,1.4,1.5,1.75,2.];start=time.monotonic();cpu=time.process_time();beg=datetime.datetime.now(datetime.timezone.utc).isoformat();counter=0;cache={};physical=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def guard():
 if time.monotonic()-start>1610:raise TimeoutError('GPU wall budget safety limit')
 if time.process_time()-cpu>1610:raise TimeoutError('CPU time budget safety limit')
def writejson(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def appendcsv(path,rows):
 if not rows:return
 with path.open('a',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
  if f.tell()==0:w.writeheader()
  w.writerows(rows);f.flush();os.fsync(f.fileno())
def writecsv(path,rows):
 if not rows:return
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def flat(x):return [float(v) for v in x]

def main():
 global counter
 if torch.cuda.device_count()!=1:raise RuntimeError('exactly one GPU must be visible')
 torch.cuda.set_device(0);torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 if sha(MODEL)!=MSHA or sha(SPLIT)!=SSHA or sha(IDS)!=ISHA:raise RuntimeError('locked file checksum mismatch')
 ck=torch.load(MODEL,map_location='cpu',weights_only=False)
 if ck['epoch']!=193 or ck['split_sha256']!=SSHA or ck['code_sha']!='03def08d9fab5c37968a2a2eaa37675666cb4110':raise RuntimeError('checkpoint provenance mismatch')
 model=model_create().cuda().eval();model.load_state_dict(ck['model']);
 for p in model.parameters():p.requires_grad_(False)
 mods=[(n,m) for n,m in model.named_modules() if isinstance(m,ATLIF)];names=[n for n,m in mods];base=[float(m.thresh.item()) for n,m in mods]
 if len(mods)!=19 or any(not math.isfinite(v) or v<=0 for v in base) or any(not torch.isfinite(p).all() for p in model.parameters()):raise RuntimeError('invalid model/thresholds')
 spl=json.loads(SPLIT.read_text());ids=json.loads(IDS.read_text())['indices']
 if len(ids)!=len(set(ids)) or len(ids)!=500 or not set(ids)<=set(spl['train']) or set(ids)&set(spl['validation']) or set(ids)&set(spl['diagnostic']):raise RuntimeError('calibration sample error')
 ds=CIFAR10(DATA,train=True,transform=transforms_pair()[1],download=False);assert ds._check_integrity()
 if any(sum(ds.targets[i]==c for i in ids)!=50 for c in range(10)):raise RuntimeError('calibration stratification mismatch')
 cal=DataLoader(Subset(ds,ids),batch_size=32,shuffle=False,num_workers=0)
 OUT.mkdir(parents=True,exist_ok=False);torch.cuda.reset_peak_memory_stats()
 meta={'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'start_utc':beg,'checkpoint_sha256':MSHA,'split_sha256':SSHA,'calibration_indices_sha256':ISHA,'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpu':torch.cuda.get_device_name(0),'calibration_indices':ids,'modules_execution_order':names,'batch_size':32,'bootstrap_seed':2027,'bootstrap_replicates':2000,'original_thresholds':base,'sop_definition':'original SOPMonitor, full network convs excluding skip/conv1, positive inputs; ones weights, includes spatial geometry and residual convolutions, no non-conv ops or effective weight sparsity; per actual sample; theoretical SOPs not energy'};writejson(OUT/'run_metadata.json',meta)
 # Verify actual invocation order without affecting model state.
 order=[];hs=[m.register_forward_hook(lambda m,a,o,n=n:order.append(n)) for n,m in mods]
 first_x,first_y=next(iter(cal));first_x=first_x.cuda()
 try:
  with isolated_eval(model):plain=model(first_x).detach().clone()
 finally:
  for h in hs:h.remove()
 if order!=names or len(set(order))!=19:raise RuntimeError('module order mismatch or repeated call')
 snapshot={k:v.detach().clone() for k,v in model.state_dict().items()};attrs=[(m.update_value,m.r,m.s,m.act_value) for _,m in mods];rng=torch.get_rng_state().clone();grng=torch.cuda.get_rng_state().clone();modes=[m.training for m in model.modules()]
 # Reuse the original counter twice; reset all records per batch explicitly.
 mon=SOPMonitor(model);sop=[];output=[]
 try:
  for j in range(2):
   with isolated_eval(model):
    mon.clear_recorded_data();z=model(first_x).detach().clone();output.append(z);sop.append(sum(float(t.item()) for t in mon.records));mon.clear_recorded_data();functional.reset_net(model)
 finally:mon.remove_hooks()
 if not torch.allclose(plain,output[0],atol=1e-5,rtol=0) or not torch.equal(output[0],output[1]) or sop[0]!=sop[1]:raise RuntimeError('baseline or SOP repeat control failed')
 # Explicit candidate threshold restoration using state isolation.
 with isolated_eval(model):
  mods[0][1].thresh.mul_(1.1);model(first_x)
 state_ok=all(torch.equal(snapshot[k],v) for k,v in model.state_dict().items()) and attrs==[(m.update_value,m.r,m.s,m.act_value) for _,m in mods] and torch.equal(rng,torch.get_rng_state()) and torch.equal(grng,torch.cuda.get_rng_state()) and modes==[m.training for m in model.modules()]
 if not state_ok:raise RuntimeError('candidate or eval state restoration failed')
 writejson(OUT/'smoke.json',{'actual_order':order,'alpha1_max_logit_difference':float((plain-output[0]).abs().max().item()),'SOP_repeat_absolute_difference':abs(sop[0]-sop[1]),'SOP_original_counter_reused_and_cleared':True,'candidate_state_params_bn_attrs_rng_restored':state_ok,'test_set_used':False})
 def evaluate(alpha,loader,ids_,phase,tag,record_samples=False):
  global counter
  guard();counter+=1;tic=time.monotonic();values=[];batch_rows=[];sop_total=0.;count=0;spikes={};states={};rates={};batch_count={};hooks=[]
  def makehook(n):
   def f(m,args,out):
    c=(out.detach()>0).sum(dtype=torch.int64);total=out.numel();spikes[n]=spikes.get(n,0)+c;states[n]=states.get(n,0)+total;rates[n]=rates.get(n,0)+c.double()/total;batch_count[n]=batch_count.get(n,0)+1
   return f
  for name,mod in mods:hooks.append(mod.register_forward_hook(makehook(name)))
  monitor=SOPMonitor(model)
  try:
   with isolated_eval(model):
    for (_,mod),ref,mul in zip(mods,base,alpha):mod.thresh.fill_(ref*mul)
    for batch,(x,y) in enumerate(loader):
     guard();n=len(y);idx=ids_[count:count+n];x=x.cuda();y=y.cuda();monitor.clear_recorded_data();logits=model(x).mean(0);loss=F.cross_entropy(logits,y,reduction='none');pred=logits.argmax(1)
     if not(torch.isfinite(logits).all() and torch.isfinite(loss).all()):raise RuntimeError('nonfinite logits/loss')
     so=float(torch.stack(monitor.records).double().sum().item()) if monitor.records else 0.;sop_total+=so;batch_rows.append({'method':tag,'phase':phase,'batch_index':batch,'sample_count':n,'first_sample_id':int(idx[0]),'last_sample_id':int(idx[-1]),'sop_count':so})
     losses=loss.cpu().tolist();preds=pred.cpu().tolist();targets=y.cpu().tolist()
     for i,sid in enumerate(idx):values.append({'method':tag,'sample_id':int(sid),'loss':float(losses[i]),'prediction':int(preds[i]),'correct':int(preds[i]==targets[i])})
     count+=n;functional.reset_net(model)
  finally:
   monitor.remove_hooks()
   for h in hooks:h.remove()
  if count!=len(ids_):raise RuntimeError('sample count mismatch')
  layer={n:{'FR':float(spikes[n].item()/states[n]),'spike_count':int(spikes[n].item()),'neuron_time_states':states[n],'author_batch_mean_fr':float((rates[n]/batch_count[n]).item())} for n in names}
  summary={'loss':float(np.mean([v['loss'] for v in values])),'accuracy':float(np.mean([v['correct'] for v in values])),'SOPs':sop_total/count,'FR_global':sum(v['spike_count'] for v in layer.values())/sum(v['neuron_time_states'] for v in layer.values()),'FR_author':float(np.mean([v['author_batch_mean_fr'] for v in layer.values()])),'samples':count,'batches':len(batch_rows),'elapsed_seconds':time.monotonic()-tic,'eval_id':counter}
  if phase=='calibration':
   event={'method':tag,'eval_id':counter,'samples':count,'batches':len(batch_rows),'elapsed_seconds':summary['elapsed_seconds']};physical.append(event);appendcsv(OUT/'physical_evaluations.csv',[event])
  if record_samples:appendcsv(OUT/'validation_per_sample.csv',values);appendcsv(OUT/'validation_batch_sops.csv',batch_rows)
  return summary,layer,values
 def calc(alpha,method):
  key=tuple(alpha)
  if key in cache and (method in ('B1','shared_initial') or (method in ('B2','M1') and sum(v!=1. for v in alpha)==1 and max(alpha)==1.1)):return cache[key],True
  summary,layer,_=evaluate(alpha,cal,ids,'calibration',method);cache[key]=(summary,layer);return cache[key],False
 def remaining():return 1710-(time.monotonic()-start)
 alpha0=[1.]*19;(basecal,baselayers),_=calc(alpha0,'shared_baseline');C0=basecal['SOPs'];writejson(OUT/'calibration_baseline.json',basecal)
 def hist(method,iteration,module,old,new,cand,current,cached,valid,dc,score,reason):
  appendcsv(OUT/'candidate_history.csv',[{'method':method,'iteration':iteration,'module':module,'from_alpha':old,'to_alpha':new,'eval_id':cand['eval_id'],'cached':cached,'calibration_loss':cand['loss'],'calibration_accuracy':cand['accuracy'],'calibration_SOPs':cand['SOPs'],'delta_loss':cand['loss']-current['loss'],'delta_c':dc,'score':score if valid else '', 'valid':valid,'reason':reason,'total_savings':1-cand['SOPs']/C0}])
 def action(method,step,name,old,new,cur,dc,score):
  appendcsv(OUT/'accepted_actions.csv',[{'method':method,'step':step,'module':name,'from_alpha':old,'to_alpha':new,'eval_id':cur['eval_id'],'calibration_loss':cur['loss'],'calibration_SOPs':cur['SOPs'],'total_savings':1-cur['SOPs']/C0,'delta_c':dc,'score':score}])
 uniform=[]
 for v in uniforms:
  if remaining()<300:raise TimeoutError('validation reserve reached during B1 calibration')
  (res,_),cached=calc([v]*19,'B1');dc=(C0-res['SOPs'])/C0;hist('B1',0,'ALL',1.,v,res,basecal,cached,True,dc,0.,'fixed_uniform_grid');uniform.append((v,res))
 choices=[(v,res) for v,res in uniform if 1-res['SOPs']/C0>=.10];u=min(choices,key=lambda t:1-t[1]['SOPs']/C0-.10) if choices else min(uniform,key=lambda t:t[1]['SOPs']);selected={'B0':{'alpha':alpha0,'calibration':basecal},'B1':{'alpha':[u[0]]*19,'calibration':u[1],'target_met':bool(choices)}}
 # Shared initial 19 single-step evaluations and static ordering.
 initial=[]
 for i in range(19):
  if remaining()<300:raise TimeoutError('validation reserve reached during initial ranking')
  vec=alpha0.copy();vec[i]=1.1;(res,_),cached=calc(vec,'shared_initial');dc=(C0-res['SOPs'])/C0;valid=dc>=.005;score=(res['loss']-basecal['loss'])/dc if valid else None
  hist('shared_initial',0,names[i],1.,1.1,res,basecal,cached,valid,dc,score,'eligible' if valid else 'below_delta_cost');initial.append((i,score,dc,valid))
 ranking=[i for i,score,dc,valid in sorted((z for z in initial if z[3]),key=lambda z:(z[1],-z[2],z[0]))]+[i for i,score,dc,valid in initial if not valid];writejson(OUT/'static_order.json',{'module_indices':ranking,'module_names':[names[i] for i in ranking],'invalid_initial':[names[i] for i,score,dc,valid in initial if not valid]})
 def search(method):
  vector=alpha0.copy();current=basecal;accepted=0;stop_reason='maximum_accepted_actions';indices=ranking if method=='B2' else list(range(19));round_index=0;cursor=0
  while accepted<12:
   if remaining()<300:stop_reason='budget_validation_reserve';break
   candidates=[];round_index+=1
   for i in indices:
    pos=levels.index(vector[i])
    if pos==len(levels)-1:continue
    if remaining()<300:stop_reason='budget_validation_reserve';break
    nxt=levels[pos+1];v=vector.copy();v[i]=nxt;(res,_),cached=calc(v,method);dc=(current['SOPs']-res['SOPs'])/C0;valid=dc>=.005;score=(res['loss']-current['loss'])/dc if valid else None
    hist(method,round_index,names[i],vector[i],nxt,res,current,cached,valid,dc,score,'eligible' if valid else 'below_delta_cost')
    if valid:candidates.append({'i':i,'vector':v,'result':res,'dc':dc,'score':score,'savings':1-res['SOPs']/C0})
   if stop_reason=='budget_validation_reserve':break
   if not candidates:stop_reason='no_valid_candidate';break
   window=[c for c in candidates if .10<=c['savings']<=.12];pool=window if window else candidates
   chosen=min(pool,key=(lambda z:(z['score'],-z['dc'],z['i'])) if method=='M1' else (lambda z:(ranking.index(z['i'])-cursor)%len(ranking)))
   i=chosen['i'];old=vector[i];vector=chosen['vector'];current=chosen['result'];accepted+=1;cursor=(ranking.index(i)+1)%len(ranking) if method=='B2' else cursor;action(method,accepted,names[i],old,vector[i],current,chosen['dc'],chosen['score']);writejson(OUT/(method+'_progress.json'),{'alpha':vector,'accepted':accepted,'calibration':current})
   print(json.dumps({'event':'accepted','method':method,'step':accepted,'module':names[i],'savings':chosen['savings'],'remaining_seconds':remaining()}),flush=True)
   if chosen['savings']>=.10:stop_reason='target_reached';break
  return {'alpha':vector,'calibration':current,'accepted_actions':accepted,'stop_reason':stop_reason,'target_met':1-current['SOPs']/C0>=.10}
 selected['B2']=search('B2');selected['M1']=search('M1')
 # Lock before constructing validation subset or looking at any validation sample.
 for k,v in selected.items():v['thresholds']={n:b*a for n,b,a in zip(names,base,v['alpha'])};v['calibration_savings']=1-v['calibration']['SOPs']/C0
 writejson(OUT/'selected_thresholds.json',selected);writejson(OUT/'selection_lock.json',{'locked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selection_inputs':'calibration only','code_sha':meta['code_sha'],'validation_touched_before_lock':False})
 valids=spl['validation'];val=DataLoader(Subset(ds,valids),batch_size=32,shuffle=False,num_workers=0);validations={};pergroup={}
 for tag in ['B0','B1','B2','M1']:
  summary,layer,values=evaluate(selected[tag]['alpha'],val,valids,'validation',tag,True);validations[tag]={'summary':summary,'layers':layer};pergroup[tag]=values;print(json.dumps({'event':'validation','method':tag,'accuracy':summary['accuracy'],'SOPs':summary['SOPs']}),flush=True)
 writejson(OUT/'validation_metrics.json',validations)
 # Keep layer, batch and sample-weighted definitions separate.
 writecsv(OUT/'layer_metrics.csv',[{'method':tag,'module':name,'alpha':selected[tag]['alpha'][i],'threshold':selected[tag]['thresholds'][name],**validations[tag]['layers'][name]} for tag in validations for i,name in enumerate(names)])
 rng=np.random.default_rng(2027);index=rng.integers(0,len(valids),(2000,len(valids)));comparisons={}
 def pair(A,B):
  if [v['sample_id'] for v in pergroup[A]]!=[v['sample_id'] for v in pergroup[B]]:raise RuntimeError('validation pairing mismatch')
  result={}
  for metric in ['loss','correct']:
   d=np.array([a[metric]-b[metric] for a,b in zip(pergroup[A],pergroup[B])]);ci=np.quantile(d[index].mean(1),[.025,.975]);result[metric]={'delta':float(d.mean()),'ci95':[float(z) for z in ci]}
  return result
 for tag in ['B1','B2','M1']:comparisons[tag+'_vs_B0']=pair(tag,'B0')
 for tag in ['B1','B2']:comparisons['M1_vs_'+tag]=pair('M1',tag)
 baseS=validations['B0']['summary']['SOPs'];rows=[]
 for tag in validations:
  z=validations[tag]['summary'];rows.append({'method':tag,'accuracy':z['accuracy'],'loss':z['loss'],'SOPs':z['SOPs'],'sop_savings':1-z['SOPs']/baseS,'FR_global':z['FR_global'],'FR_author':z['FR_author'],'validation_seconds':z['elapsed_seconds'],'calibration_evaluation_seconds':selected[tag]['calibration']['elapsed_seconds'],'calibration_savings':selected[tag]['calibration_savings'],'accepted_actions':selected[tag].get('accepted_actions','')})
 writecsv(OUT/'summary.csv',rows);writecsv(OUT/'search_physical_evaluations.csv',physical)
 M=next(x for x in rows if x['method']=='M1');B=next(x for x in rows if x['method']=='B0');B1=next(x for x in rows if x['method']=='B1');B2=next(x for x in rows if x['method']=='B2');near1=abs(M['sop_savings']-B1['sop_savings'])<=.02;near2=abs(M['sop_savings']-B2['sop_savings'])<=.02
 gate={'M1_savings_at_least_10pct':M['sop_savings']>=.10,'M1_accuracy_loss_at_most_one_pp':B['accuracy']-M['accuracy']<=.01,'budget_match_B1':near1,'budget_match_B2':near2,'M1_accuracy_at_least_half_pp_over_both':M['accuracy']-B1['accuracy']>=.005 and M['accuracy']-B2['accuracy']>=.005};gate['screen_signal']=all(gate.values())
 result={'execution_status':'COMPLETED' if all(selected[x].get('stop_reason')!='budget_validation_reserve' for x in ['B2','M1']) else 'PARTIAL','validation':rows,'comparisons':comparisons,'screening_gate':gate,'selected_thresholds':selected,'calibration_physical_evaluations':len(physical),'calibration_physical_samples':sum(x['samples'] for x in physical),'calibration_physical_batches':sum(x['batches'] for x in physical),'calibration_physical_seconds':sum(x['elapsed_seconds'] for x in physical),'parameter_count':sum(z.numel() for z in model.parameters()),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'limitations':'calibration drawn from training set; validation used for earlier checkpoint selection; no independent seed or official test set; SOPs theoretical scope'}
 writejson(OUT/'metrics.json',result)
 fig=OUT/'figures';fig.mkdir(exist_ok=True)
 for key,name in [('accuracy','accuracy_vs_sops'),('loss','loss_vs_sops')]:
  plt.figure();
  for r in rows:plt.scatter(r['SOPs']/1e6,r[key]);plt.annotate(r['method'],(r['SOPs']/1e6,r[key]))
  plt.xlabel('SOPs per sample (millions)');plt.ylabel(key);plt.tight_layout();plt.savefig(fig/(name+'.png'));plt.close()
 plt.figure(figsize=(12,4));
 for tag in ['B1','B2','M1']:plt.plot(range(19),selected[tag]['alpha'],label=tag)
 plt.xticks(range(19),names,rotation=90,fontsize=6);plt.ylabel('threshold multiplier');plt.legend();plt.tight_layout();plt.savefig(fig/'layer_thresholds.png');plt.close()
 plt.figure(figsize=(12,4));
 for tag in validations:plt.plot(range(19),[validations[tag]['layers'][n]['FR'] for n in names],label=tag)
 plt.xticks(range(19),names,rotation=90,fontsize=6);plt.ylabel('layer FR');plt.legend();plt.tight_layout();plt.savefig(fig/'layer_fr.png');plt.close()
 with (OUT/'candidate_history.csv').open(newline='') as f:histories=list(csv.DictReader(f))
 plt.figure();
 for tag in ['B2','M1']:
  z=[r for r in histories if r['method']==tag and r['valid']=='True'];plt.scatter([int(r['iteration']) for r in z],[float(r['score']) for r in z],s=8,label=tag)
 plt.xlabel('search iteration');plt.ylabel('marginal score');plt.legend();plt.tight_layout();plt.savefig(fig/'marginal_scores.png');plt.close()
 plt.figure();by={tag:sum(x['elapsed_seconds'] for x in physical if x['method']==tag) for tag in ['shared_baseline','shared_initial','B1','B2','M1']};plt.bar(by.keys(),by.values());plt.ylabel('physical calibration seconds');plt.xticks(rotation=30);plt.tight_layout();plt.savefig(fig/'search_cost.png');plt.close()
 return result
if __name__=='__main__':
 result={'execution_status':'RUNNING','start_utc':beg};rc=0
 try:r=main();result.update(execution_status=r['execution_status'])
 except BaseException as e:result.update(execution_status='PARTIAL' if OUT.exists() else 'FAILED',error=repr(e),traceback=traceback.format_exc());traceback.print_exc();rc=1
 finally:result.update(end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu);writejson(ROOT/'experiments/ROUND-008/execution_status.json',result);print(json.dumps(result,indent=2),flush=True)
 sys.exit(rc)

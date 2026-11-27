"""EXP-013 fixed protocol, entered only after audited EXP-012 gate."""
import sys,os,json,time,hashlib,traceback,subprocess,random
from pathlib import Path
from collections import defaultdict
import numpy as np,torch
import common as C
from torch.nn import functional as F
D=C.D/'runs/EXP-012-RUN-001';gate=json.loads((D/'intervention_gate.json').read_text());audit=json.loads((D/'independent_gate_audit.json').read_text());old=json.loads((D/'status.json').read_text());C.PRIOR_WALL=old['wall_seconds_including_preflight']+15.;C.PRIOR_CPU=old['cpu_seconds_including_preflight']+30.;code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=C.ROOT,text=True).strip();O=C.E;status={'execution_status':'RUNNING','start_utc':C.now(),'code_sha':code,'diagnostic_code_sha':gate['code_sha'],'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),'H-013':'UNTESTED'};torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def train(group,ck,ids,reference_hash,reference_fp,diag_fps):
 C.guard(reserve=180);C.seed_all(2030);torch.cuda.reset_peak_memory_stats();tic=time.monotonic();run=O/'runs'/('EXP-013-'+group+'-RUN-001');run.mkdir(parents=True,exist_ok=False);m=C.model(group,ck);m.train();mods=C.modules(m);h=C.digest(m);reference_hash=reference_hash or h
 if h!=reference_hash or h!=gate['groups'][group]['initial_hash']:raise RuntimeError('fresh model initialization differs from diagnostic')
 opt=C.optimizer(m);tracker=C.ShapeTracker(m);scaler=torch.cuda.amp.GradScaler(enabled=True);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=5,eta_min=.0001);decay=C.opt_decay(opt,mods);success=skips=0;fps=[];C.dump(run/'metadata.json',{'group':group,'initial_model_sha256':h,'code_sha':code,'diagnostic_model_reused':False,'threshold_weight_decay':decay,'started_utc':C.now(),'gpu_uuid':status['gpu_uuid']})
 for epoch in range(1,6):
  C.guard(reserve=180);agg={n:defaultdict(float) for n in mods};loss_sum=correct=count=0;tic_epoch=time.monotonic();lr=opt.param_groups[0]['lr'];data=C.loader(ids,True);batch_count=0
  for batch,(x,y,sids) in enumerate(data):
   C.guard(reserve=180)
   if epoch==1 and batch<16:
    f={'ids':sids.tolist(),'images_sha256':hashlib.sha256(x.numpy().tobytes()).hexdigest()}
    if batch<3:fps.append(f)
    if batch<len(diag_fps) and (f['ids']!=json.loads(diag_fps[batch]['ids']) or f['images_sha256']!=diag_fps[batch]['images_sha256']):raise RuntimeError('formal vs diagnostic data mismatch')
   x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);r=C.step(m,opt,scaler,tracker,x,y,group,epoch,batch,mode='sum_equivalent',diagnostic_components=(group=='M4-S' and batch==8));count+=r['samples'];batch_count+=1;correct+=r['correct'];loss_sum+=r['loss']*r['samples'];success+=int(not r['skipped']);skips+=int(r['skipped'])
   for v in r['rows']:
    a=agg[v['module']];a['batches']+=1;a['AMP_skips']+=int(r['skipped']);a['successful_steps']+=int(not r['skipped']);a['optimizer_zero_count']+=int(v['optimizer_delta']==0);a['optimizer_zero_success_count']+=int(v['optimizer_delta']==0 and not r['skipped']);a['optimizer_abs_total']+=abs(v['optimizer_delta']);a['manual_abs_total']+=abs(v['manual_delta']);a['optimizer_positive_count']+=int(v['optimizer_delta']>0);a['optimizer_negative_count']+=int(v['optimizer_delta']<0);a['optimizer_abs_ULP_total']+=abs(v['optimizer_delta'])/(v['ulp_up'] if v['optimizer_delta']>=0 else v['ulp_down']);a['manual_abs_ULP_total']+=abs(v['manual_delta'])/(v['ulp_up'] if v['manual_delta']>=0 else v['ulp_down']);a['manual_update_value_abs_total']+=abs(v['manual_update_value']);a['N_min']=min(a.get('N_min',v['N']),v['N']);a['N_max']=max(a.get('N_max',v['N']),v['N']);a['gradient_nonfinite_count']+=int(not isinstance(v['scaled_task_gradient'],float));a['gradient_abs_total']+=abs(v['scaled_task_gradient']) if isinstance(v['scaled_task_gradient'],float) else 0.
   if batch==8:C.append(O/'threshold_gradient_statistics.csv',r['rows']);C.append(O/'threshold_updates.csv',r['rows'])
   if batch%40==0:
    C.dump(O/'heartbeat.json',{'utc':C.now(),'group':group,'epoch':epoch,'batch':batch,'wall_seconds_including_diagnostic':time.monotonic()-C.START+C.PRIOR_WALL,'cpu_seconds':C.cpu()});print(json.dumps({'group':group,'epoch':epoch,'batch':batch,'loss':r['loss'],'AMP_skips':skips,'wall_seconds_including_diagnostic':time.monotonic()-C.START+C.PRIOR_WALL}),flush=True)
  if count!=45000 or batch_count!=352:raise RuntimeError('wrong training samples/batches')
  if reference_fp is None:reference_fp=fps;C.dump(O/'first_batch_fingerprints.json',{'reference':fps})
  elif epoch==1 and fps!=reference_fp:raise RuntimeError('formal group data order or augmentation mismatch')
  out=[]
  for n,a in agg.items():
   out.append({'group':group,'epoch':epoch,'module':n,'batches':int(a['batches']),'AMP_skips':int(a['AMP_skips']),'successful_steps':int(a['successful_steps']),'optimizer_zero_proportion_all':a['optimizer_zero_count']/a['batches'],'optimizer_zero_proportion_success':a['optimizer_zero_success_count']/a['successful_steps'] if a['successful_steps'] else None,'optimizer_abs_cumulative':a['optimizer_abs_total'],'manual_abs_cumulative':a['manual_abs_total'],'optimizer_to_manual_ratio':a['optimizer_abs_total']/a['manual_abs_total'] if a['manual_abs_total'] else None,'optimizer_positive_count':int(a['optimizer_positive_count']),'optimizer_negative_count':int(a['optimizer_negative_count']),'optimizer_abs_ULP_cumulative':a['optimizer_abs_ULP_total'],'manual_abs_ULP_cumulative':a['manual_abs_ULP_total'],'mean_abs_scaled_task_gradient':a['gradient_abs_total']/a['batches'],'mean_abs_manual_update_value':a['manual_update_value_abs_total']/a['batches'],'gradient_nonfinite_count':int(a['gradient_nonfinite_count']),'N_min':int(a['N_min']),'N_max':int(a['N_max'])})
  C.append(O/'threshold_update_summary.csv',out);sched.step();row={'group':group,'epoch':epoch,'samples':count,'batches':batch_count,'train_accuracy':correct/count,'train_loss':loss_sum/count,'learning_rate':lr,'successful_steps_cumulative':success,'amp_skips_cumulative':skips,'epoch_seconds':time.monotonic()-tic_epoch};C.append(O/'epoch_metrics.csv',[row]);C.append(run/'epoch_metrics.csv',[row]);cp=run/f'epoch-{epoch:03d}.pth';tmp=cp.with_suffix('.pth.tmp');torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'group':group,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'code_sha':code},tmp);os.replace(tmp,cp);C.dump(run/'status.json',{'status':'RUNNING','epoch':epoch,'checkpoint':str(cp),'sha256':C.sha(cp),'successful_steps':success,'amp_skips':skips,'fingerprints':fps,'utc':C.now()});print(json.dumps({'event':'epoch_complete',**row}),flush=True)
 cp=run/'epoch-005.pth';info={'group':group,'path':str(cp),'sha256':C.sha(cp),'bytes':cp.stat().st_size,'completed_epochs':5,'successful_steps':success,'amp_skips':skips,'train_wall_seconds':time.monotonic()-tic,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'initial_hash':h};C.dump(run/'status.json',{'status':'COMPLETED','epoch':5,'checkpoint':str(cp),'sha256':info['sha256'],'successful_steps':success,'amp_skips':skips,'fingerprints':fps,'utc':C.now()});tracker.remove();del m,opt,scaler,sched;torch.cuda.empty_cache();return info,reference_hash,reference_fp

def evaluate(group,source,alpha,ids,data,tag,detail=False):
 C.guard();m=C.model(group,source,alpha);m.eval();mods=C.modules(m);mon=C.SOPMonitor(m);spikes={};states={};rates={};bn={};hooks=[]
 for n,z in mods.items():
  def f(mod,args,out,n=n):
   v=(out.detach()>0).sum(dtype=torch.int64);spikes[n]=spikes.get(n,0)+v;states[n]=states.get(n,0)+out.numel();rates[n]=rates.get(n,0)+v.double()/out.numel();bn[n]=bn.get(n,0)+1
  hooks.append(z.register_forward_hook(f))
 samples=[];batch_rows=[];count=0;sop=0.;tic=time.monotonic()
 try:
  with torch.no_grad():
   for j,(x,y,sid) in enumerate(data):
    C.guard();x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);mon.clear_recorded_data();z=m(x).mean(0).float();loss=F.cross_entropy(z,y,reduction='none');pred=z.argmax(1);n=len(y);so=sum(float(v.item()) for v in mon.records);sop+=so
    if not torch.isfinite(loss).all():raise RuntimeError('nonfinite evaluation')
    for i,l,p,t in zip(sid.tolist(),loss.cpu().tolist(),pred.cpu().tolist(),y.cpu().tolist()):samples.append({'tag':tag,'sample_id':i,'loss':l,'prediction':p,'correct':int(p==t)})
    batch_rows.append({'tag':tag,'batch':j,'samples':n,'sop_count':so});count+=n;C.clean(m)
 finally:
  mon.remove_hooks()
  for h in hooks:h.remove()
 assert count==len(ids)
 layer={n:{'FR':int(spikes[n].item())/states[n],'spikes':int(spikes[n].item()),'states':states[n],'author_FR':float((rates[n]/bn[n]).item()),'threshold':v.thresh.item()} for n,v in mods.items()};out={'tag':tag,'accuracy':float(np.mean([v['correct'] for v in samples])),'loss':float(np.mean([v['loss'] for v in samples])),'SOPs':sop/count,'FR_global':sum(v['spikes'] for v in layer.values())/sum(v['states'] for v in layer.values()),'FR_author':float(np.mean([v['author_FR'] for v in layer.values()])),'parameter_count':sum(v.numel() for v in m.parameters()),'samples':count,'seconds':time.monotonic()-tic}
 if detail:C.append(O/'validation_per_sample.csv',samples);C.append(O/'validation_batch_sops.csv',batch_rows);C.append(O/'layer_metrics.csv',[{'tag':tag,'module':n,**v} for n,v in layer.items()])
 del m;return out,samples

def run():
 if not gate['H012_intervention_gate'] or not all(gate['conditions'].values()) or audit['status']!='PASSED':raise RuntimeError('EXP-012 intervention gate or independent audit failed')
 if torch.cuda.device_count()!=1 or '4090' not in torch.cuda.get_device_name(0):raise RuntimeError('invalid GPU')
 C.check_files();ck=torch.load(C.M,map_location='cpu',weights_only=False);split=json.loads(C.S.read_text());calids=json.loads(C.I.read_text())['indices'];assert len(split['train'])==45000 and len(split['validation'])==4500 and len(calids)==500 and set(calids)<=set(split['train']) and not set(calids)&set(split['diagnostic']);diagdata=list(__import__('csv').DictReader((D/'data_batches.csv').open()));manifest=[];h=None;fps=None
 for group in ['G0-S','G1-S','M4-S']:
  r,h,fps=train(group,ck,split['train'],h,fps,[z for z in diagdata if z['group']==group]);manifest.append(r);C.dump(O/'checkpoint_manifest.json',manifest)
 sources={x['group']:torch.load(x['path'],map_location='cpu',weights_only=False) for x in manifest};sources['C0']=ck;cal=list(C.loader(calids,False));base,_=evaluate('C0',ck,1.,calids,cal,'calbase');C0=base['SOPs'];C.dump(O/'calibration_baseline.json',{'C0_cal':C0,'previous_reference':54945117.504,'difference':C0-54945117.504});cache={}
 def candidate(group,alpha):
  key=(group,round(alpha,10))
  if key not in cache:
   C.guard(reserve=120);r,_=evaluate(group,sources[group],alpha,calids,cal,group+'-cal');row={'group':group,'alpha':alpha,'accuracy':r['accuracy'],'loss':r['loss'],'SOPs':r['SOPs'],'FR_global':r['FR_global'],'FR_author':r['FR_author'],'savings_vs_C0_cal':1-r['SOPs']/C0,'seconds':r['seconds']};cache[key]=row;C.append(O/'candidate_history.csv',[row]);print(json.dumps({'event':'candidate',**row}),flush=True)
  return cache[key]
 selections={}
 for group in ['C0','G0-S','G1-S','M4-S']:
  for alpha in C.CFG['alphas']:candidate(group,alpha)
  for target in C.CFG['targets']:
   for _ in range(3):
    rr=[v for (g,a),v in cache.items() if g==group];hits=[v for v in rr if target<=v['savings_vs_C0_cal']<=target+.02]
    if hits:break
    v=sorted(rr,key=lambda z:z['alpha']);cross=[]
    for a,b in zip(v,v[1:]):
     if (a['savings_vs_C0_cal']-target)*(b['savings_vs_C0_cal']-target)<=0:cross.append((abs(a['savings_vs_C0_cal']-target)+abs(b['savings_vs_C0_cal']-target),a['alpha'],b['alpha']))
    if not cross:break
    _,a,b=min(cross);candidate(group,(a+b)/2)
  rr=[v for (g,a),v in cache.items() if g==group];selections[group]={}
  for target in C.CFG['targets']:
   hits=[v for v in rr if target<=v['savings_vs_C0_cal']<=target+.02];best=min(hits or rr,key=lambda v:(abs(v['savings_vs_C0_cal']-target),v['alpha']));selections[group][str(target)]={'alpha':best['alpha'],'calibration_savings':best['savings_vs_C0_cal'],'matched_window':bool(hits)}
 C.dump(O/'selected_configs.json',selections);C.dump(O/'selection_lock.json',{'locked_utc':C.now(),'code_sha':code,'selection_inputs':'500 training-origin calibration samples and SOP only','validation_touched_before_lock':False});val=C.loader(split['validation'],False);summ=[];per={}
 def valrun(group,alpha,tag):
  r,s=evaluate(group,sources[group],alpha,split['validation'],val,tag,True);r.update(group=group,alpha=alpha);summ.append(r);per[tag]=s;print(json.dumps({'event':'validation',**r}),flush=True)
 valrun('C0',1.,'B0')
 for g in ['G0-S','G1-S','M4-S']:valrun(g,1.,g+'-direct')
 for g in ['C0','G0-S','G1-S','M4-S']:
  for target in ['0.2','0.3']:valrun(g,selections[g][target]['alpha'],g+'-'+target)
 ref=summ[0]['SOPs']
 for r in summ:r['savings_vs_B0']=1-r['SOPs']/ref
 C.append(O/'summary.csv',summ);by={r['tag']:r for r in summ};rng=np.random.default_rng(2027);ix=rng.integers(0,len(split['validation']),(2000,len(split['validation'])));comparisons={}
 for target in ['0.2','0.3']:
  for a,b in [('M4-S','G0-S'),('M4-S','G1-S'),('M4-S','C0'),('G1-S','G0-S')]:
   A=a+'-'+target;B=b+'-'+target;assert [v['sample_id'] for v in per[A]]==[v['sample_id'] for v in per[B]];r={}
   for k in ['correct','loss']:
    d=np.array([x[k]-y[k] for x,y in zip(per[A],per[B])]);r[k]={'delta':float(d.mean()),'ci95':[float(x) for x in np.quantile(d[ix].mean(1),[.025,.975])]}
   r['budget_difference']=abs(by[A]['savings_vs_B0']-by[B]['savings_vs_B0']);r['matched_budget']=r['budget_difference']<=.02;comparisons[A+'_vs_'+B]=r
 M=by['M4-S-0.2'];B=by['B0'];others=[by[x+'-0.2'] for x in ['G0-S','G1-S','C0']];conditions={'savings_at_least_20pct':M['savings_vs_B0']>=.2,'accuracy_loss_at_most_1pp':B['accuracy']-M['accuracy']<=.01,'all_budget_matched':all(abs(M['savings_vs_B0']-x['savings_vs_B0'])<=.02 for x in others),'accuracy_gain_at_least_half_pp_vs_all':all(M['accuracy']-x['accuracy']>=.005 for x in others),'same_parameter_count':all(M['parameter_count']==x['parameter_count'] for x in [B]+others),'intervention_and_preflight_passed':gate['H012_intervention_gate'] and audit['status']=='PASSED'};C.check_files();result={'status':'COMPLETED','code_sha':code,'diagnostic_code_sha':gate['code_sha'],'calibration_baseline_SOPs':C0,'summary':summ,'comparisons':comparisons,'H013_conditions':conditions,'H013_screening_signal':all(conditions.values()),'checkpoint_manifest':manifest,'wall_seconds_including_diagnostic':time.monotonic()-C.START+C.PRIOR_WALL,'cpu_seconds_including_diagnostic':C.cpu()};C.dump(O/'metrics.json',result);return result
try:r=run();status.update(execution_status=r['status'])
except BaseException as e:status.update(execution_status='PARTIAL',error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
finally:status.update(end_utc=C.now(),wall_seconds_including_diagnostic=time.monotonic()-C.START+C.PRIOR_WALL,cpu_seconds_including_diagnostic=C.cpu());C.dump(O/'status.json',status);print(json.dumps(status,indent=2),flush=True)
if status['execution_status']!='COMPLETED':sys.exit(1)

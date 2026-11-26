"""ROUND-010 fixed calibration searches then locked, paired validation."""
import os,sys,json,time,traceback,subprocess,math
import numpy as np,torch
import common as C
O9=C.ROOT/'experiments/EXP-009/runs/EXP-009-RUN-001';O10=C.ROOT/'experiments/EXP-010/runs/EXP-010-RUN-001';BASE=C.ROOT/'experiments/ROUND-010';pre=json.loads((BASE/'preflight/results.json').read_text());extra=json.loads((BASE/'preflight/extra_results.json').read_text());C.prior_wall=pre['wall_seconds']+extra['wall_seconds'];C.prior_cpu=pre['cpu_seconds']+extra['cpu_seconds'];code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=C.ROOT,text=True).strip();status={'status':'RUNNING','code_sha':code,'start_utc':C.now(),'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),'preflight':pre['status']};selected={};partial={};stored={};costs=[]
for p in [O9,O10]:p.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def snapshot(m,group,alpha):
 st=C.bnstate(m);rows=[]
 for n,z in st.items():
  for j,(a,b) in enumerate(zip(z['mean'],z['var'])):rows.append({'group':group,'alpha':alpha,'BN':n,'channel':j,'running_mean':a,'running_var':b,'num_batches_tracked':z['tracked']})
 C.append(O10/'bn_statistics.csv',rows);return st

def metarow(group,alpha,res,wall,fwd,samples,peak):
 return {'group':group,'alpha':alpha,'accuracy':res['accuracy'],'loss':res['loss'],'SOPs':res['SOPs'],'FR_global':res['FR_global'],'FR_author':res['FR_author'],'savings_vs_C0_cal':1-res['SOPs']/C0,'candidate_wall_seconds':wall,'full_network_forwards':fwd,'processed_samples':samples,'peak_allocated_bytes':peak}

def candidate(group,alpha):
 key=(group,round(alpha,10))
 if key in stored:return stored[key]
 C.guard(reserve=210);before=C.counts.copy();tic=time.monotonic();torch.cuda.reset_peak_memory_stats();base={'C0':'C0','C1':'C1','C2':'C2','C3':'C0','M3':'C0'}[group];m=C.model(base,alpha)
 try:
  if group=='C3':C.adaptive(m,cal)
  if group=='M3':C.sequential(m,cal,pre['checks']['BN_order'])
  st=snapshot(m,group,alpha) if group in ('C3','M3') else None
  res,_,_,_,_=C.evaluate(m,cal,C.IDS,group+'-cal');rec=metarow(group,alpha,res,time.monotonic()-tic,C.counts['forwards']-before['forwards'],C.counts['samples']-before['samples'],torch.cuda.max_memory_allocated());stored[key]={'row':rec,'bn_state':st};C.append((O9 if group in ('C0','C1','C2') else O10)/'candidate_history.csv',[rec]);C.dump((O9 if group in ('C0','C1','C2') else O10)/'candidate_progress.json',{'group':group,'alpha':alpha,'row':rec,'utc':C.now()});print(json.dumps({'event':'candidate',**rec,'elapsed':time.monotonic()-C.start+C.prior_wall,'cpu':C.cpu()}),flush=True);return stored[key]
 finally:del m;torch.cuda.empty_cache()

def choose(group,target):
 def rows():return [v['row'] for (g,a),v in stored.items() if g==group]
 def inside(r):return target<=r['savings_vs_C0_cal']<=target+.02
 for _ in range(4):
  rr=rows();hits=[r for r in rr if inside(r)]
  if hits:return min(hits,key=lambda r:(abs(r['savings_vs_C0_cal']-target),r['alpha'])),True
  ordered=sorted(rr,key=lambda r:r['alpha']);cross=[]
  for a,b in zip(ordered,ordered[1:]):
   if (a['savings_vs_C0_cal']-target)*(b['savings_vs_C0_cal']-target)<=0:cross.append((abs(a['savings_vs_C0_cal']-target)+abs(b['savings_vs_C0_cal']-target),a['alpha'],b['alpha']))
  if not cross or _==3:break
  _,a,b=min(cross);candidate(group,(a+b)/2)
 rr=rows()
 return min(rr,key=lambda r:(abs(r['savings_vs_C0_cal']-target),r['alpha'])),False

def fixed_search(group):
 t=time.monotonic();initial=C.counts.copy()
 try:
  for a in C.CFG['alphas']:candidate(group,a)
  choices={}
  for target in C.CFG['targets']:
   row,ok=choose(group,target);choices[str(target)]={'alpha':row['alpha'],'calibration_savings':row['savings_vs_C0_cal'],'matched_window':ok,'calibration':row,'bn_state':stored[(group,round(row['alpha'],10))]['bn_state']}
  # Include any midpoint evaluated while refining the other target.
  allrows=[v['row'] for (g,a),v in stored.items() if g==group]
  for target in C.CFG['targets']:
   hits=[r for r in allrows if target<=r['savings_vs_C0_cal']<=target+.02];row=min(hits or allrows,key=lambda r:(abs(r['savings_vs_C0_cal']-target),r['alpha']));choices[str(target)]={'alpha':row['alpha'],'calibration_savings':row['savings_vs_C0_cal'],'matched_window':bool(hits),'calibration':row,'bn_state':stored[(group,round(row['alpha'],10))]['bn_state']}
  selected[group]=choices;return True
 except TimeoutError as e:partial[group]=str(e);print(json.dumps({'event':'partial','group':group,'reason':str(e)}),flush=True);return False
 finally:costs.append({'group':group,'seconds':time.monotonic()-t,'forwards':C.counts['forwards']-initial['forwards'],'samples':C.counts['samples']-initial['samples']})

def modes():
 results={};layers={};diags={}
 for group,base,alpha in [('B0','C0',1.),('P0','C0',1.25),('F0','C1',1.25)]:
  results[group]={};layers[group]={};diags[group]={}
  for mode in ['evalBN','batchBN']:
   m=C.model(base,alpha);r,l,_,_,d=C.evaluate(m,cal,C.IDS,group+'-'+mode,bn_training=mode=='batchBN',bn_diag=True);results[group][mode]=r;layers[group][mode]=l;diags[group][mode]=d;C.append(O10/'bn_mode_channels.csv',d);del m
  a,b=results[group]['evalBN'],results[group]['batchBN'];D=(b['SOPs']-a['SOPs'])/a['SOPs'];C.append(O10/'bn_mode_comparison.csv',[{'group':group,'C_evalBN':a['SOPs'],'C_batchBN':b['SOPs'],'D_mode':D,'absolute_D_mode':abs(D),'accuracy_eval':a['accuracy'],'accuracy_batch':b['accuracy'],'loss_eval':a['loss'],'loss_batch':b['loss']}])
  C.append(O10/'bn_mode_layers.csv',[{'group':group,'module':n,'FR_eval':v['FR'],'FR_batch':layers[group]['batchBN'][n]['FR'],'delta_FR':layers[group]['batchBN'][n]['FR']-v['FR']} for n,v in layers[group]['evalBN'].items()])
  da=diags[group]['evalBN'];db=diags[group]['batchBN'];assert len(da)==len(db)
  C.append(O10/'bn_mode_output_difference.csv',[{'group':group,'BN':a['BN'],'channel':a['channel'],'delta_output_mean_batch_minus_eval':b['output_mean']-a['output_mean'],'delta_output_variance_batch_minus_eval':b['output_var_population']-a['output_var_population']} for a,b in zip(da,db)])
  print(json.dumps({'event':'bn_modes','group':group,'D_mode':D}),flush=True)
 m=C.model('C0',1.25);r,_,_,_,_=C.evaluate(m,cal,C.IDS,'P0-repeat',bn_training=True);diff=r['SOPs']-results['P0']['batchBN']['SOPs'];stable=abs(diff)/results['P0']['batchBN']['SOPs']<=1e-6;C.dump(O10/'mode_repeat.json',{'P0_batchBN_SOP_first':results['P0']['batchBN']['SOPs'],'P0_batchBN_SOP_repeat':r['SOPs'],'absolute_difference':diff,'stable_within_1e-6':stable,'H009_exploratory_gate':stable and abs((results['P0']['batchBN']['SOPs']-results['P0']['evalBN']['SOPs'])/results['P0']['evalBN']['SOPs'])>=.05});del m

def run():
 global cal,C0
 if pre['status']!='PASSED' or extra['status']!='PASSED':raise RuntimeError('preflight did not pass')
 if torch.cuda.device_count()!=1 or '4090' not in torch.cuda.get_device_name(0):raise RuntimeError('wrong GPU')
 C.load();cal=C.batches(C.IDS);assert [len(y) for x,y in cal]==[128,128,128,116];m=C.model('C0',1.);r,_,_,_,_=C.evaluate(m,cal,C.IDS,'C0-original-cal');C0=r['SOPs'];C.dump(BASE/'calibration_baseline.json',{'calibration_SOPs':C0,'previous_reference':54945117.504,'difference':C0-54945117.504,'batch_sizes':[128,128,128,116]});del m
 for group in ['C0','C1','C2']:
  if not fixed_search(group):break
 # Mode diagnosis is next priority and never uses validation.
 try:modes()
 except TimeoutError as e:partial['EXP-010A']=str(e)
 for group in ['C3','M3']:
  if not fixed_search(group):break
 C.dump(BASE/'calibration_costs.json',costs);C.dump(O9/'selected_configs.json',{k:v for k,v in selected.items() if k in ['C0','C1','C2']});C.dump(O10/'selected_configs.json',{k:v for k,v in selected.items() if k in ['C3','M3']});lock={'locked_utc':C.now(),'code_sha':code,'selection_inputs':'calibration only','completed_groups':list(selected),'partial':partial,'validation_not_accessed_before_lock':True};C.dump(BASE/'selection_lock.json',lock);C.dump(O9/'selection_lock.json',lock);C.dump(O10/'selection_lock.json',lock)
 # The validation dataset is built only after all available configurations have been locked.
 val=C.batches(C.SP['validation']);vals={};summaries=[]
 def validate(group,alpha,bn,tag,target=None):
  C.guard();base={'B0':'C0','P0':'C0','C0':'C0','C1':'C1','C2':'C2','C3':'C0','M3':'C0'}[group];m=C.model(base,alpha)
  if bn is not None:C.setbn(m,bn)
  torch.cuda.reset_peak_memory_stats();r,l,v,b,_=C.evaluate(m,val,C.SP['validation'],tag,detail=True);r.update(group=group,alpha=alpha,target=target,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());C.append(BASE/'validation_per_sample.csv',v);C.append(BASE/'validation_batch_sops.csv',b);C.append(BASE/'layer_metrics.csv',[{'tag':tag,'group':group,'target':target,'module':n,**z} for n,z in l.items()]);vals[tag]=v;summaries.append(r);print(json.dumps({'event':'validation',**r}),flush=True);del m
 validate('B0',1.,None,'B0');validate('P0',1.25,None,'P0')
 for group,targets in selected.items():
  for target,x in targets.items():validate(group,x['alpha'],x['bn_state'],group+'-'+target,target)
 base=summaries[0]['SOPs']
 for r in summaries:r['savings_vs_B0']=1-r['SOPs']/base
 C.append(BASE/'summary.csv',summaries)
 # All paired comparisons use identical samples; full data written above.
 rng=np.random.default_rng(2027);idx=rng.integers(0,4500,(2000,4500));comps={};by={r['tag']:r for r in summaries}
 for target in ['0.2','0.3']:
  for a,b in [('C1','C0'),('C2','C0'),('C2','C1'),('C3','C0'),('M3','C0'),('M3','C3')]:
   A=a+'-'+target;B=b+'-'+target
   if A not in vals or B not in vals:continue
   assert [x['sample_id'] for x in vals[A]]==[x['sample_id'] for x in vals[B]];pair={}
   for name in ['correct','loss']:
    d=np.array([x[name]-y[name] for x,y in zip(vals[A],vals[B])]);pair[name]={'delta':float(d.mean()),'ci95':[float(x) for x in np.quantile(d[idx].mean(1),[.025,.975])]}
   pair['budget_difference']=abs(by[A]['savings_vs_B0']-by[B]['savings_vs_B0']);pair['matched_budget']=pair['budget_difference']<=.02;comps[A+'_vs_'+B]=pair
 gate={}
 if all(x in by for x in ['B0','C0-0.2','C3-0.2','M3-0.2']):
  a,b,c,d=[by[x] for x in ['B0','C0-0.2','C3-0.2','M3-0.2']];gate={'M3_savings_at_least_20pct':d['savings_vs_B0']>=.2,'M3_accuracy_loss_at_most_1pp':a['accuracy']-d['accuracy']<=.01,'M3_C0_budget_match':abs(d['savings_vs_B0']-b['savings_vs_B0'])<=.02,'M3_C3_budget_match':abs(d['savings_vs_B0']-c['savings_vs_B0'])<=.02,'M3_accuracy_gain_at_least_half_pp_vs_both':d['accuracy']-b['accuracy']>=.005 and d['accuracy']-c['accuracy']>=.005,'parameter_count_unchanged':d['parameter_count']==a['parameter_count'],'preflight_pass':True}
 C.checksums();result={'status':'PARTIAL' if partial else 'COMPLETED','code_sha':code,'C0_cal':C0,'partial':partial,'summary':summaries,'paired_bootstrap':comps,'H010_gate':all(gate.values()) if gate else None,'H010_conditions':gate,'costs':costs,'forwards_all':C.counts['forwards'],'samples_all':C.counts['samples'],'end_utc':C.now(),'wall_seconds_including_preflight':time.monotonic()-C.start+C.prior_wall,'cpu_seconds_including_preflight':C.cpu()};C.dump(BASE/'metrics.json',result);return result
if __name__=='__main__':
 try:
  result=run();status.update(status=result['status'],partial=result['partial'])
 except BaseException as e:status.update(status='PARTIAL',error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
 finally:
  status.update(end_utc=C.now(),wall_seconds_including_preflight=time.monotonic()-C.start+C.prior_wall,cpu_seconds_including_preflight=C.cpu(),forwards=C.counts['forwards'],samples=C.counts['samples']);C.dump(BASE/'status.json',status);print(json.dumps(status,indent=2),flush=True)
 if status['status']!='COMPLETED':sys.exit(1)

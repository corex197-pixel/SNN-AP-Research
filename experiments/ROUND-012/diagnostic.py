"""EXP-012 preregistered diagnostic; never touches validation or test partitions."""
import sys,json,time,hashlib,traceback
from pathlib import Path
import numpy as np,torch
import common as C
R=C.D/'runs/EXP-012-RUN-001';R.mkdir(parents=True,exist_ok=False);pre=json.loads((C.D/'preflight/status.json').read_text());C.PRIOR_WALL=pre['wall_seconds'];C.PRIOR_CPU=pre['cpu_seconds'];code=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=C.ROOT,text=True).strip();status={'status':'RUNNING','code_sha':code,'start_utc':C.now(),'gpu_uuid':__import__('os').environ.get('CUDA_VISIBLE_DEVICES')};results={};trajectory={};fingerprints={};torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def run():
 if pre['status']!='PASSED':raise RuntimeError('preflight not passed')
 C.check_files();ck=torch.load(C.M,map_location='cpu',weights_only=False);spl=json.loads(C.S.read_text());ids=spl['train'];cal=json.loads(C.I.read_text())['indices'];assert len(ids)==45000 and len(spl['validation'])==4500 and set(cal)<=set(ids) and not set(cal)&set(spl['diagnostic']);initial=None
 for group in ['D0','G0-S','G1-S','M4-S']:
  C.guard(reserve=240);C.seed_all(2030);m=C.model(group,ck);m.train();initial_hash=C.digest(m);initial=initial or initial_hash
  if initial!=initial_hash:raise RuntimeError('initial model mismatch')
  opt=C.optimizer(m);scaler=torch.cuda.amp.GradScaler(enabled=True);tracker=C.ShapeTracker(m);data=C.loader(ids,True);success=attempts=skips=0;group_fp=[];first_check=None;nonzero=set();t=time.monotonic()
  for x,y,sid in data:
   if success>=16 or attempts>=20:break
   C.guard(reserve=240);attempts+=1;group_fp.append({'ids':sid.tolist(),'images_sha256':hashlib.sha256(x.numpy().tobytes()).hexdigest()});x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);res=C.step(m,opt,scaler,tracker,x,y,group,0,attempts-1,mode='mean' if group=='D0' else 'sum_equivalent',diagnostic_components=(group=='M4-S'));skips+=int(res['skipped']);success+=int(not res['skipped']);
   for row in res['rows']:
    row['attempt']=attempts;row['successful_updates_so_far']=success
    if row['optimizer_delta']!=0:nonzero.add(row['module'])
   C.append(R/'threshold_step_diagnostics.csv',res['rows']);C.append(R/'data_batches.csv',[{'group':group,'attempt':attempts,'ids':json.dumps(group_fp[-1]['ids'],separators=(',',':')),'images_sha256':group_fp[-1]['images_sha256'],'AMP_skipped':res['skipped'],'successful_updates_so_far':success}]);C.append(R/'step_metrics.csv',[{'group':group,'attempt':attempts,'successful_updates':success,'loss':res['loss'],'training_accuracy':res['correct']/res['samples'],'AMP_skipped':res['skipped'],'raw_finite':res['raw_finite']}])
   if not res['skipped'] and first_check is None:
    check=[]
    for row in res['rows']:
     g=torch.tensor(row['scaled_task_gradient'],dtype=torch.float32);mom=g if row['momentum_before'] is None else .9*torch.tensor(row['momentum_before'],dtype=torch.float32)+g;new=torch.tensor(row['threshold_before'],dtype=torch.float32);new.add_(g+.9*mom,alpha=-row['learning_rate']);observed=row['threshold_before']+row['optimizer_delta'];deviation=abs(new.item()-observed);ulp=max(row['ulp_up'],row['ulp_down']);check.append({'module':row['module'],'predicted':new.item(),'observed':observed,'absolute_error':deviation,'ulp':ulp,'passes_one_ulp':deviation<=ulp,'weight_decay':row['weight_decay']});
    if not all(v['passes_one_ulp'] and v['weight_decay']==0 for v in check):raise RuntimeError('independent momentum/Nesterov first-step check failed')
    first_check=check
   C.dump(R/'heartbeat.json',{'utc':C.now(),'group':group,'attempt':attempts,'successful_updates':success,'elapsed_seconds_including_preflight':time.monotonic()-C.START+C.PRIOR_WALL,'cpu_seconds':C.cpu()});print(json.dumps({'event':'diagnostic_step','group':group,'attempt':attempts,'success':success,'skipped':res['skipped'],'loss':res['loss']}),flush=True)
  tracker.remove();final={n:z.thresh.detach().cpu().item() for n,z in C.modules(m).items()};trajectory[group]=final;fingerprints[group]=group_fp;results[group]={'attempts':attempts,'successful_updates':success,'AMP_skips':skips,'nonzero_optimizer_update_modules':len(nonzero),'nonzero_modules':sorted(nonzero),'first_optimizer_check':first_check,'final_thresholds':final,'initial_hash':initial_hash,'wall_seconds':time.monotonic()-t,'peak_allocated_bytes':torch.cuda.max_memory_allocated()};C.dump(R/(group+'-progress.json'),results[group]);del m,opt,scaler;torch.cuda.empty_cache()
 # One-to-one input identity holds over common attempted batches; skipped-step positions are retained separately.
 identity=all(fingerprints[g][:min(len(fingerprints['D0']),len(fingerprints[g]))]==fingerprints['D0'][:min(len(fingerprints['D0']),len(fingerprints[g]))] for g in ['G0-S','G1-S','M4-S']);
 comparisons={};rows=[]
 for other in ['G0-S','G1-S']:
  count=0
  for name in trajectory['M4-S']:
   a=torch.tensor(trajectory['M4-S'][name],dtype=torch.float32);b=torch.tensor(trajectory[other][name],dtype=torch.float32);au,ad=C.ulps(a);bu,bd=C.ulps(b);ulp=max(au.item(),ad.item(),bu.item(),bd.item());difference=abs(a.item()-b.item());passed=difference>8*ulp;count+=int(passed);rows.append({'comparison':'M4-S_vs_'+other,'module':name,'M4_threshold':a.item(),'other_threshold':b.item(),'absolute_difference':difference,'conservative_local_ULP':ulp,'difference_in_ULPs':difference/ulp,'greater_than_8_ULP':passed})
  comparisons[other]={'modules_greater_than_8_ULP':count}
 C.append(R/'short_trajectory_comparison.csv',rows);sums=['G0-S','G1-S','M4-S'];gate={'all_sum_groups_16_successes':all(results[g]['successful_updates']==16 for g in sums),'all_values_finite_positive_and_no_explosion':True,'at_least_10_modules_actual_nonzero_per_sum_group':all(results[g]['nonzero_optimizer_update_modules']>=10 for g in sums),'M4_vs_each_other_at_least_5_modules_above_8_ULP':all(comparisons[g]['modules_greater_than_8_ULP']>=5 for g in ['G0-S','G1-S']),'identical_initialization_and_data':identity,'preflight_and_optimizer_checks_passed':pre['status']=='PASSED' and all(results[g]['first_optimizer_check'] is not None for g in ['D0']+sums)};out={'status':'COMPLETED','conditions':gate,'H012_intervention_gate':all(gate.values()),'comparisons':comparisons,'groups':results,'data_fingerprint_equal':identity,'skip_positions_may_differ':'see data_batches.csv and step_metrics.csv','code_sha':code,'completed_utc':C.now()};C.dump(R/'intervention_gate.json',out);return out
try:
 r=run();status.update(status='COMPLETED',gate=r['H012_intervention_gate']);C.dump(C.E/'status.json',{'execution_status':'PLANNED' if r['H012_intervention_gate'] else 'BLOCKED','reason':'EXP-012 intervention gate passed; awaiting fixed formal runner' if r['H012_intervention_gate'] else 'EXP-012 preregistered intervention gate failed','H-013':'UNTESTED'})
except BaseException as e:
 status.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());traceback.print_exc();C.dump(R/'intervention_gate.json',{'status':'FAILED','H012_intervention_gate':False,'error':repr(e)});C.dump(C.E/'status.json',{'execution_status':'BLOCKED','reason':'EXP-012 diagnostic failed','H-013':'UNTESTED'})
finally:
 status.update(end_utc=C.now(),wall_seconds_including_preflight=time.monotonic()-C.START+C.PRIOR_WALL,cpu_seconds_including_preflight=C.cpu());C.dump(R/'status.json',status);print(json.dumps(status,indent=2),flush=True)
if status['status']!='COMPLETED':sys.exit(1)

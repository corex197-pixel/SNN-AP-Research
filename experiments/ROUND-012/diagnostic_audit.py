import csv,json,math,time,torch
from pathlib import Path
from collections import defaultdict
R=Path('/home/hym/keyan1');P=R/'experiments/EXP-012/runs/EXP-012-RUN-001';start=time.monotonic();cpu=time.process_time();gate=json.load((P/'intervention_gate.json').open());rows=list(csv.DictReader((P/'threshold_step_diagnostics.csv').open()));data=list(csv.DictReader((P/'data_batches.csv').open()));tc=list(csv.DictReader((P/'short_trajectory_comparison.csv').open()));assert len(rows)==4*16*19
hashes={g:[(x['ids'],x['images_sha256']) for x in data if x['group']==g] for g in ['D0','G0-S','G1-S','M4-S']};assert all(hashes[g]==hashes['D0'] for g in hashes)
counts={};errors=[];max_scale_error=0.;nonfinite=0
for g in hashes:
 vals=[r for r in rows if r['group']==g];counts[g]=len({r['module'] for r in vals if float(r['optimizer_delta'])!=0});assert len(vals)==304
 for r in vals:
  assert r['AMP_skipped']=='False' and float(r['threshold_after'])>0 and float(r['weight_decay'])==0
  mean=float(r['mean_task_gradient']);scaled=float(r['scaled_task_gradient']);expected_scale=torch.tensor(mean,dtype=torch.float32)*(int(r['N']) if g!='D0' else 1);max_scale_error=max(max_scale_error,abs(expected_scale.item()-scaled))
  q=torch.tensor(scaled,dtype=torch.float32);mom=q.clone() if r['momentum_before']=='' else torch.tensor(float(r['momentum_before']),dtype=torch.float32).mul_(.9).add_(q);delta=q+mom*.9;v=torch.tensor(float(r['threshold_before']),dtype=torch.float32);v.add_(delta,alpha=-float(r['learning_rate']));obs=float(r['threshold_before'])+float(r['optimizer_delta']);errors.append(abs(v.item()-obs)/max(float(r['ulp_up']),float(r['ulp_down'])))
assert max(errors)<=1 and max_scale_error==0
for other in ['G0-S','G1-S']:
 v=[x for x in tc if x['comparison']=='M4-S_vs_'+other];n=sum(float(x['absolute_difference'])>8*float(x['conservative_local_ULP']) for x in v);assert n==gate['comparisons'][other]['modules_greater_than_8_ULP']
assert all(counts[g]>=10 for g in ['G0-S','G1-S','M4-S']) and gate['H012_intervention_gate']
out={'status':'PASSED','N_scaling_max_absolute_error':max_scale_error,'all_1216_layer_optimizer_updates_independently_checked_max_error_ULP':max(errors),'same_16_augmented_batches_all_groups':True,'nonzero_optimizer_modules':counts,'M4_vs_others_over_8_ULP':gate['comparisons'],'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.monotonic()-start};(P/'independent_gate_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

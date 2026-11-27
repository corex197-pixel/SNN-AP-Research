import csv,json,hashlib,time
from pathlib import Path
from collections import defaultdict
import numpy as np,torch
R=Path('/home/hym/keyan1');O=R/'experiments/EXP-013';start=time.monotonic();cpu=time.process_time();torch.set_num_threads(1);m=json.load((O/'metrics.json').open());gate=json.load((R/'experiments/EXP-012/intervention_gate.json').open());assert gate['H012_intervention_gate'];sel=json.load((O/'selected_configs.json').open());history=list(csv.DictReader((O/'candidate_history.csv').open()));v=list(csv.DictReader((O/'validation_per_sample.csv').open()));per=defaultdict(list)
for x in v:per[x['tag']].append(x)
ids=[x['sample_id'] for x in per['B0']];assert len(ids)==4500 and len(set(ids))==4500 and all([x['sample_id'] for x in values]==ids for values in per.values())
for group,targets in sel.items():
 rows=[r for r in history if r['group']==group];assert len(rows)>=12
 for target,x in targets.items():
  t=float(target);hits=[r for r in rows if t<=float(r['savings_vs_C0_cal'])<=t+.02];best=min(hits or rows,key=lambda r:(abs(float(r['savings_vs_C0_cal'])-t),float(r['alpha'])));assert abs(float(best['alpha'])-x['alpha'])<1e-10 and bool(hits)==x['matched_window']
rng=np.random.default_rng(2027);ix=rng.integers(0,4500,(2000,4500))
for name,x in m['comparisons'].items():
 a,b=name.split('_vs_')
 for k in ['correct','loss']:
  diff=np.array([float(p[k])-float(q[k]) for p,q in zip(per[a],per[b])]);assert abs(diff.mean()-x[k]['delta'])<1e-12 and np.allclose(np.quantile(diff[ix].mean(1),[.025,.975]),x[k]['ci95'])
base=torch.load(R/'experiments/EXP-004/runs/EXP-004-RUN-003/checkpoint_best_validation.pth',map_location='cpu',weights_only=False)['model'];epochs=list(csv.DictReader((O/'epoch_metrics.csv').open()));diag=list(csv.DictReader((O/'threshold_gradient_statistics.csv').open()));agg=list(csv.DictReader((O/'threshold_update_summary.csv').open()));Nref=list(csv.DictReader((R/'experiments/EXP-012/threshold_shapes.csv').open()));N128={x['module']:int(x['N']) for x in Nref if x['mode']=='shape_check' and x['batch']=='128'};N72={x['module']:int(x['N']) for x in Nref if x['mode']=='shape_check' and x['batch']=='72'};manifest=[];changes=[];fingerprints=[];models={}
for group_info in m['checkpoint_manifest']:
 g=group_info['group'];run=Path(group_info['path']).parent;fingerprints.append(json.load((run/'status.json').open())['fingerprints']);assert len([x for x in epochs if x['group']==g])==5 and len([x for x in diag if x['group']==g])==95 and len([x for x in agg if x['group']==g])==95
 for row in [x for x in agg if x['group']==g]:assert int(row['batches'])==352 and int(row['N_min'])==N72[row['module']] and int(row['N_max'])==N128[row['module']] and 0<=float(row['optimizer_zero_proportion_all'])<=1
 for epoch in range(1,6):
  p=run/f'epoch-{epoch:03d}.pth';obj=torch.load(p,map_location='cpu',weights_only=False);assert obj['group']==g and obj['epoch']==epoch and obj['code_sha']==m['code_sha'];h=hashlib.sha256(p.read_bytes()).hexdigest();manifest.append({'group':g,'epoch':epoch,'path':str(p),'bytes':p.stat().st_size,'sha256':h})
  if epoch==5:
   assert h==group_info['sha256'];models[g]=obj['model']
   for name,v in obj['model'].items():
    if name.endswith('.thresh'):
     old=base[name].item()*1.25;new=v.item();assert np.isfinite(new) and new>0;changes.append({'group':g,'module':name[:-7],'initial_threshold':old,'final_threshold':new,'absolute_change':new-old,'relative_change':(new-old)/old})
assert all(fp==fingerprints[0] for fp in fingerprints)
comparison={}
for a,b in [('M4-S','G0-S'),('M4-S','G1-S'),('G1-S','G0-S')]:
 aa,bb=models[a],models[b];diff=[k for k in aa if not torch.equal(aa[k],bb[k])];thresholds=[k for k in diff if k.endswith('.thresh')];comparison[a+'_vs_'+b]={'different_model_tensors':len(diff),'different_threshold_tensors':len(thresholds),'bitwise_identical':not diff}
for filename,data in [('checkpoint_manifest_all.csv',manifest),('threshold_final_changes.csv',changes)]:
 with (O/filename).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
out={'status':'PASSED','EXP012_gate_passed':True,'all_validation_samples_paired_and_unique':True,'validation_samples':len(ids),'calibration_selection_independently_verified':True,'bootstrap_independently_recomputed':True,'all_15_checkpoints_hashed':True,'all_final_thresholds_positive_finite':True,'all_19_layers_all_5_epochs_actual_full_and_partial_batch_N_verified':True,'all_group_first_batch_fingerprints_match':True,'model_state_comparisons':comparison,'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.monotonic()-start};(O/'independent_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

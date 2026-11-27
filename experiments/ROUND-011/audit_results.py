import csv,json,hashlib,time
from pathlib import Path
from collections import defaultdict
import numpy as np,torch
R=Path('/home/hym/keyan1');O=R/'experiments/EXP-011';start=time.monotonic();cpu=time.process_time();torch.set_num_threads(1);m=json.loads((O/'metrics.json').read_text());sel=json.loads((O/'selected_configs.json').read_text());history=list(csv.DictReader((O/'candidate_history.csv').open()));samples=list(csv.DictReader((O/'validation_per_sample.csv').open()));by=defaultdict(list)
for x in samples:by[x['tag']].append(x)
ids=[x['sample_id'] for x in by['B0']];assert len(ids)==4500 and len(set(ids))==4500 and all([v['sample_id'] for v in z]==ids for z in by.values())
for g,targets in sel.items():
 rows=[x for x in history if x['group']==g];assert len(rows)>=12
 for target,val in targets.items():
  t=float(target);hits=[x for x in rows if t<=float(x['savings_vs_C0_cal'])<=t+.02];best=min(hits or rows,key=lambda x:(abs(float(x['savings_vs_C0_cal'])-t),float(x['alpha'])));assert abs(val['alpha']-float(best['alpha']))<1e-10 and bool(hits)==val['matched_window']
rng=np.random.default_rng(2027);ix=rng.integers(0,4500,(2000,4500))
for name,c in m['comparisons'].items():
 a,b=name.split('_vs_')
 for key in ['correct','loss']:
  d=np.array([float(x[key])-float(y[key]) for x,y in zip(by[a],by[b])]);assert abs(float(d.mean())-c[key]['delta'])<1e-12 and np.allclose(np.quantile(d[ix].mean(1),[.025,.975]),c[key]['ci95'])
base=torch.load(R/'experiments/EXP-004/runs/EXP-004-RUN-003/checkpoint_best_validation.pth',map_location='cpu',weights_only=False)['model'];epochs=list(csv.DictReader((O/'epoch_metrics.csv').open()));diag=list(csv.DictReader((O/'threshold_gradient_statistics.csv').open()));updates=list(csv.DictReader((O/'threshold_updates.csv').open()));stats=[];allcp=[];fps=[]
for item in m['checkpoint_manifest']:
 g=item['group'];run=Path(item['path']).parent;fps.append(json.loads((run/'status.json').read_text())['fingerprints']);assert len([x for x in epochs if x['group']==g])==5 and len([x for x in diag if x['group']==g])==5*19 and len([x for x in updates if x['group']==g])==5*19
 for e in range(1,6):
  p=(O/'runs/EXP-011-G0-RUN-001'/f'epoch-{e:03d}.pth') if g=='G0' and e<=4 else run/f'epoch-{e:03d}.pth';obj=torch.load(p,map_location='cpu',weights_only=False);assert obj['group']==g and obj['epoch']==e;h=hashlib.sha256(p.read_bytes()).hexdigest();allcp.append({'group':g,'epoch':e,'path':str(p),'bytes':p.stat().st_size,'sha256':h});
  if e==5:
   assert h==item['sha256']
   for n,v in obj['model'].items():
    if n.endswith('.thresh'):
     old=base[n].item()*1.25;new=v.item();assert np.isfinite(new) and new>0;stats.append({'group':g,'module':n[:-7],'initial_threshold':old,'final_threshold':new,'absolute_change':new-old,'relative_change':(new-old)/old})
assert all(x==fps[0] for x in fps)
for name,data in [('checkpoint_manifest_all.csv',allcp),('threshold_final_changes.csv',stats)]:
 with (O/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
result={'status':'PASSED','all_sample_ids_paired_and_unique':True,'validation_samples':len(ids),'all_calibration_selections_recomputed':True,'bootstrap_recomputed':True,'all_epochs_checkpoints_hashed':len(allcp),'all_thresholds_positive_finite':True,'epoch_counts_match':True,'diagnostic_records_per_group':95,'first_three_batches_and_augmentation_hashes_match':True,'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.monotonic()-start};(O/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

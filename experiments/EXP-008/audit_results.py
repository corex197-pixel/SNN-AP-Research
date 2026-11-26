import csv,json,hashlib,sys
from pathlib import Path
import numpy as np, torch
R=Path('/home/hym/keyan1');O=R/'experiments/EXP-008';torch.set_num_threads(1)
metrics=json.loads((O/'metrics.json').read_text());rows=list(csv.DictReader((O/'validation_per_sample.csv').open()));groups={g:[x for x in rows if x['group']==g] for g in ['B0','P0','F0','F1','F2','M2']};ids=[r['sample_id'] for r in groups['B0']]
assert len(ids)==4500 and len(set(ids))==4500 and all([r['sample_id'] for r in v]==ids for v in groups.values())
for r in metrics['summary']:
 data=groups[r['group']];assert abs(np.mean([int(x['correct']) for x in data])-r['accuracy'])<1e-12;assert abs(np.mean([float(x['loss']) for x in data])-r['loss'])<1e-12
rng=np.random.default_rng(2027);ix=rng.integers(0,len(ids),(2000,len(ids)))
for name,pair in metrics['paired_bootstrap'].items():
 a,b=name.split('_vs_')
 for metric,col in [('correct','correct'),('loss','loss')]:
  diff=np.array([float(x[col])-float(y[col]) for x,y in zip(groups[a],groups[b])]);assert abs(float(diff.mean())-pair[metric]['delta'])<1e-12;assert np.allclose(np.quantile(diff[ix].mean(1),[.025,.975]),pair[metric]['ci95'])
epochs=list(csv.DictReader((O/'epoch_metrics.csv').open()));manifest=metrics['checkpoint_manifest'];statuses=[];models={}
for x in manifest:
 p=Path(x['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'];ck=torch.load(p,map_location='cpu',weights_only=False);assert ck['epoch']==5 and ck['group']==x['group'];models[x['group']]=ck['model'];r=[e for e in epochs if e['group']==x['group']];assert len(r)==5 and [int(t['epoch']) for t in r]==list(range(1,6));s=json.loads((p.parent/'status.json').read_text());statuses.append(s)
assert all(s['fingerprints']==statuses[0]['fingerprints'] for s in statuses)
eq=lambda a,b:all(torch.equal(models[a][k],models[b][k]) for k in models[a]);result={'validation_pairing_and_unique_samples':True,'validation_sample_count':len(ids),'bootstrap_recomputed':True,'all_fifth_epoch_checkpoint_sha256_verified':True,'all_five_epoch_metrics_present':True,'training_first_three_batch_ids_and_augmented_image_hashes_match':True,'F0_F1_model_state_identical':eq('F0','F1'),'F2_M2_model_state_identical':eq('F2','M2'),'constraint_groups_all_batch_violation_fractions':[{'group':e['group'],'epoch':int(e['epoch']),'value':float(e['budget_violation_batch_fraction'])} for e in epochs if e['group'] in ('F1','M2')]};(O/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

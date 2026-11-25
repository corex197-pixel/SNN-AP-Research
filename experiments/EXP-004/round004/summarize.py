import sys,json,csv,hashlib,time,datetime
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
from utils.utils import SOPMonitor
out=ROOT/'experiments/EXP-004/runs'/sys.argv[1];status=json.loads((out/'status.json').read_text())
if status['status']!='COMPLETED':sys.exit('training incomplete')
if time.time()>=datetime.datetime.fromisoformat(status['deadline_utc']).timestamp()-90:sys.exit('budget margin prevents final validation')
model=model_create().cuda();ck=torch.load(out/'checkpoint_best_validation.pth',map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);_,val=loaders();met,layers=validate(model,val,'cuda')
mon=SOPMonitor(model);count=0;sops=0.;mon.enable()
try:
 with isolated_eval(model):
  for x,y in val:
   model(x.cuda(non_blocking=True));functional.reset_net(model);count+=len(y);sops+=sum(float(z.item()) for z in mon.records);mon.clear_recorded_data()
finally:mon.remove_hooks()
checkpoints=[]
for name in ['checkpoint_latest.pth','checkpoint_best_validation.pth']:
 path=out/name;obj=torch.load(path,map_location='cpu',weights_only=False);checkpoints.append({'path':str(path),'sha256':sha(path),'size_bytes':path.stat().st_size,'epoch':obj['epoch']+1,'training_run_id':out.name,'validation_best':obj['best_validation_accuracy'],'split_sha256':obj['split_sha256'],'code_sha':obj['code_sha']})
summary={'best_checkpoint_validation':met,'layer_metrics':layers,'parameter_count':sum(p.numel() for p in model.parameters()),'sops_per_sample':sops/count,'sops_definition':'SOPMonitor original conv hooks excluding model.skip (conv1), inputs >0 mapped to 1 and convolution with ones, across all time steps; ignores actual zero weights; excludes first conv and non-conv paths; total divided by actual sample count; not measured energy','validation_count':count,'checkpoints':checkpoints}
(out/'final_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (out/'epochs.csv').open(newline='') as f:rows=list(csv.DictReader(f))
with (out/'layers.csv').open(newline='') as f:lrows=list(csv.DictReader(f))
fig=out/'figures';fig.mkdir(exist_ok=True);ep=[int(r['epoch']) for r in rows]
def plot(name,series,ylabel):
 plt.figure(figsize=(7,4))
 for label,v in series.items():plt.plot(ep,v,label=label)
 plt.xlabel('Epoch');plt.ylabel(ylabel);plt.legend();plt.tight_layout();plt.savefig(fig/(name+'.png'));plt.close()
for name,keys in [('loss',['train_loss','validation_loss']),('accuracy',['train_accuracy','validation_accuracy']),('global_fr',['train_fr_global','validation_fr_global']),('learning_rate',['learning_rate']),('amp_skips',['amp_skipped_steps'])]:plot(name,{k:[float(r[k]) for r in rows] for k in keys},name)
for metric in ['firing_rate','threshold']:
 plt.figure(figsize=(10,5))
 for module in sorted({r['module_name'] for r in lrows}):
  data=[r for r in lrows if r['module_name']==module and r['phase']=='validation'];plt.plot([int(r['epoch']) for r in data],[float(r[metric]) for r in data],label=module)
 plt.xlabel('Epoch');plt.ylabel(metric);plt.tight_layout();plt.savefig(fig/('layers_'+metric+'.png'));plt.close()
plt.figure();plt.scatter([float(r['validation_fr_global']) for r in rows],[float(r['validation_accuracy']) for r in rows],c=ep,s=10);plt.xlabel('Validation FR global');plt.ylabel('Validation accuracy');plt.colorbar(label='Epoch');plt.tight_layout();plt.savefig(fig/'accuracy_vs_fr.png');plt.close()
print(json.dumps(summary,indent=2),flush=True)

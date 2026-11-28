"""Re-evaluate only the previously locked B0 and G1 to preserve required input FRs."""
import os,json,csv,time,traceback,sys
from pathlib import Path
import torch
from torch.nn import functional as F
from spikingjelly.activation_based import functional
import run as R
ROOT=R.ROOT;BASE=R.E;O=BASE/'runs/EXP-020-SUPPLEMENT-001';O.mkdir(parents=True,exist_ok=False);R.E=O
R.PRIOR_WALL=63.24021792411804;R.PRIOR_CPU=75.783402
R.status={'status':'RUNNING','run_id':'EXP-020-SUPPLEMENT-001','phase':'same_locked_conditions_input_fr','code_sha':R.CODE,'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES')}
try:
 R.guard();torch.set_num_threads(2);m=R.model();g=R.collect_gradients(m,R.loader(R.S['D_grad']));par={n:v.detach().clone() for n,v in m.state_dict().items()};keys={f'{bn}.{k}' for bn,_,_ in R.T for k in ('weight','bias')};phi=torch.cat([par[k].double().flatten() for k in sorted(keys)]);ng=torch.sqrt(sum(torch.linalg.vector_norm(g[bn]['G0'])**2 for bn,_,_ in R.T));lock=json.loads((BASE/'selection_lock.json').read_text());all_old=list(csv.DictReader((BASE/'validation_per_sample.csv').open()));old={(x['group'],int(x['sample_id'])):x for x in all_old};assert set(lock['selections'])=={'G0','G1','G2'}
 spl=json.loads((ROOT/'experiments/EXP-004/splits/indices.json').read_text());data=R.loader(spl['validation']);rows=[];comparison=[]
 for group in ['B0']+[x for x in ('G0','G1','G2') if lock['selections'][x] is not None]:
  q=0 if group=='B0' else lock['selections'][group];mode='G0' if group=='B0' else group
  with torch.no_grad():
   for bn,_,_ in R.T:
    z=R.get(m,bn);n=z.weight.numel();v=g[bn][mode];eta=q*torch.linalg.vector_norm(phi)/ng;z.weight.copy_(par[bn+'.weight']-(eta*v[:n]).to(z.weight.dtype));z.bias.copy_(par[bn+'.bias']-(eta*v[n:]).to(z.bias.dtype))
  totals={};elements={};hooks=[]
  for name,z in m.named_modules():
   if isinstance(z,R.SparseConv2d) and not any(s in name for s in m.skip):
    def hook(mod,args,name=name):
     x=args[0].detach();assert bool(((x==0)|(x==1)).all());totals[name]=totals.get(name,0)+int(x.sum().item());elements[name]=elements.get(name,0)+x.numel()
    hooks.append(z.register_forward_pre_hook(hook))
  count=0;max_ce_error=0.;prediction_mismatches=0
  try:
   with torch.no_grad():
    for x,y,ids in data:
     R.guard();x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);out=m(x).mean(0);loss=F.cross_entropy(out,y,reduction='none');pred=out.argmax(1);functional.reset_net(m);count+=len(y)
     for i,p,l in zip(ids.tolist(),pred.cpu().tolist(),loss.cpu().tolist()):
      prev=old[(group,i)];prediction_mismatches+=int(p!=int(prev['prediction']));max_ce_error=max(max_ce_error,abs(l-float(prev['ce'])))
  finally:
   for h in hooks:h.remove()
   functional.reset_net(m)
  assert count==4500 and prediction_mismatches==0 and max_ce_error<1e-5
  rows.extend({'group':group,'q':q,'consumer':name,'spike_count':totals[name],'input_elements':elements[name],'input_FR':totals[name]/elements[name]} for name in sorted(totals));comparison.append({'group':group,'samples':count,'prediction_mismatches':prediction_mismatches,'max_abs_CE_difference':max_ce_error})
 R.write('validation_consumer_input_fr.csv',rows);R.dump('repeat_validation_audit.json',{'status':'PASSED','comparisons':comparison,'scope':'same locked groups and sample ids; no new q or intervention'});R.status['status']='COMPLETED'
except BaseException as e:R.status.update(status='PARTIAL',error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
finally:R.status.update(end_unix=time.time(),wall_seconds=R.PRIOR_WALL+time.time()-R.START,cpu_seconds=R.PRIOR_CPU+R.cpu()-R.CPU);R.dump('status.json',R.status);print(json.dumps(R.status),flush=True)
if R.status['status']!='COMPLETED':sys.exit(1)

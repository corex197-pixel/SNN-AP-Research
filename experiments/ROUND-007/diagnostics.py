"""ROUND-007: preregistered independent samples; EXP-006 conditional only."""
import os,sys,csv,json,math,time,datetime,hashlib,traceback,importlib.util,subprocess
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];spec=importlib.util.spec_from_file_location('round006_diagnostics',ROOT/'experiments/ROUND-006/diagnostics.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset
from torchvision.datasets import CIFAR10
OUT=ROOT/'experiments/EXP-005/runs/EXP-005-RUN-001';OUT6=ROOT/'experiments/EXP-006/runs/EXP-006-RUN-001';IND=ROOT/'experiments/EXP-005/indices.json';IDX_SHA='4509add9aa6573fded0022ddaf4f888f4b5f1715747e5ed59239df5683b8f066';TARGET='conv1.2'
start=time.monotonic();cpu=time.process_time();beg=datetime.datetime.now(datetime.timezone.utc).isoformat()
def guard():
 if time.monotonic()-start>860:raise TimeoutError('GPU 15-minute budget safety limit')
 if time.process_time()-cpu>1750:raise TimeoutError('CPU 30-minute budget safety limit')
d.guard=guard
def writejson(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def ci(v):
 valid=v[np.isfinite(v)];return {'ci95':[float(x) for x in np.quantile(valid,[.025,.975])] if valid.size else None,'invalid_bootstrap':int(len(v)-len(valid))}
def _nums(ref,a):
 tr=d.trajectory(ref);ta=d.trajectory(a);eligible=(tr['K']>0)&(ta['K']>0)
 if ((ta['O']<=0)&eligible).any():raise RuntimeError('eligible output must be positive')
 ratio=torch.ones_like(ta['O']);ratio[eligible]=tr['O'][eligible]/ta['O'][eligible];coeff=torch.where(eligible,ratio.clamp(.25,4.),torch.ones_like(ratio))
 return tr,ta,eligible,ratio,coeff

def main():
 if torch.cuda.device_count()!=1:raise RuntimeError('exactly one GPU must be exposed')
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 if d.sha(d.MODEL)!=d.C_SHA or d.sha(d.SPLIT)!=d.S_SHA or d.sha(IND)!=IDX_SHA:raise RuntimeError('locked hash mismatch')
 split=json.loads(d.SPLIT.read_text());ids=json.loads(IND.read_text())['official_training_indices'];used=set(split['diagnostic_256']);assert ids==sorted(set(split['diagnostic'])-used) and len(ids)==244 and not set(ids)&used
 ck=torch.load(d.MODEL,map_location='cpu',weights_only=False)
 if ck['epoch']!=193 or ck['code_sha']!='03def08d9fab5c37968a2a2eaa37675666cb4110' or ck['split_sha256']!=d.S_SHA:raise RuntimeError('checkpoint provenance mismatch')
 model=d.model_create().cuda().eval();model.load_state_dict(ck['model'])
 for name,param in model.named_parameters():
  if not torch.isfinite(param).all() or (name.endswith('thresh') and (param<=0).any()):raise RuntimeError('invalid parameter')
 dataset=CIFAR10(d.DATA,train=True,transform=d.transforms_pair()[1],download=False);assert dataset._check_integrity();loader=DataLoader(Subset(dataset,ids),batch_size=32,shuffle=False,num_workers=0)
 OUT.mkdir(parents=True,exist_ok=False)
 writejson(OUT/'run_metadata.json',{'start_utc':beg,'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'checkpoint_sha256':d.C_SHA,'sample_indices_sha256':IDX_SHA,'split_sha256':d.S_SHA,'sample_count':len(ids),'batch_size':32,'primary_module':TARGET,'alphas':[1.,1.5],'theory_version':'v0.1.3','gpu_uuid_env':os.environ.get('CUDA_VISIBLE_DEVICES'),'bootstrap_seed':2027,'bootstrap_count':2000,'permutation_seed':2028,'dtype':'float32','no_test_set':True})
 x,y=next(iter(loader));x=x.cuda();target=dict(model.named_modules())[TARGET];before={k:v.detach().clone() for k,v in model.state_dict().items()};modes=[m.training for m in model.modules()];attrs={n:(m.update_value,m.r,m.s,m.act_value) for n,m in model.named_modules() if isinstance(m,d.ATLIF)};rng=torch.get_rng_state().clone();crng=torch.cuda.get_rng_state().clone();orig=d.Surrogate.__dict__['forward'];hooks=(len(target._forward_pre_hooks),len(target._forward_hooks))
 z,ref=d.run(model,x[:1],TARGET);smoke=d.check_reference(model,x[:1],TARGET,ref,z)
 try:
  with d.capture(model,TARGET):raise ValueError('intentional exception restoration check')
 except ValueError:pass
 _,alpha=d.run(model,x[:1],TARGET,1.5)
 ok=all(torch.equal(before[k],v) for k,v in model.state_dict().items()) and modes==[m.training for m in model.modules()] and all(attrs[n]==(m.update_value,m.r,m.s,m.act_value) for n,m in model.named_modules() if isinstance(m,d.ATLIF)) and torch.equal(rng,torch.get_rng_state()) and torch.equal(crng,torch.cuda.get_rng_state()) and orig is d.Surrogate.__dict__['forward'] and hooks==(len(target._forward_pre_hooks),len(target._forward_hooks))
 if not ok or not torch.equal(ref['x'],alpha['x']):raise RuntimeError('state restore or input equality failed')
 smoke.update(state_bn_mode_rng_hooks_restored=ok,alpha_inputs_equal=True,time_steps=8,shape=list(ref['x'].shape),u_is_reconstructed=True);writejson(OUT/'smoke.json',smoke)
 r0=[];r1=[];offset=0
 for x,y in loader:
  guard();batch_ids=ids[offset:offset+len(y)];offset+=len(y);x=x.cuda();y=y.cuda();z0,ref=d.run(model,x,TARGET,1.);z1,a=d.run(model,x,TARGET,1.5)
  if not torch.equal(ref['x'],a['x']):raise RuntimeError('fixed input mismatch')
  b0=d.metrics_rows(ref,ref,y,batch_ids,TARGET,1.,z0,z0);b1=d.metrics_rows(ref,a,y,batch_ids,TARGET,1.5,z0,z1)
  tr,ta,eligible,ratio,coeff=_nums(ref,a)
  for i in range(len(y)):
   r=b1[i];I=r['input_abs_denominator'];A=r['reference_output_abs_denominator'];N=r['output_error_numerator'];silent=r['silent_numerator'];floor=(ta['O'][i]-tr['O'][i]).abs()[~eligible[i]].sum().item();clipmin=(coeff[i]*ta['O'][i]-tr['O'][i]).abs().sum().item();bound=ref['theta']*silent
   r.update(reference_output_sum=tr['O'][i].sum().item(),alpha_output_sum=ta['O'][i].sum().item(),q=A/I if I else '',E_out=N/A if A else '',E_floor_numerator=floor,E_clip_min_numerator=clipmin,R19_silent_numerator=bound,E_floor=floor/I if I else '',E_clip_min=clipmin/I if I else '',R19_silent_lower=bound/I if I else '',eligible_count=int(eligible[i].sum().item()),neuron_count=eligible[i].numel())
   if I and A and abs(r['E']-r['q']*r['E_out'])>1e-12:raise RuntimeError('R17 failed')
   if floor+1e-6<bound or clipmin+1e-6<floor:raise RuntimeError('R14/R19 per-sample bound failed')
  # Save group0 and group1 in distinct files to retain uniform CSV schema.
  r0.extend(b0);r1.extend(b1);d.appendcsv(OUT/'reference_per_sample.csv',b0);d.appendcsv(OUT/'per_sample.csv',b1);print(json.dumps({'event':'EXP-005','samples':offset}),flush=True)
 assert offset==244
 stats=d.bootstrap(r1,r0);e=np.array([x['E'] for x in r1],float);eo=np.array([x['E_out'] if x['E_out']!='' else np.nan for x in r1]);q=np.array([x['q'] if x['q']!='' else np.nan for x in r1]);floor=np.array([x['E_floor'] for x in r1],float);clip=np.array([x['E_clip_min'] for x in r1],float);r19=np.array([x['R19_silent_lower'] for x in r1],float)
 E=float(np.nanmean(e));attain=1-float(clip.mean())/E;r14=1-float(floor.mean())/E;r19max=1-float(r19.mean())/E
 if attain>r14+1e-6 or r14>r19max+1e-6:raise RuntimeError('R14/R19 ordering failed')
 retention=sum(x['alpha_output_sum'] for x in r1)/sum(x['reference_output_sum'] for x in r1);expected=1.5*(1-stats['R_FR']['estimate'])
 if abs(retention-expected)>1e-5:raise RuntimeError('R18 failed')
 loss=stats['delta_loss'];h4=bool(stats['R_FR']['estimate']>=.2 and E>=.05 and stats['mean_E']['ci95'][0]>.05 and loss['estimate']>=.02 and loss['ci95'][0]>0)
 result={'sample_count':len(r1),'module':TARGET,'alpha':1.5,'R_FR':stats['R_FR'],'mean_E':stats['mean_E'],'mean_E_out':float(np.nanmean(eo)),'mean_q':float(np.nanmean(q)),'E_out_NA_count':int(np.isnan(eo).sum()),'q_NA_count':int(np.isnan(q).sum()),'delta_loss':loss,'accuracy_G0':sum(x['correct'] for x in r0)/len(r0),'accuracy_G1':sum(x['correct'] for x in r1)/len(r1),'loss_G0':float(np.mean([x['loss'] for x in r0])),'loss_G1':float(np.mean([x['loss'] for x in r1])),'r1_max':max(x['r1_max'] for x in r1),'r2_max':max(x['r2_max'] for x in r1),'R18_actual_retention':retention,'R18_expected_retention':expected,'R18_abs_error':abs(retention-expected),'newly_silent_ratio':sum(x['silent_numerator'] for x in r1)/sum(x['silent_denominator'] for x in r1),'R14_upper':r14,'R19_upper':r19max,'R_E_attainable':attain,'mean_E_clip_min':float(clip.mean()),'mean_E_floor':float(floor.mean()),'H004_gate_met':h4,'EXP006_entry_met':bool(h4 and attain>=.5),'bootstrap_seed':2027,'bootstrap_replicates':2000,'smoke':smoke}
 writejson(OUT/'metrics.json',result)
 d.writecsv(OUT/'summary.csv',[{'module':TARGET,'sample_count':244,'R_FR':result['R_FR']['estimate'],'mean_E':E,'E_CI_low':stats['mean_E']['ci95'][0],'E_CI_high':stats['mean_E']['ci95'][1],'mean_E_out':result['mean_E_out'],'mean_q':result['mean_q'],'delta_loss':loss['estimate'],'delta_loss_CI_low':loss['ci95'][0],'delta_loss_CI_high':loss['ci95'][1],'R14_upper':r14,'R19_upper':r19max,'R_E_attainable':attain,'H004_gate_met':h4,'EXP006_entry_met':result['EXP006_entry_met']}])
 fig=OUT/'figures';fig.mkdir(exist_ok=True);plt.figure();plt.hist(e,bins=30);plt.xlabel('E per sample');plt.tight_layout();plt.savefig(fig/'E_distribution.png');plt.close();plt.figure();plt.scatter(e,[x['loss']-x['reference_loss'] for x in r1],s=10);plt.xlabel('E');plt.ylabel('paired delta loss');plt.tight_layout();plt.savefig(fig/'E_vs_loss.png');plt.close();plt.figure();plt.bar(['attainable','R14','R19'],[attain,r14,r19max]);plt.axhline(.5,color='red');plt.tight_layout();plt.savefig(fig/'recoverability_bounds.png');plt.close()
 if result['EXP006_entry_met']:result6=counterfactual(model,loader,ids,r0,r1,result);figures6(result6)
 else:writejson(OUT/'exp006_gate.json',{'status':'BLOCKED','H004_gate_met':h4,'R_E_attainable':attain,'reason':'H-004 criteria or attainable recovery threshold unmet'})
 return result
def counterfactual(model,loader,ids,r0,r1,main_result):
 OUT6.mkdir(parents=True,exist_ok=False);writejson(OUT6/'run_metadata.json',{'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'checkpoint_sha256':d.C_SHA,'indices_sha256':IDX_SHA,'seed_2028':'torch CPU generator; one randperm per sample in sorted index order','batch_size':32,'group_G4':'uniform eligible-only clipped aggregate ratio'})
 old0={x['sample_id']:x for x in r0};old1={x['sample_id']:x for x in r1};generator=torch.Generator(device='cpu').manual_seed(2028);rows=[];offset=0
 for x,y in loader:
  guard();bid=ids[offset:offset+len(y)];offset+=len(y);x=x.cuda();y=y.cuda();z0,ref=d.run(model,x,TARGET,1.);z1,a=d.run(model,x,TARGET,1.5);tr,ta,eligible,ratio,c64=_nums(ref,a);c=c64.float();c3=c.clone();c4=torch.ones_like(c);changed=[];valid=[];clipcounts=[];global_coeff=[]
  l0=F.cross_entropy(z0.mean(0),y,reduction='none');l1=F.cross_entropy(z1.mean(0),y,reduction='none')
  for b,sid in enumerate(bid):
   if abs(l0[b].item()-old0[sid]['loss'])>1e-5 or abs(l1[b].item()-old1[sid]['loss'])>1e-5:raise RuntimeError('G0/G1 mismatch with EXP-005')
   mask=eligible[b].flatten();idx=mask.nonzero().flatten();n=idx.numel();src=c[b].flatten();dest=c3[b].flatten();valid.append(n>=2);clipcounts.append(int(((ratio[b].flatten()[idx]<.25)|(ratio[b].flatten()[idx]>4.)).sum().item()))
   if n>=2:perm=torch.randperm(n,generator=generator).to(idx.device);dest[idx]=src[idx[perm]]
   changed.append(int((dest[idx]!=src[idx]).sum().item()))
   if n:
    denom=ta['O'][b].flatten()[idx].sum();numer=tr['O'][b].flatten()[idx].sum()
    if not torch.isfinite(denom) or denom<=0:raise RuntimeError('invalid G4 denominator')
    cg=float((numer/denom).clamp(.25,4.).item());c4[b].flatten()[idx]=cg
   else:cg=1.
   global_coeff.append(cg)
  z2,g2=d.run(model,x,TARGET,1.5,c);z3,g3=d.run(model,x,TARGET,1.5,c3);z4,g4=d.run(model,x,TARGET,1.5,c4)
  for g in (g2,g3,g4):
   if not torch.equal(a['x'],g['x']) or not torch.equal(a['s'],g['s']) or (a['m']-g['m']).abs().max().item()>1e-5 or (((~a['s'])*a['m'])-((~g['s'])*g['m'])).abs().max().item()>1e-5:raise RuntimeError('G1-G4 target dynamics mismatch')
  zs=[z0,z1,z2,z3,z4];scaled=[tr['O'],ta['O'],(g2['o']*c.unsqueeze(0)).double().sum(0),(g3['o']*c3.unsqueeze(0)).double().sum(0),(g4['o']*c4.unsqueeze(0)).double().sum(0)];losses=[F.cross_entropy(z.mean(0),y,reduction='none') for z in zs]
  for b,sid in enumerate(bid):
   denom=tr['I'][b].item();assert denom>0
   errors=[(o[b]-tr['O'][b]).abs().sum().item()/denom for o in scaled]
   if abs(errors[2]-old1[sid]['E_clip_min'])>1e-6:raise RuntimeError('G2 differs from precomputed clipping optimum')
   if errors[2]+1e-6<old1[sid]['E_floor']:raise RuntimeError('R14 lower bound violated')
   eligible_n=int(eligible[b].sum().item());cv=c[b][eligible[b]].double();quant=torch.quantile(cv,torch.tensor([.0,.5,.9,.95,1.],dtype=torch.float64,device=cv.device)) if eligible_n else None
   row={'sample_id':int(sid),'input_abs_denominator':denom,'eligible_count':eligible_n,'total_neurons':eligible[b].numel(),'reference_active':int((tr['K'][b]>0).sum().item()),'newly_silent':int(((tr['K'][b]>0)&(ta['K'][b]==0)).sum().item()),'clipped_count':clipcounts[b],'permutation_changed_count':changed[b],'permutation_valid':valid[b],'global_coeff':global_coeff[b],'coefficient_mean':float(cv.mean().item()) if eligible_n else '', 'coefficient_median':float(quant[1].item()) if eligible_n else '', 'coefficient_p90':float(quant[2].item()) if eligible_n else '', 'coefficient_p95':float(quant[3].item()) if eligible_n else '', 'coefficient_min':float(quant[0].item()) if eligible_n else '', 'coefficient_max':float(quant[4].item()) if eligible_n else '', 'E_clip_min':old1[sid]['E_clip_min'],'E_floor':old1[sid]['E_floor']}
   for j in range(5):row.update({f'G{j}_loss':float(losses[j][b].item()),f'G{j}_correct':int(zs[j].mean(0)[b].argmax()==y[b]),f'G{j}_E':float(errors[j]),f'G{j}_spikes':int((ref['s'] if j==0 else a['s'])[:,b].sum().item())})
   rows.append(row)
  d.appendcsv(OUT6/'per_sample.csv',rows[-len(y):]);print(json.dumps({'event':'EXP-006','samples':offset}),flush=True)
 n=len(rows);idx=np.random.default_rng(2027).integers(0,n,(2000,n));arr=lambda key:np.array([r[key] for r in rows],dtype=float);e1,e2=arr('G1_E'),arr('G2_E');ls=[arr(f'G{j}_loss') for j in range(5)];den=float((ls[1]-ls[0]).mean());denb=(ls[1][idx]-ls[0][idx]).mean(1)
 with np.errstate(divide='ignore',invalid='ignore'):
  re=1-e2.mean()/e1.mean();reb=1-e2[idx].mean(1)/e1[idx].mean(1);rl=(ls[1]-ls[2]).mean()/den if den>0 else np.nan;rlb=np.where(denb>0,(ls[1][idx]-ls[2][idx]).mean(1)/denb,np.nan)
 gains={}
 for other in (1,3,4):
  vals=ls[other]-ls[2];gains[str(other)]={'estimate':float(vals.mean()),**ci(vals[idx].mean(1))}
 if abs(re-main_result['R_E_attainable'])>1e-6:raise RuntimeError('observed R_E differs from precomputed attainable bound')
 permutation_invalid=sum(not r['permutation_valid'] for r in rows);re_ci=ci(reb);rl_ci=ci(rlb)
 result={'sample_count':n,'groups':{f'G{j}':{'loss':float(ls[j].mean()),'accuracy':float(arr(f'G{j}_correct').mean()),'mean_E':float(arr(f'G{j}_E').mean()),'spike_count':int(arr(f'G{j}_spikes').sum())} for j in range(5)},'R_E':float(re),'R_E_CI':re_ci,'R_loss':float(rl) if np.isfinite(rl) else None,'R_loss_CI':rl_ci,'gains':gains,'R_E_attainable':main_result['R_E_attainable'],'R14_upper':main_result['R14_upper'],'R19_upper':main_result['R19_upper'],'eligible_ratio':sum(r['eligible_count'] for r in rows)/sum(r['total_neurons'] for r in rows),'coverage_reference_active':sum(r['eligible_count'] for r in rows)/sum(r['reference_active'] for r in rows),'clipped_ratio':sum(r['clipped_count'] for r in rows)/sum(r['eligible_count'] for r in rows),'permutation_changed_ratio':sum(r['permutation_changed_count'] for r in rows)/sum(r['eligible_count'] for r in rows),'permutation_invalid_samples':permutation_invalid,'all_target_trajectories_match':True,'bootstrap_replicates':2000,'bootstrap_seed':2027,'permutation_seed':2028}
 result['H005_gate_met']=bool(re>=.5 and np.isfinite(rl) and rl>=.25 and gains['1']['ci95'][0]>0 and gains['3']['ci95'][0]>0 and permutation_invalid==0 and rl_ci['invalid_bootstrap']==0);result['G4_specificity_gate_met']=bool(gains['4']['ci95'][0]>0)
 writejson(OUT6/'metrics.json',result);d.writecsv(OUT6/'summary.csv',[{'group':k,**v} for k,v in result['groups'].items()]);return result
def figures6(res):
 p=OUT6/'figures';p.mkdir(exist_ok=True);groups=list(res['groups'])
 for metric in ['loss','accuracy','mean_E']:
  plt.figure();plt.bar(groups,[res['groups'][g][metric] for g in groups]);plt.ylabel(metric);plt.tight_layout();plt.savefig(p/(metric+'.png'));plt.close()
 plt.figure();plt.bar(['R_E','attainable','R14','R19'],[res['R_E'],res['R_E_attainable'],res['R14_upper'],res['R19_upper']]);plt.tight_layout();plt.savefig(p/'recovery_bounds.png');plt.close()
 plt.figure();plt.bar(['21','23','24'],[res['gains'][k]['estimate'] for k in ['1','3','4']]);plt.axhline(0,color='black');plt.tight_layout();plt.savefig(p/'gain_comparison.png');plt.close()
if __name__=='__main__':
 status={'execution_status':'RUNNING','start_utc':beg};rc=0
 try:r=main();status.update(execution_status='COMPLETED',EXP006_entry_met=r['EXP006_entry_met'])
 except BaseException as e:status.update(execution_status='FAILED',error=repr(e),traceback=traceback.format_exc());traceback.print_exc();rc=1
 finally:status.update(end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu);writejson(ROOT/'experiments/ROUND-007/execution_status.json',status);print(json.dumps(status,indent=2),flush=True)
 sys.exit(rc)

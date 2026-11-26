"""Preregistered ROUND-006 real-network diagnostics; no training, no test set."""
import os,sys,json,csv,time,datetime,hashlib,contextlib,traceback,math,subprocess,random
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/EXP-004/controlled'))
from core import model_create,isolated_eval,transforms_pair,DATA,ATLIF
from models.submodules.layers import Surrogate
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset,DataLoader
from torch.nn import functional as F
MODEL=ROOT/'experiments/EXP-004/runs/EXP-004-RUN-003/checkpoint_best_validation.pth';SPLIT=ROOT/'experiments/EXP-004/splits/indices.json'
C_SHA='c068f54379cb5d29bcfa5fe2d9de676cd3cc153433ada1ed5aa1c66355d0b32a';S_SHA='69dccfe992236766aea575a0a66495802ca889ca329d61f51e8de3c1ab7f3165'
TARGET='conv3_x.1.residual_function.2'; LAYERS=['conv1.2',TARGET,'conv4_x.2.act'];ALPHAS=[1.,1.25,1.5,2.];TOL=1e-5
OUT=ROOT/'experiments/EXP-002/runs/EXP-002-RUN-001';OUT3=ROOT/'experiments/EXP-003/runs/EXP-003-RUN-001'; SMOKE=ROOT/'experiments/ROUND-006/SMOKE-001'
start=time.monotonic();cpu=time.process_time(); started=datetime.datetime.now(datetime.timezone.utc).isoformat()
def guard():
 if time.monotonic()-start>1700:raise TimeoutError('GPU wall-budget safety limit')
 if time.process_time()-cpu>1700:raise TimeoutError('CPU budget safety limit')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def writejson(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def writecsv(p,rows):
 if not rows:return
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def appendcsv(p,rows):
 if not rows:return
 with p.open('a',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
  if f.tell()==0:w.writeheader()
  w.writerows(rows);f.flush();os.fsync(f.fileno())
@contextlib.contextmanager
def capture(model,name,alpha=1.,coeff=None):
 """Capture original output before any intervention, restoring hook & function even on error."""
 mod=dict(model.named_modules())[name];seen={'m':[],'calls':0};active=[False];desc=Surrogate.__dict__['forward'];original=Surrogate.forward
 def surrogate(ctx,input,thre,sp):
  if active[0]:seen['m'].append(input.detach().clone())
  return original(ctx,input,thre,sp)
 def pre(m,args):
  seen['calls']+=1;seen['x']=args[0].detach().clone();active[0]=True
 def post(m,args,out):
  active[0]=False;seen['o']=out.detach().clone()
  return out if coeff is None else out*coeff.unsqueeze(0)
 with isolated_eval(model):
  with torch.no_grad():mod.thresh.mul_(alpha)
  seen['theta']=float(mod.thresh.item());seen['tau']=float(mod.tau)
  h1=mod.register_forward_pre_hook(pre);h2=mod.register_forward_hook(post);Surrogate.forward=staticmethod(surrogate)
  try:yield seen
  finally:h1.remove();h2.remove();Surrogate.forward=desc

def run(model,x,name,alpha=1.,coeff=None):
 guard()
 with capture(model,name,alpha,coeff) as seen:logits=model(x).detach().clone()
 if seen['calls']!=1 or len(seen['m'])!=8:raise RuntimeError('target calls or time dimension mismatch')
 seen['m']=torch.stack(seen['m']);seen['s']=(seen['o']>0)
 if not (torch.isfinite(logits).all() and torch.isfinite(seen['x']).all() and torch.isfinite(seen['m']).all() and torch.isfinite(seen['o']).all() and seen['theta']>0):raise RuntimeError('non-finite state or non-positive threshold')
 if not torch.equal(seen['s'],seen['m']>=seen['theta']):raise RuntimeError('spikes disagree with observed membrane and threshold')
 return logits,seen

def trajectory(d):
 x=d['x'].to(torch.float64);m=d['m'].to(torch.float64);o=d['o'].to(torch.float64);s=d['s'];u=(~s)*m;prev=torch.cat([torch.zeros_like(u[:1]),u[:-1]],0)
 X=x.sum(0);O=o.sum(0);L=((1-d['tau'])*prev).sum(0);D=(s*(m-d['theta'])).sum(0);R=u[-1];e=(X-O-L-D-R).abs()/(1+X.abs()+O.abs()+L.abs()+D.abs()+R.abs())
 return {'X':X,'O':O,'L':L,'D':D,'R':R,'r1':e,'r1_residual':X-O-L-D-R,'u':u,'I':x.abs().sum((0,*range(2,x.ndim))),'K':s.sum(0)}
def check_reference(model,x,name,ref,logits):
 # Baseline with only output hook, not wrapped. Eval snapshot restores all state.
 mod=dict(model.named_modules())[name];got=[]
 with isolated_eval(model):
  hook=mod.register_forward_hook(lambda m,a,o:got.append(o.detach().clone()))
  try:plain=model(x).detach().clone()
  finally:hook.remove()
 if len(got)!=1 or not torch.equal(got[0]>0,ref['s']) or not torch.allclose(plain,logits,atol=1e-5,rtol=0) or not torch.allclose(got[0],ref['o'],atol=1e-5,rtol=0):raise RuntimeError('alpha1 no-capture control differs')
 # Replay real effective input in independently initialized actual ATLIF.
 local=ATLIF(T=8,thresh=ref['theta'],tau=ref['tau'],sp=0.).to(x.device).eval()
 with torch.no_grad():replayed=local(ref['x'])
 replay_diff=(replayed-ref['o']).abs().max().item()
 if replay_diff>1e-5:raise RuntimeError('actual neuron local replay differs')
 # Reconstructed state provides a separate next-step charge check for t>0.
 u=(~ref['s'])*ref['m'];res=(ref['m'][1:]-(ref['tau']*u[:-1]+ref['x'][1:])).abs().max().item()
 if res>1e-5:raise RuntimeError('next-step charge reconstruction mismatch')
 return {'baseline_logit_max_diff':(plain-logits).abs().max().item(),'replay_output_max_diff':replay_diff,'reconstructed_charge_max_diff':res}

def bootstrap(rows,refrows):
 n=len(rows);gen=np.random.default_rng(2027);ix=gen.integers(0,n,(2000,n));out={}
 def vals(key,source=rows):return np.array([float(r[key]) if r[key] not in ('',None) else np.nan for r in source],dtype=np.float64)
 def stat(name,a):
  valid=a[np.isfinite(a)];out[name]={'estimate':None,'ci95':None,'invalid_bootstrap':int(len(a)-len(valid))}
  if len(valid):out[name]['ci95']=[float(v) for v in np.quantile(valid,[.025,.975])]
  return out[name]
 e=vals('E');er=stat('mean_E',np.nanmean(e[ix],axis=1));er['estimate']=float(np.nanmean(e)) if np.isfinite(e).any() else None
 delta=vals('loss')-vals('loss',refrows);d=stat('delta_loss',np.mean(delta[ix],axis=1));d['estimate']=float(delta.mean())
 a=vals('spike_count');b=vals('spike_count',refrows);na=vals('neuron_time_states');nb=vals('neuron_time_states',refrows)
 with np.errstate(divide='ignore',invalid='ignore'):
  ba=a[ix].sum(1)/na[ix].sum(1);bb=b[ix].sum(1)/nb[ix].sum(1);rr=(bb-ba)/bb;point=(b.sum()/nb.sum()-a.sum()/na.sum())/(b.sum()/nb.sum())
 k=stat('R_FR',rr);k['estimate']=float(point) if np.isfinite(point) else None
 return out

def metrics_rows(ref,a,labels,ids,name,alpha,ref_logits,logits):
 if not torch.equal(ref['x'],a['x']):raise RuntimeError('target effective input changed')
 tr=trajectory(ref);ta=trajectory(a);dO=ta['O']-tr['O'];dL=ta['L']-tr['L'];dD=ta['D']-tr['D'];dR=ta['R']-tr['R'];r2raw=dO+dL+dD+dR;r2=r2raw.abs()/(1+dO.abs()+dL.abs()+dD.abs()+dR.abs())
 if max(tr['r1'].max().item(),ta['r1'].max().item(),r2.max().item())>TOL:raise RuntimeError('R1/R2 tolerance failure')
 loss=F.cross_entropy(logits.mean(0),labels,reduction='none');sloss=F.cross_entropy(ref_logits.mean(0),labels,reduction='none');rows=[]
 for b in range(len(ids)):
  iabs=tr['I'][b].item();err=dO[b].abs().sum().item();oref=tr['O'][b].abs().sum().item();sref=ref['s'][:,b];ss=a['s'][:,b];n=sref.numel();silent=((tr['K'][b]>0)&(ta['K'][b]==0)).sum().item();active=(tr['K'][b]>0).sum().item()
  row={'sample_id':int(ids[b]),'module_name':name,'alpha':alpha,'seed':2026,'spike_count':int(ss.sum().item()),'neuron_time_states':n,'fr':float(ss.sum().item()/n),'reference_spike_count':int(sref.sum().item()),'output_error_numerator':err,'input_abs_denominator':iabs,'reference_output_abs_denominator':oref,'E':err/iabs if iabs else '', 'E_reference':err/oref if oref else '', 'delta_output_signed_sum':dO[b].sum().item(),'flip_1to0':int((sref&~ss).sum().item()),'flip_0to1':int((~sref&ss).sum().item()),'silent_numerator':int(silent),'silent_denominator':int(active),'loss':loss[b].item(),'reference_loss':sloss[b].item(),'prediction':int(logits.mean(0)[b].argmax()),'correct':int(logits.mean(0)[b].argmax()==labels[b]),'r1_max':max(tr['r1'][b].max().item(),ta['r1'][b].max().item()),'r2_max':r2[b].max().item(),'r1_residual_signed_mean':ta['r1_residual'][b].mean().item(),'r1_residual_abs_mean':ta['r1_residual'][b].abs().mean().item(),'r2_residual_signed_mean':r2raw[b].mean().item(),'r2_residual_abs_mean':r2raw[b].abs().mean().item(),'input_equal':True}
  for key,val in [('D_leak',ta['L']),('D_reset',ta['D']),('D_res',ta['R']),('delta_O',dO),('delta_D_leak',dL),('delta_D_reset',dD),('delta_D_res',dR)]:
   v=val[b].flatten();q=torch.quantile(v,torch.tensor([.5,.9,.95],device=v.device,dtype=v.dtype));row.update({key+'_signed_mean':v.mean().item(),key+'_abs_mean':v.abs().mean().item(),key+'_median':q[0].item(),key+'_p90':q[1].item(),key+'_p95':q[2].item()})
  rows.append(row)
 return rows
def counterfactual(model,loader,ids,existing):
 OUT3.mkdir(parents=True,exist_ok=False);rows=[];permgen=torch.Generator(device='cpu').manual_seed(2028);offset=0
 for x,y in loader:
  guard();batch_ids=ids[offset:offset+len(y)];offset+=len(y);x=x.cuda();y=y.cuda()
  z0,ref=run(model,x,TARGET,1.);z1,a=run(model,x,TARGET,1.5);tr=trajectory(ref);ta=trajectory(a)
  for b,sid in enumerate(batch_ids):
   old=existing[(int(sid),1.5)];
   if abs(float(old['loss'])-F.cross_entropy(z1.mean(0),y,reduction='none')[b].item())>1e-5:raise RuntimeError('G1 recomputation inconsistent')
  eligible=(tr['K']>0)&(ta['K']>0)
  if ((ta['O']<=0)&eligible).any() or not torch.isfinite(ta['O']).all():raise RuntimeError('invalid eligible cumulative output')
  ratio=torch.ones_like(ta['O']);ratio[eligible]=tr['O'][eligible]/ta['O'][eligible];clipped=ratio.clamp(.25,4.);c=torch.where(eligible,clipped,torch.ones_like(clipped)).float();c3=c.clone();changed=[];eligible_n=[];clip_n=[];valid_perm=[]
  for b in range(len(y)):
   mask=eligible[b].flatten();idx=mask.nonzero().flatten();flat=c[b].flatten();dest=c3[b].flatten();n=idx.numel();eligible_n.append(n);clip_n.append(int(((ratio[b].flatten()[idx]<.25)|(ratio[b].flatten()[idx]>4.)).sum().item()));valid_perm.append(n>=2)
   if n>=2:
    pi=torch.randperm(n,generator=permgen).to(idx.device);dest[idx]=flat[idx[pi]]
   changed.append(int((dest[idx]!=flat[idx]).sum().item()))
  z2,g2=run(model,x,TARGET,1.5,c);z3,g3=run(model,x,TARGET,1.5,c3)
  for g in (g2,g3):
   if not torch.equal(a['x'],g['x']) or not torch.equal(a['s'],g['s']) or (a['m']-g['m']).abs().max().item()>TOL or ((~a['s'])*a['m']-(~g['s'])*g['m']).abs().max().item()>TOL:raise RuntimeError('counterfactual module trajectory mismatch')
  zs=[z0,z1,z2,z3];outs=[tr['O'],ta['O'],(g2['o']*c.unsqueeze(0)).to(torch.float64).sum(0),(g3['o']*c3.unsqueeze(0)).to(torch.float64).sum(0)]
  losses=[F.cross_entropy(z.mean(0),y,reduction='none') for z in zs]
  for b,sid in enumerate(batch_ids):
   denom=tr['I'][b].item();floor=(ta['O'][b]-tr['O'][b]).abs()[~eligible[b]].sum().item()/denom if denom else None
   e=[(o[b]-tr['O'][b]).abs().sum().item()/denom if denom else None for o in outs]
   if e[2] is not None and e[2]+1e-6<floor:raise RuntimeError('R14 floor violation')
   refactive=(tr['K'][b]>0).sum().item();alln=tr['K'][b].numel();silent=((tr['K'][b]>0)&(ta['K'][b]==0)).sum().item();newactive=((tr['K'][b]==0)&(ta['K'][b]>0)).sum().item()
   cv=c[b].flatten()[eligible[b].flatten()].to(torch.float64);q=torch.quantile(cv,torch.tensor([.0,.5,.9,.95,1.],device=cv.device,dtype=cv.dtype)) if cv.numel() else None
   row={'coefficient_hist_16':json.dumps(torch.histc(cv.float(),bins=16,min=.25,max=4.).to(torch.int64).cpu().tolist()),'coefficient_mean':cv.mean().item() if cv.numel() else '', 'coefficient_min':q[0].item() if q is not None else '', 'coefficient_median':q[1].item() if q is not None else '', 'coefficient_p90':q[2].item() if q is not None else '', 'coefficient_p95':q[3].item() if q is not None else '', 'coefficient_max':q[4].item() if q is not None else '', 'sample_id':int(sid),'input_abs_denominator':denom,'E_floor':floor if floor is not None else '', 'eligible_count':eligible_n[b],'total_neurons':alln,'reference_active':int(refactive),'newly_silent':int(silent),'newly_active_from_silent':int(newactive),'clipped_count':clip_n[b],'permutation_changed_count':changed[b],'permutation_valid':valid_perm[b]}
   for j in range(4):row.update({f'G{j}_loss':losses[j][b].item(),f'G{j}_correct':int(zs[j].mean(0)[b].argmax()==y[b]),f'G{j}_E':e[j] if e[j] is not None else '',f'G{j}_spikes':int((ref['s'] if j==0 else a['s'])[:,b].sum().item())})
   rows.append(row)
  appendcsv(OUT3/'per_sample.csv',rows[-len(y):])
 n=len(rows);idx=np.random.default_rng(2027).integers(0,n,(2000,n));
 def arr(k):return np.array([float(r[k]) if r[k]!='' else np.nan for r in rows])
 e1,e2,floor=arr('G1_E'),arr('G2_E'),arr('E_floor');l0,l1,l2,l3=[arr(f'G{j}_loss') for j in range(4)]
 def ci(x):
  v=x[np.isfinite(x)];return {'ci95':[float(z) for z in np.quantile(v,[.025,.975])] if len(v) else None,'invalid_bootstrap':int(len(x)-len(v))}
 with np.errstate(invalid='ignore',divide='ignore'):
  re=1-np.nanmean(e2)/np.nanmean(e1);rmax=1-np.nanmean(floor)/np.nanmean(e1);rl=(l1.mean()-l2.mean())/(l1.mean()-l0.mean());b_re=1-np.nanmean(e2[idx],1)/np.nanmean(e1[idx],1);den=(l1[idx]-l0[idx]).mean(1);b_rl=np.where(den>0,(l1[idx]-l2[idx]).mean(1)/den,np.nan)
 if re>rmax+1e-5:raise RuntimeError('R14 upper-bound violation')
 summary={'n':n,'groups':{f'G{j}':{'mean_loss':float(arr(f'G{j}_loss').mean()),'accuracy':float(arr(f'G{j}_correct').mean()),'mean_E':float(np.nanmean(arr(f'G{j}_E')))} for j in range(4)},'R_E':float(re),'R_E_CI':ci(b_re),'R_loss':float(rl) if l1.mean()-l0.mean()>=.02 else None,'R_loss_CI':ci(b_rl),'gain_G2_vs_G1':float((l1-l2).mean()),'gain_G2_vs_G1_CI':ci((l1[idx]-l2[idx]).mean(1)),'gain_G2_vs_G3':float((l3-l2).mean()),'gain_G2_vs_G3_CI':ci((l3[idx]-l2[idx]).mean(1)),'R_E_max':float(rmax),'mean_E_floor':float(np.nanmean(floor)),'eligible_ratio':float(sum(r['eligible_count'] for r in rows)/sum(r['total_neurons'] for r in rows)),'coverage_ref_active':float(sum(r['eligible_count'] for r in rows)/sum(r['reference_active'] for r in rows)),'newly_silent_ratio':float(sum(r['newly_silent'] for r in rows)/sum(r['reference_active'] for r in rows)),'clipped_ratio_eligible':float(sum(r['clipped_count'] for r in rows)/sum(r['eligible_count'] for r in rows)),'permutation_changed_ratio_eligible':float(sum(r['permutation_changed_count'] for r in rows)/sum(r['eligible_count'] for r in rows)),'permutation_invalid_samples':int(sum(not r['permutation_valid'] for r in rows)),'module_trajectories_match':True,'bootstrap_seed':2027,'permutation_seed':2028}
 summary['H002_gate']=bool(summary['R_E']>=.5 and summary['R_loss'] is not None and summary['R_loss']>=.25 and summary['gain_G2_vs_G1_CI']['ci95'][0]>0 and summary['gain_G2_vs_G3_CI']['ci95'][0]>0 and summary['permutation_invalid_samples']==0)
 writejson(OUT3/'metrics.json',summary);return summary
def figures2(rows,summaries):
 fig=OUT/'figures';fig.mkdir(exist_ok=True);data=[summaries[str(a)] for a in ALPHAS]
 for name,key,label in [('alpha_FR','FR','FR'),('alpha_E','mean_E','mean E'),('alpha_loss','delta_loss','mean delta loss'),('alpha_flips','flip_rate','flip rate'),('alpha_silent','silent_ratio','newly silent ratio')]:
  plt.figure();plt.plot(ALPHAS,[r[key] if r[key] is not None else np.nan for r in data],marker='o');plt.xlabel('alpha');plt.ylabel(label);plt.tight_layout();plt.savefig(fig/(name+'.png'));plt.close()
 plt.figure(figsize=(8,4))
 for field in ['delta_O','delta_D_leak','delta_D_reset','delta_D_res']:
  vals=[np.mean([float(r[field+'_signed_mean']) for r in rows if r['alpha']==a]) for a in ALPHAS];plt.plot(ALPHAS,vals,label=field)
 plt.legend();plt.xlabel('alpha');plt.ylabel('per-neuron signed mean');plt.tight_layout();plt.savefig(fig/'decomposition_signed.png');plt.close()
 plt.figure(figsize=(8,4))
 for field in ['delta_O','delta_D_leak','delta_D_reset','delta_D_res']:
  vals=[np.mean([float(r[field+'_abs_mean']) for r in rows if r['alpha']==a]) for a in ALPHAS];plt.plot(ALPHAS,vals,label=field)
 plt.legend();plt.xlabel('alpha');plt.ylabel('per-neuron absolute mean');plt.tight_layout();plt.savefig(fig/'decomposition_absolute.png');plt.close()
 one=[r for r in rows if r['alpha']==1.5 and r['E']!=''];plt.figure();plt.scatter([r['E'] for r in one],[r['loss']-r['reference_loss'] for r in one],s=8);plt.xlabel('E');plt.ylabel('paired delta cross entropy');plt.tight_layout();plt.savefig(fig/'E_vs_loss.png');plt.close()
def figures3(rows,summary):
 fig=OUT3/'figures';fig.mkdir(exist_ok=True)
 for name,key in [('loss','mean_loss'),('accuracy','accuracy'),('E','mean_E')]:
  plt.figure();plt.bar(['G0','G1','G2','G3'],[summary['groups'][f'G{i}'][key] for i in range(4)]);plt.ylabel(key);plt.tight_layout();plt.savefig(fig/(name+'.png'));plt.close()
 plt.figure();plt.bar(['R_E','R_E_max'],[summary['R_E'],summary['R_E_max']]);plt.tight_layout();plt.savefig(fig/'recovery_bound.png');plt.close()
 plt.figure();v=[float(r['coefficient_mean']) for r in rows if r['coefficient_mean']!=''];plt.hist(v,bins=30);plt.xlabel('mean eligible coefficient per sample');plt.tight_layout();plt.savefig(fig/'coefficients_sample_means.png');plt.close()
 plt.figure();plt.hist([r['G1_loss']-r['G2_loss'] for r in rows],bins=30);plt.xlabel('paired G1 minus G2 loss');plt.tight_layout();plt.savefig(fig/'paired_gain.png');plt.close()
 hist=np.sum([json.loads(r['coefficient_hist_16']) for r in rows],axis=0);plt.figure();plt.bar(np.linspace(.25,4.,17)[:-1],hist,width=3.75/16,align='edge');plt.xlabel('eligible compensation coefficient');plt.ylabel('count');plt.tight_layout();plt.savefig(fig/'coefficient_distribution.png');plt.close()
 plt.figure();plt.bar(['eligible/all','coverage/ref','clipped/eligible','silent/ref'],[summary['eligible_ratio'],summary['coverage_ref_active'],summary['clipped_ratio_eligible'],summary['newly_silent_ratio']]);plt.xticks(rotation=20);plt.tight_layout();plt.savefig(fig/'coverage.png');plt.close()
 plt.figure();plt.bar(['R_E','R_loss' if summary['R_loss'] is not None else 'R_loss NA'],[summary['R_E'],summary['R_loss'] or 0]);plt.tight_layout();plt.savefig(fig/'recovery_ratios.png');plt.close()
 plt.figure();vals=[summary['gain_G2_vs_G1'],summary['gain_G2_vs_G3']];cis=[summary['gain_G2_vs_G1_CI']['ci95'],summary['gain_G2_vs_G3_CI']['ci95']];plt.errorbar([0,1],vals,yerr=[[vals[i]-cis[i][0] for i in range(2)],[cis[i][1]-vals[i] for i in range(2)]],fmt='o',capsize=4);plt.xticks([0,1],['G2 vs G1','G2 vs G3']);plt.axhline(0,color='black',linewidth=.5);plt.tight_layout();plt.savefig(fig/'paired_gain_ci.png');plt.close()
def main():
 if torch.cuda.device_count()!=1:raise RuntimeError('exactly one GPU must be visible')
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 if sha(MODEL)!=C_SHA or sha(SPLIT)!=S_SHA:raise RuntimeError('checkpoint or split checksum mismatch')
 cp=torch.load(MODEL,map_location='cpu',weights_only=False)
 if cp.get('epoch')!=193 or cp.get('code_sha')!='03def08d9fab5c37968a2a2eaa37675666cb4110':raise RuntimeError('checkpoint epoch or training code mismatch')
 config=cp.get('config',{});required={'model':'ResNet20','T':8,'eta':.002,'eta2':0.,'seed':42}
 if any(config.get(k)!=v for k,v in required.items()) or cp.get('split_sha256')!=S_SHA:raise RuntimeError('checkpoint config or split mismatch')
 model=model_create().cuda();model.load_state_dict(cp['model']);model.eval()
 for name,p in model.named_parameters():
  if not torch.isfinite(p).all() or (name.endswith('thresh') and (p<=0).any()):raise RuntimeError('nonfinite model or invalid threshold')
 data=json.loads(SPLIT.read_text());ids=data['diagnostic_256'];pool=set(data['diagnostic']);assert len(ids)==len(set(ids))==256 and set(ids)<=pool
 ds=CIFAR10(DATA,train=True,transform=transforms_pair()[1],download=False);assert ds._check_integrity()
 loader=DataLoader(Subset(ds,ids),batch_size=32,shuffle=False,num_workers=0)
 OUT.mkdir(parents=True,exist_ok=False);SMOKE.mkdir(parents=True,exist_ok=False)
 info={'checkpoint_sha256':C_SHA,'checkpoint_epoch':194,'training_code_sha':cp['code_sha'],'split_sha256':S_SHA,'sample_indices':ids,'batch_size':32,'dtype':'float32','T':8,'seed':2026,'bootstrap_seed':2027,'bootstrap_replicates':2000,'gpu':torch.cuda.get_device_name(0),'gpu_uuid_env':os.environ.get('CUDA_VISIBLE_DEVICES'),'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'start_utc':started,'theory_version':'v0.1.2'};writejson(OUT/'run_metadata.json',info)
 x,y=next(iter(loader));x=x.cuda();y=y.cuda();order=[];hooks=[]
 for name,m in model.named_modules():
  if isinstance(m,ATLIF):hooks.append(m.register_forward_hook(lambda mod,a,o,n=name:order.append(n)))
 try:
  with isolated_eval(model):model(x[:1])
 finally:
  for h in hooks:h.remove()
 if len(order)!=19 or len(set(order))!=19 or [order[i] for i in (0,9,18)]!=LAYERS:raise RuntimeError('execution order mismatch')
 # Snapshot all externally visible diagnostic state, model buffers, parameters and RNG.
 modes=[m.training for m in model.modules()];before={k:v.detach().clone() for k,v in model.state_dict().items()};attrs={n:(m.update_value,m.r,m.s,m.act_value) for n,m in model.named_modules() if isinstance(m,ATLIF)};rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone();orig=Surrogate.__dict__['forward'];target=dict(model.named_modules())[TARGET];hooks_before=(len(target._forward_pre_hooks),len(target._forward_hooks))
 z,ref=run(model,x[:1],TARGET,1.);checks=check_reference(model,x[:1],TARGET,ref,z)
 try:
  with capture(model,TARGET,1.):raise ValueError('intentional restoration test')
 except ValueError:pass
 state_equal=all(torch.equal(before[k],v) for k,v in model.state_dict().items()) and modes==[m.training for m in model.modules()] and all(attrs[n]==(m.update_value,m.r,m.s,m.act_value) for n,m in model.named_modules() if isinstance(m,ATLIF)) and torch.equal(rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state()) and orig is Surrogate.__dict__['forward'] and hooks_before==(len(target._forward_pre_hooks),len(target._forward_hooks))
 if not state_equal:raise RuntimeError('model state/hook/RNG restoration failed')
 zhi,hi=run(model,x[:1],TARGET,1.5)
 if not torch.equal(ref['x'],hi['x']):raise RuntimeError('smoke input changed')
 checks.update(order=order,target_input_shape=list(ref['x'].shape),target_time_steps=len(ref['m']),state_and_exception_restored=state_equal,effective_input_equal=True,raw_spikes_match=True,derived_u_is_reconstructed=True);writejson(SMOKE/'metrics.json',checks)
 all_by_layer={};main_stats=None
 # Primary first; supplementary layers are deferred until after the EXP-003 decision.
 def one_layer(name,path):
  allrows=[];offset=0
  for x,y in loader:
   guard();batch_ids=ids[offset:offset+len(y)];offset+=len(y);x=x.cuda();y=y.cuda();zref,ref=run(model,x,name,1.)
   for alpha in ALPHAS:
    z,a=(zref,ref) if alpha==1. else run(model,x,name,alpha)
    batch=metrics_rows(ref,a,y,batch_ids,name,alpha,zref,z);allrows.extend(batch);appendcsv(path,batch)
   print(json.dumps({'event':'EXP-002_batch','layer':name,'samples_done':offset}),flush=True)
  result={};refs=[r for r in allrows if r['alpha']==1.]
  for alpha in ALPHAS:
   current=[r for r in allrows if r['alpha']==alpha];st=bootstrap(current,refs);den=sum(r['neuron_time_states'] for r in current);sil=sum(r['silent_denominator'] for r in current)
   result[str(alpha)]={'FR':sum(r['spike_count'] for r in current)/den,'R_FR':st['R_FR'],'mean_E':st['mean_E']['estimate'],'mean_E_CI':st['mean_E']['ci95'],'E_NA_count':sum(r['E']=='' for r in current),'delta_loss':st['delta_loss']['estimate'],'delta_loss_CI':st['delta_loss']['ci95'],'accuracy':sum(r['correct'] for r in current)/len(current),'r1_max':max(r['r1_max'] for r in current),'r2_max':max(r['r2_max'] for r in current),'flip_rate':sum(r['flip_1to0']+r['flip_0to1'] for r in current)/den,'flip_1to0_count':sum(r['flip_1to0'] for r in current),'flip_0to1_count':sum(r['flip_0to1'] for r in current),'silent_ratio':sum(r['silent_numerator'] for r in current)/sil if sil else None,'bootstrap':st}
  return allrows,result
 rows,main_stats=one_layer(TARGET,OUT/'per_sample.csv');all_by_layer[TARGET]=main_stats;figures2(rows,main_stats)
 a=main_stats['1.5'];h1=bool(a['R_FR']['estimate'] is not None and a['R_FR']['estimate']>=.2 and a['mean_E'] is not None and a['mean_E']>=.05 and a['mean_E_CI'][0]>.05 and a['r1_max']<=TOL and a['r2_max']<=TOL);task_gate=bool(a['delta_loss']>=.02 and a['delta_loss_CI'][0]>0);gate=h1 and task_gate
 main_out={'primary_module':TARGET,'primary_alpha':1.5,'conditions':main_stats,'H001_gate_met':h1,'EXP003_entry_met':gate,'entry_loss_gate_met':task_gate,'bootstrap_seed':2027,'bootstrap_replicates':2000,'n_samples':256,'smoke':checks};writejson(OUT/'metrics.json',main_out)
 writecsv(OUT/'summary.csv',[{'module':TARGET,'alpha':a,**{k:v for k,v in main_stats[str(a)].items() if k not in ['bootstrap','R_FR']},'R_FR':main_stats[str(a)]['R_FR']['estimate']} for a in ALPHAS])
 if gate:
  existing={(r['sample_id'],r['alpha']):r for r in rows};s3=counterfactual(model,loader,ids,existing)
  with (OUT3/'per_sample.csv').open(newline='') as f:r3=list(csv.DictReader(f))
  for r in r3:
   for k in list(r):
    if k!='coefficient_hist_16' and r[k] not in ('','True','False'):r[k]=float(r[k])
  figures3(r3,s3)
 else:writejson(OUT/'exp003_gate.json',{'status':'BLOCKED','reason':'original EXP-002 gate not met','H001_gate_met':h1,'loss_gate_met':task_gate})
 for name in [LAYERS[0],LAYERS[2]]:
  if time.monotonic()-start>1300 or time.process_time()-cpu>1300:print('supplementary omitted due to budget',name,flush=True);continue
  _,result=one_layer(name,OUT/('supplementary_'+name.replace('.','_')+'.csv'));all_by_layer[name]=result
 writejson(OUT/'layer_summary.json',all_by_layer)
 writecsv(OUT/'layer_summary.csv',[{'module':name,'alpha':alpha,'FR':v['FR'],'R_FR':v['R_FR']['estimate'],'mean_E':v['mean_E'],'delta_loss':v['delta_loss'],'accuracy':v['accuracy'],'silent_ratio':v['silent_ratio'],'r1_max':v['r1_max'],'r2_max':v['r2_max']} for name,d in all_by_layer.items() for alpha,v in d.items()])
 return main_out
if __name__=='__main__':
 result={'execution_status':'RUNNING','start_utc':started};code=0
 try:res=main();result.update(execution_status='COMPLETED',EXP003_entry_met=res['EXP003_entry_met'])
 except BaseException as e:result.update(execution_status='FAILED',error=repr(e),traceback=traceback.format_exc());traceback.print_exc();code=1
 finally:
  result.update(end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu);writejson(ROOT/'experiments/ROUND-006/execution_status.json',result);print(json.dumps(result,indent=2),flush=True)
 sys.exit(code)

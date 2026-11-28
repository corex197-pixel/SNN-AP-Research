"""EXP-020 preregistered, single GPU, local-gradient evaluation; never trains."""
import os,sys,time,json,csv,hashlib,random,resource,signal,traceback,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT);sys.path[:0]=[str(ROOT/'vendor/UniSparse'),str(ROOT/'experiments/ROUND-014'),str(ROOT/'experiments/ROUND-018')]
import numpy as np,torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from spikingjelly.activation_based import functional
from models.cifarnet import cifarnet
from models.submodules.layers import SparseConv2d,SparseNeuron
from controlled_data import TrainFileOnly
from utils.utils import SOPMonitor
import math_core
E=ROOT/'experiments/EXP-020';C=json.loads((E/'config.json').read_text());S=json.loads((E/'data_split.json').read_text());T=C['targets'];Q=C['q'];M=('G0','G1','G2');CODE=os.popen('git rev-parse HEAD').read().strip()
def dump(n,v):
 p=E/n;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def write(n,rows):
 if not rows:return
 p=E/n
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def cpu():
 r=resource.getrusage(resource.RUSAGE_SELF);c=resource.getrusage(resource.RUSAGE_CHILDREN);v=r.ru_utime+r.ru_stime+c.ru_utime+c.ru_stime
 try:
  import psutil
  for p in psutil.Process().children(recursive=True):
   try:z=p.cpu_times();v+=z.user+z.system
   except psutil.Error:pass
 except ImportError:pass
 return v
START=time.time();CPU=cpu();PRIOR_WALL=float(os.environ.get('EXP020_PRIOR_WALL','0'));PRIOR_CPU=float(os.environ.get('EXP020_PRIOR_CPU','0'));status={'status':'RUNNING','run_id':os.environ.get('EXP020_RUN_ID',C['run_id']),'code_sha':CODE,'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),'start_unix':START,'phase':'preflight'}
def guard():
 wall=PRIOR_WALL+time.time()-START;c=PRIOR_CPU+cpu()-CPU
 if wall>1740 or c>3540:raise TimeoutError('resource budget reserve reached')
 status.update(wall_seconds=wall,cpu_seconds=c);dump('status.json',status)
def abort(s,f):raise TimeoutError('watchdog signal '+str(s))
signal.signal(signal.SIGTERM,abort)
def loader(ids):return DataLoader(TrainFileOnly(ids,False),batch_size=64,shuffle=False,num_workers=4,pin_memory=True,drop_last=False)
def get(m,n):return m.get_submodule(n)
def model():
 assert sha(C['parent_path'])==C['parent_sha256'];ck=torch.load(C['parent_path'],map_location='cpu',weights_only=False);assert ck['epoch']==300 and ck['group']=='U2'
 m=cifarnet(T=4,base_channels=256,num_classes=10);m.load_state_dict(ck['model']);del ck;m.cuda().eval()
 for p in m.parameters():p.requires_grad_(False)
 return m

def collect_gradients(m,data):
 grads={n:{k:None for k in ('G0','G1','G2','h')} for n,_,_ in T};ds={};co={};perms={};fanrows=[];coefrows=[];graph=[]
 for idx,(bn,act,conv) in enumerate(T):
  b,n,v=get(m,bn),get(m,act),get(m,conv);assert len(n.next_convs)==1 and n.next_convs[0] is v and len(n.prev_bns)==1 and n.prev_bns[0] is b and v.groups==1 and v.in_channels==b.num_features and n.flatten is None and b.affine
  d=v.sparse_mask().detach().sum((0,2,3)).double();a=math_core.coefficients(d,.25);p=torch.randperm(len(d),generator=torch.Generator().manual_seed(20261128+idx));a2=a[p.to(a.device)];assert torch.equal(a.sort().values,a2.sort().values);d0=v.out_channels*v.kernel_size[0]*v.kernel_size[1];ds[bn]=(d,d0);co[bn]={'G0':torch.ones_like(a),'G1':a,'G2':a2};perms[bn]=p.tolist();graph.append({'bn':bn,'neuron':act,'consumer':conv,'registered_direct':True,'groups':v.groups,'channels':len(d),'nominal':d0,'mean':d.mean().item(),'std':d.std(unbiased=False).item(),'zero_mean_fallback':d.mean().item()==0})
  for j in range(len(d)):
   fanrows.append({'bn':bn,'consumer':conv,'channel':j,'effective_fanout':d[j].item(),'nominal':d0})
   coefrows.append({'bn':bn,'channel':j,**{k:co[bn][k][j].item() for k in M}})
 dump('target_dependency_audit.json',{'status':'PASSED','targets':graph,'scope':'registered paths and static CifarNet.forward; actual hook output compared with consumer input'});dump('permutation.json',perms);write('fanout_vectors.csv',fanrows);write('coefficient_vectors.csv',coefrows)
 replay=[];capture=[];n_total=0
 for batch,(x,y,ids) in enumerate(data):
  guard();x=x.cuda(non_blocking=True);cache={};hs=[]
  for bn,act,conv in T:
   hs.append(get(m,bn).register_forward_pre_hook(lambda mod,args,n=bn:cache.__setitem__(n,args[0].detach().clone())))
   hs.append(get(m,act).register_forward_hook(lambda mod,args,out,n=act:cache.__setitem__(n,out.detach().clone())))
   hs.append(get(m,conv).register_forward_pre_hook(lambda mod,args,n=conv:cache.__setitem__(n,args[0].detach().clone())))
  with torch.no_grad():out=m(x).detach();functional.reset_net(m)
  for h in hs:h.remove()
  if batch==0:
   with torch.no_grad():out2=m(x).detach();functional.reset_net(m)
   capture.append({'capture_does_not_change_output':bool(torch.equal(out,out2))});assert torch.equal(out,out2)
  count=len(x);n_total+=count
  for bn,act,conv in T:
   z=get(m,bn);node=get(m,act).activation;u=cache[bn];target=cache[act];assert torch.equal(target,cache[conv]);node.reset();node.train();w=z.weight.detach().clone().requires_grad_();b=z.bias.detach().clone().requires_grad_();shape=u.shape
   xhat=F.batch_norm(u.flatten(0,1),z.running_mean,z.running_var,w,b,False,z.momentum,z.eps).view(shape);sp=node(xhat);assert torch.equal(sp.detach(),target);replay.append({'batch':batch,'bn':bn,'exact_spike_match':True});rate=sp.mean(dim=(0,1,3,4));d,d0=ds[bn]
   vals={k:d0*(rate*co[bn][k].to(rate.dtype)).sum() for k in M};vals['h']=(rate*d.to(rate.dtype)).sum()
   for k in ('G0','G1','G2','h'):
    gg=torch.autograd.grad(vals[k],(w,b),retain_graph=k!='h');g=torch.cat([v.detach().double() for v in gg]);grads[bn][k]=g*count if grads[bn][k] is None else grads[bn][k]+g*count
   node.reset();node.eval();del xhat,sp,rate,vals,w,b,u,target
  del cache,x,out
 assert n_total==512
 stats=[];matched={};formula=[]
 for bn,_,_ in T:
  g={k:v/n_total for k,v in grads[bn].items()};d,d0=ds[bn];expected=.75*g['G0']+.25*d0/d.mean()*g['h'] if d.mean()>0 else g['G0'];err=torch.linalg.vector_norm(g['G1']-expected)/(torch.linalg.vector_norm(expected)+1e-30);assert err.item()<1e-5
  formula.append({'bn':bn,'R78_relative_error':err.item(),'passed':True});matched[bn]={}
  for k in M:
   v=math_core.match(g[k],g['G0']);matched[bn][k]=v;norm=torch.linalg.vector_norm(g[k]).item();mn=torch.linalg.vector_norm(v).item();ref=torch.linalg.vector_norm(g['G0']).item();assert abs(mn-ref)<=1e-5*max(ref,1e-30)
   n=len(d);mag=torch.sqrt(g[k][:n]**2+g[k][n:]**2);cor=spearmanr(d.cpu().numpy(),mag.cpu().numpy()).statistic
   stats.append({'bn':bn,'group':k,'raw_norm':norm,'matched_norm':mn,'cosine_g0':F.cosine_similarity(g[k][None],g['G0'][None]).item() if norm and ref else 0,'fanout_abs_gradient_spearman':float(cor) if np.isfinite(cor) else 'NA'})
 dump('local_replay_tests.json',{'status':'PASSED','replay':replay,'capture':capture,'state_reset_per_forward':True,'local_node_training_for_surrogate_only':True});dump('gradient_formula_tests.json',{'synthetic':math_core.tests(),'real_R78':formula,'status':'PASSED'});write('gradient_statistics.csv',stats)
 return matched

def main():
 guard();assert torch.cuda.device_count()==1 and '4090' in torch.cuda.get_device_name();torch.set_num_threads(2);torch.manual_seed(20261128);np.random.seed(20261128);random.seed(20261128)
 assert sha('experiments/EXP-004/splits/indices.json')==C['split_sha256'];spl=json.loads(Path('experiments/EXP-004/splits/indices.json').read_text());assert len(S['D_grad'])==len(S['D_cal'])==512 and not set(S['D_grad'])&set(S['D_cal']) and not (set(S['D_grad'])|set(S['D_cal']))&(set(spl['validation'])|set(spl['diagnostic']))
 m=model();dump('parent_checkpoint_audit.json',{'before':sha(C['parent_path']),'expected':C['parent_sha256'],'match':True});print('parent loaded',flush=True)
 data=loader(S['D_grad']);g=collect_gradients(m,data);del data
 # Snapshot every parameter and buffer once, including actual masks; only target affine parameters may differ.
 parent={n:v.detach().clone() for n,v in m.state_dict().items()};targetkeys={f'{bn}.{k}' for bn,_,_ in T for k in ('weight','bias')};params=torch.cat([parent[k].double().flatten() for k in sorted(targetkeys)]);normphi=torch.linalg.vector_norm(params);normg=torch.sqrt(sum(torch.linalg.vector_norm(g[bn]['G0'])**2 for bn,_,_ in T));assert torch.isfinite(normphi) and torch.isfinite(normg) and normg>0
 mask0={n:z.sparse_mask().detach().clone() for n,z in m.named_modules() if isinstance(z,SparseConv2d)};nweights=sum(v.numel() for v in mask0.values());nonzero=sum(int(v.sum().item()) for v in mask0.values());integrity=[];normrows=[]
 def intervene(group,q):
  with torch.no_grad():
   for bn,_,_ in T:
    z=get(m,bn);n=z.weight.numel();v=g[bn][group];eta=q*normphi/normg;z.weight.copy_(parent[bn+'.weight']-(eta*v[:n]).to(z.weight.dtype));z.bias.copy_(parent[bn+'.bias']-(eta*v[n:]).to(z.bias.dtype))
   for n,v in m.state_dict().items():
    if n not in targetkeys and not torch.equal(v,parent[n]):raise RuntimeError('unauthorized state change '+n)
    if not torch.isfinite(v).all():raise RuntimeError('nonfinite state '+n)
   for n,z in m.named_modules():
    if isinstance(z,SparseConv2d) and not torch.equal(z.sparse_mask(),mask0[n]):raise RuntimeError('mask change '+n)
   diffs=[]
   for bn,_,_ in T:
    z=get(m,bn);actual=torch.cat([(z.weight-parent[bn+'.weight']).double(),(z.bias-parent[bn+'.bias']).double()]);diffs.append(actual);normrows.append({'group':group,'q':q,'bn':bn,'actual_update_norm':torch.linalg.vector_norm(actual).item(),'planned_update_norm':(eta*torch.linalg.vector_norm(g[bn][group])).item()})
   norm=torch.linalg.vector_norm(torch.cat(diffs)).item();integrity.append({'group':group,'q':q,'non_target_exact':True,'masks_exact':True,'bn_running_exact':True,'finite':True,'update_norm':norm});return norm
 def evaluate(data,group,q,save_samples=False):
  guard();up=intervene(group,q);functional.reset_net(m);mon=SOPMonitor(m);handles=[];layers={n:0. for n in mon.monitored_layers};reference={};spikes={};elements={};channels={};samples=[];correct=0;loss=0.;total=0;batch_sop=[]
  for n,z in m.named_modules():
   if isinstance(z,SparseNeuron):
    def h(mod,args,out,n=n):
     o=out.detach();spikes[n]=spikes.get(n,0)+int((o!=0).sum().item());elements[n]=elements.get(n,0)+o.numel()
     if n in [t[1] for t in T]:
      v=o.sum((0,1,3,4)).double().cpu();channels[n]=channels.get(n,torch.zeros_like(v))+v
    handles.append(z.register_forward_hook(h))
  for n in mon.monitored_layers:
   def check(mod,args,n=n):
    x=args[0];
    if bool(((x!=0)&(x!=1)).any()):raise RuntimeError('nonbinary input '+n)
    if total==0 and group=='G0' and q==0:
     reference[n]=float(F.conv2d(x.flatten(0,1).float(),mod.sparse_mask(),None,mod.stride,mod.padding,mod.dilation,mod.groups).sum(dtype=torch.float64).item())
   handles.append(get(m,n).register_forward_pre_hook(check))
  try:
   with torch.no_grad():
    for x,y,ids in data:
     guard();x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);mon.clear_recorded_data();z=m(x).mean(0);losses=F.cross_entropy(z,y,reduction='none');assert torch.isfinite(losses).all();pred=z.argmax(1);n=len(y);total+=n;correct+=int((pred==y).sum());loss+=float(losses.sum());so=0.
     for name in mon.monitored_layers:
      v=sum(float(t.item()) for t in mon[name]);layers[name]+=v;so+=v
      if name in reference and total==n:
       err=abs(v-reference[name])/(1+abs(reference[name]));assert err<1e-5;reference[name]=err
     batch_sop.append(so);functional.reset_net(m)
     if save_samples:samples.extend({'group':group,'q':q,'sample_id':int(i),'label':int(t),'prediction':int(p),'correct':int(p==t),'ce':float(l)} for i,t,p,l in zip(ids.tolist(),y.cpu().tolist(),pred.cpu().tolist(),losses.cpu().tolist()))
  finally:
   mon.remove_hooks()
   for h in handles:h.remove()
   functional.reset_net(m)
  assert total==len(data.dataset)
  if reference:dump('sop_reference_audit.json',{'status':'PASSED','relative_errors_first_batch':reference,'method':'independent float64 reduction of convolution counts'})
  result={'group':group,'q':q,'samples':total,'accuracy':correct/total,'ce':loss/total,'FR_global':sum(spikes.values())/sum(elements.values()),'SOP_sample_weighted':sum(batch_sop)/total,'SOP_author':sum(batch_sop)/len(batch_sop)/64,'target_sops':(layers[T[0][2]]+layers[T[1][2]])/total,'update_norm':up,'weight_nonzero':nonzero,'weight_sparsity':1-nonzero/nweights}
  lr=[{'group':group,'q':q,'layer':n,'sops':v/total} for n,v in layers.items()];rates=[{'group':group,'q':q,'neuron':n,'channel':i,'rate':float(v[i])/(total*4*32*32)} for n,v in channels.items() for i in range(len(v))]
  return result,lr,rates,samples
 status['phase']='calibration';dump('status.json',status);cal=loader(S['D_cal']);responses=[];layerrows=[];chrows=[]
 b0,lr,cr,_=evaluate(cal,'G0',0);b0['group']='B0';responses.append(b0);layerrows+=lr;chrows+=cr
 for group in M:
  intervene(group,0)
  assert all(torch.equal(v,parent[n]) for n,v in m.state_dict().items())
  responses.append(dict(b0,group=group,q=0))
 for group in M:
  for q in Q[1:]:
   r,lr,cr,_=evaluate(cal,group,q);responses.append(r);layerrows+=lr;chrows+=cr;write('calibration_response.csv',responses);write('channel_activity.csv',chrows);dump('parameter_integrity_audit.json',{'status':'PASSED','checks':integrity});write('update_norm_audit.csv',normrows);print('calibration',group,q,r['SOP_sample_weighted'],flush=True)
 del cal
 for q in Q[1:]:
  for bn,_,_ in T:
   v=[x['actual_update_norm'] for x in normrows if x['q']==q and x['bn']==bn and x['group'] in M];assert len(v)==3 and (max(v)-min(v))/max(max(v),1e-30)<1e-5
 bcost=b0['SOP_sample_weighted'];selections={}
 for group in M:
  candidates=[r for r in responses if r['group']==group and .04<=1-r['SOP_sample_weighted']/bcost<=.06]
  selections[group]=None if not candidates else min(candidates,key=lambda r:(abs(1-r['SOP_sample_weighted']/bcost-.05),r['q']))['q']
 lock={'locked_before_validation':True,'metric':'D_cal actual SOP only','selections':selections,'unmatched':[g for g in M if selections[g] is None]};dump('selection_lock.json',lock)
 primary={r['group']:b0['target_sops']-r['target_sops'] for r in responses if r['q']==.02};gate=primary['G1']>primary['G0'] and primary['G1']>primary['G2'];dump('hypothesis_gate_results.json',{'H019_q':.02,'target_sop_reductions':primary,'direction_passed':gate,'H019_scientific_status':'UNTESTED_pending_theory_review','H017_scientific_status':'UNTESTED'})
 status['phase']='validation';dump('status.json',status);val=loader(spl['validation']);results=[];allsamples=[]
 for group in ['B0']+list(M):
  q=0 if group=='B0' else selections[group]
  if q is None:continue
  r,lr,_,samples=evaluate(val,'G0' if group=='B0' else group,q,True);r['group']=group
  for z in lr:z['group']=group
  for z in samples:z['group']=group
  results.append(r);allsamples.extend(samples);layerrows.extend(lr);write('validation_summary.csv',results);write('validation_per_sample.csv',allsamples);write('layer_sops.csv',layerrows)
 boot={};randomgen=np.random.default_rng(20261128);by={x:{r['sample_id']:r for r in allsamples if r['group']==x} for x in ['B0']+list(M)}
 for other in ['G0','G2']:
  if not by['G1'] or not by[other]:boot[other]={'status':'NA_unmatched_budget'};continue
  ids=sorted(by['G1']);assert ids==sorted(by[other]);da=np.array([by['G1'][i]['correct']-by[other][i]['correct'] for i in ids]);dc=np.array([by['G1'][i]['ce']-by[other][i]['ce'] for i in ids]);ix=randomgen.integers(0,len(ids),size=(2000,len(ids)));boot[other]={'accuracy_delta':float(da.mean()),'accuracy_ci95':np.quantile(da[ix].mean(1),[.025,.975]).tolist(),'ce_delta':float(dc.mean()),'ce_ci95':np.quantile(dc[ix].mean(1),[.025,.975]).tolist()}
 dump('paired_bootstrap.json',{'seed':20261128,'resamples':2000,'comparisons':boot});dump('parameter_integrity_audit.json',{'status':'PASSED','checks':integrity});write('update_norm_audit.csv',normrows)
 dump('parent_checkpoint_audit.json',{'before':C['parent_sha256'],'after':sha(C['parent_path']),'expected':C['parent_sha256'],'match':sha(C['parent_path'])==C['parent_sha256']});assert sha(C['parent_path'])==C['parent_sha256'];return 'COMPLETED'
if __name__ == '__main__':
 try:status['status']=main()
 except BaseException as e:status.update(status='PARTIAL',error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
 finally:
  status.update(end_unix=time.time(),wall_seconds=PRIOR_WALL+time.time()-START,cpu_seconds=PRIOR_CPU+cpu()-CPU);dump('status.json',status);print(json.dumps(status),flush=True)
 if status['status']!='COMPLETED':sys.exit(1)

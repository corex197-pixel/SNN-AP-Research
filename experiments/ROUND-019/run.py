"""EXP-021 finite cost-only matching, selection commit, then paired validation."""
import os,sys,json,csv,hashlib,time,traceback,subprocess,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import core as K
from core import *
from torch.nn import functional as F
status=K.status;status.update(run_id='EXP-021-RUN-001',phase='preflight');K.status=status
CODE=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();status['code_sha']=CODE

def data_audit():
 spl=json.loads((ROOT/'experiments/EXP-004/splits/indices.json').read_text());assert sha(ROOT/'experiments/EXP-004/splits/indices.json')==C['split_sha256']
 expected={'D_grad':'93c4c834c3fd3c8b92a64caa88da8a671e27be23dc2df11999c8e5f3dfbe47d8','D_cal':'b671cbd803b13cc76b02f5db4bad311ef9492276622051df01af56a931301baa'}
 for k,h in expected.items():assert hashlib.sha256(json.dumps(S[k],separators=(',',':')).encode()).hexdigest()==h and len(S[k])==512 and len(set(S[k]))==512
 assert not(set(S['D_grad'])&set(S['D_cal'])) and not (set(S['D_grad'])|set(S['D_cal']))&(set(spl['validation'])|set(spl['diagnostic']))
 dump('data_split_audit.json',{'status':'PASSED','hashes':expected,'validation_count':len(spl['validation']),'disjoint':True,'samples_previously_trained':True});return spl

def main():
 guard();assert torch.cuda.device_count()==1 and '4090' in torch.cuda.get_device_name();torch.set_num_threads(2);torch.manual_seed(20261128);np.random.seed(20261128);random.seed(20261128);spl=data_audit();m=model();dump('parent_checkpoint_audit.json',{'before':sha(C['parent_path']),'expected':C['parent_sha256']})
 g=collect_gradients(m,loader(S['D_grad']));stats=list(csv.DictReader((E/'gradient_statistics.csv').open()));old=list(csv.DictReader((OLD/'gradient_statistics.csv').open()));errors=[]
 for x in stats:
  y=next(v for v in old if v['bn']==x['bn'] and v['group']==x['group'])
  for k in ['raw_norm','matched_norm','cosine_g0']:
   err=abs(float(x[k])-float(y[k]))/max(1,abs(float(y[k])));errors.append({'bn':x['bn'],'group':x['group'],'metric':k,'normalized_error':err});assert err<1e-5
 for file in ['fanout_vectors.csv','coefficient_vectors.csv']:
  a=list(csv.DictReader((E/file).open()));b=list(csv.DictReader((OLD/file).open()));assert a==b
 assert json.loads((E/'permutation.json').read_text())==json.loads((OLD/'permutation.json').read_text());dump('coefficient_audit.json',{'status':'PASSED','fanout_and_coefficients_identical':True,'saved_permutation_identical':True,'gradient_statistics_errors':errors,'elementwise_gradient_comparison_possible':False})
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
  guard();up=intervene(group,q);functional.reset_net(m);mon=SOPMonitor(m);handles=[];layers={n:0. for n in mon.monitored_layers};input_ones={};input_n={};reference={};spikes={};elements={};channels={};samples=[];correct=0;loss=0.;total=0;batch_sop=[]
  for n,z in m.named_modules():
   if isinstance(z,SparseNeuron):
    def h(mod,args,out,n=n):
     o=out.detach();spikes[n]=spikes.get(n,0)+int((o!=0).sum().item());elements[n]=elements.get(n,0)+o.numel()
     if n in [t[1] for t in T]:
      v=o.sum((0,1,3,4)).double().cpu();channels[n]=channels.get(n,torch.zeros_like(v))+v
    handles.append(z.register_forward_hook(h))
  for n in mon.monitored_layers:
   def check(mod,args,n=n):
    x=args[0];input_ones[n]=input_ones.get(n,0)+int(x.sum().item());input_n[n]=input_n.get(n,0)+x.numel()
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
  for n,v in m.state_dict().items():
   if n not in targetkeys and not torch.equal(v,parent[n]):raise RuntimeError('post-evaluation state changed '+n)
  if reference:dump('sop_reference_audit.json',{'status':'PASSED','relative_errors_first_batch':reference,'method':'independent float64 reduction of convolution counts'})
  result={'group':group,'q':q,'samples':total,'accuracy':correct/total,'ce':loss/total,'FR_global':sum(spikes.values())/sum(elements.values()),'SOP_sample_weighted':sum(batch_sop)/total,'SOP_author':sum(batch_sop)/len(batch_sop)/64,'target_sops':(layers[T[0][2]]+layers[T[1][2]])/total,'update_norm':up,'weight_nonzero':nonzero,'weight_sparsity':1-nonzero/nweights}
  lr=[{'group':group,'q':q,'layer':n,'sops':v/total,'input_FR':input_ones[n]/input_n[n]} for n,v in layers.items()];rates=[{'group':group,'q':q,'neuron':n,'channel':i,'rate':float(v[i])/(total*4*32*32)} for n,v in channels.items() for i in range(len(v))]
  return result,lr,rates,samples
 status['phase']='calibration';dump('status.json',status);cal=loader(S['D_cal']);calrows=[];layerrows=[];trace=[]
 oldcal=list(csv.DictReader((OLD/'calibration_response.csv').open()))
 def measure(group,q,method):
  r,lr,cr,_=evaluate(cal,group,q);r.update(method=method);calrows.append(r);layerrows.extend([dict(x,dataset='D_cal') for x in lr]);write('calibration_response.csv',calrows);write('layer_sops.csv',layerrows);write('update_norm_audit.csv',normrows);dump('parameter_integrity_audit.json',{'status':'PASSED','checks':integrity});print('measured',group,q,r['SOP_sample_weighted'],flush=True);return r
 b0=measure('G0',0.,'B0');b0['group']='B0';g1=measure('G1',.04,'anchor');bcost=b0['SOP_sample_weighted'];target=1-g1['SOP_sample_weighted']/bcost;assert .04<=target<=.06
 history=[]
 for x,og,oq in [(b0,'B0',0.),(g1,'G1',.04)]:
  old=next(z for z in oldcal if z['group']==og and float(z['q'])==oq)
  for k in ['accuracy','ce','SOP_sample_weighted','target_sops']:
   diff=abs(float(x[k])-float(old[k]));relative=diff/max(1,abs(float(old[k])));history.append({'group':og,'metric':k,'old':float(old[k]),'now':float(x[k]),'normalized_error':relative});assert relative<1e-5
 dump('historical_reconstruction_audit.json',{'status':'PASSED','calibration_comparison':history,'gradient_statistics':errors,'limitation':'EXP-020 did not save full gradient tensors; elementwise identity to history cannot be asserted'})
 selections={'G1':.04};matched={};
 for group in ['G0','G2']:
  lo=.04;hi=.08;a=measure(group,lo,'lower_bound');b=measure(group,hi,'upper_bound');ra=1-a['SOP_sample_weighted']/bcost;rb=1-b['SOP_sample_weighted']/bcost
  def record(r,method,l,h):
   value=1-r['SOP_sample_weighted']/bcost;err=abs(value-target);trace.append({'group':group,'method':method,'q':r['q'],'sops':r['SOP_sample_weighted'],'reduction':value,'target_reduction':target,'absolute_reduction_error':err,'interval_low':l,'interval_high':h,'within_tolerance':err<=.001 and .04<=value<=.06});write('budget_matching_trace.csv',trace);return value,err
  record(a,'lower_bound',lo,hi);record(b,'upper_bound',lo,hi);candidates=[a,b]
  if ra<=target<=rb and rb>ra:
   q=lo+(target-ra)*(hi-lo)/(rb-ra);r=measure(group,q,'interpolation');candidates.append(r);rv,err=record(r,'interpolation',lo,hi)
   if err>.001 and ra<=rv<=rb:
    if rv<target:lo=q
    else:hi=q
    for i in range(4):
     q=(lo+hi)/2;r=measure(group,q,'bisection_'+str(i+1));candidates.append(r);rv,err=record(r,'bisection_'+str(i+1),lo,hi)
     if err<=.001:break
     if rv<target:lo=q
     else:hi=q
  else:trace.append({'group':group,'method':'non_bracketing_stop','q':'NA','sops':'NA','reduction':'NA','target_reduction':target,'absolute_reduction_error':'NA','interval_low':lo,'interval_high':hi,'within_tolerance':False});write('budget_matching_trace.csv',trace)
  good=[r for r in candidates if .04<=1-r['SOP_sample_weighted']/bcost<=.06 and abs(1-r['SOP_sample_weighted']/bcost-target)<=.001];best=min(good,key=lambda x:(abs(1-x['SOP_sample_weighted']/bcost-target),x['q'])) if good else None;selections[group]=best['q'] if best else None;matched[group]=None if best is None else 1-best['SOP_sample_weighted']/bcost
  print('locked candidate',group,selections[group],flush=True)
 for x in calrows:x['R81_target_sop_per_update']='NA' if not x['update_norm'] else (b0['target_sops']-x['target_sops'])/x['update_norm']
 write('calibration_response.csv',calrows)
 lock={'locked_before_validation':True,'B0_sop':bcost,'G1_anchor_sop':g1['SOP_sample_weighted'],'target_reduction':target,'selections':selections,'matched_reductions':matched,'selection_uses_only':'D_cal SOP error; tie smaller q','gradient_vectors_sha256':sha(E/'gradient_vectors.csv'),'coefficients_sha256':sha(E/'coefficient_vectors.csv'),'permutation_sha256':sha(E/'permutation.json'),'trace_sha256':sha(E/'budget_matching_trace.csv')};dump('selection_lock.json',lock)
 # Commit the actual candidates and their determining evidence before reading validation.
 subprocess.run(['git','add','experiments/EXP-021/selection_lock.json','experiments/EXP-021/budget_matching_trace.csv','experiments/EXP-021/gradient_vectors.csv','experiments/EXP-021/coefficient_vectors.csv','experiments/EXP-021/permutation.json'],check=True)
 subprocess.run(['git','commit','-m','lock(EXP-021): fix matched calibration candidates'],check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);locksha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();guard();dump('lock_commit.json',{'sha':locksha,'before_validation':True});print('selection lock committed',locksha,flush=True)
 status['phase']='validation';dump('status.json',status);val=loader(spl['validation']);results=[];allsamples=[]
 for group in ['B0','G0','G1','G2']:
  q=0 if group=='B0' else selections[group]
  if q is None:continue
  r,lr,_,samples=evaluate(val,'G0' if group=='B0' else group,q,True);r['group']=group
  for z in lr:z.update(group=group,dataset='validation')
  for z in samples:z['group']=group
  results.append(r);allsamples.extend(samples);layerrows.extend(lr);write('validation_summary.csv',results);write('validation_per_sample.csv',allsamples);write('layer_sops.csv',layerrows)
 oldsamples=list(csv.DictReader((OLD/'validation_per_sample.csv').open()));oldmap={(x['group'],int(x['sample_id'])):x for x in oldsamples};newmap={(x['group'],int(x['sample_id'])):x for x in allsamples};compare={}
 for group in ['B0','G1']:
  aa=[x for x in allsamples if x['group']==group];assert len(aa)==4500;diff=0.;mismatch=0
  for x in aa:
   z=oldmap[(group,x['sample_id'])];assert int(z['label'])==x['label'];mismatch+=int(int(z['prediction'])!=x['prediction']);diff=max(diff,abs(float(z['ce'])-x['ce']))
  oldsum=next(x for x in csv.DictReader((OLD/'validation_summary.csv').open()) if x['group']==group);newsum=next(x for x in results if x['group']==group);sdiff=abs(float(oldsum['SOP_sample_weighted'])-newsum['SOP_sample_weighted'])/newsum['SOP_sample_weighted'];compare[group]={'prediction_mismatches':mismatch,'max_absolute_ce_difference':diff,'relative_SOP_difference':sdiff};assert mismatch==0 and diff<1e-5 and sdiff<1e-5
 h=json.loads((E/'historical_reconstruction_audit.json').read_text());h['validation_comparison']=compare;dump('historical_reconstruction_audit.json',h)
 by={group:{r['sample_id']:r for r in allsamples if r['group']==group} for group in ['B0','G0','G1','G2']};rng=np.random.default_rng(20261128);boot={}
 for exp,ref in [('G1','G0'),('G1','G2'),('G1','B0'),('G0','B0'),('G2','B0')]:
  key=exp+'-'+ref
  if not by[exp] or not by[ref]:boot[key]={'status':'NA_unmatched'};continue
  ids=sorted(by[exp]);assert ids==sorted(by[ref]);assert all(by[exp][i]['label']==by[ref][i]['label'] for i in ids)
  da=np.array([by[exp][i]['correct']-by[ref][i]['correct'] for i in ids]);dc=np.array([by[exp][i]['ce']-by[ref][i]['ce'] for i in ids]);ix=rng.integers(0,len(ids),size=(2000,len(ids)));ep=next(z for z in results if z['group']==exp);rp=next(z for z in results if z['group']==ref)
  boot[key]={'status':'COMPLETED','accuracy_delta_pp':100*float(da.mean()),'accuracy_delta_ci95_pp':(100*np.quantile(da[ix].mean(1),[.025,.975])).tolist(),'CE_delta':float(dc.mean()),'CE_delta_ci95':np.quantile(dc[ix].mean(1),[.025,.975]).tolist(),'reference_wrong_experiment_right':int((da==1).sum()),'reference_right_experiment_wrong':int((da==-1).sum()),'prediction_disagree':sum(by[exp][i]['prediction']!=by[ref][i]['prediction'] for i in ids),'SOP_delta':ep['SOP_sample_weighted']-rp['SOP_sample_weighted'],'per_sample_sop':'NA_not_collected'}
 dump('paired_bootstrap.json',{'seed':20261128,'resamples':2000,'rng':'numpy.default_rng PCG64','interval':'percentile','comparisons':boot});dump('parameter_integrity_audit.json',{'status':'PASSED','checks':integrity});write('update_norm_audit.csv',normrows)
 after=sha(C['parent_path']);assert after==C['parent_sha256'];dump('parent_checkpoint_audit.json',{'before':after,'after':after,'expected':C['parent_sha256'],'match':True})
 return 'COMPLETED'
try:status['status']=main()
except BaseException as e:status.update(status='PARTIAL',error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
finally:status.update(end_unix=time.time(),wall_seconds=time.time()-K.START,cpu_seconds=K.cpu()-K.CPU);dump('status.json',status);print(json.dumps(status),flush=True)
if status['status']!='COMPLETED':sys.exit(1)

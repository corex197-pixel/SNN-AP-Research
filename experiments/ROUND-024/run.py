"""ROUND-024: fixed support reconstruction; no SGD. See TASK.md."""
import os,sys,time,json,hashlib,subprocess,traceback,copy,gc,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT)
sys.path[:0]=[str(ROOT/'experiments/ROUND-019'),str(ROOT/'experiments/ROUND-022')]
# Reuse precisely the previously accepted full-network implementation without executing its main.
src=(ROOT/'experiments/ROUND-022/run.py').read_text().split('def main():')[0]
exec(compile(src,str(ROOT/'experiments/ROUND-022/run.py'),'exec'),globals())
E=ROOT/'experiments/EXP-026';K.E=E;K.START=time.time();K.CPU=K.cpu();K.PRIOR_WALL=float(os.environ.get('ROUND024_PRIOR_WALL','0'));K.PRIOR_CPU=float(os.environ.get('ROUND024_PRIOR_CPU_RESERVE','60'))
CODE=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();status.clear();status.update(status='RUNNING',run_id=os.environ.get('ROUND024_RUN_ID','EXP-026-RUN-001'),phase='admission',code_sha=CODE,start_unix=K.START,gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'))
# Conservative bound: whole process wall from launch counts toward GPU budget, including lock push.
def guard(reserve=60):
 wall=K.PRIOR_WALL+time.time()-K.START;cpu=K.PRIOR_CPU+K.cpu()-K.CPU
 status.update(wall_seconds=wall,cpu_budget_accounted_seconds=cpu)
 K.dump('status.json',status)
 if wall>=1200-reserve or cpu>=2700-reserve:raise TimeoutError('budget reserve reached')
K.guard=guard
signal.signal(signal.SIGTERM,lambda a,b:(_ for _ in ()).throw(TimeoutError('supervisor stop')))
def hash_array(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def explicit_test(name,passed,inputs,assertion,actual):
 row={'name':name,'status':'PASSED' if bool(passed) else 'FAILED','inputs':inputs,'assertion':assertion,'actual':actual};extra.append(row);dump('formula_tests.json',{'status':'RUNNING','tests':extra})
 if not passed:raise AssertionError(name)
def solve_support(rate,a,keep,deleted,mask):
 Xall=rate[:,keep].double();err=rate[:,deleted].double()@a[:,deleted].T;delta=torch.zeros((10,len(keep)),device=rate.device,dtype=torch.float64);details=[]
 for k in range(10):
  guard();sel=(mask[k*10:(k+1)*10,:].sum(0)>0);x=Xall[:,sel];e=err[:,k];lam=.001*float((x.square().sum(0)/len(x)).mean()) if x.shape[1] else 0.;t=time.monotonic()
  if x.shape[1] and lam>0:
   if x.shape[1]>len(x):
    mat=x@x.T+len(x)*lam*torch.eye(len(x),device=x.device,dtype=x.dtype);v=x.T@torch.linalg.solve(mat,e);form='dual'
   else:
    mat=x.T@x+len(x)*lam*torch.eye(x.shape[1],device=x.device,dtype=x.dtype);v=torch.linalg.solve(mat,x.T@e);form='primal'
   assert bool(torch.isfinite(v).all());delta[k,sel]=v
   residual=float(torch.linalg.vector_norm(x.T@(x@v-e)+len(x)*lam*v)/(torch.linalg.vector_norm(x.T@e)+1e-30))
  else:form='degenerate_zero';residual=0.
  details.append({'class':k,'support_indices':torch.where(sel)[0].cpu().tolist(),'support_size':int(sel.sum()),'lambda':lam,'form':form,'normal_equation_relative_residual':residual,'solve_seconds':time.monotonic()-t,'finite':bool(torch.isfinite(delta[k]).all())})
 return delta,details,err

def apply_delta(model,delta):
 with torch.no_grad():
  old=model.fc2.sparse_weight().detach().clone();mask=old!=0;theta=model.fc2.threshold.detach();assert torch.isfinite(theta).all() and bool((theta>=0).all());before=model.fc2.weight.detach().clone();target=old.double().clone();support=mask[:,:,0,0].reshape(10,10,-1);m=support.sum(1);add=torch.where(m>0,10*delta/m.clamp_min(1),torch.zeros_like(delta));target[:,:,0,0]+= (support*add[:,None,:]).reshape(100,-1);target=target.to(old.dtype);newraw=target.sign()*(target.abs()+theta);newraw=torch.where(target==0,torch.zeros_like(newraw),newraw);model.fc2.weight.copy_(torch.where(mask,newraw,before));actual=model.fc2.sparse_weight().detach();newmask=actual!=0
  assert not bool((newmask & ~mask).any()) and torch.equal(model.fc2.weight[~mask],before[~mask]);err=normerr(actual,target);assert err<=1e-5
  return {'old_support':int(mask.sum()),'new_support':int(newmask.sum()),'added':int((newmask&~mask).sum()),'removed':int((mask&~newmask).sum()),'old_sha256':hash_array(old),'new_sha256':hash_array(actual),'effective_l2_change':float(torch.linalg.vector_norm((actual-old).double())),'mapping_normalized_error':err}

def added_checks(m,a,b):
 # Deterministic toy problem with independently calculated primal and dual ridge solutions.
 x=torch.tensor([[1.,2.],[2.,-1.],[0.5,1.]],dtype=torch.float64,device='cuda');e=torch.tensor([1.,-.5,.25],dtype=torch.float64,device='cuda');lam=.1;n=3
 p=torch.linalg.solve(x.T@x+n*lam*torch.eye(2,device='cuda',dtype=torch.float64),x.T@e);d=x.T@torch.linalg.solve(x@x.T+n*lam*torch.eye(3,device='cuda',dtype=torch.float64),e);er=float((p-d).abs().max());explicit_test('ridge_primal_dual',er<1e-10,{'X':x.cpu().tolist(),'e':e.cpu().tolist(),'lambda':lam},'max absolute difference < 1e-10',{'primal':p.cpu().tolist(),'dual':d.cpu().tolist(),'error':er})
 r=torch.zeros((3,4),device='cuda',dtype=torch.float64);aa=torch.zeros((10,4),device='cuda',dtype=torch.float64);mask=torch.zeros((100,2),device='cuda',dtype=torch.bool);v,details,_=solve_support(r,aa,[0,1],[2,3],mask);explicit_test('empty_support',bool((v==0).all()) and all(q['form']=='degenerate_zero' for q in details),{'rate_shape':[3,4],'mask_shape':[100,2],'mask':'all zero'},'all zero correction and degenerate status',{'details':details})
 mask[:]=True;v,details,_=solve_support(r,aa,[0,1],[2,3],mask);explicit_test('zero_input',bool((v==0).all()) and all(q['lambda']==0 for q in details),{'rate':'all zero','mask':'all valid','shape':[3,4]},'lambda zero and zero correction',{'details':details})
 explicit_test('nonfinite_detection',not bool(torch.isfinite(torch.tensor([float('nan'),float('inf')])).all()),{'values':['NaN','Inf']},'nonfinite detection triggers',{'finite':False})
 # Soft-threshold mapping for signs, exact boundary, and inactive weights.
 vals=torch.tensor([-2.,-.5,0.,.5,2.],device='cuda');t=torch.tensor(.5,device='cuda');eff=vals.sign()*torch.relu(vals.abs()-t);explicit_test('soft_threshold_boundaries',torch.equal(eff,torch.tensor([-1.5,0.,0.,0.,1.5],device='cuda')),{'raw':vals.cpu().tolist(),'threshold':.5},'sign and boundary match',{'effective':eff.cpu().tolist()})
 ids=S['D_grad'][:3];x,y,sid=next(iter(loader(ids)));x=x.cuda();z,sp=fetch(m,x,True);rate=sp[:,:,:,0,0].double().mean(0);deleted=list(range(1024));compact,keep=esc.compact(m,deleted);oldstate={k:v.detach().clone() for k,v in compact.state_dict().items()};zz,ss=fetch(compact,x,True);Ecalc=rate[:,deleted]@a[:,deleted].T;er=normerr(z.double()-zz.double(),Ecalc);explicit_test('R116_real_compact',er<=1e-5,{'sample_ids':sid.tolist(),'batch_sizes':[3],'deleted_count':1024},'deleted contribution vs real full-compact logits <=1e-5',{'normalized_error':er})
 delta=torch.zeros((10,len(keep)),device='cuda',dtype=torch.float64);oldmask=compact.fc2.sparse_weight().detach()[:,:,0,0]!=0;support=oldmask.reshape(10,10,-1).any(1)
 for k in range(10):
  ix=torch.where(support[k])[0]
  if len(ix):delta[k,ix[0]]=.001*(k+1)
 before=compact.fc2.sparse_weight().detach()[:,:,0,0].reshape(10,10,-1).double().mean(1);audit=apply_delta(compact,delta);after=compact.fc2.sparse_weight().detach()[:,:,0,0].reshape(10,10,-1).double().mean(1);em=float((after-before-delta).abs().max());znew,spnew=fetch(compact,x,True);expect=zz.double()+rate[:,keep]@delta.T;el=normerr(znew,expect);unchanged=all(torch.equal(v,compact.state_dict()[k]) for k,v in oldstate.items() if k!='fc2.weight');same=torch.equal(ss,spnew);params=sum(p.numel() for p in compact.parameters());expected=36720484-1024*16485
 explicit_test('R120_real_mapping',em<=1e-5 and el<=1e-5 and audit['added']==0 and unchanged and same and params==expected,{'sample_ids':sid.tolist(),'deleted_count':1024,'delta_first_supported_per_class':True},'boost compensation and logits <=1e-5; no support growth; other state and spikes exact; expected parameter count',{'boost_max_abs_error':em,'logit_normalized_error':el,'support':audit,'other_state_exact':unchanged,'spikes_equal':same,'parameters':params,'expected':expected})
 del compact,x,sp,ss,spnew;torch.cuda.empty_cache()

def main24():
 guard();assert torch.cuda.device_count()==1 and '4090' in torch.cuda.get_device_name();torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 dump('precision_audit.json',{'cudnn_allow_tf32':False,'matmul_allow_tf32':False,'amp':False,'forward':'FP32','algebra':'FP64','device':torch.cuda.get_device_name(),'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES']})
 spl=data_audit();m=K.model();dump('parent_checkpoint_audit.json',{'before':sha(C['parent_path']),'expected':C['parent_sha256']});architecture(m);a,b=preflight(m);added_checks(m,a,b)
 existing=json.loads((E/'admission_gate_audit.json').read_text());assert existing['status']=='PASSED' and all(x['status']=='PASSED' for x in extra) and len(extra)==7
 dump('formula_tests.json',{'status':'PASSED','tests':extra});dump('admission_gate_audit.json',{'status':'PASSED','exp024_tests':existing['tests'],'round024_formula_tests':extra,'actual_tests_recorded_before_selection':True})
 status['phase']='selection';dump('status.json',status);rate,y,z,ids,ct=capture_dataset(m,S['D_grad']);assert normerr(esc.reconstruct(rate,a,b),z)<=1e-5
 freq=rate.mean(0).cpu().numpy();l1=m.fc1.sparse_weight().detach().abs().sum((1,2,3)).double().cpu().numpy();idx=np.arange(2048);ord1=np.lexsort((idx,l1));ord2=np.lexsort((idx,freq));perms={str(seed):np.random.default_rng(seed).permutation(2048).tolist() for seed in [20261128,20261129,20261130]}
 write('neuron_statistics.csv',[{'neuron':i,'effective_L1':float(l1[i]),'D_grad_firing_rate':float(freq[i])} for i in range(2048)]);selected={};coeff=[];supportdetails={};deltas={};supportaudits={}
 for ratio,n in [('50',1024),('75',1536)]:
  groups={f'R0-{seed}':sorted(perms[str(seed)][:n]) for seed in [20261128,20261129,20261130]};groups.update(R1=sorted(map(int,ord1[:n])),R2=sorted(map(int,ord2[:n])));groups['R3']=groups['R1']
  selected[ratio]={g:{'deleted':v,'retained':[i for i in range(2048) if i not in set(v)]} for g,v in groups.items()}
  keep=selected[ratio]['R1']['retained'];mask=m.fc2.sparse_mask().detach()[:,:,0,0][:,keep].bool();delta,details,err=solve_support(rate,a,keep,groups['R1'],mask);deltas[ratio]=delta;supportdetails[ratio]=details
  for k in range(10):
   for j,orig in enumerate(keep):
    if j in set(details[k]['support_indices']):coeff.append({'ratio':ratio,'class':k,'retained_index':j,'original_neuron':orig,'delta':float(delta[k,j])})
  compact,_=esc.compact(m,groups['R1']);supportaudits[ratio]=apply_delta(compact,delta);del compact;torch.cuda.empty_cache()
 dump('selected_neurons.json',{'permutations':perms,'ratios':selected});write('reconstruction_coefficients.csv',coeff);dump('readout_support_audit.json',{'ratios':supportaudits,'fit_class_details':supportdetails});dump('selection_lock.json',{'code_sha':CODE,'parent_sha256':C['parent_sha256'],'D_grad_sha256':'93c4c834c3fd3c8b92a64caa88da8a671e27be23dc2df11999c8e5f3dfbe47d8','D_cal_sha256':'b671cbd803b13cc76b02f5db4bad311ef9492276622051df01af56a931301baa','selected_sha256':sha(E/'selected_neurons.json'),'coefficients_sha256':sha(E/'reconstruction_coefficients.csv'),'support_audit_sha256':sha(E/'readout_support_audit.json'),'admission_sha256':sha(E/'admission_gate_audit.json'),'fixed_lambda_scale':.001,'D_cal_or_validation_used':False})
 paths=['admission_gate_audit.json','formula_tests.json','neuron_statistics.csv','selected_neurons.json','reconstruction_coefficients.csv','readout_support_audit.json','selection_lock.json']
 subprocess.run(['git','add']+['experiments/EXP-026/'+p for p in paths],check=True);subprocess.run(['git','commit','-m','EXP-026 lock selections and fixed-support reconstruction'],check=True);locksha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();subprocess.run(['git','push','origin','HEAD:research/round-024-structural-reconstruction'],check=True,timeout=90);remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/research/round-024-structural-reconstruction'],text=True,timeout=60).split()[0];assert remote==locksha;dump('lock_commit.json',{'sha':locksha,'verified_remote':remote,'before_Dcal_validation':True});guard();status['phase']='evaluation';dump('status.json',status)
 # Strict FP32 B0 is rebuilt before all pruned validation comparisons.
 val=[];layers=[];samples=[];recon=[];counts={};manifest=[];groups=[('B0',None,None)]+[(f'{ratio}-{g}',ratio,g) for ratio in ['50','75'] for g in ['R0-20261128','R0-20261129','R0-20261130','R1','R2','R3']]
 for label,ratio,g in groups:
  guard();model=m if ratio is None else esc.compact(m,selected[ratio][g]['deleted'])[0];audit=None
  if g=='R3':audit=apply_delta(model,deltas[ratio]);assert audit['added']==0
  info=model_metrics(model);expected=36720484-(0 if ratio is None else int(ratio)*2048//100*16485);assert info['parameter_count']==expected;info['compression_fraction']=1-info['parameter_count']/36720484
  weightpath=E/'models'/f'{label}.pth';weightpath.parent.mkdir(exist_ok=True);torch.save({'group':label,'parent_sha256':C['parent_sha256'],'retained':None if ratio is None else selected[ratio][g]['retained'],'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}},weightpath);info['inference_file_bytes']=weightpath.stat().st_size;manifest.append({'group':label,'path':str(weightpath),'sha256':sha(weightpath),'bytes':weightpath.stat().st_size});counts[label]=info
  if g in ['R1','R3']:
   for split,splitids in [('D_grad',S['D_grad']),('D_cal',S['D_cal'])]:
    errs=[];match=correct=tot=0;sumce=0.
    for xx,yy,ii in loader(splitids):
     guard();xx=xx.cuda();yy=yy.cuda();zt,_=fetch(m,xx,False);zp,_=fetch(model,xx,False);errs.append((zp.double()-zt.double()).square().sum().item());match+=int((zp.argmax(1)==zt.argmax(1)).sum());correct+=int((zp.argmax(1)==yy).sum());sumce+=F.cross_entropy(zp,yy,reduction='sum').item();tot+=len(yy)
    mse=sum(errs)/(tot*10);recon.append({'group':label,'split':split,'samples':tot,'mse':mse,'rmse':mse**.5,'teacher_prediction_agreement':match/tot,'CE':sumce/tot,'effective_l2_change':0 if g=='R1' else audit['effective_l2_change']});write('reconstruction_summary.csv',recon)
  vr,lr,fr,ss=evaluate(model,spl['validation'],label,'validation',True);vr.update(info);val.append(vr);layers.extend(lr);samples.extend(ss);write('validation_summary.csv',val);write('layer_sops.csv',layers);write('validation_per_sample.csv',samples);dump('parameter_count_audit.json',counts);dump('checkpoint_manifest.json',manifest);print('EVALUATED',label,vr['accuracy'],vr['CE'],flush=True)
  if label=='B0':
   da=abs(vr['accuracy']-.9422222222222222);dc=abs(vr['CE']-.1727328626844618);ds=abs(vr['SOP_sample_weighted']-86009551.08977778)/86009551.08977778;ok=da<=1e-7 and dc<=1e-5 and ds<=1e-5;dump('baseline_reconstruction_audit.json',{'status':'PASSED' if ok else 'FAILED','accuracy_error':da,'CE_abs_error':dc,'SOP_relative_error':ds,'tolerances':{'accuracy':1e-7,'CE':1e-5,'SOP_relative':1e-5},'actual':vr});assert ok
  if ratio is not None:del model;gc.collect();torch.cuda.empty_cache()
 by={g:{r['sample_id']:r for r in samples if r['group']==g} for g,_,_ in groups};rng=np.random.default_rng(20261128);boot={}
 comparisons=[(f'{r}-R1','B0') for r in ['50','75']]+[(f'{r}-R3',f'{r}-{g}') for r in ['50','75'] for g in ['R1','R2','R0-20261128','R0-20261129','R0-20261130']]
 for ga,gb in comparisons:
  ix=sorted(by[ga]);assert ix==sorted(by[gb]) and all(by[ga][i]['label']==by[gb][i]['label'] for i in ix);aa=np.array([by[ga][i]['correct']-by[gb][i]['correct'] for i in ix]);cc=np.array([by[ga][i]['ce']-by[gb][i]['ce'] for i in ix]);sel=rng.integers(0,len(ix),(2000,len(ix)));boot[ga+' minus '+gb]={'accuracy_delta_pp':float(100*aa.mean()),'accuracy_ci95_pp':(100*np.quantile(aa[sel].mean(1),[.025,.975])).tolist(),'CE_delta':float(cc.mean()),'CE_ci95':np.quantile(cc[sel].mean(1),[.025,.975]).tolist(),'SOP_ci':'NA per-sample SOP not collected'}
 dump('paired_bootstrap.json',{'seed':20261128,'rng':'numpy.default_rng PCG64','replicates':2000,'method':'paired percentile','comparisons':boot});v={x['group']:x for x in val};rr={(x['group'],x['split']):x for x in recon};gates={}
 for r in ['50','75']:
  x=v[r+'-R1'];y=v[r+'-R3'];checks={'Dcal_mse_decreases':rr[(r+'-R3','D_cal')]['mse']<rr[(r+'-R1','D_cal')]['mse'],'accuracy_gain_at_least_0_5pp':y['accuracy']-x['accuracy']>=.005,'CE_decreases':y['CE']<x['CE'],'parameter_equal':y['parameter_count']==x['parameter_count'],'support_not_expanded':supportaudits[r]['added']==0,'SOP_at_most_101pct':y['SOP_sample_weighted']<=1.01*x['SOP_sample_weighted'],'admission_and_isolation':True};gates[r]={'H026_drop_at_least_1pp':v['B0']['accuracy']-x['accuracy']>=.01,'H027_checks':checks,'H027_all':all(checks.values())}
 dump('hypothesis_gate_results.json',{'ratios':gates,'primary_ratio':'50','H026_scientific_status':'UNTESTED_pending_theory_review','H027_scientific_status':'UNTESTED_pending_theory_review'});return 'COMPLETED'

extra=[]
try:status['status']=main24()
except BaseException as e:status.update(status='PARTIAL',error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
finally:
 try:
  before=C['parent_sha256'];after=sha(C['parent_path']);dump('parent_checkpoint_audit.json',{'before':before,'after':after,'expected':before,'match':before==after})
 except BaseException as e:status['parent_final_audit_error']=repr(e)
 status.update(end_unix=time.time(),wall_seconds=K.PRIOR_WALL+time.time()-K.START,cpu_accounted_seconds=K.PRIOR_CPU+K.cpu()-K.CPU,process_tree_cpu_seconds=K.cpu()-K.CPU,cpu_prior_conservative_reserve_seconds=K.PRIOR_CPU,peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None);dump('status.json',status);print(json.dumps(status),flush=True)
if status['status']!='COMPLETED':sys.exit(1)

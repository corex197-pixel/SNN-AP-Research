"""EXP-024 machine-audited exact readout and real structured compact pruning."""
import os,sys,time,json,csv,hashlib,traceback,signal,subprocess,random,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'experiments/ROUND-019'),str(ROOT/'experiments/ROUND-022')]
import core as K
K.E=ROOT/'experiments/EXP-024';K.START=time.time();K.CPU=K.cpu();K.PRIOR_WALL=K.PRIOR_CPU=0;K.status.update(run_id='EXP-024-RUN-001',phase='admission')
from core import *
# core resolved from ROUND-019; load local algebra explicitly to avoid ambiguous module names.
import importlib.util
spec=importlib.util.spec_from_file_location('esc_algebra',ROOT/'experiments/ROUND-022/core.py');esc=importlib.util.module_from_spec(spec);spec.loader.exec_module(esc)
from torch.nn import functional as F
from models.submodules.layers import SparseConv2d,SparseNeuron,IdentityBN
from spikingjelly.activation_based import functional,neuron
from utils.utils import SOPMonitor
E=K.E;status=K.status;CODE=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();status.update(code_sha=CODE,start_unix=K.START)
# Override inherited 15 minute guard with this round's 20 minute budget.
def guard(reserve=45):
 wall=time.time()-K.START;cpu=K.cpu()-K.CPU
 if wall>=1200-reserve or cpu>=2700-reserve:raise TimeoutError('ROUND-022 cumulative budget reserve')
 status.update(wall_seconds=wall,cpu_seconds=cpu);K.dump('status.json',status);return wall,cpu
K.guard=guard
signal.signal(signal.SIGTERM,lambda a,b:(_ for _ in ()).throw(TimeoutError('watchdog signal')))
REQUIRED=['architecture','data_split','capture_switch','readout_zero_set','tail_2_plus_1','single_gate','single_compact','block16_gate','block16_compact','set512_gate','set512_compact','preserved_spikes','prediction_CE','physical_deletion']
admissions=[]
def check(name,passed,inputs,assertion,actual):
 entry={'name':name,'status':'PASSED' if bool(passed) else 'FAILED','inputs':inputs,'assertion':assertion,'actual':actual};admissions.append(entry);dump('admission_gate_audit.json',{'status':'RUNNING','required':REQUIRED,'tests':admissions});
 if not passed:raise AssertionError('admission failed '+name)
def digest_tensor(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def normerr(a,b):return (a.double()-b.double()).abs().max().item()/max(1.,b.double().abs().max().item())
def fetch(model,x,capture=True,gate=None):
 cache={};hooks=[]
 if capture or gate is not None:
  def h(mod,args,out):
   cache['spikes']=out.detach().clone()
   if gate is None:return out
   mask=torch.ones(out.shape[2],device=out.device,dtype=out.dtype);mask[torch.as_tensor(gate,device=out.device,dtype=torch.long)]=0;return out*mask[None,None,:,None,None]
  hooks.append(model.activation_fc1.register_forward_hook(h))
 try:
  functional.reset_net(model)
  with torch.no_grad():z=model(x).mean(0).detach().clone()
 finally:
  functional.reset_net(model)
  for hook in hooks:hook.remove()
 return z,cache.get('spikes')
def data_audit():
 spl=json.loads((ROOT/'experiments/EXP-004/splits/indices.json').read_text());assert sha(ROOT/'experiments/EXP-004/splits/indices.json')==C['split_sha256'];expected={'D_grad':'93c4c834c3fd3c8b92a64caa88da8a671e27be23dc2df11999c8e5f3dfbe47d8','D_cal':'b671cbd803b13cc76b02f5db4bad311ef9492276622051df01af56a931301baa'};actual={}
 for n,h in expected.items():actual[n]=hashlib.sha256(json.dumps(S[n],separators=(',',':')).encode()).hexdigest();assert actual[n]==h and len(S[n])==len(set(S[n]))==512
 passed=not set(S['D_grad'])&set(S['D_cal']) and not (set(S['D_grad'])|set(S['D_cal']))&(set(spl['validation'])|set(spl['diagnostic'])) and len(spl['validation'])==4500;details={'hashes':actual,'counts':{n:len(S[n]) for n in actual},'validation_count':len(spl['validation']),'disjoint':bool(passed),'train_only_loader':True};dump('data_split_audit.json',details);check('data_split',passed,details,'fixed hashes, counts and disjointness',details);return spl

def architecture(m):
 from spikingjelly.activation_based import layer
 b=m.boost;act=m.activation_fc1;spec={'fc1_in':m.fc1.in_channels,'fc1_out':m.fc1.out_channels,'fc2_in':m.fc2.in_channels,'fc2_out':m.fc2.out_channels,'fc1_bias':m.fc1.bias is not None,'fc2_bias':m.fc2.bias is not None,'kernel_fc1':list(m.fc1.kernel_size),'kernel_fc2':list(m.fc2.kernel_size),'boost_kernel':b.kernel_size,'boost_stride':b.stride,'BN_type':type(m.bn1).__name__,'last_neuron_type':type(act.activation).__name__,'groups':m.fc1.groups,'registered_fc1_consumer':m.activation5.next_convs==[m.fc1],'registered_fc2_consumer':act.next_convs==[m.fc2],'T':m.T}
 passed=(m.fc1.in_channels,m.fc1.out_channels,m.fc2.in_channels,m.fc2.out_channels)==(16384,2048,2048,100) and isinstance(m.bn1,IdentityBN) and isinstance(act.activation,neuron.LIFNode) and m.fc1.groups==m.fc2.groups==1 and m.fc1.kernel_size==m.fc2.kernel_size==(1,1) and b.kernel_size in (10,(10,)) and b.stride in (10,(10,)) and act.next_convs==[m.fc2] and m.activation5.next_convs==[m.fc1]
 dump('architecture_audit.json',{'actual':spec,'source_path':'vendor/UniSparse/models/cifarnet.py','execution_path':'activation5/maxpool5/flatten/fc1/IdentityBN/elementwise LIF/fc2/flatten/boost/time mean','parameter_coupling':'original LIF is elementwise, source manually audited'});check('architecture',passed,spec,'fixed direct last hidden structure and grouped mean',spec)

def preflight(m):
 torch.set_num_threads(2);ids=S['D_grad'][:32];dl=loader(ids);x,y,sid=next(iter(dl));assert sid.tolist()==ids;x=x.cuda();y=y.cuda();z,sp=fetch(m,x,True);z0,_=fetch(m,x,False);check('capture_switch',torch.equal(z,z0),{'sample_ids':ids,'batch_sizes':[32]},'capture on/off exact logits',{'max_abs_logit_diff':float((z-z0).abs().max())});assert sp.shape==(4,32,2048,1,1) and bool(((sp==0)|(sp==1)).all());rate=sp[:,:, :,0,0].double().mean(0);a,b=esc.readout(m);rr=esc.reconstruct(rate,a,b);err=normerr(rr,z);check('readout_zero_set',err<=1e-5,{'sample_ids':ids,'batch_sizes':[32],'hidden_shape':list(sp.shape)},'R103 relative infinity error <= 1e-5',{'normalized_error':err,'bias_present':m.fc2.bias is not None});predok=torch.equal(rr.argmax(1),z.argmax(1));cediff=abs(F.cross_entropy(rr,y).item()-F.cross_entropy(z.double(),y).item());check('prediction_CE',predok and cediff<=1e-5,{'sample_ids':ids,'batch_sizes':[32]},'reconstructed prediction equality and CE error <=1e-5',{'predictions_equal':predok,'absolute_CE_error':cediff})
 # Real unequal 2+1 partition, not a label-only synthetic test.
 z3,_=fetch(m,x[:3],False);za,_=fetch(m,x[:2],False);zb,_=fetch(m,x[2:3],False);errtail=normerr(torch.cat([za,zb]),z3);full=F.cross_entropy(z3.double(),y[:3],reduction='none');weighted=(F.cross_entropy(za.double(),y[:2],reduction='sum')+F.cross_entropy(zb.double(),y[2:3],reduction='sum'))/3;d=float(abs(weighted-full.mean()));check('tail_2_plus_1',errtail<=1e-5 and d<=1e-5,{'sample_ids':ids[:3],'batch_sizes':[2,1],'reference_batch_size':3},'unequal batches reproduce logits and sample-weighted CE <=1e-5',{'logit_normalized_error':errtail,'CE_abs_error':d})
 tests=[];spikechecks=[]
 for name,n in [('single',1),('block16',16),('set512',512)]:
  guard();deleted=list(range(n));expect=esc.reconstruct(rate,a,b,deleted);gated,gs=fetch(m,x,True,deleted);eg=normerr(expect,gated);check(name+'_gate',eg<=1e-5,{'sample_ids':ids,'batch_sizes':[32],'deleted_original_indices':deleted},'R104 vs true zero-spike gated full network <=1e-5',{'normalized_error':eg});t=time.monotonic();compact,keep=esc.compact(m,deleted);actual,sp2=fetch(compact,x,True);ec=normerr(actual,gated);spmatch=torch.equal(gs.index_select(2,keep),sp2);spikechecks.append(spmatch);pp=torch.equal(actual.argmax(1),gated.argmax(1));ce=abs(F.cross_entropy(actual.double(),y).item()-F.cross_entropy(gated.double(),y).item());params=sum(p.numel() for p in compact.parameters());original=sum(p.numel() for p in m.parameters());delta=original-params;expected=n*(16384+1+100);ok=ec<=1e-5 and spmatch and pp and ce<=1e-5 and delta==expected;check(name+'_compact',ok,{'sample_ids':ids,'batch_sizes':[32],'deleted_original_indices':deleted},'compact vs gated logits <=1e-5, kept spikes exact, prediction/CE match, expected physical parameter removal',{'normalized_error':ec,'kept_spikes_equal':spmatch,'predictions_equal':pp,'CE_abs_error':ce,'parameter_count_before':original,'after':params,'actual_removed':delta,'expected_removed':expected,'construction_seconds':time.monotonic()-t});tests.append({'case':name,'deleted_count':n,'gated_error':eg,'compact_error':ec,'kept_spikes_equal':spmatch,'predictions_equal':pp,'CE_error':ce,'parameter_before':original,'parameter_after':params});del compact;torch.cuda.empty_cache()
 check('preserved_spikes',all(spikechecks),{'sample_ids':ids,'batch_sizes':[32],'deleted_counts':[1,16,512]},'kept LIF spikes elementwise exact in all fixed deletion cases',{'per_case':spikechecks});last=tests[-1];check('physical_deletion',last['parameter_before']-last['parameter_after']==8440320,{'sample_ids':ids,'deleted_count':512},'actual parameter count drops by 8440320',{'before':last['parameter_before'],'after':last['parameter_after']})
 names=[x['name'] for x in admissions];missing=[n for n in REQUIRED if names.count(n)!=1];assert not missing and all(x['status']=='PASSED' for x in admissions);dump('exact_readout_tests.json',{'status':'PASSED','R103_normalized_error':err,'prediction_CE_error':cediff});dump('structural_equivalence_tests.json',{'status':'PASSED','fixed_cases':tests});dump('admission_gate_audit.json',{'status':'PASSED','required':REQUIRED,'tests':admissions,'no_missing_or_duplicate_test':True,'verified_before_selection':True});return a,b
def capture_dataset(m,ids):
 t=time.monotonic();rows=[];labels=[];sample_ids=[];refs=[]
 for x,y,sid in loader(ids):
  guard();x=x.cuda(non_blocking=True);z,sp=fetch(m,x,True);assert sp.shape[0]==4 and sp.shape[2]==2048 and bool(((sp==0)|(sp==1)).all());rows.append(sp[:,:,:,0,0].double().mean(0));labels.append(y.cuda());sample_ids.extend(sid.tolist());refs.append(z)
 assert sample_ids==list(ids);return torch.cat(rows),torch.cat(labels),torch.cat(refs),sample_ids,time.monotonic()-t

def select(m,rate,a,b,y,ids):
 base=esc.reconstruct(rate,a,b);loss0=esc.losses(base,y).mean();N=2048;neurons=np.arange(N,dtype=np.int64);t0=time.monotonic();single=esc.scores(base,rate,a,y,neurons.tolist(),64);s3time=time.monotonic()-t0;freq=rate.mean(0).cpu().numpy();l1=m.fc1.sparse_weight().detach().abs().sum((1,2,3)).double().cpu().numpy();rng=np.random.default_rng(20261128);picked={'S0':sorted(int(i) for i in rng.choice(N,size=512,replace=False)),'S1':sorted(int(i) for i in esc.stable_order(neurons,l1)[:512]),'S2':sorted(int(i) for i in esc.stable_order(neurons,freq)[:512]),'S3':sorted(int(i) for i in esc.stable_order(neurons,single)[:512])};trace=[];remaining=neurons.copy();removed=[];cur=base.clone();start=time.monotonic();peak_before=torch.cuda.max_memory_allocated()
 for roundno in range(1,33):
  guard();st=time.monotonic();score=esc.scores(cur,rate,a,y,remaining.tolist(),64);order=np.lexsort((remaining,score));choose=remaining[order[:16]];oldloss=esc.losses(cur,y).mean().item();cur=cur-rate[:,choose.tolist()]@a[:,choose.tolist()].T;removed.extend(choose.tolist());actual=esc.losses(cur,y).mean().item();direct=esc.reconstruct(rate,a,b,removed);error=normerr(cur,direct);assert error<=1e-10
  for j,ix in enumerate(order[:16]):trace.append({'round':roundno,'within_block_rank':j+1,'neuron':int(remaining[ix]),'conditional_score':float(score[ix]),'loss_before_block':oldloss,'actual_joint_loss_after_block':actual,'joint_vs_direct_error':error,'elapsed_seconds':time.monotonic()-st})
  remaining=np.delete(remaining,order[:16]);write('greedy_selection_trace.csv',trace)
 picked['M5']=sorted(int(i) for i in removed);m5time=time.monotonic()-start;assert all(len(v)==512 and len(set(v))==512 for v in picked.values());stats=[{'neuron':i,'spike_rate':float(freq[i]),'effective_fc1_L1':float(l1[i]),'single_delta_CE':float(single[i])} for i in range(N)];write('neuron_statistics.csv',stats);write('single_neuron_loss.csv',[{'neuron':i,'delta_CE':float(single[i])} for i in range(N)])
 interaction=[]
 for group in ['S3','M5']:
  chosen=picked[group];joint=esc.losses(esc.reconstruct(rate,a,b,chosen),y).mean().item()-loss0.item();independent=float(single[chosen].sum());interaction.append({'group':group,'chosen_neurons':len(chosen),'joint_delta_CE':joint,'sum_single_deltas':independent,'joint_minus_single_sum':joint-independent})
 write('interaction_diagnostics.csv',interaction);dump('selection_timing.json',{'S3_seconds':s3time,'M5_seconds':m5time,'M5_peak_additional_allocated_bytes':max(0,torch.cuda.max_memory_allocated()-peak_before),'greedy_rounds':32,'candidates_initial':2048,'chunk':64});return picked

def model_metrics(m):
 params=sum(x.numel() for x in m.parameters());nonzero=total=0
 for n,z in m.named_modules():
  if isinstance(z,SparseConv2d):mask=z.sparse_mask();nonzero+=int(mask.sum().item());total+=mask.numel()
 return {'parameter_count':params,'effective_nonzero_weights':nonzero,'effective_weight_sparsity':1-nonzero/total,'FP32_dense_parameter_bytes':params*4,'fc1_out':m.fc1.out_channels,'fc2_in':m.fc2.in_channels}

def evaluate(m,ids,group,split,save_samples):
 guard();m.eval();mon=SOPMonitor(m);handles=[];ones={};elements={};spikes={};states={};layers={n:0. for n in mon.monitored_layers};batch_sops=[];total=correct=0;sumce=0.;samples=[];start=time.monotonic();batch_sizes=[]
 for n,z in m.named_modules():
  if isinstance(z,SparseNeuron):
   def spike_hook(mod,args,out,n=n):spikes[n]=spikes.get(n,0)+int((out!=0).sum().item());states[n]=states.get(n,0)+out.numel()
   handles.append(z.register_forward_hook(spike_hook))
 for n in mon.monitored_layers:
  def input_hook(mod,args,n=n):
   x=args[0].detach();assert bool(((x==0)|(x==1)).all());ones[n]=ones.get(n,0)+int(x.sum().item());elements[n]=elements.get(n,0)+x.numel()
  handles.append(m.get_submodule(n).register_forward_pre_hook(input_hook))
 try:
  with torch.no_grad():
   for x,y,sid in loader(ids):
    guard();x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);mon.clear_recorded_data();functional.reset_net(m);z=m(x).mean(0);loss=F.cross_entropy(z,y,reduction='none');assert torch.isfinite(loss).all();pred=z.argmax(1);cnt=len(y);total+=cnt;batch_sizes.append(cnt);sumce+=float(loss.sum());correct+=int((pred==y).sum());batch=0.
    for n in mon.monitored_layers:
     v=sum(float(x.item()) for x in mon[n]);layers[n]+=v;batch+=v
    batch_sops.append(batch);functional.reset_net(m)
    if save_samples:samples.extend({'group':group,'sample_id':int(i),'label':int(t),'prediction':int(p),'correct':int(p==t),'ce':float(l)} for i,t,p,l in zip(sid.tolist(),y.cpu().tolist(),pred.cpu().tolist(),loss.cpu().tolist()))
 finally:
  mon.remove_hooks()
  for h in handles:h.remove()
  functional.reset_net(m)
 assert total==len(ids);elapsed=time.monotonic()-start;result={'group':group,'split':split,'samples':total,'accuracy':correct/total,'CE':sumce/total,'FR_global':sum(spikes.values())/sum(states.values()),'SOP_sample_weighted':sum(batch_sops)/total,'SOP_author':sum(batch_sops)/len(batch_sops)/64,'fc1_SOP':layers['fc1']/total,'fc2_SOP':layers['fc2']/total,'validation_seconds':elapsed,'last_batch_size':batch_sizes[-1]};lr=[{'group':group,'split':split,'layer':n,'sops':layers[n]/total,'input_FR':ones[n]/elements[n]} for n in mon.monitored_layers];fr=[{'group':group,'split':split,'neuron':n,'FR':spikes[n]/states[n]} for n in spikes];return result,lr,fr,samples

def main():
 guard();assert torch.cuda.device_count()==1 and '4090' in torch.cuda.get_device_name();torch.set_num_threads(2);torch.manual_seed(20261128);np.random.seed(20261128);random.seed(20261128);spl=data_audit();m=K.model();dump('parent_checkpoint_audit.json',{'before':sha(C['parent_path']),'expected':C['parent_sha256']});architecture(m);a,b=preflight(m)
 admitted=json.loads((E/'admission_gate_audit.json').read_text());assert admitted['status']=='PASSED' and set(REQUIRED)=={x['name'] for x in admitted['tests']} and all(x['status']=='PASSED' and x['inputs'] and x['assertion'] and x['actual'] for x in admitted['tests']);status['phase']='selection';dump('status.json',status)
 rate,y,ref,ids,capture_seconds=capture_dataset(m,S['D_grad']);z=esc.reconstruct(rate,a,b);err=normerr(z,ref);assert err<=1e-5
 picked=select(m,rate,a,b,y,ids);hashes={'rates_sha256':digest_tensor(rate),'A_sha256':digest_tensor(a),'bias_sha256':digest_tensor(b),'labels_sha256':digest_tensor(y),'sample_ids_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()};selected={g:{'deleted':v,'retained':[i for i in range(2048) if i not in set(v)]} for g,v in picked.items()};dump('selected_neurons.json',selected);lock={'code_sha':CODE,'parent_sha256':C['parent_sha256'],'fixed_seed':20261128,'rng':'numpy.default_rng PCG64','block_size':16,'deleted_count':512,'selected_neurons_sha256':sha(E/'selected_neurons.json'),'admission_sha256':sha(E/'admission_gate_audit.json'),'data_hashes':hashes,'capture_seconds':capture_seconds,'D_cal_or_validation_used_for_selection':False,'full_Dgrad_R103_error':err};dump('selection_lock.json',lock)
 subprocess.run(['git','add','experiments/EXP-024/admission_gate_audit.json','experiments/EXP-024/selection_lock.json','experiments/EXP-024/selected_neurons.json','experiments/EXP-024/neuron_statistics.csv','experiments/EXP-024/single_neuron_loss.csv','experiments/EXP-024/greedy_selection_trace.csv'],check=True);subprocess.run(['git','commit','-m','lock(EXP-024): fix audited structured pruning selections'],check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);locksha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();dump('lock_commit.json',{'sha':locksha,'before_Dcal_validation':True});print('selection locked',locksha,flush=True);del rate,y,ref,z;torch.cuda.empty_cache()
 status['phase']='selected_model_equivalence';dump('status.json',status);auditrows=[]
 # Each actual locked set is independently checked against the same real gated reference before D_cal/validation.
 xcheck,ycheck,idcheck=next(iter(loader(S['D_grad'][:32])));xcheck=xcheck.cuda();ycheck=ycheck.cuda()
 for group in ['S0','S1','S2','S3','M5']:
  guard();deleted=picked[group];ref,spref=fetch(m,xcheck,True,deleted);compact,keep=esc.compact(m,deleted);out,sout=fetch(compact,xcheck,True);err=normerr(out,ref);spikes_equal=torch.equal(spref.index_select(2,keep),sout);pred_equal=torch.equal(out.argmax(1),ref.argmax(1));ce_error=abs(F.cross_entropy(out.double(),ycheck).item()-F.cross_entropy(ref.double(),ycheck).item());passed=err<=1e-5 and spikes_equal and pred_equal and ce_error<=1e-5
  auditrows.append({'group':group,'status':'PASSED' if passed else 'FAILED','sample_ids':idcheck.tolist(),'batch_sizes':[32],'deleted_indices_sha256':hashlib.sha256(json.dumps(deleted,separators=(',',':')).encode()).hexdigest(),'assertion':'locked compact vs same-set gated logits<=1e-5; kept spikes exact; prediction and CE match','normalized_error':err,'kept_spikes_equal':spikes_equal,'prediction_equal':pred_equal,'CE_abs_error':ce_error});dump('selected_structure_audit.json',{'status':'RUNNING','tests':auditrows});assert passed;del compact,out,sout,ref,spref;gc.collect();torch.cuda.empty_cache()
 dump('selected_structure_audit.json',{'status':'PASSED','tests':auditrows,'complete_before_Dcal_or_validation':True});del xcheck,ycheck
 status['phase']='evaluation';dump('status.json',status);allcal=[];allval=[];alllayers=[];allfr=[];allsamples=[];counts={};manifest=[];original_params=sum(x.numel() for x in m.parameters())
 for group in ['B0','S0','S1','S2','S3','M5']:
  guard();t=time.monotonic();model=m if group=='B0' else esc.compact(m,picked[group])[0];build=time.monotonic()-t;info=model_metrics(model);assert info['parameter_count']==original_params-(0 if group=='B0' else 8440320);info['compression_fraction']=1-info['parameter_count']/original_params;info['construction_seconds']=build;weightpath=E/'models'/f'{group}.pth';weightpath.parent.mkdir(exist_ok=True);torch.save({'format':'pure_inference_state_dict','group':group,'parent_sha256':C['parent_sha256'],'hidden_retained':2048 if group=='B0' else 1536,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}},weightpath);info['inference_file_bytes']=weightpath.stat().st_size;info['inference_file_sha256']=sha(weightpath);manifest.append({'group':group,'path':str(weightpath),'sha256':info['inference_file_sha256'],'bytes':info['inference_file_bytes'],'includes_optimizer':False});counts[group]=info;dump('parameter_count_audit.json',{'original_parameter_count':original_params,'groups':counts});dump('checkpoint_manifest.json',manifest)
  cr,lr,fr,_=evaluate(model,S['D_cal'],group,'D_cal',False);vr,lr2,fr2,ss=evaluate(model,spl['validation'],group,'validation',True);cr.update(info);vr.update(info);allcal.append(cr);allval.append(vr);alllayers+=lr+lr2;allfr+=fr+fr2;allsamples+=ss;write('calibration_summary.csv',allcal);write('validation_summary.csv',allval);write('layer_sops.csv',alllayers);write('layer_FR.csv',allfr);write('validation_per_sample.csv',allsamples);print('completed group',group,vr['accuracy'],vr['SOP_sample_weighted'],flush=True)
  if group!='B0':del model;gc.collect();torch.cuda.empty_cache()
 # Matched sample-level comparisons.
 by={g:{r['sample_id']:r for r in allsamples if r['group']==g} for g in ['B0','S0','S1','S2','S3','M5']};rng=np.random.default_rng(20261128);boot={}
 for refgroup in ['S1','S2','S3','B0','S0']:
  ids=sorted(by['M5']);assert ids==sorted(by[refgroup]) and all(by['M5'][i]['label']==by[refgroup][i]['label'] for i in ids);da=np.array([by['M5'][i]['correct']-by[refgroup][i]['correct'] for i in ids]);dc=np.array([by['M5'][i]['ce']-by[refgroup][i]['ce'] for i in ids]);ix=rng.integers(0,len(ids),size=(2000,len(ids)));boot['M5-'+refgroup]={'accuracy_delta_pp':100*float(da.mean()),'accuracy_ci95_pp':(100*np.quantile(da[ix].mean(1),[.025,.975])).tolist(),'CE_delta':float(dc.mean()),'CE_ci95':np.quantile(dc[ix].mean(1),[.025,.975]).tolist(),'reference_wrong_M5_right':int((da==1).sum()),'reference_right_M5_wrong':int((da==-1).sum()),'prediction_disagree':sum(by['M5'][i]['prediction']!=by[refgroup][i]['prediction'] for i in ids),'per_sample_SOP':'NA_not_collected'}
 dump('paired_bootstrap.json',{'rng':'numpy.default_rng PCG64','seed':20261128,'resamples':2000,'method':'percentile','comparisons':boot});vr={x['group']:x for x in allval};checks={'all_admission_checks':json.loads((E/'admission_gate_audit.json').read_text())['status']=='PASSED' and json.loads((E/'selected_structure_audit.json').read_text())['status']=='PASSED','physical_parameters_drop_ge_20pct':counts['M5']['compression_fraction']>=.2,'M5_accuracy_drop_B0_le_0_5pp':vr['B0']['accuracy']-vr['M5']['accuracy']<=.005,'M5_accuracy_gain_S1_ge_0_3pp':boot['M5-S1']['accuracy_delta_pp']>=.3,'M5_accuracy_gain_S2_ge_0_3pp':boot['M5-S2']['accuracy_delta_pp']>=.3,'M5_accuracy_gain_S3_ge_0_3pp':boot['M5-S3']['accuracy_delta_pp']>=.3,'M5_SOP_at_most_B0_101pct':vr['M5']['SOP_sample_weighted']<=1.01*vr['B0']['SOP_sample_weighted'],'equal_pruned_parameter_budget':len({counts[g]['parameter_count'] for g in ['S0','S1','S2','S3','M5']})==1,'samples_and_data_isolation':True};dump('hypothesis_gate_results.json',{'checks':checks,'progression_passed':all(checks.values()),'H024_scientific_status':'UNTESTED_pending_theory_review','H025_scientific_status':'UNTESTED_pending_theory_review'});return 'COMPLETED'

try:status['status']=main()
except BaseException as e:status.update(status='PARTIAL',error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
finally:
 try:
  after=sha(C['parent_path']);dump('parent_checkpoint_audit.json',{'before':C['parent_sha256'],'after':after,'expected':C['parent_sha256'],'match':after==C['parent_sha256']})
 except BaseException:pass
 status.update(end_unix=time.time(),wall_seconds=time.time()-K.START,cpu_seconds=K.cpu()-K.CPU,peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None);dump('status.json',status);print(json.dumps(status),flush=True)
if status['status']!='COMPLETED':sys.exit(1)

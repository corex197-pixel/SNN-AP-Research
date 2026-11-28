"""Stage D only: guarded fixed-validation entry, seven actual models and temporal outputs."""
import os,time,json,csv,traceback
from common import *
from utils.utils import SOPMonitor
import gate
stage=os.environ.get('ROUND027_STAGE_ID','evaluate-manual');started=time.monotonic();cpu0=K.cpu()
def guard():
 if time.monotonic()-started>=float(os.environ['ROUND027_REMAIN_WALL'])-35 or K.cpu()-cpu0>=float(os.environ['ROUND027_REMAIN_CPU'])-60:raise TimeoutError('evaluation resource reserve')
def eval_one(m,dl,group):
 m.eval();mon=SOPMonitor(m);handles=[];spikes={};states={};layers={n:0. for n in mon.monitored_layers};ss=[];zs=[];allids=[];labels=[];total=correct=0;ce=0.;batches=[]
 for name,module in m.named_modules():
  if isinstance(module,SparseNeuron):
   def hook(mod,args,out,n=name):spikes[n]=spikes.get(n,0)+int((out!=0).sum());states[n]=states.get(n,0)+out.numel()
   handles.append(module.register_forward_hook(hook))
 try:
  with torch.no_grad():
   for x,y,ids in dl:
    guard();x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);functional.reset_net(m);mon.clear_recorded_data();zt=m(x);z=zt.mean(0);loss=F.cross_entropy(z,y,reduction='none');assert bool(torch.isfinite(z).all()) and bool(torch.isfinite(loss).all());pred=z.argmax(1);n=len(y);total+=n;correct+=int((pred==y).sum());ce+=float(loss.sum());batches.append(n);zs.append(zt.detach().float().cpu().numpy());allids.extend(ids.tolist());labels+=y.cpu().tolist()
    for layer in mon.monitored_layers:layers[layer]+=sum(float(v.item()) for v in mon[layer])
    ss.extend({'group':group,'sample_id':int(i),'label':int(t),'prediction':int(p),'correct':int(p==t),'ce':float(l)} for i,t,p,l in zip(ids.tolist(),y.cpu().tolist(),pred.cpu().tolist(),loss.cpu().tolist()));functional.reset_net(m)
 finally:
  mon.remove_hooks()
  for h in handles:h.remove()
  functional.reset_net(m)
 assert total==4500;nonzero=totalw=0
 for module in m.modules():
  if isinstance(module,SparseConv2d):mask=module.sparse_mask();nonzero+=int(mask.sum());totalw+=mask.numel()
 params=sum(p.numel() for p in m.parameters());result={'group':group,'samples':total,'accuracy':correct/total,'CE':ce/total,'FR_global':sum(spikes.values())/sum(states.values()),'SOP_sample_weighted':sum(layers.values())/total,'SOP_author':sum(layers.values())/len(batches)/64,'parameter_count':params,'compression_fraction':1-params/36720484,'effective_nonzero_weights':nonzero,'effective_weight_sparsity':1-nonzero/totalw,'validation_seconds':time.monotonic()-tgroup,'last_batch_size':batches[-1]};lr=[{'group':group,'layer':n,'SOP_sample_weighted':v/total} for n,v in layers.items()];fr=[{'group':group,'layer':n,'FR':spikes[n]/states[n]} for n in spikes];zarr=np.concatenate(zs,axis=1);assert zarr.shape==(4,4500,10);path=E/'models'/'temporal'/f'{group}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,logits=zarr,ids=np.asarray(allids,dtype=np.int64),labels=np.asarray(labels,dtype=np.int64));return result,lr,fr,ss,{'path':str(path),'sha256':sha(path),'shape':list(zarr.shape),'dtype':str(zarr.dtype),'ids_sha256':idhash(allids)},zarr

def actual_evaluation(dl):
 global tgroup
 init_runtime();teacher=model_teacher();p0,_,_=model_p0(teacher);basehash=statehash(teacher);p0hash=statehash(p0);manifest=json.loads((E/'checkpoint_manifest.json').read_text());groups=['B0','P0']+NAMES;vals=[];layers=[];frs=[];samples=[];temporal={};arrays={}
 for group in groups:
  guard();tgroup=time.monotonic()
  if group=='B0':m=teacher;size=B0.stat().st_size
  elif group=='P0':m=p0;size=P0.stat().st_size
  else:
   m,_,_=model_p0(teacher);p=Path(manifest[group]['path']);assert sha(p)==manifest[group]['sha256'];c=torch.load(p,map_location='cpu',weights_only=False);assert c['group']==group and c['epoch']==5;m.load_state_dict(c['model']);size=p.stat().st_size
  r,lr,fr,ss,tm,za=eval_one(m,dl,group);r['inference_file_bytes']=size;vals.append(r);layers+=lr;frs+=fr;samples+=ss;temporal[group]=tm;arrays[group]=za;write('validation_summary.csv',vals);write('layer_sops.csv',layers);write('layer_FR.csv',frs);write('validation_per_sample.csv',samples);dump('temporal_logits_manifest.json',temporal);print('EVALUATED',group,r['accuracy'],r['CE'],flush=True)
  if group not in ('B0','P0'):del m,c;torch.cuda.empty_cache()
 assert statehash(teacher)==basehash and statehash(p0)==p0hash
 # Temporal loss/margin diagnostics are post-lock computations on saved true logits.
 teacherz=torch.from_numpy(arrays['B0']);p0z=torch.from_numpy(arrays['P0']);labs=torch.tensor([x['label'] for x in samples if x['group']=='B0']);d,w,competitor,mt,mp=weights(teacherz,p0z,labs);diags=[]
 for g in ['P0']+NAMES:
  z=torch.from_numpy(arrays[g]);pt=F.softmax(teacherz/4,dim=-1);lt=F.log_softmax(teacherz/4,dim=-1);ls=F.log_softmax(z/4,dim=-1);kl=(pt*(lt-ls)).sum(-1);mtmean=teacherz.mean(0)/4;zm=z.mean(0)/4;p=F.softmax(mtmean,-1);mean_kl=(p*(F.log_softmax(mtmean,-1)-F.log_softmax(zm,-1))).sum(-1);ix=labs.view(1,-1,1).expand(4,-1,1);mg=z.gather(2,ix).squeeze(2)-z.gather(2,competitor.unsqueeze(2)).squeeze(2)
  for t in range(4):diags.append({'group':g,'t':t,'teacher_student_KL':float(kl[t].mean()),'mean_logits_KL':float(mean_kl.mean()),'mean_margin_loss':float((mt[t]-mg[t]).mean()),'P0_damage_weight_mean':float(w[t].mean()),'P0_damage_mean':float(d[t].mean()),'P0_weighted_KL_mean':float((w[t]*kl[t]).mean())})
 write('temporal_distillation_diagnostics.csv',diags);by={g:{r['sample_id']:r for r in samples if r['group']==g} for g in groups};hist={int(r['sample_id']):r for r in csv.DictReader((ROOT/'experiments/EXP-028/validation_per_sample.csv').open()) if r['group']=='F0'};assert set(hist)==set(by['K0']);prior=next(r for r in csv.DictReader((ROOT/'experiments/EXP-028/validation_summary.csv').open()) if r['group']=='F0');cur=next(r for r in vals if r['group']=='K0');dump('K0_historical_reproduction_audit.json',{'accuracy_delta':cur['accuracy']-float(prior['accuracy']),'CE_delta':cur['CE']-float(prior['CE']),'FR_delta':cur['FR_global']-float(prior['FR_global']),'SOP_delta':cur['SOP_sample_weighted']-float(prior['SOP_sample_weighted']),'prediction_disagreement_count':sum(int(hist[i]['prediction'])!=by['K0'][i]['prediction'] for i in hist),'scope':'same parent/config and single-seed historical rerun; not independent training-seed evidence'})
 rng=np.random.default_rng(20261128);boot={};pairs=[('K1','K0'),('K2','K1')]+[('M6',g) for g in ['K0','K1','K2','K3','B0']]+[(g,'P0') for g in NAMES]
 for a,b in pairs:
  ids=sorted(by[a]);assert ids==sorted(by[b]) and all(by[a][i]['label']==by[b][i]['label'] for i in ids);da=np.array([by[a][i]['correct']-by[b][i]['correct'] for i in ids]);dc=np.array([by[a][i]['ce']-by[b][i]['ce'] for i in ids]);ix=rng.integers(0,len(ids),(2000,len(ids)));boot[a+'-'+b]={'accuracy_delta_pp':float(100*da.mean()),'accuracy_ci95_pp':(100*np.quantile(da[ix].mean(1),[.025,.975])).tolist(),'CE_delta':float(dc.mean()),'CE_ci95':np.quantile(dc[ix].mean(1),[.025,.975]).tolist(),'reference_wrong_candidate_right':int((da==1).sum()),'reference_right_candidate_wrong':int((da==-1).sum()),'prediction_disagree':sum(by[a][i]['prediction']!=by[b][i]['prediction'] for i in ids),'SOP_ci':'NA'}
 dump('paired_bootstrap.json',{'seed':20261128,'rng':'numpy.default_rng PCG64','replicates':2000,'interval':'uncorrected exploratory percentile','comparisons':boot});vv={r['group']:r for r in vals};admission=not gate.verify(lock,'evaluation',identity,config_sha,remote_sha,commit['sha'],lock['expected_samples']);checks={g:vv['M6']['accuracy']-vv[g]['accuracy']>=.003 for g in ['K0','K1','K2','K3']};conditions={'M6_gain_vs_all_ge_0_3pp':all(checks.values()),'B0_M6_gap_le_0_5pp':vv['B0']['accuracy']-vv['M6']['accuracy']<=.005,'M6_CE_le_K2':vv['M6']['CE']<=vv['K2']['CE'],'parameter_fixed':vv['M6']['parameter_count']==11399524,'SOP_le_101pct_P0':vv['M6']['SOP_sample_weighted']<=1.01*vv['P0']['SOP_sample_weighted'],'admissions_valid':admission,'data_and_training_controls':json.loads((E/'endpoint_tests.json').read_text())['status']=='PASSED'};den=vv['B0']['accuracy']-vv['P0']['accuracy'];dump('hypothesis_gate_results.json',{'H030_K1_gain_ge_0_3pp':vv['K1']['accuracy']-vv['K0']['accuracy']>=.003,'H031_K2_gain_ge_0_3pp':vv['K2']['accuracy']-vv['K1']['accuracy']>=.003,'H032_M6_vs_K2_K3_ge_0_3pp':checks['K2'] and checks['K3'],'M6_vs_comparators':checks,'overall_M6_checks':conditions,'overall_M6_pass':all(conditions.values()),'recovery_ratio':{g:(vv[g]['accuracy']-vv['P0']['accuracy'])/den if den>0 else None for g in NAMES},'scientific_status':'UNTESTED_pending_theory_review'});dump('evaluation_teacher_protection_audit.json',{'teacher_before':basehash,'teacher_after':statehash(teacher),'P0_before':p0hash,'P0_after':statehash(p0),'manual_penalty_calls':manual_calls['count']});dump('status.json',{'status':'COMPLETED','validity':'VALID','phase':'evaluate','stage_id':stage,'code_sha':CODE});print('EVALUATION_COMPLETED',flush=True)

def main():
 global lock,identity,config_sha,remote_sha,commit
 lock=json.loads((E/'evaluation_admission_lock.json').read_text());commit=json.loads((E/'evaluation_lock_commit.json').read_text());assert sha(E/'evaluation_admission_lock.json')==commit['lock_sha256'] and sha(E/'checkpoint_manifest.json')==lock['checkpoint_manifest_sha256'] and sha(E/'epoch_metrics.csv')==lock['epoch_metrics_sha256'] and sha(E/'training_sample_audit.csv')==lock['sample_audit_sha256'];identity={g:sha(Path(x['path'])) for g,x in json.loads((E/'checkpoint_manifest.json').read_text()).items()};assert sha(B0)==CFG['teacher_sha256'] and sha(P0)==CFG['p0_sha256'] and sha(SPL)==CFG['split_sha256'];config_sha=sha(E/'config.json');remote_sha=subprocess.check_output(['git','ls-remote','origin','refs/heads/research/round-027-pruning-damage-distillation'],text=True,timeout=60).split()[0]
 def make_validation_loader():
  # Only executed by the real fail-closed entry after candidate lock and remote checks.
  ids=json.loads(SPL.read_text())['validation'];assert len(ids)==4500;dump('evaluation_entry_audit.json',{'validated_lock_sha256':commit['lock_sha256'],'remote_commit':remote_sha,'model_hashes':identity,'loader_created_after_entry_validation':True});return eval_loader(ids)
 return gate.stage_entry(lock,'evaluation',identity,config_sha,remote_sha,commit['sha'],lock['expected_samples'],make_validation_loader,actual_evaluation)
try:main()
except BaseException as ex:dump('status.json',{'status':'PARTIAL','validity':'NOT_ESTABLISHED','phase':'evaluate','stage_id':stage,'code_sha':CODE,'error':repr(ex),'traceback':traceback.format_exc()});traceback.print_exc();raise

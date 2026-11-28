"""EXP-022: full-network task-gradient conflict and conditional protected evaluation."""
import os,sys,json,csv,time,hashlib,subprocess,traceback,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'experiments/ROUND-019'),str(ROOT/'experiments/ROUND-020')]
import core as K
K.E=ROOT/'experiments/EXP-022';K.status.update(run_id='EXP-022-RUN-001',phase='preflight');K.CPU=K.cpu();K.START=time.time();K.PRIOR_CPU=K.PRIOR_WALL=0
from core import *
from torch.nn import functional as F
from spikingjelly.activation_based import neuron
import projection
E=K.E;status=K.status;CODE=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();status.update(code_sha=CODE,start_unix=K.START)
Q0=0.04546005512832126;Q1=.04

def hash_cpu(x):
 t=x.detach().to('cpu').contiguous().numpy();return hashlib.sha256(t.tobytes()).hexdigest()
def data_audit():
 s=json.loads((ROOT/'experiments/EXP-004/splits/indices.json').read_text());assert sha(ROOT/'experiments/EXP-004/splits/indices.json')==C['split_sha256'];expected={'D_grad':'93c4c834c3fd3c8b92a64caa88da8a671e27be23dc2df11999c8e5f3dfbe47d8','D_cal':'b671cbd803b13cc76b02f5db4bad311ef9492276622051df01af56a931301baa'}
 for n,h in expected.items():assert hashlib.sha256(json.dumps(S[n],separators=(',',':')).encode()).hexdigest()==h and len(S[n])==512
 assert not set(S['D_grad'])&set(S['D_cal']) and not(set(S['D_grad'])|set(S['D_cal']))&(set(s['validation'])|set(s['diagnostic']));dump('data_split_audit.json',{'status':'PASSED','hashes':expected,'disjoint':True,'train_origin':True});return s

def read_activity():
 p=ROOT/'experiments/EXP-021/gradient_vectors.csv';digest=sha(p);assert digest=='19426f0b80dd5bcce4c09a4cc911a6cb3b5279b52decf21b44cb6834cc1a591c';rows=list(csv.DictReader(p.open()));assert len(rows)==3072;activity={};output=[]
 for bn,_,_ in T:
  activity[bn]={}
  for group in ['G0','G1']:
   a=[r for r in rows if r['bn']==bn and r['group']==group];assert len(a)==512
   for j,z in enumerate(a):assert z['parameter']==('gamma' if j<256 else 'beta') and int(z['channel'])==j%256
   v=torch.tensor([float(r['raw']) for r in a],dtype=torch.float64,device='cuda');ref=torch.tensor([float(r['matched']) for r in a],dtype=torch.float64,device='cuda');assert torch.isfinite(v).all() and torch.isfinite(ref).all();activity[bn][group]=v
   output.extend({'bn':bn,'group':group,'parameter':z['parameter'],'channel':z['channel'],'raw':z['raw'],'historical_matched':z['matched']} for z in a)
  for group in ['G0','G1']:
   x=activity[bn][group];r=torch.tensor([float(z['matched']) for z in rows if z['bn']==bn and z['group']==group],dtype=torch.float64,device='cuda');expected=math_core.match(x,activity[bn]['G0']);assert torch.equal(r,expected)
 write('activity_gradient_vectors.csv',output);dump('historical_reconstruction_audit.json',{'status':'PASSED','activity_vector_source_sha256':digest,'row_order_and_raw_matched_verified':True});return activity

def task_grad(m,data):
 base={n:v.detach().clone() for n,v in m.state_dict().items()};params=[]
 for bn,_,_ in T:
  z=get(m,bn);z.weight.requires_grad_(True);z.bias.requires_grad_(True);params.extend([z.weight,z.bias])
 sums={bn:torch.zeros(512,dtype=torch.float64,device='cuda') for bn,_,_ in T};halves=[{bn:torch.zeros(512,dtype=torch.float64,device='cuda') for bn,_,_ in T} for _ in range(2)];batchrows=[];forward=[];count=0
 for i,(x,y,ids) in enumerate(data):
  guard();x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);refhash={};trainhash={};hooks=[]
  for n,z in m.named_modules():
   if isinstance(z,SparseNeuron):hooks.append(z.register_forward_hook(lambda mod,args,out,n=n:refhash.__setitem__(n,hash_cpu(out))))
  m.eval();functional.reset_net(m)
  with torch.no_grad():baseout=m(x).detach().clone()
  functional.reset_net(m)
  for h in hooks:h.remove()
  hooks=[]
  for n,z in m.named_modules():
   if isinstance(z,SparseNeuron):hooks.append(z.register_forward_hook(lambda mod,args,out,n=n:trainhash.__setitem__(n,hash_cpu(out))))
  for z in m.modules():
   if isinstance(z,neuron.BaseNode):z.train()
  assert all(not z.training for z in m.modules() if isinstance(z,torch.nn.modules.batchnorm._BatchNorm))
  out=m(x);assert torch.equal(out.detach(),baseout) and refhash==trainhash;loss=F.cross_entropy(out.mean(0),y);assert torch.isfinite(loss);gg=torch.autograd.grad(loss,params);n=len(y);count+=n
  for j,(bn,_,_) in enumerate(T):
   v=torch.cat([gg[2*j].detach().double(),gg[2*j+1].detach().double()]);assert torch.isfinite(v).all();sums[bn]+=v*n;halves[0 if i<4 else 1][bn]+=v*n;batchrows.append({'batch':i,'samples':n,'bn':bn,'task_grad_norm':torch.linalg.vector_norm(v).item(),'loss':loss.item()})
  forward.append({'batch':i,'samples':n,'logits_exact':True,'all_neuron_spike_hashes_match':True,'neuron_count':len(refhash)});functional.reset_net(m);m.eval()
  for h in hooks:h.remove()
  del out,gg,loss,baseout,x,y,refhash,trainhash
 assert count==512
 for n,v in m.state_dict().items():assert torch.equal(v,base[n]),'task gradient mutated '+n
 for p in params:p.requires_grad_(False)
 ts={bn:v/count for bn,v in sums.items()};rows=[];stability=[]
 for bn,_,_ in T:
  a=halves[0][bn]/256;b=halves[1][bn]/256;na=torch.linalg.vector_norm(a);nb=torch.linalg.vector_norm(b);cs=torch.dot(a,b)/(na*nb) if na>0 and nb>0 else None;stability.append({'bn':bn,'first_norm':na.item(),'second_norm':nb.item(),'cosine_first_second':cs.item() if cs is not None else None});v=ts[bn]
  for j in range(512):rows.append({'bn':bn,'parameter':'gamma' if j<256 else 'beta','channel':j%256,'task_gradient':v[j].item(),'first_256':a[j].item(),'second_256':b[j].item()})
 write('task_gradient_vectors.csv',rows);write('task_gradient_batches.csv',batchrows);dump('gradient_stability.json',{'status':'COMPLETED','halves':stability});dump('forward_equivalence_audit.json',{'status':'PASSED','batches':forward,'all_BN_eval':True,'original_cupy_atan_nodes_only_training':True,'model_state_exact':True});return ts

def diagnose(t,g):
 rows=[];channels=[];projected={};pvec=[];pa=[];passes=[]
 for bn,_,_ in T:
  projected[bn]={};task=t[bn];tn=torch.linalg.vector_norm(task)
  for group in ['G0','G1']:
   a=g[bn][group];an=torch.linalg.vector_norm(a);dot=torch.dot(task,a);cos=dot/(tn*an) if tn>0 and an>0 else None;v=projection.project(task,a);vn=torch.linalg.vector_norm(v);removed=torch.linalg.vector_norm(a-v);post=torch.dot(task,v)/(tn*vn) if tn>0 and vn>0 else None
   valid=not (torch.linalg.vector_norm(g[bn]['G0'])>0 and vn/torch.linalg.vector_norm(g[bn]['G0'])<=1e-8)
   if post is not None:assert post.item()>=-1e-8
   if group=='G1':passes.append(cos is not None and cos.item() < -1e-6)
   tag='P0' if group=='G0' else 'P1';projected[bn][tag]=v;projected[bn][group]=a
   rows.append({'bn':bn,'group':group,'task_norm':tn.item(),'activity_norm':an.item(),'dot':dot.item(),'cosine':cos.item() if cos is not None else 'NA','conflict_component_fraction':(removed/an).item() if an>0 else 0,'negative_channel_fraction':sum((task[:256]*a[:256]+task[256:]*a[256:])<0).item()/256});pa.append({'bn':bn,'group':tag,'before_dot':dot.item(),'before_cosine':cos.item() if cos is not None else None,'after_dot':torch.dot(task,v).item(),'after_cosine':post.item() if post is not None else None,'removed_norm':removed.item(),'removed_fraction':(removed/an).item() if an>0 else 0,'nondegenerate':bool(valid)})
   for j in range(256):channels.append({'bn':bn,'group':group,'channel':j,'task_activity_dot':(task[j]*a[j]+task[j+256]*a[j+256]).item()})
   for j in range(512):pvec.append({'bn':bn,'group':tag,'parameter':'gamma' if j<256 else 'beta','channel':j%256,'original':a[j].item(),'projected':v[j].item(),'removed':(a[j]-v[j]).item()})
 write('gradient_conflict_statistics.csv',rows);write('channel_conflict_statistics.csv',channels);write('projected_gradient_vectors.csv',pvec);dump('projection_audit.json',{'status':'PASSED' if all(x['nondegenerate'] for x in pa) else 'DEGENERATE','layers':pa});dump('formula_tests.json',projection.unit_tests());dump('hypothesis_gate_results.json',{'H020_direction_passed':any(passes),'H020_scientific_status':'UNTESTED_pending_theory_review','H021_scientific_status':'UNTESTED','projection_nondegenerate':all(x['nondegenerate'] for x in pa)});return projected,any(passes),all(x['nondegenerate'] for x in pa)
def performance(m,t,raw,proj,spl):
 parent={n:v.detach().clone() for n,v in m.state_dict().items()};targetkeys={f'{bn}.{k}' for bn,_,_ in T for k in ('weight','bias')};params=torch.cat([parent[k].double().flatten() for k in sorted(targetkeys)]);normphi=torch.linalg.vector_norm(params);normg=torch.sqrt(sum(torch.linalg.vector_norm(raw[bn]['G0'])**2 for bn,_,_ in T));assert torch.isfinite(normphi) and torch.isfinite(normg) and normg>0
 g={bn:{k:projection.normalized(proj[bn][k],raw[bn]['G0']) for k in ['G0','G1','P0','P1']} for bn,_,_ in T}
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
 status['phase']='same_scale';dump('status.json',status);cal=loader(S['D_cal']);layerrows=[];results=[];responses=[];trace=[]
 def measure(group,q,method):
  r,lr,_,_=evaluate(cal,'G0' if group=='B0' else group,q);r['group']=group;r['method']=method;r['predicted_task_delta']=sum((-q*normphi/normg*torch.dot(t[bn],g[bn]['G0' if group=='B0' else group])).item() for bn,_,_ in T);responses.append(r);layerrows.extend(dict(x,group=group,dataset='D_cal') for x in lr);write('calibration_response.csv',responses);write('layer_sops.csv',layerrows);dump('parameter_integrity_audit.json',{'status':'PASSED','checks':integrity});write('update_norm_audit.csv',normrows);print('D_cal',group,q,r['SOP_sample_weighted'],flush=True);return r
 b0=measure('B0',0,'baseline');same={group:measure(group,.02,'primary_same_q') for group in ['G0','G1','P0','P1']}
 b0['actual_CE_delta_vs_B0']=0.0
 for x in same.values():x['actual_CE_delta_vs_B0']=x['ce']-b0['ce']
 write('mechanism_same_q.csv',[b0]+list(same.values()))
 hist=list(csv.DictReader((ROOT/'experiments/EXP-021/calibration_response.csv').open()));oldb=next(x for x in hist if x['group']=='B0');
 def check_history(now,group,q):
  old=next(x for x in hist if x['group']==group and abs(float(x['q'])-q)<1e-13)
  for k in ['SOP_sample_weighted','accuracy','ce']:assert abs(now[k]-float(old[k]))/max(1,abs(float(old[k])))<1e-5
 check_history(b0,'B0',0.);g0=measure('G0',Q0,'historical_fixed');g1=measure('G1',Q1,'historical_anchor');check_history(g0,'G0',Q0);check_history(g1,'G1',Q1);bcost=b0['SOP_sample_weighted'];target=1-g1['SOP_sample_weighted']/bcost;assert .04<=target<=.06
 selected={'G0':Q0,'G1':Q1};matched={}
 for group in ['P0','P1']:
  vals=[dict(b0,group=group),same[group]]
  for q in [.04,.08,.16]:
   vals.append(measure(group,q,'fixed_bound'));ordered=sorted(vals,key=lambda x:x['q']);rr=[1-x['SOP_sample_weighted']/bcost for x in ordered]
   if any(rr[i]>rr[i+1]+1e-8 for i in range(len(rr)-1)):break
   if any(rr[i]<=target<=rr[i+1] for i in range(len(rr)-1)):break
  ordered=sorted(vals,key=lambda x:x['q']);rr=[1-x['SOP_sample_weighted']/bcost for x in ordered];monotone=all(rr[i]<=rr[i+1]+1e-8 for i in range(len(rr)-1));bracket=next(((ordered[i],ordered[i+1]) for i in range(len(rr)-1) if rr[i]<=target<=rr[i+1] and rr[i+1]>rr[i]),None)
  if monotone and bracket:
   a,b=bracket;lo=a['q'];hi=b['q'];ra=1-a['SOP_sample_weighted']/bcost;rb=1-b['SOP_sample_weighted']/bcost;q=lo+(target-ra)*(hi-lo)/(rb-ra);v=measure(group,q,'R82_interpolation');vals.append(v);rv=1-v['SOP_sample_weighted']/bcost
   if abs(rv-target)>.001 and ra<=rv<=rb:
    if rv<target:lo=q
    else:hi=q
    for i in range(4):
     q=(lo+hi)/2;v=measure(group,q,'bisection_'+str(i+1));vals.append(v);rv=1-v['SOP_sample_weighted']/bcost
     if abs(rv-target)<=.001:break
     if rv<target:lo=q
     else:hi=q
  for v in vals:
   rr=1-v['SOP_sample_weighted']/bcost;trace.append({'group':group,'method':v['method'],'q':v['q'],'sops':v['SOP_sample_weighted'],'reduction':rr,'target':target,'error':abs(rr-target),'monotone':monotone,'matched':.04<=rr<=.06 and abs(rr-target)<=.001})
  good=[v for v in vals if .04<=1-v['SOP_sample_weighted']/bcost<=.06 and abs(1-v['SOP_sample_weighted']/bcost-target)<=.001];best=min(good,key=lambda v:(abs(1-v['SOP_sample_weighted']/bcost-target),v['q'])) if good else None;selected[group]=None if best is None else best['q'];matched[group]=None if best is None else 1-best['SOP_sample_weighted']/bcost
 write('budget_matching_trace.csv',trace)
 lock={'parent_sha256':C['parent_sha256'],'data_grad_sha256':'93c4c834c3fd3c8b92a64caa88da8a671e27be23dc2df11999c8e5f3dfbe47d8','data_cal_sha256':'b671cbd803b13cc76b02f5db4bad311ef9492276622051df01af56a931301baa','task_gradient_sha256':sha(E/'task_gradient_vectors.csv'),'activity_gradient_sha256':sha(E/'activity_gradient_vectors.csv'),'projected_gradient_sha256':sha(E/'projected_gradient_vectors.csv'),'trace_sha256':sha(E/'budget_matching_trace.csv'),'G1_anchor_reduction':target,'selections':selected,'matched_reductions':matched,'locked_before_validation':True};dump('selection_lock.json',lock)
 subprocess.run(['git','add','experiments/EXP-022/selection_lock.json','experiments/EXP-022/budget_matching_trace.csv','experiments/EXP-022/task_gradient_vectors.csv','experiments/EXP-022/activity_gradient_vectors.csv','experiments/EXP-022/projected_gradient_vectors.csv'],check=True);subprocess.run(['git','commit','-m','lock(EXP-022): fix protected matched-budget directions'],check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);locksha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();dump('lock_commit.json',{'sha':locksha,'before_validation':True});print('locked',locksha,selected,flush=True)
 if selected['P0'] is None or selected['P1'] is None:return 'PARTIAL_BUDGET_MATCH'
 status['phase']='validation';dump('status.json',status);val=loader(spl['validation']);summary=[];samples=[]
 for group in ['B0','G0','G1','P0','P1']:
  q=0 if group=='B0' else selected[group];r,lr,_,ss=evaluate(val,'G0' if group=='B0' else group,q,True);r['group']=group
  for x in lr:x.update(group=group,dataset='validation')
  for x in ss:x['group']=group
  summary.append(r);samples.extend(ss);layerrows.extend(lr);write('validation_summary.csv',summary);write('validation_per_sample.csv',samples);write('layer_sops.csv',layerrows)
 old=list(csv.DictReader((ROOT/'experiments/EXP-021/validation_per_sample.csv').open()));oldmap={(x['group'],int(x['sample_id'])):x for x in old};recon={}
 for group in ['B0','G0','G1']:
  rr=[x for x in samples if x['group']==group];mismatch=0;maxce=0.
  for x in rr:
   z=oldmap[(group,x['sample_id'])];assert int(z['label'])==x['label'];mismatch+=int(int(z['prediction'])!=x['prediction']);maxce=max(maxce,abs(float(z['ce'])-x['ce']))
  osum=next(x for x in csv.DictReader((ROOT/'experiments/EXP-021/validation_summary.csv').open()) if x['group']==group);nsum=next(x for x in summary if x['group']==group);err=abs(float(osum['SOP_sample_weighted'])-nsum['SOP_sample_weighted'])/nsum['SOP_sample_weighted'];recon[group]={'prediction_mismatches':mismatch,'max_CE_abs_difference':maxce,'SOP_relative_error':err};assert mismatch==0 and maxce<1e-5 and err<1e-5
 h=json.loads((E/'historical_reconstruction_audit.json').read_text());h['validation_reconstruction']=recon;dump('historical_reconstruction_audit.json',h)
 by={z:{x['sample_id']:x for x in samples if x['group']==z} for z in ['B0','G0','G1','P0','P1']};rng=np.random.default_rng(20261128);boot={}
 for exp,ref in [('P1','G0'),('P1','G1'),('P1','P0'),('P0','G0'),('G1','G0'),('P1','B0')]:
  ids=sorted(by[exp]);assert ids==sorted(by[ref]) and all(by[exp][i]['label']==by[ref][i]['label'] for i in ids);da=np.array([by[exp][i]['correct']-by[ref][i]['correct'] for i in ids]);dc=np.array([by[exp][i]['ce']-by[ref][i]['ce'] for i in ids]);ix=rng.integers(0,len(ids),size=(2000,len(ids)));boot[exp+'-'+ref]={'accuracy_delta_pp':100*float(da.mean()),'accuracy_ci95_pp':(100*np.quantile(da[ix].mean(1),[.025,.975])).tolist(),'CE_delta':float(dc.mean()),'CE_ci95':np.quantile(dc[ix].mean(1),[.025,.975]).tolist(),'reference_wrong_experiment_right':int((da==1).sum()),'reference_right_experiment_wrong':int((da==-1).sum()),'prediction_disagree':sum(by[exp][i]['prediction']!=by[ref][i]['prediction'] for i in ids),'per_sample_sop':'NA_not_collected'}
 dump('paired_bootstrap.json',{'rng':'numpy.default_rng PCG64','seed':20261128,'resamples':2000,'method':'percentile','comparisons':boot})
 ids=sorted(by['B0']);
 for group in ['G0','G1','P0','P1']:assert ids==sorted(by[group]) and all(by[group][i]['label']==by['B0'][i]['label'] for i in ids)
 da=np.array([(by['P1'][i]['correct']-by['G1'][i]['correct'])-(by['P0'][i]['correct']-by['G0'][i]['correct']) for i in ids]);dc=np.array([(by['P1'][i]['ce']-by['G1'][i]['ce'])-(by['P0'][i]['ce']-by['G0'][i]['ce']) for i in ids]);ix=rng.integers(0,len(ids),size=(2000,len(ids)));dump('factorial_interaction.json',{'accuracy_interaction_pp':100*float(da.mean()),'accuracy_ci95_pp':(100*np.quantile(da[ix].mean(1),[.025,.975])).tolist(),'CE_interaction':float(dc.mean()),'CE_ci95':np.quantile(dc[ix].mean(1),[.025,.975]).tolist(),'rng':'PCG64 continuation','seed':20261128,'resamples':2000,'method':'percentile'})
 dump('parameter_integrity_audit.json',{'status':'PASSED','checks':integrity});write('update_norm_audit.csv',normrows);return 'COMPLETED'

def main():
 guard();assert torch.cuda.device_count()==1 and '4090' in torch.cuda.get_device_name();torch.set_num_threads(2);torch.manual_seed(20261128);np.random.seed(20261128);random.seed(20261128);spl=data_audit();dump('formula_tests.json',projection.unit_tests());m=model();dump('parent_checkpoint_audit.json',{'before':sha(C['parent_path']),'expected':C['parent_sha256']});raw=read_activity();status['phase']='task_gradient';dump('status.json',status);task=task_grad(m,loader(S['D_grad']));proj,gate,nondeg=diagnose(task,raw)
 if not gate:return 'PARTIAL_NO_CONFLICT'
 if not nondeg:return 'PARTIAL_PROJECTION_DEGENERATE'
 return performance(m,task,raw,proj,spl)
try:status['status']=main()
except BaseException as e:status.update(status='PARTIAL',error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
finally:
 try:
  after=sha(C['parent_path']);dump('parent_checkpoint_audit.json',{'before':C['parent_sha256'],'after':after,'expected':C['parent_sha256'],'match':after==C['parent_sha256']})
 except BaseException:pass
 status.update(end_unix=time.time(),wall_seconds=time.time()-K.START,cpu_seconds=K.cpu()-K.CPU,peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None);dump('status.json',status);print(json.dumps(status),flush=True)
if status['status'] not in ['COMPLETED','PARTIAL_NO_CONFLICT','PARTIAL_BUDGET_MATCH','PARTIAL_PROJECTION_DEGENERATE']:sys.exit(1)

"""Pre-registered EXP-014. No training, test set, threshold updates, or BN calibration."""
import os,sys,time,json,copy,traceback,subprocess,resource,hashlib,math
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F
from scipy.stats import spearmanr
sys.path.insert(0,'experiments/ROUND-012')
import common as C
from spikingjelly.activation_based import layer
ROOT=C.ROOT; O=ROOT/'experiments/EXP-014'; R=O/'runs/EXP-014-RUN-001'; NAMES=[f'conv{i}_x.1.residual_function.2' for i in [2,3,4]]; SIZES=[32,64,128]; K=[8,16,32]; START=time.monotonic(); CPU0=resource.getrusage(resource.RUSAGE_SELF); CODE=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(); status={'execution_status':'RUNNING','start_utc':C.now(),'code_sha':CODE,'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),'run_id':'EXP-014-RUN-001'}
torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def cpu():
 r=resource.getrusage(resource.RUSAGE_SELF);c=resource.getrusage(resource.RUSAGE_CHILDREN);return r.ru_utime+r.ru_stime-CPU0.ru_utime-CPU0.ru_stime+c.ru_utime+c.ru_stime
def guard(reserve=0):
 if time.monotonic()-START>860-reserve:raise TimeoutError('GPU wall budget guard including startup')
 if cpu()>1700-reserve:raise TimeoutError('CPU cumulative budget guard')
def dump(p,x):C.dump(p,x)
def log(x):print(json.dumps(x,default=str),flush=True)
def clean(m):C.clean(m)
def fresh(ck):
 m=C.model_create().cuda();m.load_state_dict(ck['model']);m.eval();C.validate_thresholds(m);return m

def compact(m,sel):
 m=copy.deepcopy(m)
 for n,removed in sel.items():
  s=m.get_submodule(n.rsplit('.',1)[0]); a,b,neuron,c=s[0],s[1],s[2],s[3];assert a.groups==c.groups==1 and a.bias is None and c.bias is None and a.step_mode==c.step_mode=='m';keep=[i for i in range(a.out_channels) if i not in removed]
  aa=layer.Conv2d(a.in_channels,len(keep),a.kernel_size,a.stride,a.padding,a.dilation,a.groups,bias=False,padding_mode=a.padding_mode,step_mode='m').to(device=a.weight.device,dtype=a.weight.dtype)
  bb=layer.BatchNorm2d(len(keep),eps=b.eps,momentum=b.momentum,affine=b.affine,track_running_stats=b.track_running_stats,step_mode='m').to(device=b.weight.device,dtype=b.weight.dtype)
  cc=layer.Conv2d(len(keep),c.out_channels,c.kernel_size,c.stride,c.padding,c.dilation,c.groups,bias=False,padding_mode=c.padding_mode,step_mode='m').to(device=c.weight.device,dtype=c.weight.dtype)
  with torch.no_grad():
   aa.weight.copy_(a.weight[keep]);bb.weight.copy_(b.weight[keep]);bb.bias.copy_(b.bias[keep]);bb.running_mean.copy_(b.running_mean[keep]);bb.running_var.copy_(b.running_var[keep]);bb.num_batches_tracked.copy_(b.num_batches_tracked);cc.weight.copy_(c.weight[:,keep]);
  assert torch.equal(aa.weight,a.weight[keep]) and torch.equal(cc.weight,c.weight[:,keep]) and torch.equal(bb.running_mean,b.running_mean[keep]) and torch.equal(bb.running_var,b.running_var[keep]) and torch.equal(bb.weight,b.weight[keep]) and torch.equal(bb.bias,b.bias[keep]);s[0]=aa;s[1]=bb;s[3]=cc
 m.eval();return m

def masks(m,sel):
 hooks=[]
 for name,idx in sel.items():
  def f(mod,inp,out,idx=idx):
   out=out.clone();out[:,:,idx]=0;return out
  hooks.append(m.get_submodule(name).register_forward_hook(f))
 return hooks

def preflight(m,x):
 dep={}; hooks=[];seen={}
 for name,size in zip(NAMES,SIZES):
  s=m.get_submodule(name.rsplit('.',1)[0]);assert len(s)==5 and isinstance(s[2],C.ATLIF) and s[0].out_channels==size and s[3].in_channels==size and s[0].groups==s[3].groups==1 and s[0].bias is None and s[3].bias is None
  def f(mod,inp,out,n=name):seen[n]={'input_shape':list(inp[0].shape),'output_shape':list(out.shape),'tensor_id':id(out)}
  def g(mod,inp,n=name):seen[n]['consumer_input_id']=id(inp[0]);seen[n]['consumer_shape']=list(inp[0].shape)
  hooks += [s[2].register_forward_hook(f),s[3].register_forward_pre_hook(g)]
 with torch.no_grad():out=m(x[:2]);assert torch.isfinite(out).all()
 clean(m)
 for h in hooks:h.remove()
 for n in NAMES:
  d=seen[n];assert d.pop('tensor_id')==d.pop('consumer_input_id');d.update(type='ATLIF',consumer=n.rsplit('.',1)[0]+'.3',consumer_evidence='nn.Sequential forward order plus identical output/input object; BasicBlock.forward uses sequential result and separate shortcut',extra_consumers=False);dep[n]=d
 dump(O/'channel_dependency_map.json',dep)
 # Synthetic module exercises the same physical slicing routine, with an independent residual.
 class Tiny(torch.nn.Module):
  def __init__(self):
   super().__init__();self.block=torch.nn.Sequential(layer.Conv2d(4,6,3,padding=1,bias=False,step_mode='m'),layer.BatchNorm2d(6,step_mode='m'),C.ATLIF(T=8,tau=.9,sp=.002),layer.Conv2d(6,4,3,padding=1,bias=False,step_mode='m'),layer.BatchNorm2d(4,step_mode='m'))
  def forward(self,z):return self.block(z)+z
 t=Tiny().cuda().eval();z=torch.randn(8,2,4,5,5,device='cuda');ss={'block.2':[1,4]};h=masks(t,ss)
 with torch.no_grad():a=t(z)
 for i in h:i.remove()
 clean(t);u=compact(t,ss)
 with torch.no_grad():b=u(z)
 e=(a-b).abs().max().item();assert e<=1e-5 and a.shape==b.shape and torch.equal(t.block[2].thresh,u.block[2].thresh) and sum(p.numel() for p in u.parameters())<sum(p.numel() for p in t.parameters())
 dump(O/'structural_equivalence_tests.json',{'synthetic':{'passed':True,'max_abs':e,'output_shape':list(a.shape),'threshold_equal':True,'residual_addition_compatible':True,'original_parameters':sum(p.numel() for p in t.parameters()),'compact_parameters':sum(p.numel() for p in u.parameters())},'real':{}})

def collect(m,data,tag):
 counts={n:None for n in NAMES};den={n:0 for n in NAMES};hs=[]
 for name in NAMES:
  def f(mod,inp,out,n=name):
   v=(out>0).sum(dim=(0,1,3,4),dtype=torch.int64);counts[n]=v if counts[n] is None else counts[n]+v;den[n]+=out.shape[0]*out.shape[1]*out.shape[3]*out.shape[4]
  hs.append(m.get_submodule(name).register_forward_hook(f))
 losses=[]
 with torch.no_grad():
  for x,y,ids in data:
   guard();z=m(x).mean(0);assert torch.isfinite(z).all();losses.extend(F.cross_entropy(z,y,reduction='none').cpu().tolist());clean(m)
 for h in hs:h.remove()
 return {n:(counts[n].cpu().numpy()/den[n]) for n in NAMES},np.array(losses)

def importance(m,data,fr,ref,tag):
 result={}; first=time.monotonic();times=[]
 for n,size in zip(NAMES,SIZES):
  w=m.get_submodule(n.rsplit('.',1)[0]+'.0').weight.detach().abs().sum((1,2,3)).cpu().numpy();rows=[]
  for c in range(size):
   guard(reserve=45); t=time.monotonic();h=masks(m,{n:[c]});loss=[]
   try:
    with torch.no_grad():
     for x,y,ids in data:
      z=m(x).mean(0);assert torch.isfinite(z).all();loss.extend(F.cross_entropy(z,y,reduction='none').cpu().tolist());clean(m)
   finally:
    for hh in h:hh.remove()
   d=float(np.mean(np.array(loss)-ref));row={'module':n,'channel':c,'FR':float(fr[n][c]),'L1':float(w[c]),'D':d,'samples':len(loss)};C.append(O/f'channel_importance_{tag}.csv',[row]);rows.append(row);times.append(time.monotonic()-t)
   if len(times)==3:log({'event':'candidate_timing','split':tag,'seconds_each':times,'projected_seconds_for_remaining_A_and_B':float(np.mean(times))*(448-len(times)),'budget_remaining_seconds':860-(time.monotonic()-START)})
   if c%16==0:log({'event':'candidate','split':tag,'module':n,'channel':c,'elapsed':time.monotonic()-START})
  result[n]=rows
 return result

def real_equivalence(m,compact_model,sel,x):
 h=masks(m,sel);a_sp={};b_sp={};hs=[]
 for n,z in C.modules(m).items():
  def record(mod,inp,out,n=n):a_sp[n]=out.detach().clone()
  hs.append(z.register_forward_hook(record))
 for n,z in C.modules(compact_model).items():
  def record(mod,inp,out,n=n):b_sp[n]=out.detach().clone()
  hs.append(z.register_forward_hook(record))
 with torch.no_grad():a=m(x).mean(0);b=compact_model(x).mean(0)
 for q in h+hs:q.remove()
 clean(m);clean(compact_model);diff=(a-b).abs();rel=(diff/(1+a.abs())).max().item();changes={}
 for n in a_sp:
  v=a_sp[n];w=b_sp[n]
  if n in sel:v=v[:,:,[i for i in range(v.shape[2]) if i not in sel[n]]]
  count=int(((v>0)!=(w>0)).sum().item())
  if count:changes[n]=count
 return {'passed':rel<=1e-5 and bool(torch.isfinite(b).all()),'max_normalized_logits_error':rel,'max_abs_logits_error':diff.max().item(),'prediction_disagreements':int((a.argmax(1)!=b.argmax(1)).sum().item()),'spike_difference_counts':changes,'difference_cause':'unresolved: numerical threshold boundary or dependency/implementation issue' if changes else 'none observed'}

def evaluate(m,ids,tag):
 mon=C.SOPMonitor(m);spikes={};states={};rates={};num={};hooks=[]
 for n,z in C.modules(m).items():
  def f(mod,inp,out,n=n):
   v=(out>0).sum(dtype=torch.int64);spikes[n]=spikes.get(n,0)+v;states[n]=states.get(n,0)+out.numel();rates[n]=rates.get(n,0)+v.double()/out.numel();num[n]=num.get(n,0)+1
  hooks.append(z.register_forward_hook(f))
 result=[];sop=0;start=time.monotonic()
 try:
  with torch.no_grad():
   for j,(x,y,sid) in enumerate(C.loader(ids,False)):
    guard(reserve=10);x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);mon.clear_recorded_data();z=m(x).mean(0);loss=F.cross_entropy(z,y,reduction='none');assert torch.isfinite(loss).all();pred=z.argmax(1);so=sum(float(v.item()) for v in mon.records);sop+=so
    rows=[{'group':tag,'sample_id':i,'label':t,'loss':l,'prediction':p,'correct':int(p==t)} for i,l,p,t in zip(sid.tolist(),loss.cpu().tolist(),pred.cpu().tolist(),y.cpu().tolist())];C.append(O/'validation_per_sample.csv',rows);result+=rows;C.append(O/'validation_batch_sops.csv',[{'group':tag,'batch':j,'samples':len(y),'sop_count':so}]);clean(m)
 finally:
  mon.remove_hooks()
  for h in hooks:h.remove()
 assert len(result)==len(ids)
 layerfr={n:int(spikes[n].item())/states[n] for n in spikes};return {'group':tag,'samples':len(result),'accuracy':float(np.mean([r['correct'] for r in result])),'loss':float(np.mean([r['loss'] for r in result])),'SOPs':sop/len(result),'parameters':sum(p.numel() for p in m.parameters()),'FR_global':sum(int(v.item()) for v in spikes.values())/sum(states.values()),'FR_author':float(np.mean([(rates[n]/num[n]).item() for n in spikes])),'target_FR':{n:layerfr[n] for n in NAMES},'seconds':time.monotonic()-start},result

def run():
 C.check_files();assert torch.cuda.device_count()==1 and '4090' in torch.cuda.get_device_name(0);split=json.loads((O/'calibration_split.json').read_text());ids=json.loads(C.S.read_text());ck=torch.load(C.M,map_location='cpu',weights_only=False);m=fresh(ck);original=C.digest(m);nparams=sum(p.numel() for p in m.parameters());assert nparams==1084605
 data={}
 for tag in ['A','B']:data[tag]=[(x.cuda(),y.cuda(),sid) for x,y,sid in C.loader(split[tag],False)]
 x32=data['A'][0][0][:32];preflight(m,x32);guard(reserve=45)
 fr={};ref={};imp={}
 for tag in ['A','B']:
  fr[tag],ref[tag]=collect(m,data[tag],tag);imp[tag]=importance(m,data[tag],fr[tag],ref[tag],tag)
 if C.digest(m)!=original:raise RuntimeError('original model state modified')
 rng=np.random.default_rng(2032);selection={}
 for group in ['P-Random','P-L1','P-FR','P-Loss']:
  selection[group]={}
  for n,size,k in zip(NAMES,SIZES,K):
   if group=='P-Random':chosen=sorted(rng.choice(size,k,replace=False).tolist())
   else:
    key={'P-L1':'L1','P-FR':'FR','P-Loss':'D'}[group];chosen=sorted(r['channel'] for r in sorted(imp['A'][n],key=lambda r:(r[key],r['channel']))[:k])
   selection[group][n]=chosen
 dump(O/'selected_channels.json',selection);dump(O/'selection_lock.json',{'locked_utc':C.now(),'code_sha':CODE,'inputs':'A only, weight, and RNG 2032','validation_accessed':False,'selected_sha256':C.sha(O/'selected_channels.json')})
 stats=[]
 for n,k in zip(NAMES,K):
  a=imp['A'][n];b=imp['B'][n];da=np.array([z['D'] for z in a]);db=np.array([z['D'] for z in b]);rr={'module':n,'bottom_D_overlap':len(set(np.argsort(da,kind='stable')[:k])&set(np.argsort(db,kind='stable')[:k]))/k}
  for key in ['FR','L1','D']:
   z=spearmanr([x[key] for x in a],db);rr['spearman_'+key+'_A_vs_D_B']=float(z.statistic) if np.isfinite(z.statistic) else 'NA_constant_input';sel=sorted(range(len(a)),key=lambda i:(a[i][key],i))[:k];rr['bottom_'+key+'_D_B_mean']=float(db[sel].mean());rr['bottom_'+key+'_D_B_values']=json.dumps(db[sel].tolist())
  stats.append(rr)
 C.append(O/'ranking_statistics.csv',stats);eq=json.loads((O/'structural_equivalence_tests.json').read_text());models={};audit=[]
 for g in selection:
  cm=compact(m,selection[g]);p=sum(v.numel() for v in cm.parameters());assert p==987725 and nparams-p==96880
  test=real_equivalence(m,cm,selection[g],x32);eq['real'][g]=test;dump(O/'structural_equivalence_tests.json',eq);audit.append({'group':g,'original':nparams,'compact':p,'deleted':nparams-p,'expected_deleted':96880});log({'event':'equivalence','group':g,**test})
  if test['passed']:models[g]=cm
 C.append(O/'parameter_count_audit.csv',audit)
 if len(models)<4:raise RuntimeError('real structure equivalence failed; formal validation stopped for safety')
 per={};summary=[]
 for g,mm in [('B0',m)]+list(models.items()):
  guard(reserve=10);s,v=evaluate(mm,ids['validation'],g);summary.append(s);per[g]=v;log({'event':'validation',**s})
 base=summary[0]
 for s in summary:s['SOP_savings']=1-s['SOPs']/base['SOPs'];s['parameter_reduction']=1-s['parameters']/base['parameters'];s['target_FR']=json.dumps(s['target_FR'])
 C.append(O/'summary.csv',summary);rng=np.random.default_rng(2027);ix=rng.integers(0,len(ids['validation']),(2000,len(ids['validation'])));pairs=[(g,'B0') for g in selection]+[('P-Loss',g) for g in ['P-FR','P-L1','P-Random']];comparisons={}
 for a,b in pairs:
  assert [x['sample_id'] for x in per[a]]==[x['sample_id'] for x in per[b]];comparisons[a+'_vs_'+b]={}
  for k in ['correct','loss']:
   d=np.array([x[k]-y[k] for x,y in zip(per[a],per[b])]);comparisons[a+'_vs_'+b][k]={'delta':float(d.mean()),'ci95':np.quantile(d[ix].mean(1),[.025,.975]).tolist()}
 by={x['group']:x for x in summary};h14=any(base['accuracy']-x['accuracy']<=.01 and x['SOPs']<base['SOPs'] for x in summary[1:]);loss=by['P-Loss'];h15=all(loss['accuracy']-by[g]['accuracy']>=.005 and loss['SOP_savings']>=by[g]['SOP_savings']-.02 for g in ['P-FR','P-L1'])
 C.check_files();assert C.digest(m)==original;dump(O/'metrics.json',{'summary':summary,'comparisons':comparisons,'H014_screening_gate':h14,'H015_screening_gate':h15,'formal_hypotheses':'UNTESTED pending theory review','code_sha':CODE});return 'COMPLETED'
try:status['execution_status']=run()
except BaseException as e:status.update(execution_status='PARTIAL',error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
finally:
 status.update(end_utc=C.now(),gpu_wall_seconds=time.monotonic()-START,cpu_seconds=cpu(),peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0);dump(O/'status.json',status);dump(R/'status.json',status);log(status)
if status['execution_status']!='COMPLETED':sys.exit(1)

import json,time,traceback,sys,copy,datetime
import torch
from torch import nn
import common as C
OUT=C.ROOT/'experiments/ROUND-010/preflight';OUT.mkdir(parents=True,exist_ok=False);result={'status':'RUNNING','checks':{},'start_utc':C.now()};torch.set_num_threads(2)
def save():C.dump(OUT/'results.json',result)
def main():
 if torch.cuda.device_count()!=1 or '4090' not in torch.cuda.get_device_name(0):raise RuntimeError('need exactly one approved RTX 4090')
 C.load();cal=C.batches(C.IDS);small=[(cal[0][0][:4],cal[0][1][:4]),(cal[0][0][4:5],cal[0][1][4:5])]
 # Known 5D values and uneven batches: independent concatenation establishes population variance.
 torch.manual_seed(1);x=torch.randn(2,5,3,4,4,dtype=torch.float64);mom=C.Moments();mom.add(x[:,:4]);mom.add(x[:,4:]);mean,var=mom.result();reference=x.permute(2,0,1,3,4).reshape(3,-1);dmean=(mean-reference.mean(1)).abs().max().item();dvar=(var-reference.var(1,correction=0)).abs().max().item()
 if max(dmean,dvar)>1e-12:raise RuntimeError('moment merge failed')
 result['checks']['synthetic_population_mean_max_error']=dmean;result['checks']['synthetic_population_var_max_error']=dvar
 m=C.model('C0',1.25);bn=C.bnmap(m);calls=[];layers=[];shapes={};hooks=[]
 for n,z in bn.items():
  def hook(mod,a,o,n=n):calls.append(n);shapes.setdefault(n,list(a[0].shape))
  hooks.append(z.register_forward_hook(hook))
 for n,z in m.named_modules():
  if isinstance(z,C.ATLIF):hooks.append(z.register_forward_hook(lambda z,a,o,n=n:layers.append(n)))
 with torch.no_grad():z0=C.forward(m,small[0][0].cuda()).detach().clone();C.clean(m)
 for h in hooks:h.remove()
 if len(calls)!=len(bn) or len(set(calls))!=len(bn) or set(calls)!=set(bn) or len(layers)!=19 or len(set(layers))!=19:raise RuntimeError('non-feed-forward or repeated BN/ATLIF invocation')
 if any(len(shapes[n])!=5 or shapes[n][2]!=bn[n].num_features for n in bn):raise RuntimeError('BN input layout mismatch')
 calls2=[];hs=[z.register_forward_hook(lambda z,a,o,n=n:calls2.append(n)) for n,z in bn.items()]
 with torch.no_grad():C.forward(m,small[1][0].cuda());C.clean(m)
 for h in hs:h.remove()
 if calls2!=calls:raise RuntimeError('dynamic order changes')
 mom=C.Moments();h=bn[calls[0]].register_forward_pre_hook(lambda z,a:mom.add(a[0]))
 with torch.no_grad():z1=C.forward(m,small[0][0].cuda()).detach().clone();C.clean(m)
 h.remove();logitdiff=(z0-z1).abs().max().item()
 if logitdiff!=0:raise RuntimeError('statistics hook changes model')
 mon=C.SOPMonitor(m);tot=[]
 for _ in range(2):
  with torch.no_grad():mon.clear_recorded_data();C.forward(m,small[0][0].cuda());tot.append(sum(q.item() for q in mon.records));C.clean(m)
 mon.remove_hooks()
 if tot[0]!=tot[1]:raise RuntimeError('SOP records contaminated')
 state={k:v.detach().clone() for k,v in m.state_dict().items()};allowed={n+'.'+key for n in bn for key in ['running_mean','running_var','num_batches_tracked']}
 t=time.monotonic();C.adaptive(m,small);adaptive_seconds=time.monotonic()-t
 for k,v in m.state_dict().items():
  if k not in allowed and not torch.equal(v,state[k]):raise RuntimeError('adaptive changed forbidden state '+k)
 m2=C.model('C0',1.25);initial={k:v.detach().clone() for k,v in m2.state_dict().items()};t=time.monotonic();C.sequential(m2,small,calls);sequential_seconds=time.monotonic()-t
 for k,v in m2.state_dict().items():
  if k not in allowed and not torch.equal(v,initial[k]):raise RuntimeError('sequential changed forbidden state '+k)
 for n,b in C.bnmap(m2).items():
  if not torch.isfinite(b.running_mean).all() or not torch.isfinite(b.running_var).all() or (b.running_var<0).any():raise RuntimeError('invalid BN buffer')
 m3=C.model('C0',1.25)
 if any(not torch.equal(v,initial[k]) for k,v in m3.state_dict().items()):raise RuntimeError('temporary model polluted another')
 C.checksums();result['checks'].update({'BN_order':calls,'ATLIF_order':layers,'BN_input_shapes':shapes,'BN_layout':'T,B,C,H,W','hook_max_logit_difference':logitdiff,'SOP_repeat_absolute_difference':abs(tot[0]-tot[1]),'only_allowed_BN_buffers_modified':True,'original_weight_affine_threshold_unchanged':True,'independent_candidates_unpolluted':True,'original_checkpoint_hashes_unchanged':True,'alpha_1_25_from_baseline_verified':True,'small_calibration_batches':[4,1],'small_adaptive_wall_seconds':adaptive_seconds,'small_sequential_wall_seconds':sequential_seconds,'state_reset_update_values_zero':all(z.update_value==0. for z in m2.modules() if isinstance(z,C.ATLIF))});result['status']='PASSED'
try:main()
except BaseException as e:result.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
finally:result.update(end_utc=C.now(),wall_seconds=time.monotonic()-C.start,cpu_seconds=C.cpu(),forwards=C.counts['forwards'],samples=C.counts['samples']);save();print(json.dumps(result,indent=2),flush=True)
if result['status']!='PASSED':sys.exit(1)

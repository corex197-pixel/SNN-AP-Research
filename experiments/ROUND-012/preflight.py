"""All required aggregation, AMP and real-network checks before EXP-012."""
import sys,json,math,traceback,time,torch
import numpy as np
import common as C
from models.submodules.layers import Surrogate
from artg import Eq13,Artg
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
OUT=C.D/'preflight';OUT.mkdir(parents=True,exist_ok=False);status={'status':'RUNNING','start_utc':C.now()};torch.set_num_threads(2)
def err(a,b):return abs(a-b)/(1+abs(b))
def manual():
 cases=[]
 for dtype in [torch.float64,torch.float32]:
  tol=1e-10 if dtype==torch.float64 else 1e-5
  for shape in [(4,3,2,2),(3,3,2,2),(1,5,3,2)]:
   N=math.prod(shape);gen=torch.Generator().manual_seed(1024+N);theta0=torch.tensor(1.3,dtype=dtype);r=torch.randn((8,)+shape,dtype=dtype,generator=gen)*.9+1.;x0=r*theta0;a=torch.randn((8,)+shape,dtype=dtype,generator=gen);b=torch.randn((8,)+shape,dtype=dtype,generator=gen)
   for group in ['G0','G1','M4']:
    for path in (['combined'] if group!='M4' else ['amplitude','reset','combined']):
     th=theta0.clone().requires_grad_();terms=[];expected_sum=0.
     for t in range(8):
      x=x0[t].clone().requires_grad_();g=(1-((x.detach()-th.detach())/th.detach()).abs()).clamp_min(0);rr=x.detach()/th.detach();sp=(x.detach()>=th.detach()).float()
      if group=='M4':
       o,s,_=Artg.apply(x,th,.002,'test');aa=a[t].to(o.dtype) if path!='reset' else torch.zeros_like(o);bb=b[t].to(s.dtype) if path!='amplitude' else torch.zeros_like(s);terms.append((o*aa).sum()+(s*bb).sum());aa=aa.to(dtype);bb=bb.to(dtype);expected_sum+=(aa*(sp.to(dtype)-rr*g)-bb*rr*g/th.detach()).sum().item()
      else:
       o,_=(Surrogate if group=='G0' else Eq13).apply(x,th,.002);q=a[t].to(o.dtype);terms.append((o*q).sum());q=q.to(dtype);expected_sum+=(-(q*g*(g if group=='G0' else 1))).sum().item()
     mean_grad=torch.autograd.grad(sum(terms),th)[0].item();actual=mean_grad*N;abserr=abs(actual-expected_sum);relative=err(actual,expected_sum)
     cases.append({'dtype':str(dtype),'shape':list(shape),'T':8,'group':group,'path':path,'N':N,'mean_time_accumulated':mean_grad,'scaled':actual,'independent_sum':expected_sum,'absolute_error':abserr,'normalized_error':relative,'tolerance':tol});assert relative<=tol,(dtype,shape,group,path,relative)
 C.dump(OUT/'gradient_aggregation_tests.json',{'status':'PASSED','cases':cases,'includes_last_incomplete_batch_shape':True,'temporal_gradient_accumulated_without_extra_T':True,'shared_theta_scalar':True,'positive_negative_gradients':True})
def amp_guard():
 # scaler finds nonfinite raw gradients; optimizer is skipped.
 p=torch.nn.Parameter(torch.tensor(1.,device='cuda'));opt=torch.optim.SGD([p],lr=.005);sc=torch.cuda.amp.GradScaler(enabled=True);opt.zero_grad();sc.scale(p*torch.tensor(float('inf'),device='cuda')).backward();sc.unscale_(opt);raw=C.finite_grads(opt);before=p.detach().clone();sc.step(opt);old=sc.get_scale();sc.update();skipped=sc.get_scale()<old;assert not raw and skipped and torch.equal(p,before)
 # Guard rejects nonfinite created only by post-unscale multiplication, before step.
 class Mini:
  def __init__(self):self.thresh=torch.nn.Parameter(torch.tensor(1.,device='cuda'))
 z=Mini();z.thresh.grad=torch.tensor(3e38,device='cuda');rejected=False
 try:C.scale_thresholds({'x':z},{'x':{'N':2}})
 except FloatingPointError:rejected=True
 assert rejected and z.thresh.item()==1.
 # Previous M4 float16 upstream overflow regression on each path.
 mixed=[]
 for case,av,bv in [('amplitude',100000.,0.),('reset',0.,-100000.),('both',100000.,-100000.)]:
  x=torch.tensor([2.],device='cuda',dtype=torch.float16,requires_grad=True);th=torch.tensor(1.3,device='cuda',dtype=torch.float32,requires_grad=True);o,s,_=Artg.apply(x,th,.002,'test');a=torch.tensor([av],device='cuda');b=torch.tensor([bv],device='cuda');_,gt=torch.autograd.grad((o,s),(x,th),(a,b));r=(x.detach()/th.detach()).float();g=(1-((x.detach()-th.detach())/th.detach()).abs()).clamp_min(0).float();expected=(a*(s.detach()-r*g)-b*r*g/th.detach()).mean();e=err(gt.item(),expected.item());mixed.append({'case':case,'absolute_error':abs(gt.item()-expected.item()),'normalized_error':e,'gradient':gt.item()});assert torch.isfinite(gt) and e<1e-5
 C.dump(OUT/'mixed_precision_tests.json',{'status':'PASSED','raw_nonfinite_correctly_skips_optimizer':skipped,'post_scaling_nonfinite_stops_before_optimizer':rejected,'mixed_precision_large_upstream':mixed})

def real():
 ck=torch.load(C.M,map_location='cpu',weights_only=False);spl=json.loads(C.S.read_text());ids=json.loads(C.I.read_text())['indices'];assert len(ids)==500 and set(ids)<=set(spl['train']) and not set(ids)&set(spl['validation']) and not set(ids)&set(spl['diagnostic']);ds=CIFAR10(C.DATA,train=True,download=False,transform=C.transforms_pair()[1]);x=torch.stack([ds[i][0] for i in ids[:8]]).cuda();y=torch.tensor([ds[i][1] for i in ids[:8]],device='cuda');modes={};shape_rows=[];opt_rows=[];initial_hash=None
 for enabled in [False,True]:
  label='AMP' if enabled else 'FP32';reference=None;modes[label]={}
  for group in ['D0','G0-S','G1-S','M4-S']:
   C.seed_all(2030);m=C.model(group,ck);m.train();init=C.digest(m);initial_hash=initial_hash or init;assert init==initial_hash;mods=C.modules(m);tracker=C.ShapeTracker(m);spikes={};hs=[];out=[]
   for n,z in mods.items():hs.append(z.register_forward_hook(lambda z,a,o,n=n:spikes.update({n:(o.detach()>0).cpu()})))
   hs.append(m.register_forward_hook(lambda z,a,o:out.append(o.detach().cpu())));mon=C.SOPMonitor(m);opt=C.optimizer(m);sc=torch.cuda.amp.GradScaler(enabled=enabled,init_scale=1.);before={n:p.detach().cpu().clone() for n,p in m.named_parameters() if not n.endswith('thresh')};res=C.step(m,opt,sc,tracker,x,y,group,0,0,mode='mean' if group=='D0' else 'sum_equivalent',diagnostic_components=(group=='M4-S'));sop=sum(float(v.item()) for v in mon.records);grads={n:p.grad.detach().cpu().clone() for n,p in m.named_parameters() if p.grad is not None and not n.endswith('thresh')};updated={n:p.detach().cpu().clone() for n,p in m.named_parameters() if not n.endswith('thresh')}
   for h in hs:h.remove()
   mon.remove_hooks();tracker.remove();assert not res['skipped'] and res['raw_finite'] and all(torch.isfinite(v).all() for v in grads.values());assert all(v['threshold_after']>0 for v in res['rows']);shape_rows.extend({'mode':label,'group':group,'module':n,'shape':json.dumps(z['shape']),'N':z['N'],'batch':z['batch']} for n,z in res['shapes'].items())
   if reference is None:reference={'logits':out[0],'spikes':spikes,'sop':sop,'grads':grads,'updated':updated,'shapes':res['shapes']}
   else:
    assert torch.equal(out[0],reference['logits']) and all(torch.equal(v,reference['spikes'][n]) for n,v in spikes.items()) and sop==reference['sop'];assert all(torch.equal(v,reference['grads'][n]) for n,v in grads.items());assert all(torch.equal(v,reference['updated'][n]) for n,v in updated.items());assert res['shapes']==reference['shapes']
   for r in res['rows']:
    assert r['momentum_before'] is None and r['weight_decay']==0.;q=torch.tensor(r['scaled_task_gradient'],dtype=torch.float32);before_value=torch.tensor(r['threshold_before'],dtype=torch.float32);pred=before_value.clone();pred.add_(q+0.9*q,alpha=-.005);observed=r['threshold_before']+r['optimizer_delta'];absolute=abs(pred.item()-observed);ulp=max(r['ulp_up'],r['ulp_down']);assert absolute<=ulp,(group,r['module'],absolute,ulp);r['predicted_first_optimizer_threshold']=pred.item();r['first_optimizer_absolute_error']=absolute;opt_rows.append({'mode':label,**r})
   modes[label][group]={'same_initial_hash':init,'loss':res['loss'],'SOPs':sop,'successful_step':True,'threshold_count':len(mods),'nonthreshold_gradients_and_parameter_update_equal_D0':True,'threshold_weight_decay':sorted(set(r['weight_decay'] for r in res['rows']))};del m,opt,sc;torch.cuda.empty_cache()
 # Actual complete and last partial batch dimensions; no optimizer update.
 real_shapes=[];m=C.model('D0',ck);m.eval();tracker=C.ShapeTracker(m)
 for batch in [128,72]:
  tracker.reset()
  with torch.no_grad():m(torch.zeros((batch,3,32,32),device='cuda'))
  tracker.validate(m,batch);real_shapes.extend({'mode':'shape_check','group':'D0','module':n,'shape':json.dumps(v['shape']),'N':v['N'],'batch':batch} for n,v in tracker.shapes.items());C.clean(m)
 tracker.remove();C.append(OUT/'threshold_shapes.csv',shape_rows+real_shapes);C.dump(OUT/'real_optimizer_step_check.json',{'status':'PASSED','groups':modes,'rows':opt_rows,'first_optimizer_step_independently_checked_within_one_ULP':True,'same_initialization_hash':initial_hash,'full_batch_and_partial_batch_shapes_checked':[128,72]})
try:
 C.check_files();assert torch.cuda.device_count()==1 and '4090' in torch.cuda.get_device_name(0);manual();amp_guard();real();status.update(status='PASSED')
except BaseException as e:status.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
finally:status.update(end_utc=C.now(),wall_seconds=time.monotonic()-C.START,cpu_seconds=C.cpu());C.dump(OUT/'status.json',status);print(json.dumps(status,indent=2),flush=True)
if status['status']!='PASSED':sys.exit(1)

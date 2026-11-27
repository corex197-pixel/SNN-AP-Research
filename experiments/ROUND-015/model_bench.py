"""EXP-016 full-model gates and short benchmark; fixed config; train-file only."""
import os,sys,time,json,traceback,resource,datetime,csv,hashlib,gc,subprocess
from pathlib import Path
start=time.monotonic();ROOT=Path(__file__).resolve().parents[2];O=ROOT/'experiments/EXP-016';R=O/'runs/EXP-016-MODEL-001';R.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(ROOT/'vendor/UniSparse'));sys.path.insert(0,str(ROOT/'experiments/ROUND-014'))
import numpy as np,torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from timm.optim import create_optimizer_v2
from spikingjelly.activation_based import functional
from models.cifarnet import cifarnet
from models.submodules.layers import SparseConv2d,SparseBN,SparseNeuron
from utils.utils import calc_penalty,penalize,clear_penalty,SOPMonitor
from controlled_data import TrainFileOnly
import random
cpu0=resource.getrusage(resource.RUSAGE_SELF);torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
status={'status':'RUNNING','run_id':R.name,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'prior_gpu_wall_seconds':9.2813}
def dump(name,x):
 p=O/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def guard(reserve=10):
 c=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN);cpu=c.ru_utime+c.ru_stime-cpu0.ru_utime-cpu0.ru_stime+children.ru_utime+children.ru_stime
 if time.monotonic()-start>860-status['prior_gpu_wall_seconds']-reserve or cpu>2600-reserve:raise TimeoutError('resource guard')
def seed():random.seed(12450);np.random.seed(12450);torch.manual_seed(12450);torch.cuda.manual_seed(12450)
def sync():torch.cuda.synchronize()
def state(m):return {n:v for n,v in m.named_modules() if isinstance(v,SparseConv2d)}
def rows_save(row):
 p=O/'resource_benchmark.csv'
 with p.open('a',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(row),lineterminator='\n');
  if f.tell()==0:w.writeheader()
  w.writerow(row);f.flush();os.fsync(f.fileno())
def fresh():
 seed();m=cifarnet(T=4,base_channels=256,num_classes=10).cuda().train();opt=create_optimizer_v2(m,opt='sgd',lr=.1);return m,opt

def main():
 assert json.loads((O/'backend_status.json').read_text())['status']=='PASSED';assert torch.cuda.device_count()==1 and '4090' in torch.cuda.get_device_name(0);torch.cuda.set_per_process_memory_fraction(.88);assert hashlib.sha256((ROOT/'experiments/EXP-004/splits/indices.json').read_bytes()).hexdigest()=='69dccfe992236766aea575a0a66495802ca889ca329d61f51e8de3c1ab7f3165'
 ids=json.loads((ROOT/'experiments/EXP-004/splits/indices.json').read_text())['train'];assert len(ids)==45000
 # Use a deterministic prefix; no validation/test. Each path starts from same seed.
 ds=TrainFileOnly(ids[:256*5],training=True);loader=DataLoader(ds,batch_size=256,shuffle=False,num_workers=2,pin_memory=True,drop_last=True)
 m,opt=fresh();conv=state(m);names={id(v):n for n,v in m.named_modules()};dependency={n:{'prev_bns':[names.get(id(x),'unregistered') for x in v.prev_bns],'next_convs':[names.get(id(x),'unregistered') for x in v.next_convs],'flatten':v.flatten,'activation':type(v.activation).__name__} for n,v in m.named_modules() if isinstance(v,SparseNeuron)};dump('consumer_dependency_map.json',{'neuron_relations':dependency,'skip':m.skip,'architecture':'activation2/5 followed by pooling; activation5 flattened before fc1; IdentityBN has no affine parameters','evidence':'official source graph and registered module references'})
 optim_param_ids={id(p) for g in opt.param_groups for p in g['params']};param_count=sum(p.numel() for p in m.parameters());thresholds={n:{'is_parameter':isinstance(v.threshold,torch.nn.Parameter),'initial':v.threshold.item(),'mask_matches_weight':bool(torch.equal(v.sparse_mask(),(v.sparse_weight()!=0).float())),'zero_threshold_matches_raw':bool(torch.equal(v.sparse_weight(),v.weight))} for n,v in conv.items()};assert all(z['mask_matches_weight'] and z['zero_threshold_matches_raw'] and not z['is_parameter'] for z in thresholds.values());bns={n:{'gamma_in_optimizer':id(v.weight) in optim_param_ids,'beta_in_optimizer':id(v.bias) in optim_param_ids} for n,v in m.named_modules() if isinstance(v,SparseBN)};assert all(z['gamma_in_optimizer'] and z['beta_in_optimizer'] for z in bns.values())
 t=time.monotonic();x,y,sid=next(iter(loader));load=time.monotonic()-t;x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True);scaler=torch.amp.GradScaler('cuda');mon=SOPMonitor(m);mon.enable();sync();t=time.monotonic()
 with torch.autocast('cuda'):z=m(x);loss=F.cross_entropy(z.mean(0),y)
 sync();fwd=time.monotonic()-t;assert list(z.shape)==[4,256,10] and torch.isfinite(loss);scaler.scale(loss).backward();sync();scaler.unscale_(opt);assert all(torch.isfinite(p.grad).all().item() for p in m.parameters() if p.grad is not None);sop=sum(float(q.item()) for q in mon.records);mon.remove_hooks();functional.reset_net(m)
 # Verify local penalty preserves current unscaled task grads, without changing optimizer ordering in benchmark.
 pre={n:(v.weight.grad.detach().clone() if v.weight.grad is not None else None,v.bias.grad.detach().clone() if v.bias.grad is not None else None) for n,v in m.named_modules() if isinstance(v,SparseBN)};calc_penalty(m);restore=all((a is None and v.weight.grad is None or a is not None and torch.equal(a,v.weight.grad)) and (b is None and v.bias.grad is None or b is not None and torch.equal(b,v.bias.grad)) for n,v in m.named_modules() if isinstance(v,SparseBN) for a,b in [pre[n]]);assert restore
 nonzero_conv=sum(v.penalty.item()!=0 for v in conv.values());before={n:v.threshold.item() for n,v in conv.items()};before_bn={n:(v.weight.detach().clone(),v.bias.detach().clone()) for n,v in m.named_modules() if isinstance(v,SparseBN)};penalize(m,.000001,.000001);changed=sum(v.threshold.item()!=before[n] for n,v in conv.items());bnchanges=sum(not torch.equal(v.weight,before_bn[n][0]) or not torch.equal(v.bias,before_bn[n][1]) for n,v in m.named_modules() if isinstance(v,SparseBN));finite=all(torch.isfinite(p).all().item() for p in m.parameters()) and all(torch.isfinite(v.threshold).item() for v in conv.values());clear_penalty(m);cleared=all(v.penalty.item()==0 for v in conv.values()) and all(torch.count_nonzero(v.penalty_gamma).item()==0 and torch.count_nonzero(v.penalty_beta).item()==0 for v in m.modules() if isinstance(v,SparseBN));assert nonzero_conv and changed and bnchanges and finite and cleared
 # Explicitly probe non-zero threshold without changing the model used in any benchmark.
 q=next(iter(conv.values()));original=q.threshold.clone();n0=int((q.sparse_weight()!=0).sum().item());q.threshold.fill_(.05);n1=int((q.sparse_weight()!=0).sum().item());q.threshold.copy_(original);assert n1<n0
 dump('complete_model_smoke.json',{'status':'PASSED','kind':'SMOKE_TEST','output_shape':list(z.shape),'task_loss_finite':True,'gradient_finite':True,'parameter_count':param_count,'first_batch_data_load_seconds':load,'first_forward_seconds_including_compile_and_sop_monitor':fwd,'monitor_sops_for_batch':sop,'train_samples_only':True,'groups':[{'lr':g['lr'],'weight_decay':g['weight_decay'],'momentum':g.get('momentum'),'num_parameters':sum(p.numel() for p in g['params'])} for g in opt.param_groups],'threshold_buffers':thresholds,'bn_parameters_in_optimizer':bns,'initial_compile_cost_approx':fwd})
 dump('sparse_mechanism_tests.json',{'status':'PASSED','kind':'SMOKE_TEST','conv_penalty_nonzero_modules':nonzero_conv,'threshold_changed_modules':changed,'bn_affine_changed_modules':bnchanges,'local_calc_penalty_restored_task_gradients':restore,'buffers_cleared':cleared,'all_parameters_finite':finite,'nonzero_threshold_probe':{'original_nonzero':n0,'test_nonzero':n1,'threshold':.05,'probe_only':True},'manual_coefficient':1e-6,'note':'synthetic numerical probe; not paper lambda; local pre-forward cache may precede optimizer update in author sequence'})
 del z,loss,pre,before_bn,mon,m,opt,scaler,x,y;gc.collect();torch.cuda.empty_cache()
 for path in ['zero','nonzero']:
  guard(reserve=30);m,opt=fresh();scaler=torch.amp.GradScaler('cuda');torch.cuda.reset_peak_memory_stats();iterator=iter(loader);initial_digest=hashlib.sha256()
  for n,p in m.state_dict().items():initial_digest.update(n.encode());initial_digest.update(p.detach().cpu().numpy().tobytes())
  digest=initial_digest.hexdigest();completed=0
  for batch in range(5):
   guard(reserve=15);whole=time.monotonic();t=time.monotonic();x,y,sid=next(iterator);load=time.monotonic()-t;x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True);sync();t=time.monotonic()
   with torch.autocast('cuda'):z=m(x);loss=F.cross_entropy(z.mean(0),y)
   sync();forward=time.monotonic()-t;assert torch.isfinite(loss);t=time.monotonic();scaler.scale(loss).backward();sync();back=time.monotonic()-t;functional.reset_net(m);t=time.monotonic();scale0=scaler.get_scale();scaler.step(opt);scaler.update();sync();optimizer=time.monotonic()-t;skip=scaler.get_scale()<scale0;opt.zero_grad(set_to_none=True)
   penalty=manual=0.;updates=0
   if path=='nonzero':
    t=time.monotonic();calc_penalty(m);sync();penalty=time.monotonic()-t;t=time.monotonic();penalize(m,.01*.1,.01*.1);clear_penalty(m);sync();manual=time.monotonic()-t;updates=1
   assert all(torch.isfinite(v.threshold).item() for v in state(m).values()) and all(torch.isfinite(p).all().item() for p in m.parameters());whole_sec=time.monotonic()-whole;row={'path':path,'batch':batch,'phase':'warmup' if batch<2 else 'measured','samples':len(y),'data_load_seconds':load,'forward_seconds':forward,'backward_seconds':back,'optimizer_seconds':optimizer,'penalty_forward_backward_combined_seconds':penalty,'manual_clear_seconds':manual,'total_seconds':whole_sec,'amp_skipped':skip,'optimizer_success':not skip,'penalty_updates':updates,'allocated_peak_bytes':torch.cuda.max_memory_allocated(),'reserved_peak_bytes':torch.cuda.max_memory_reserved(),'initial_state_sha256':digest};rows_save(row);completed+=1;print(json.dumps(row),flush=True)
  del m,opt,scaler,z,loss,x,y,iterator;gc.collect();torch.cuda.empty_cache()
 return 'COMPLETED'
try:status['status']=main()
except BaseException as e:status.update(status='PARTIAL',error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
finally:
 r=resource.getrusage(resource.RUSAGE_SELF);c=resource.getrusage(resource.RUSAGE_CHILDREN);status.update(end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.monotonic()-start,cpu_seconds=r.ru_utime+r.ru_stime-cpu0.ru_utime-cpu0.ru_stime+c.ru_utime+c.ru_stime,peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0);dump('model_run_status.json',status);(R/'status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status),flush=True)
if status['status']!='COMPLETED':sys.exit(1)

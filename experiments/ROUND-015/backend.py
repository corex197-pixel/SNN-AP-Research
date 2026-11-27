"""EXP-016 official CuPy backend checks; no model training or test data."""
import os,sys,time,json,traceback,resource
from pathlib import Path
start=time.monotonic();ROOT=Path(__file__).resolve().parents[2];O=ROOT/'experiments/EXP-016';sys.path.insert(0,str(ROOT/'vendor/UniSparse'))
import numpy as np,torch,cupy as cp
from spikingjelly.activation_based import neuron,surrogate,functional
from models.submodules.layers import LIF
c0=resource.getrusage(resource.RUSAGE_SELF);torch.set_num_threads(1)
def dump(n,x):(O/n).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def rel(a,b):return float(((a-b).abs()/(1+b.abs())).max().item())
status={'status':'RUNNING','start_time_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')}
try:
 assert torch.cuda.device_count()==1 and '4090' in torch.cuda.get_device_name(0) and cp.cuda.runtime.getDeviceCount()==1
 torch.cuda.init();cp.cuda.Device(0).use();t=time.monotonic();kernel=cp.RawKernel(r'extern "C" __global__ void plus1(const float* x,float* y){int i=threadIdx.x;y[i]=x[i]+1;}','plus1');a=cp.arange(4,dtype=cp.float32);b=cp.empty_like(a);kernel((1,),(4,),(a,b));cp.cuda.runtime.deviceSynchronize();first=time.monotonic()-t;t=time.monotonic();kernel((1,),(4,),(a,b));cp.cuda.runtime.deviceSynchronize();second=time.monotonic()-t;assert cp.asnumpy(b).tolist()==[1,2,3,4];tx=torch.utils.dlpack.from_dlpack(b);assert tx.device.index==0 and torch.equal(tx,torch.tensor([1.,2.,3.,4.],device='cuda'));tx.add_(2);cp.cuda.runtime.deviceSynchronize();assert cp.asnumpy(b).tolist()==[3,4,5,6]
 runtime={'status':'PASSED','gpu_name':torch.cuda.get_device_name(0),'torch_cuda':torch.version.cuda,'cupy_version':cp.__version__,'cupy_runtime':cp.cuda.runtime.runtimeGetVersion(),'cupy_driver':cp.cuda.runtime.driverGetVersion(),'first_kernel_seconds':first,'cached_kernel_seconds':second,'dlpack_shared_data_verified':True,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'cupy_device_properties':str(cp.cuda.runtime.getDeviceProperties(0).get('name'))};dump('cupy_runtime_tests.json',runtime)
 # Inputs cover subthreshold, repeated firing, accumulation, reset, signed and near-boundary states.
 x=torch.tensor([[0.,3.,1.5,-1.,1.999999,2.000001],[0.,3.,1.5,-1.,0.,0.],[0.,3.,1.5,4.,0.,0.],[0.,3.,0.,0.,0.,0.]],device='cuda').reshape(4,1,6,1,1);up=torch.tensor([1.,-2.,.3,1.5]).reshape(4,1,1,1,1).cuda();results={}
 for dtype,tol in [(torch.float32,1e-5),(torch.float16,1e-3)]:
  outputs=[]
  for backend in ['cupy','torch']:
   net=neuron.LIFNode(tau=2.,decay_input=True,v_threshold=1.,v_reset=0.,surrogate_function=surrogate.ATan(),detach_reset=True,step_mode='m',backend=backend,store_v_seq=True).cuda();z=x.to(dtype).detach().requires_grad_(True);y=net(z);v=net.v_seq.detach().clone();last=net.v.detach().clone();(y.float()*up).sum().backward();grad=z.grad.detach().clone();functional.reset_net(net);reset=net.v;outputs.append((y.detach(),v,last,grad,reset))
  a,b=outputs;spike_diff=int((a[0]!=b[0]).sum().item());entry={'spike_mismatches':spike_diff,'voltage_error':rel(a[1].float(),b[1].float()),'final_voltage_error':rel(a[2].float(),b[2].float()),'gradient_error':rel(a[3].float(),b[3].float()),'reset_cupy':str(a[4]),'reset_torch':str(b[4]),'near_threshold_inputs':x[:,0,4:,0,0].cpu().tolist(),'tolerance':tol};entry['passed']=spike_diff==0 and max(entry['voltage_error'],entry['final_voltage_error'],entry['gradient_error'])<=tol;results[str(dtype)]=entry
 dump('neuron_backend_comparison.json',{'kind':'UNIT_TEST','diagnostic_store_v_seq':True,'cases':'silent, repeated, temporal accumulation, hard reset, signed and threshold-adjacent','results':results})
 official=LIF().cuda();z=x.detach().requires_grad_(True);out=official(z);out.sum().backward();assert torch.isfinite(z.grad).all();functional.reset_net(official)
 # AMP skip plus actual official penalize order with synthetic fixed buffers.
 from models.submodules.layers import SparseBN,SparseConv2d
 from utils.utils import penalize,clear_penalty
 class Small(torch.nn.Module):
  def __init__(self):super().__init__();self.bn=SparseBN(1);self.conv=SparseConv2d(1,1,1)
 m=Small().cuda();opt=torch.optim.SGD(m.parameters(),lr=.1);scaler=torch.amp.GradScaler('cuda',init_scale=8.);opt.zero_grad();loss=scaler.scale(m.conv.weight.sum()*torch.tensor(float('inf'),device='cuda'));loss.backward();before=m.conv.weight.detach().clone();t0=m.conv.threshold.item();gamma=m.bn.weight.detach().clone();beta=m.bn.bias.detach().clone();sc0=scaler.get_scale();scaler.step(opt);scaler.update();skip=torch.equal(before,m.conv.weight) and scaler.get_scale()<sc0;opt.zero_grad();m.conv.penalty.fill_(2);m.bn.penalty_gamma.fill_(3);m.bn.penalty_beta.fill_(4);penalize(m,.01,.01);manual={'threshold_changed':m.conv.threshold.item()!=t0,'gamma_changed':not torch.equal(gamma,m.bn.weight),'beta_changed':not torch.equal(beta,m.bn.bias)};finite=all(torch.isfinite(p).all().item() for p in m.parameters()) and torch.isfinite(m.conv.threshold).item();clear_penalty(m);cleared=m.conv.penalty.item()==0 and torch.all(m.bn.penalty_gamma==0).item() and torch.all(m.bn.penalty_beta==0).item();dump('amp_update_order_tests.json',{'kind':'GPU_UNIT_TEST','skipped':skip,'scale_before':sc0,'scale_after':scaler.get_scale(),'manual_changes':manual,'finite':bool(finite),'penalties_cleared':bool(cleared),'coefficients':'synthetic .01, not paper config'});dump('mixed_precision_tests.json',{'normal_fp16':results['torch.float16'],'forced_overflow':{'optimizer_skipped':skip,'finite_manual_state':bool(finite)}})
 assert all(z['passed'] for z in results.values()) and skip and finite and cleared
 status.update(status='PASSED',official_lif_forward_backward=True)
except BaseException as e:
 status.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
finally:
 r=resource.getrusage(resource.RUSAGE_SELF);status.update(end_time_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),wall_seconds=time.monotonic()-start,cpu_seconds=r.ru_utime+r.ru_stime-c0.ru_utime-c0.ru_stime,peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0);dump('backend_status.json',status);print(json.dumps(status,indent=2),flush=True)
if status['status']!='PASSED':sys.exit(1)

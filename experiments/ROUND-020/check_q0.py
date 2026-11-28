"""Verify q=0 full model output on identical calibration inputs."""
import os,sys,time,traceback,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/ROUND-019'));import core as K
import torch
from spikingjelly.activation_based import functional
base=ROOT/'experiments/EXP-022';out=base/'runs/EXP-022-Q0-CHECK-001';out.mkdir(parents=True,exist_ok=False);K.E=out;K.START=time.time();K.CPU=K.cpu();K.PRIOR_WALL=97.34428524971008;K.PRIOR_CPU=113.409794;K.status={'status':'RUNNING','run_id':'EXP-022-Q0-CHECK-001','gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),'code_sha':os.popen('git rev-parse HEAD').read().strip(),'phase':'q0_same_input'}
try:
 K.guard();torch.set_num_threads(2);m=K.model();x,_,ids=next(iter(K.loader(K.S['D_cal'][:64])));x=x.cuda();original={n:v.detach().clone() for n,v in m.state_dict().items()}
 with torch.no_grad():
  y0=m(x).detach().clone();functional.reset_net(m)
  for bn,_,_ in K.T:
   z=K.get(m,bn);z.weight.copy_(original[bn+'.weight']-torch.zeros_like(z.weight));z.bias.copy_(original[bn+'.bias']-torch.zeros_like(z.bias))
  assert all(torch.equal(v,original[n]) for n,v in m.state_dict().items());y1=m(x).detach().clone();functional.reset_net(m)
 assert torch.equal(y0,y1);K.dump('q0_output_audit.json',{'status':'PASSED','samples':len(ids),'sample_ids':ids.tolist(),'same_input_exact_logits':True,'max_abs_logit_difference':float((y0-y1).abs().max().item()),'state_equal':True});K.status['status']='COMPLETED'
except BaseException as e:K.status.update(status='PARTIAL',error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
finally:K.status.update(end_unix=time.time(),wall_seconds=K.PRIOR_WALL+time.time()-K.START,cpu_seconds=K.PRIOR_CPU+K.cpu()-K.CPU,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());K.dump('status.json',K.status);print(json.dumps(K.status),flush=True)
if K.status['status']!='COMPLETED':sys.exit(1)

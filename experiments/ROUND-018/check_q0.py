"""Same-input parent versus q=0 model-output verification, no intervention search."""
import os,sys,time,traceback,json
from pathlib import Path
import torch
from spikingjelly.activation_based import functional
import run as R
base=R.E;out=base/'runs/EXP-020-Q0-CHECK-001';out.mkdir(parents=True,exist_ok=False);R.E=out;R.PRIOR_WALL=83.70039081573486;R.PRIOR_CPU=100.525364;R.status={'status':'RUNNING','run_id':'EXP-020-Q0-CHECK-001','gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),'code_sha':R.CODE,'phase':'q0_same_input'}
try:
 R.guard();torch.set_num_threads(2);m=R.model();x,_,ids=next(iter(R.loader(R.S['D_cal'][:64])));x=x.cuda();original={n:v.detach().clone() for n,v in m.state_dict().items()}
 with torch.no_grad():
  y0=m(x).detach().clone();functional.reset_net(m)
  for bn,_,_ in R.T:
   z=R.get(m,bn);z.weight.copy_(original[bn+'.weight']-torch.zeros_like(z.weight));z.bias.copy_(original[bn+'.bias']-torch.zeros_like(z.bias))
  assert all(torch.equal(v,original[n]) for n,v in m.state_dict().items());y1=m(x).detach().clone();functional.reset_net(m)
 assert torch.equal(y0,y1)
 R.dump('q0_output_audit.json',{'status':'PASSED','samples':len(ids),'sample_ids':ids.tolist(),'same_input_exact_logits':True,'max_abs_logit_difference':float((y0-y1).abs().max().item()),'state_equal':True});R.status['status']='COMPLETED'
except BaseException as e:R.status.update(status='PARTIAL',error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
finally:R.status.update(end_unix=time.time(),wall_seconds=R.PRIOR_WALL+time.time()-R.START,cpu_seconds=R.PRIOR_CPU+R.cpu()-R.CPU);R.dump('status.json',R.status);print(json.dumps(R.status),flush=True)
if R.status['status']!='COMPLETED':sys.exit(1)

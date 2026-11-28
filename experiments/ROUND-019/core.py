"""EXP-020 preregistered, single GPU, local-gradient evaluation; never trains."""
import os,sys,time,json,csv,hashlib,random,resource,signal,traceback,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT);sys.path[:0]=[str(ROOT/'vendor/UniSparse'),str(ROOT/'experiments/ROUND-014'),str(ROOT/'experiments/ROUND-018')]
import numpy as np,torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from spikingjelly.activation_based import functional
from models.cifarnet import cifarnet
from models.submodules.layers import SparseConv2d,SparseNeuron
from controlled_data import TrainFileOnly
from utils.utils import SOPMonitor
import math_core
E=ROOT/'experiments/EXP-021';OLD=ROOT/'experiments/EXP-020';C=json.loads((OLD/'config.json').read_text());S=json.loads((OLD/'data_split.json').read_text());T=C['targets'];Q=C['q'];M=('G0','G1','G2');CODE=os.popen('git rev-parse HEAD').read().strip()
def dump(n,v):
 p=E/n;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def write(n,rows):
 if not rows:return
 p=E/n
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def cpu():
 r=resource.getrusage(resource.RUSAGE_SELF);c=resource.getrusage(resource.RUSAGE_CHILDREN);v=r.ru_utime+r.ru_stime+c.ru_utime+c.ru_stime
 try:
  import psutil
  for p in psutil.Process().children(recursive=True):
   try:z=p.cpu_times();v+=z.user+z.system
   except psutil.Error:pass
 except ImportError:pass
 return v
START=time.time();CPU=cpu();PRIOR_WALL=float(os.environ.get('EXP020_PRIOR_WALL','0'));PRIOR_CPU=float(os.environ.get('EXP020_PRIOR_CPU','0'));status={'status':'RUNNING','run_id':os.environ.get('EXP020_RUN_ID',C['run_id']),'code_sha':CODE,'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),'start_unix':START,'phase':'preflight'}
def guard():
 wall=PRIOR_WALL+time.time()-START;c=PRIOR_CPU+cpu()-CPU
 if wall>840 or c>1740:raise TimeoutError('resource budget reserve reached')
 status.update(wall_seconds=wall,cpu_seconds=c);dump('status.json',status)
def abort(s,f):raise TimeoutError('watchdog signal '+str(s))
signal.signal(signal.SIGTERM,abort)
def loader(ids):return DataLoader(TrainFileOnly(ids,False),batch_size=64,shuffle=False,num_workers=4,pin_memory=True,drop_last=False)
def get(m,n):return m.get_submodule(n)
def model():
 assert sha(C['parent_path'])==C['parent_sha256'];ck=torch.load(C['parent_path'],map_location='cpu',weights_only=False);assert ck['epoch']==300 and ck['group']=='U2'
 m=cifarnet(T=4,base_channels=256,num_classes=10);m.load_state_dict(ck['model']);del ck;m.cuda().eval()
 for p in m.parameters():p.requires_grad_(False)
 return m

def collect_gradients(m,data):
 grads={n:{k:None for k in ('G0','G1','G2','h')} for n,_,_ in T};ds={};co={};perms={};fanrows=[];coefrows=[];graph=[]
 for idx,(bn,act,conv) in enumerate(T):
  b,n,v=get(m,bn),get(m,act),get(m,conv);assert len(n.next_convs)==1 and n.next_convs[0] is v and len(n.prev_bns)==1 and n.prev_bns[0] is b and v.groups==1 and v.in_channels==b.num_features and n.flatten is None and b.affine
  d=v.sparse_mask().detach().sum((0,2,3)).double();a=math_core.coefficients(d,.25);p=torch.tensor(json.loads((OLD/'permutation.json').read_text())[bn],dtype=torch.long);a2=a[p.to(a.device)];assert torch.equal(a.sort().values,a2.sort().values);d0=v.out_channels*v.kernel_size[0]*v.kernel_size[1];ds[bn]=(d,d0);co[bn]={'G0':torch.ones_like(a),'G1':a,'G2':a2};perms[bn]=p.tolist();graph.append({'bn':bn,'neuron':act,'consumer':conv,'registered_direct':True,'groups':v.groups,'channels':len(d),'nominal':d0,'mean':d.mean().item(),'std':d.std(unbiased=False).item(),'zero_mean_fallback':d.mean().item()==0})
  for j in range(len(d)):
   fanrows.append({'bn':bn,'consumer':conv,'channel':j,'effective_fanout':d[j].item(),'nominal':d0})
   coefrows.append({'bn':bn,'channel':j,**{k:co[bn][k][j].item() for k in M}})
 dump('target_dependency_audit.json',{'status':'PASSED','targets':graph,'scope':'registered paths and static CifarNet.forward; actual hook output compared with consumer input'});dump('permutation.json',perms);write('fanout_vectors.csv',fanrows);write('coefficient_vectors.csv',coefrows)
 replay=[];capture=[];n_total=0
 for batch,(x,y,ids) in enumerate(data):
  guard();x=x.cuda(non_blocking=True);cache={};hs=[]
  for bn,act,conv in T:
   hs.append(get(m,bn).register_forward_pre_hook(lambda mod,args,n=bn:cache.__setitem__(n,args[0].detach().clone())))
   hs.append(get(m,act).register_forward_hook(lambda mod,args,out,n=act:cache.__setitem__(n,out.detach().clone())))
   hs.append(get(m,conv).register_forward_pre_hook(lambda mod,args,n=conv:cache.__setitem__(n,args[0].detach().clone())))
  with torch.no_grad():out=m(x).detach();functional.reset_net(m)
  for h in hs:h.remove()
  if batch==0:
   with torch.no_grad():out2=m(x).detach();functional.reset_net(m)
   capture.append({'capture_does_not_change_output':bool(torch.equal(out,out2))});assert torch.equal(out,out2)
  count=len(x);n_total+=count
  for bn,act,conv in T:
   z=get(m,bn);node=get(m,act).activation;u=cache[bn];target=cache[act];assert torch.equal(target,cache[conv]);node.reset();node.train();w=z.weight.detach().clone().requires_grad_();b=z.bias.detach().clone().requires_grad_();shape=u.shape
   xhat=F.batch_norm(u.flatten(0,1),z.running_mean,z.running_var,w,b,False,z.momentum,z.eps).view(shape);sp=node(xhat);assert torch.equal(sp.detach(),target);replay.append({'batch':batch,'bn':bn,'exact_spike_match':True});rate=sp.mean(dim=(0,1,3,4));d,d0=ds[bn]
   vals={k:d0*(rate*co[bn][k].to(rate.dtype)).sum() for k in M};vals['h']=(rate*d.to(rate.dtype)).sum()
   for k in ('G0','G1','G2','h'):
    gg=torch.autograd.grad(vals[k],(w,b),retain_graph=k!='h');g=torch.cat([v.detach().double() for v in gg]);grads[bn][k]=g*count if grads[bn][k] is None else grads[bn][k]+g*count
   node.reset();node.eval();del xhat,sp,rate,vals,w,b,u,target
  del cache,x,out
 assert n_total==512
 stats=[];matched={};formula=[];vectors=[]
 for bn,_,_ in T:
  g={k:v/n_total for k,v in grads[bn].items()};d,d0=ds[bn];expected=.75*g['G0']+.25*d0/d.mean()*g['h'] if d.mean()>0 else g['G0'];err=torch.linalg.vector_norm(g['G1']-expected)/(torch.linalg.vector_norm(expected)+1e-30);assert err.item()<1e-5
  formula.append({'bn':bn,'R78_relative_error':err.item(),'passed':True});matched[bn]={}
  for k in M:
   v=math_core.match(g[k],g['G0']);matched[bn][k]=v;norm=torch.linalg.vector_norm(g[k]).item();mn=torch.linalg.vector_norm(v).item();ref=torch.linalg.vector_norm(g['G0']).item();assert abs(mn-ref)<=1e-5*max(ref,1e-30)
   n=len(d)
   for j in range(2*n):vectors.append({'bn':bn,'group':k,'parameter':'gamma' if j<n else 'beta','channel':j%n,'raw':g[k][j].item(),'matched':v[j].item()})
   mag=torch.sqrt(g[k][:n]**2+g[k][n:]**2);cor=spearmanr(d.cpu().numpy(),mag.cpu().numpy()).statistic
   stats.append({'bn':bn,'group':k,'raw_norm':norm,'matched_norm':mn,'cosine_g0':F.cosine_similarity(g[k][None],g['G0'][None]).item() if norm and ref else 0,'fanout_abs_gradient_spearman':float(cor) if np.isfinite(cor) else 'NA'})
 dump('local_replay_tests.json',{'status':'PASSED','replay':replay,'capture':capture,'state_reset_per_forward':True,'local_node_training_for_surrogate_only':True});dump('gradient_formula_tests.json',{'synthetic':math_core.tests(),'real_R78':formula,'status':'PASSED'});write('gradient_statistics.csv',stats);write('gradient_vectors.csv',vectors)
 return matched


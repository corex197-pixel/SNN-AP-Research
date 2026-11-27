"""Audits adapted from EXP-017, with per-epoch compressed validation shards."""
import os,sys,time,json,csv,gzip
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent));import common as C
import torch,numpy as np
from torch.nn import functional as F
from utils.utils import SOPMonitor
from spikingjelly.activation_based import functional
O=C.O;K=json.loads((C.ROOT/'experiments/EXP-017/lambda_derivation.json').read_text())['K_actual_batch2_and3'][0]
def write_shard(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
 with gzip.open(tmp,'wt',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
 os.replace(tmp,path)
def link_ck(src,name):
 dst=src.parent/name;tmp=dst.with_suffix('.pth.tmp');
 if tmp.exists():tmp.unlink()
 os.link(src,tmp);os.replace(tmp,dst);return dst
def layer_audit(m,group,epoch):
 rows=[];bn=[];fan=[];raw=[]
 mods=dict(m.named_modules())
 for n,z in mods.items():
  if isinstance(z,C.SparseConv2d):
   w=z.sparse_weight().detach();count=int((w!=0).sum().item());a=z.weight.detach().abs().flatten();num=a.numel();q10=float(a.kthvalue(max(1,int(num*.1)))[0].item());q50=float(a.kthvalue(max(1,int(num*.5)))[0].item());q90=float(a.kthvalue(max(1,int(num*.9)))[0].item());raw.append({'group':group,'epoch':epoch,'layer':n,'q10_abs':q10,'median_abs':q50,'q90_abs':q90,'threshold_to_median':z.threshold.item()/q50 if q50 else 'NA_zero_median','quantile_method':'exact kthvalue lower order statistic'});rows.append({'group':group,'epoch':epoch,'layer':n,'penalized':bool(K[n]>0),'threshold':z.threshold.item(),'nonzero':count,'total':w.numel(),'density':count/w.numel()});assert torch.isfinite(z.threshold).item() and torch.isfinite(w).all().item()
  if isinstance(z,C.SparseBN):bn.append({'group':group,'epoch':epoch,'layer':n,'gamma_mean':z.weight.mean().item(),'gamma_std':z.weight.std(unbiased=False).item(),'beta_mean':z.bias.mean().item(),'beta_std':z.bias.std(unbiased=False).item()})
 for neuron,consumer in [('static_conv.activation','conv1.conv'),('conv1.activation','conv2'),('conv3.activation','conv4.conv'),('conv4.activation','conv5')]:
  v=mods[consumer];d=v.sparse_mask().sum(dim=(0,2,3)).detach().float().cpu().numpy();mean=float(d.mean());fan.append({'group':group,'epoch':epoch,'neuron':neuron,'consumer':consumer,'nominal':v.out_channels*v.kernel_size[0]*v.kernel_size[1],'mean':mean,'std':float(d.std()),'CV':float(d.std()/mean) if mean else 'NA_zero_mean','min':float(d.min()),'p10':float(np.quantile(d,.1)),'p50':float(np.quantile(d,.5)),'p90':float(np.quantile(d,.9)),'max':float(d.max()),'zero_fraction':float((d==0).mean()),'channel_fanout_json':json.dumps(d.tolist())})
 C.append(O/'layer_sparsity.csv',rows);C.append(O/'threshold_trajectory.csv',[{'group':r['group'],'epoch':r['epoch'],'layer':r['layer'],'threshold':r['threshold'],'penalized':r['penalized']} for r in rows]);C.append(O/'bn_statistics.csv',bn);C.append(O/'raw_weight_statistics.csv',raw)
 if epoch in [20,60,100,200,300]:C.append(O/'effective_fanout_statistics.csv',fan)
 total=sum(r['total'] for r in rows);nonzero=sum(r['nonzero'] for r in rows);pen=[r for r in rows if r['penalized']];return {'all_sparse_weight_sparsity':1-nonzero/total,'penalized_weight_sparsity':1-sum(r['nonzero'] for r in pen)/sum(r['total'] for r in pen),'min_penalized_density':min(r['density'] for r in pen)}
def evaluate(m,data,group,epoch):
 pre_rng=C.rng();pre_modes={z:z.training for z in m.modules()};pre_buffers={n:v.detach().clone() for n,v in m.named_buffers()};saved_cache={z:z.pre_norm_input for z in m.modules() if hasattr(z,'pre_norm_input')};m.eval();mon=SOPMonitor(m);spikes={};states={};batch_rates={};batches={};first_numel={};hooks=[];nonbinary={}
 for n,z in m.named_modules():
  if isinstance(z,C.SparseNeuron):
   def hook(mod,inp,out,n=n):
    a=(out!=0).sum(dtype=torch.int64);spikes[n]=spikes.get(n,0)+a;states[n]=states.get(n,0)+out.numel();batch_rates[n]=batch_rates.get(n,0)+a.double()/out.numel();batches[n]=batches.get(n,0)+1;first_numel.setdefault(n,out.numel())
   hooks.append(z.register_forward_hook(hook))
 # Check monitored layer inputs are binary on the first validation batch.
 for name in mon.monitored_layers:
  def hook(mod,args,name=name):
   if name not in nonbinary:
    a=args[0];nonbinary[name]=bool(((a!=0)&(a!=1)).any().item())
  hooks.append(m.get_submodule(name).register_forward_pre_hook(hook))
 total_sop=0.;batch_sops=[];samples=[];count=0;loss_sum=0.;correct=0
 try:
  with torch.no_grad():
   for batch,(x,y,sid) in enumerate(data):
    C.guard(reserve=300);x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);mon.clear_recorded_data();z=m(x).mean(0);loss=F.cross_entropy(z,y,reduction='none');assert torch.isfinite(loss).all();pred=z.argmax(1);so=sum(float(v.item()) for v in mon.records);total_sop+=so;count+=len(y);loss_sum+=float(loss.sum().item());correct+=int((pred==y).sum().item());batch_sops.append({'group':group,'epoch':epoch,'batch':batch,'samples':len(y),'sop_count':so});samples.extend({'group':group,'epoch':epoch,'sample_id':i,'label':t,'prediction':p,'correct':int(p==t),'loss':l} for i,t,p,l in zip(sid.tolist(),y.cpu().tolist(),pred.cpu().tolist(),loss.cpu().tolist()));functional.reset_net(m)
 finally:
  mon.remove_hooks()
  for h in hooks:h.remove()
  functional.reset_net(m)
  for n,v in m.named_buffers():
   if not torch.equal(v,pre_buffers[n]):raise RuntimeError('eval mutated buffer '+n)
  for z,val in saved_cache.items():z.pre_norm_input=val
  for z,mode in pre_modes.items():z.training=mode
  C.set_rng(pre_rng)
 assert count==4500;write_shard(O/'validation_batch_sops'/group/f'epoch-{epoch:03d}.csv.gz',batch_sops);write_shard(O/'validation_per_sample'/group/f'epoch-{epoch:03d}.csv.gz',samples);C.append(O/'layer_firing_rate.csv',[{'group':group,'epoch':epoch,'neuron':n,'FR_sample_weighted':int(spikes[n].item())/states[n],'FR_author':float((batch_rates[n]/batches[n]).item())} for n in spikes]);weighted=sum(int(v.item()) for v in spikes.values())/sum(states.values());author=sum(float((batch_rates[n]/batches[n]).item())*first_numel[n] for n in spikes)/sum(first_numel.values());return {'validation_accuracy':correct/count,'validation_loss':loss_sum/count,'FR_sample_weighted':weighted,'FR_author':author,'SOP_author':sum(r['sop_count'] for r in batch_sops)/len(batch_sops)/256,'SOP_sample_weighted':total_sop/count,'nonbinary_monitored_inputs':json.dumps([k for k,v in nonbinary.items() if v]),'validation_samples':count}

"""Exact readout algebra, fixed sorting, physically compact CifarNet."""
import copy,time
import torch,numpy as np
from torch.nn import functional as F
from models.submodules.layers import SparseConv2d

def readout(m):
 w=m.fc2.sparse_weight().detach()[:,:,0,0].double();a=w.reshape(10,10,2048).mean(1);b=m.fc2.bias.detach().double().reshape(10,10).mean(1) if m.fc2.bias is not None else torch.zeros(10,device=w.device,dtype=torch.float64);return a,b

def reconstruct(rate,a,b,removed=()):
 z=rate@a.T+b
 if len(removed):z-=rate[:,removed]@a[:,removed].T
 return z

def losses(z,y):return torch.logsumexp(z,dim=-1)-z.gather(-1,y.view(*([1]*(z.ndim-2)),-1,1).expand(*z.shape[:-1],1)).squeeze(-1)

def scores(z,rate,a,y,indices,chunk=64):
 current=losses(z,y).mean();result=[]
 for start in range(0,len(indices),chunk):
  ix=indices[start:start+chunk];zz=z[None,:,:]-rate[:,ix].T[:,:,None]*a[:,ix].T[:,None,:];result.extend((losses(zz,y).mean(1)-current).cpu().tolist())
 return np.asarray(result,dtype=np.float64)

def stable_order(ids,vals):return np.asarray(ids,dtype=np.int64)[np.lexsort((np.asarray(ids,dtype=np.int64),np.asarray(vals,dtype=np.float64)))]

def compact(m,deleted):
 out=copy.deepcopy(m);dev=next(m.parameters()).device;deleted=set(int(i) for i in deleted);keep=torch.tensor([i for i in range(2048) if i not in deleted],device=dev,dtype=torch.long);assert len(keep)==2048-len(deleted)
 old1,old2=out.fc1,out.fc2
 f1=SparseConv2d(old1.in_channels,len(keep),1,bias=old1.bias is not None).to(dev);f2=SparseConv2d(len(keep),old2.out_channels,1,bias=old2.bias is not None).to(dev)
 with torch.no_grad():
  f1.weight.copy_(old1.weight.index_select(0,keep));f1.threshold.copy_(old1.threshold);f1.penalty.copy_(old1.penalty)
  if f1.bias is not None:f1.bias.copy_(old1.bias.index_select(0,keep))
  f2.weight.copy_(old2.weight.index_select(1,keep));f2.threshold.copy_(old2.threshold);f2.penalty.copy_(old2.penalty)
  if f2.bias is not None:f2.bias.copy_(old2.bias)
 out.fc1=f1;out.fc2=f2;out.activation5.next_convs=[out.fc1];out.activation_fc1.next_convs=[out.fc2];out.eval();assert out.activation_fc1.next_convs==[out.fc2] and out.activation5.next_convs==[out.fc1];return out,keep

"""Differentiable fanout accounting tied to actual ATLIF output -> Conv2d edges."""
import torch
from torch import nn
from torch.nn import functional as F
from models.submodules.layers import ATLIF

def fanout(conv, shape, device):
    c,h,w=shape
    kh,kw=conv.kernel_size; sh,sw=conv.stride; ph,pw=conv.padding; dh,dw=conv.dilation
    oh=(h+2*ph-dh*(kh-1)-1)//sh+1; ow=(w+2*pw-dw*(kw-1)-1)//sw+1
    oph=h-((oh-1)*sh-2*ph+dh*(kh-1)+1); opw=w-((ow-1)*sw-2*pw+dw*(kw-1)+1)
    if not (0<=oph<sh and 0<=opw<sw): raise RuntimeError('invalid transpose output padding')
    with torch.autocast(device_type=device.type,enabled=False):
        out=F.conv_transpose2d(torch.ones(1,conv.out_channels,oh,ow,device=device),torch.ones_like(conv.weight,dtype=torch.float32),stride=conv.stride,padding=conv.padding,output_padding=(oph,opw),groups=conv.groups,dilation=conv.dilation)
    if out.shape!=(1,c,h,w): raise RuntimeError('fanout shape mismatch')
    return out.detach()

class Proxy:
    def __init__(self,model):
        self.model=model;self.source={};self.mappings={};self.cache={};self.terms=[];self.hooks=[]
        for n,m in model.named_modules():
            if isinstance(m,ATLIF): self.hooks.append(m.register_forward_hook(self.out(n)))
            if isinstance(m,nn.Conv2d) and not any(s in n for s in model.skip): self.hooks.append(m.register_forward_pre_hook(self.conv(n)))
    def out(self,name):
        def fn(m,args,out):
            if m.thresh.detach().item()<=0: raise RuntimeError('non-positive threshold')
            self.source[id(out)]=(name,m)
        return fn
    def conv(self,name):
        def fn(m,args):
            x=args[0]
            if id(x) not in self.source: raise RuntimeError('unmapped convolution input '+name)
            src,neuron=self.source[id(x)]; old=self.mappings.setdefault(name,src)
            if old!=src: raise RuntimeError('convolution source changed')
            if x.ndim!=5: raise RuntimeError('expected T,B,C,H,W')
            key=(name,tuple(x.shape[2:]),x.device)
            if key not in self.cache:self.cache[key]=fanout(m,x.shape[2:],x.device)
            with torch.autocast(device_type=x.device.type,enabled=False):
                self.terms.append((x.float()/neuron.thresh.detach().float()*self.cache[key]).sum()/x.shape[1])
        return fn
    def clear(self):self.source.clear();self.terms.clear()
    def value(self):
        if not self.terms:raise RuntimeError('no counted convolutions')
        return torch.stack(self.terms).sum()
    def remove(self):
        for h in self.hooks:h.remove()
        self.clear()

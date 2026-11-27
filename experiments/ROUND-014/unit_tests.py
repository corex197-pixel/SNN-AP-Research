"""EXP-015 CPU synthetic tests; torch neuron backend is a compatibility probe only."""
import os,sys,json,traceback,time,resource,hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from spikingjelly.activation_based import neuron,surrogate,functional
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'vendor/UniSparse'))
from models.submodules.layers import SparseConv2d,SparseBN,SparseNeuron
from utils.utils import SOPMonitor,calc_penalty,penalize,clear_penalty
O=ROOT/'experiments/EXP-015'; torch.set_num_threads(1); torch.manual_seed(12450); start=time.monotonic(); CPU0=resource.getrusage(resource.RUSAGE_SELF)
def out(name,x):(O/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
class ProbeNeuron(SparseNeuron):
 def __init__(self):
  super().__init__();self.activation=neuron.LIFNode(tau=2.,decay_input=True,v_threshold=1.,v_reset=0.,surrogate_function=surrogate.ATan(),detach_reset=True,step_mode='m',backend='torch')
class Tiny(nn.Module):
 def __init__(self):
  super().__init__();self.bn=SparseBN(1);self.neuron=ProbeNeuron();self.conv=SparseConv2d(1,1,1);self.neuron.register_prev_bns([self.bn]);self.neuron.register_next_convs([self.conv]);self.skip=[]
 def forward(self,x):return self.conv(self.neuron(self.bn(x)))
class ConvNet(nn.Module):
 def __init__(self):super().__init__();self.conv=SparseConv2d(1,1,2,stride=1,padding=1);self.skip=[]
 def forward(self,x):return self.conv(x)
def count_binary(x,mask,stride,padding,dilation):
 # Independent loops: input [T,B,C,H,W], kernel [out,in,kh,kw].
 t,b,ci,h,w=x.shape;co,c,kh,kw=mask.shape;sh,sw=stride;ph,pw=padding;dh,dw=dilation;oh=(h+2*ph-dh*(kh-1)-1)//sh+1;ow=(w+2*pw-dw*(kw-1)-1)//sw+1;total=0
 for ti in range(t):
  for bi in range(b):
   for oc in range(co):
    for oy in range(oh):
     for ox in range(ow):
      for ic in range(ci):
       for ky in range(kh):
        for kx in range(kw):
         iy=oy*sh-ph+ky*dh;ix=ox*sw-pw+kx*dw
         if 0<=iy<h and 0<=ix<w:total+=float(x[ti,bi,ic,iy,ix])*float(mask[oc,ic,ky,kx])
 return total

def tests():
 c=SparseConv2d(1,1,2);w=torch.tensor([[[[.25,1.],[-1.,.75]]]]);c.weight.data.copy_(w);x=torch.tensor([[[[[1.,0.],[0.,1.]]]]]);dense=c(x);zero_equal=torch.equal(c.sparse_weight(),w) and torch.allclose(dense,F.conv2d(x.flatten(0,1),w).view_as(dense));c.threshold.fill_(.5);expected=torch.sign(w)*torch.relu(w.abs()-.5);actual=c(x);mask=(expected!=0).float();same_mask=torch.equal(c.sparse_mask(),mask);conv_equal=torch.allclose(actual,F.conv2d(x.flatten(0,1),expected).view_as(actual))
 case=[];n=ConvNet().eval();n.conv.weight.data.copy_(w)
 for label,th,z in [('A',0.,x),('B',.5,x),('C',.5,1.-x)]:
  n.conv.threshold.fill_(th);mon=SOPMonitor(n);n(z);author=float(sum(v.item() for v in mon.records));mon.remove_hooks();truth=count_binary(z,n.conv.sparse_mask(),n.conv.stride,n.conv.padding,n.conv.dilation);full=count_binary(z,torch.ones_like(n.conv.weight),n.conv.stride,n.conv.padding,n.conv.dilation);case.append({'case':label,'author_sops':author,'independent_effective_sops':truth,'independent_dense_sops':full,'absolute_error':abs(author-truth),'relative_error':abs(author-truth)/truth if truth else None,'input_binary':True,'threshold':th})
 assert zero_equal and same_mask and conv_equal and all(i['absolute_error']==0 for i in case)
 out('sop_count_tests.json',{'kind':'UNIT_TEST','cases':case,'scope':'one SparseConv2d; skip empty; includes all positions; groups=1; binary input','decomposition':{'old_old':case[0]['independent_effective_sops'],'new_old':case[1]['independent_effective_sops'],'new_new':case[2]['independent_effective_sops'],'structure_delta':case[1]['independent_effective_sops']-case[0]['independent_effective_sops'],'activity_delta':case[2]['independent_effective_sops']-case[1]['independent_effective_sops'],'identity_passed':(case[1]['independent_effective_sops']-case[0]['independent_effective_sops'])+(case[2]['independent_effective_sops']-case[1]['independent_effective_sops'])==case[2]['independent_effective_sops']-case[0]['independent_effective_sops']},'author_input_semantics':'author sums raw input, not explicit x>0; validity requires binary activity for event counts'})
 m=Tiny().train();z=torch.randn(4,3,1,3,3)*2+1;m(z);functional.reset_net(m);calc_penalty(m);cp=m.conv.penalty.item();pg=m.bn.penalty_gamma.clone();pb=m.bn.penalty_beta.clone();oldth=m.conv.threshold.item();oldg=m.bn.weight.detach().clone();oldb=m.bn.bias.detach().clone();penalize(m,.01,.01);changed={'conv.threshold':m.conv.threshold.item()!=oldth,'bn.weight':not torch.equal(oldg,m.bn.weight),'bn.bias':not torch.equal(oldb,m.bn.bias)};clear_penalty(m);cleared=bool(m.conv.penalty.item()==0 and torch.all(m.bn.penalty_gamma==0) and torch.all(m.bn.penalty_beta==0) and m.bn.pre_norm_input is None)
 # Record eval state before and after a new eval forward. Neuron memory is reset explicitly.
 m.eval();before={k:v.detach().clone() for k,v in m.state_dict().items()};m(z);after={k:v.detach().clone() for k,v in m.state_dict().items()};changed_eval=[k for k in before if not torch.equal(before[k],after[k])];eval_pre_input_is_none=m.bn.pre_norm_input is None;v_before_reset=torch.is_tensor(m.neuron.activation.v);functional.reset_net(m);v_after_reset=m.neuron.activation.v
 # CPU GradScaler overflow simulates the official unconditional post-step penalty order.
 p=nn.Parameter(torch.tensor(1.));opt=torch.optim.SGD([p],lr=.1);scaler=torch.amp.GradScaler('cpu',init_scale=8.);opt.zero_grad();scaler.scale(p*torch.tensor(float('inf'))).backward();before_p=p.item();scale0=scaler.get_scale();scaler.step(opt);scaler.update();skipped=(p.item()==before_p and scaler.get_scale()<scale0);before_t=m.conv.threshold.item();m.conv.penalty.fill_(2.);penalize(m,.01,0.);manual_changed=m.conv.threshold.item()!=before_t;clear_penalty(m)
 out('sparse_mechanism_tests.json',{'kind':'UNIT_TEST','zero_threshold_forward_equal':zero_equal,'mask_equals_nonzero_effective_weight':same_mask,'sparse_conv_output_matches_reference':conv_equal,'trainable_parameters':{n:list(v.shape) for n,v in m.named_parameters()},'registered_buffers':{n:list(v.shape) for n,v in m.named_buffers()},'synthetic_penalty_coefficients':{'weight':.01,'neuron':.01,'not_paper_config':True},'conv_penalty_before_clear':cp,'bn_penalty_gamma':pg.tolist(),'bn_penalty_beta':pb.tolist(),'actual_updated_states':changed,'clear_penalty_passed':cleared,'neuron_output_values':sorted(set(m.neuron.activation(z).detach().flatten().tolist())),'neuron_backend':'torch CPU compatibility probe only; official CuPy equivalence NOT TESTED','amp_cpu_overflow_step_skipped':skipped,'manual_penalty_changed_threshold_after_skipped_step':manual_changed,'amp_note':'controlled CPU simulation of official order, not official GPU CuPy AMP test'})
 out('state_isolation_tests.json',{'kind':'UNIT_TEST','eval_state_dict_changed':changed_eval,'eval_pre_norm_input_remains_none_after_clear':eval_pre_input_is_none,'neuron_memory_tensor_before_reset':v_before_reset,'neuron_memory_after_reset':str(v_after_reset),'backend':'torch CPU probe; CuPy behavior not tested','official_eval': 'does not call calc_penalty; explicit reset_net required','stale_training_pre_norm_input':'if not cleared, eval forward does not replace cached train input; avoid relying on it'})
 assert cp>0 and any(changed.values()) and changed['conv.threshold'] and cleared and skipped and manual_changed and not changed_eval
 return {'unit_status':'PASSED','official_cupy_semantics':'NOT_TESTED','gpu_benchmark':'BLOCKED: CuPy unavailable, backend equivalence not verified'}
try:
 result=tests();out('unit_test_status.json',result);print(json.dumps(result))
except BaseException as e:
 traceback.print_exc();out('unit_test_status.json',{'unit_status':'FAILED','error':repr(e),'traceback':traceback.format_exc()});raise
finally:
 r=resource.getrusage(resource.RUSAGE_SELF);out('cpu_unit_resources.json',{'wall_seconds':time.monotonic()-start,'cpu_seconds':r.ru_utime+r.ru_stime-CPU0.ru_utime-CPU0.ru_stime,'gpu_seconds':0})

"""Random-initialized synthetic engineering smoke only, not a model performance experiment."""
import copy,sys,json,time,datetime,traceback
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'Activity-Pruning-SNN-main'))
from models.resnet_cifar import ResNet20
from models.submodules.layers import ATLIF,Surrogate
from spikingjelly.activation_based import functional
OUT=ROOT/'experiments/ROUND-002-SMOKE/runs/SMOKE-RUN-001'
OUT.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4);torch.manual_seed(2026)
start=time.perf_counter();cpu=time.process_time();beg=datetime.datetime.now(datetime.timezone.utc).isoformat()
model=ResNet20(T=8,activation=ATLIF,activation_kwargs={'T':8,'sp':.002,'tau':.9}).eval(); reference=copy.deepcopy(model)
x=torch.randn(1,3,32,32); modules=dict(model.named_modules()); records={}; order=[]; active=[None]; handles=[]
descriptor=Surrogate.__dict__['forward']; orig=Surrogate.forward

def cap(ctx,input,thre,sp):
    records[active[0]]['m'].append(input.detach().clone())
    return orig(ctx,input,thre,sp)
def before(name):
    def fn(mod,args):
        order.append(name); active[0]=name; records[name]={'x':args[0].detach().clone(),'m':[],'before':mod.update_value,'theta':mod.thresh.item()}
    return fn
def after(name):
    def fn(mod,args,out):
        records[name]['out']=out.detach().clone();records[name]['after']=mod.update_value;active[0]=None
    return fn
for name,m in model.named_modules():
    if isinstance(m,ATLIF):
        handles.extend([m.register_forward_pre_hook(before(name)),m.register_forward_hook(after(name))])
try:
    with torch.no_grad():
        y0=reference(x)
        Surrogate.forward=staticmethod(cap)
        try: y1=model(x)
        finally: Surrogate.forward=descriptor
finally:
    for h in handles:h.remove()
with torch.no_grad():
    y2=model(x)
unique=len(set(order))==len(order); diff=max((y0-y1).abs().max().item(),(y0-y2).abs().max().item())
layer_details=[]; max_replay=0.
for name in order:
    r=records[name]; mod=modules[name]; old=mod.update_value
    replay=ATLIF(T=8,thresh=r['theta'],tau=mod.tau,sp=mod.sp).eval()
    with torch.no_grad(): out=replay(r['x'])
    d=(out-r['out']).abs().max().item();max_replay=max(max_replay,d)
    layer_details.append({'name':name,'input_shape':list(r['x'].shape),'membrane_steps':len(r['m']),'replay_max_diff':d,'threshold':r['theta'],'update_before':r['before'],'update_after_first':r['after'],'update_after_second':old})
# Restore diagnostic accumulators to pre-forward values; module has no persistent membrane state.
for name in order: modules[name].update_value=records[name]['before']
functional.reset_net(model)
restored=all(modules[n].update_value==records[n]['before'] and modules[n].thresh.item()==records[n]['theta'] for n in order)
result={'seed':2026,'batch':1,'device':'cpu','random_initialized':True,'training':False,'output_shape':list(y0.shape),'module_count':len(order),'unique_module_calls':unique,'max_output_difference':diff,'max_local_replay_difference':max_replay,'all_captured_t8':all(len(records[n]['m'])==8 for n in order),'restored_thresholds_and_update_value':restored,'surrogate_restored':Surrogate.__dict__['forward'] is descriptor,'static_architecture':'ResNet20 forward is feedforward with residual additions; runtime hooks observed no repeated ATLIF calls; not a general proof for all architectures','output_isolation':'forward hooks offer a downstream output boundary; active intervention and unchanged internal trajectories not tested','end_time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'start_time_utc':beg,'wall_seconds':time.perf_counter()-start,'process_cpu_seconds':time.process_time()-cpu,'layers':layer_details}
(OUT/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if not(unique and diff<=1e-5 and max_replay<=1e-5 and restored and result['all_captured_t8']):sys.exit(1)

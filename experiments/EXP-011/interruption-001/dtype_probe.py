import sys,json,torch
from pathlib import Path
R=Path('/home/hym/keyan1');sys.path.insert(0,str(R/'experiments/EXP-004/controlled'));from core import model_create,ATLIF
m=model_create().cuda().train();out={};hs=[]
for n,z in m.named_modules():
 if isinstance(z,ATLIF):hs.append(z.register_forward_hook(lambda z,a,o,n=n:out.update({n:{'input':str(a[0].dtype),'output':str(o.dtype),'theta':str(z.thresh.dtype)}})))
with torch.no_grad(),torch.cuda.amp.autocast():m(torch.randn(2,3,32,32,device='cuda'))
print(json.dumps(out))

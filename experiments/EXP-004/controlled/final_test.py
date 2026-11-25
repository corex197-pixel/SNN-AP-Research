"""Single locked-checkpoint final test; never used by training."""
import argparse,hashlib,json
from pathlib import Path
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from core import DATA,model_create,transforms_pair,evaluate
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--locked-sha256',required=True);p.add_argument('--output',required=True);a=p.parse_args()
ck=Path(a.checkpoint);digest=hashlib.sha256(ck.read_bytes()).hexdigest()
if digest!=a.locked_sha256:raise SystemExit('checkpoint does not match prelocked SHA256')
out=Path(a.output)
if out.exists():raise SystemExit('result exists; refusing overwrite')
model=model_create().cuda();model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['model']);ds=CIFAR10(DATA,train=False,transform=transforms_pair()[1],download=False);result=evaluate(model,DataLoader(ds,batch_size=128,shuffle=False,num_workers=4),'cuda');out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x') as f:json.dump({'checkpoint_sha256':digest,'test':result},f,indent=2)

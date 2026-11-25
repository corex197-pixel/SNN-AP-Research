import argparse,json,sys,subprocess
from pathlib import Path
import torch
import yaml, hashlib
from core import ROOT,seed_all,model_create,opt_create,sched_create,loaders,step,evaluate,save,restore
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--resume');p.add_argument('--confirm-formal-training',action='store_true');a=p.parse_args()
if not a.confirm_formal_training:sys.exit('Formal 200-epoch training requires explicit resource approval and --confirm-formal-training')
cfg=yaml.safe_load(Path(a.config).read_text());required={'dataset':'CIFAR10','model':'ResNet20','seed':42,'T':8,'eta':0.002,'eta2':0.0,'batch_size':128,'accumulation_steps':1,'epochs':200,'optimizer':'sgd','lr':0.1,'lr_scheduler':'cosine','min_lr':1e-6,'weight_decay':0.0005,'amp':True,'TET':False}
if any(cfg.get(k)!=v for k,v in required.items()):sys.exit('Config does not match controlled fixed protocol')
if not torch.cuda.is_available():sys.exit('CUDA unavailable')
out=Path(a.output);out.mkdir(parents=True,exist_ok=bool(a.resume));(out/'run_metadata.json').write_text(json.dumps({'command':sys.argv,'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'config':cfg,'split_sha256':hashlib.sha256((ROOT/'experiments/EXP-004/splits/indices.json').read_bytes()).hexdigest()},indent=2));seed_all();device='cuda';model=model_create().to(device);opt=opt_create(model);sched=sched_create(opt);scaler=torch.cuda.amp.GradScaler();train,val=loaders();start=0;best=-1.
if a.resume:start,best=restore(a.resume,model,opt,sched,scaler,device);start+=1
for epoch in range(start,200):
 model.train()
 for x,y in train:step(model,opt,scaler,x.to(device),y.to(device))
 sched.step(epoch+1);m=evaluate(model,val,device)
 improved=m['top1']>best
 if improved:best=m['top1']
 save(out/'checkpoint_latest.pth',model,opt,sched,scaler,epoch,best)
 if improved:save(out/'checkpoint_best_validation.pth',model,opt,sched,scaler,epoch,best)
 print(json.dumps({'epoch':epoch,'validation':m,'best':best}),flush=True)

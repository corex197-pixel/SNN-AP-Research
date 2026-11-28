"""Stage B only: locked five-group training. No validation loader or evaluator."""
import os,time,json,copy,traceback
from common import *
import gate
stage=os.environ.get('ROUND027_STAGE_ID','train-manual');started=time.monotonic();cpu0=K.cpu();run=E/'runs'/stage;run.mkdir(parents=True,exist_ok=True)
def guard():
 if time.monotonic()-started>=float(os.environ['ROUND027_REMAIN_WALL'])-100 or K.cpu()-cpu0>=float(os.environ['ROUND027_REMAIN_CPU'])-120:raise TimeoutError('remaining resource reserve')
def remote():return subprocess.check_output(['git','ls-remote','origin','refs/heads/research/round-027-pruning-damage-distillation'],text=True,timeout=60).split()[0]
def run_training(dataset):
 guard();init_runtime();teacher=model_teacher();p0,_,_=model_p0(teacher);teacher.eval();p0.eval();t0=statehash(teacher);p0hash=statehash(p0);split=json.loads(SPL.read_text());ids=split['train'];refnames={n:z.sparse_mask().detach().bool().cpu() for n,z in p0.named_modules() if isinstance(z,SparseConv2d)};epochs=[];samples=[];dist=[];frozen={};thresholds=[];top=[];manifest={};training_started=time.monotonic()
 for group in NAMES:
  guard();random.seed(12450);np.random.seed(12450);torch.manual_seed(12450);torch.cuda.manual_seed(12450);m,_,_=model_p0(teacher);before=snapshot(m);initial=statehash(m);assert initial==p0hash;opt,sch,sc=setup_student(m);assert sorted(n for n,p in m.named_parameters() if p.requires_grad)==['fc2.bias','fc2.weight'];success=skips=0
  for epoch in range(1,6):
   guard();dl,seq,drop=training_loader(dataset,ids,epoch);srow={'group':group,'epoch':epoch,'sampler_seed':12450+epoch,'worker_seed':99912450+epoch,'sample_order_sha256':idhash(seq),'dropped_count':len(drop),'dropped_ids':json.dumps(drop,separators=(',',':'))};samples.append(srow)
   if group!='K0':
    old=next(q for q in samples if q['group']=='K0' and q['epoch']==epoch);assert all(old[k]==srow[k] for k in ('sampler_seed','worker_seed','sample_order_sha256','dropped_ids'))
   write('training_sample_audit.csv',samples);lr=opt.param_groups[0]['lr'];assert abs(lr-.01*(1+math.cos(math.pi*(epoch-1)/5))/2)<1e-14;st=time.monotonic();count=correct=good=skip=0;sumce=sumkd=sumloss=0.;seen=[];sumw=sumw2=0.;nw=nonuniform=zero=shifted=0;wmin=float('inf');wmax=-float('inf');sumd=torch.zeros(4,device='cuda',dtype=torch.float64);sumkl=torch.zeros(4,device='cuda',dtype=torch.float64)
   for step,(x,y,sid) in enumerate(dl):
    if step%10==0:guard()
    x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True)
    with torch.no_grad(),torch.autocast('cuda'):
     tt,_=forward(teacher,x) if group!='K0' else (None,None);pp,_=forward(p0,x) if group in ('K3','M6') else (None,None)
    with torch.autocast('cuda'):
     zz,_=forward(m,x);loss,ce,kd,d,w,kl=losses(group,zz,y,tt,pp)
    assert bool(torch.isfinite(loss)) and bool(torch.isfinite(zz).all());sc.scale(loss).backward();sc.unscale_(opt);assert all(bool(torch.isfinite(p.grad).all()) for p in m.parameters() if p.grad is not None);old=sc.get_scale();sc.step(opt);sc.update();is_skip=sc.get_scale()<old;functional.reset_net(m)
    n=len(y);count+=n;seen.extend(sid.tolist());correct+=int((zz.mean(0).argmax(1)==y).sum());sumloss+=float(loss)*n;sumce+=float(ce)*n;sumkd+=float(kd)*n;good+=int(not is_skip);skip+=int(is_skip)
    if w is not None:
     sumw+=float(w.sum());sumw2+=float(w.square().sum());nw+=w.numel();wmin=min(wmin,float(w.min()));wmax=max(wmax,float(w.max()));nonuniform+=int(((w.max(0).values-w.min(0).values)>1e-6).sum());zero+=int((d.sum(0)==0).sum());shifted+=int((w!=w.roll(-1,0)).sum());sumd+=d.double().sum(1);sumkl+=kl.double().sum(1)
   assert seen==seq and count==44800 and step==174 and manual_calls['count']==0 and all(bool(torch.isfinite(p).all()) for p in m.parameters()) and sum(p.numel() for p in m.parameters())==11399524;state=m.state_dict();changed=[n for n,v in before.items() if n not in ('fc2.weight','fc2.bias') and not torch.equal(v,state[n].cpu())];thbad=[n for n,v in before.items() if n.endswith('.threshold') and not torch.equal(v,state[n].cpu())];assert not changed and not thbad and statehash(teacher)==t0 and statehash(p0)==p0hash and all(q.grad is None for q in teacher.parameters()) and all(q.grad is None for q in p0.parameters());frozen[group+str(epoch)]={'checked_tensors':len(before)-2,'differences':changed,'teacher_and_P0_state_exact':True,'threshold_changes':thbad};dump('frozen_parameter_audit.json',frozen)
   for n,z in m.named_modules():
    if isinstance(z,SparseConv2d):
     mask=z.sparse_mask().detach().bool().cpu();base=refnames[n];thresholds.append({'group':group,'epoch':epoch,'layer':n,'threshold':float(z.threshold),'unchanged':n+'.threshold' not in thbad});top.append({'group':group,'epoch':epoch,'layer':n,'activated':int((mask&~base).sum()),'deactivated':int((base&~mask).sum()),'effective_nonzero':int(mask.sum()),'total':mask.numel()})
   write('threshold_audit.csv',thresholds);write('topology_transition.csv',top);success+=good;skips+=skip;sch.step();path=E/'models'/group/f'epoch-{epoch:03d}-recovery.pth';atomic_torch(path,{'group':group,'epoch':epoch,'model':m.state_dict(),'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),'scaler':sc.state_dict(),'parent_sha256':CFG['p0_sha256'],'config_sha256':sha(E/'config.json'),'code_sha':CODE,'teacher_sha256':CFG['teacher_sha256']});row={'group':group,'epoch':epoch,'learning_rate':lr,'CE':sumce/count,'KD':sumkd/count,'loss':sumloss/count,'accuracy':correct/count,'samples':count,'batches':175,'successful_steps':good,'amp_skips':skip,'cumulative_successful':success,'cumulative_skips':skips,'manual_updates':manual_calls['count'],'parameters':11399524,'train_seconds':time.monotonic()-st,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'checkpoint_path':str(path),'checkpoint_sha256':sha(path)};epochs.append(row);write('epoch_metrics.csv',epochs)
   if group in ('K3','M6'):
    mu=sumw/nw;dist.append({'group':group,'epoch':epoch,'w_mean':mu,'w_variance':max(0.,sumw2/nw-mu*mu),'w_min':wmin,'w_max':wmax,'zero_damage_fraction':zero/count,'nonuniform_fraction':nonuniform/count,'shift_different_elements':shifted,'mean_damage_per_timestep':json.dumps((sumd/count).cpu().tolist()),'mean_KL_per_timestep':json.dumps((sumkl/count).cpu().tolist())});write('distillation_metrics.csv',dist)
   dump('training_status.json',{'status':'RUNNING','group':group,'epoch':epoch,'all_completed':{n:sum(1 for r in epochs if r['group']==n) for n in NAMES},'training_elapsed_seconds':time.monotonic()-training_started});print('EPOCH',group,epoch,'CE',sumce/count,'KD',sumkd/count,'skips',skip,flush=True)
  p=E/'models'/group/'epoch-005-inference.pth';atomic_torch(p,{'group':group,'epoch':5,'model':{n:v.detach().cpu() for n,v in m.state_dict().items()},'parent_sha256':CFG['p0_sha256'],'teacher_sha256':CFG['teacher_sha256'],'config_sha256':sha(E/'config.json'),'code_sha':CODE});manifest[group]={'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'recovery_path':str(path),'recovery_sha256':sha(path),'epochs':5,'successful_updates':success,'amp_skips':skips};dump('checkpoint_manifest.json',manifest);del m,opt,sch,sc,before;torch.cuda.empty_cache()
 assert statehash(teacher)==t0 and statehash(p0)==p0hash and sha(B0)==CFG['teacher_sha256'] and sha(P0)==CFG['p0_sha256'];dump('training_teacher_protection_audit.json',{'teacher_before':t0,'teacher_after':statehash(teacher),'P0_before':p0hash,'P0_after':statehash(p0),'manual_calls':manual_calls['count']});dump('training_status.json',{'status':'COMPLETED','code_sha':CODE,'stage_id':stage,'completed_epochs':{g:5 for g in NAMES},'training_seconds':time.monotonic()-training_started});print('TRAINING_COMPLETED',flush=True)
def main():
 # Actual train entry checks remote lock before constructing even its training dataset.
 assert sha(SPL)==CFG['split_sha256'] and idhash(DGRAD)==CFG['D_grad_sha256'];lock=json.loads((E/'training_admission_lock.json').read_text());commit=json.loads((E/'training_admission_lock_commit.json').read_text());assert sha(E/'training_admission_lock.json')==commit['lock_sha256'];identity={'teacher':sha(B0),'P0':sha(P0)};conf=sha(E/'config.json');rsha=remote();expected=lock['expected_samples'];return gate.stage_entry(lock,'training',identity,conf,rsha,commit['sha'],expected,lambda:K.TrainFileOnly(json.loads(SPL.read_text())['train'],True),run_training)
try:main()
except BaseException as ex:dump('training_status.json',{'status':'PARTIAL','code_sha':CODE,'stage_id':stage,'error':repr(ex),'traceback':traceback.format_exc()});traceback.print_exc();raise

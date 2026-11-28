"""EXP-023 locked exposure admission followed by gated 3x10 epoch training."""
import os,sys,time,json,csv,hashlib,subprocess,signal,traceback,gc,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT);sys.path[:0]=[str(ROOT/'experiments/ROUND-017'),str(ROOT/'experiments/ROUND-021')]
import common as C,audit_functions as A,exposure
import torch,numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset
from spikingjelly.activation_based import functional
from utils.utils import calc_penalty,penalize,clear_penalty
E=ROOT/'experiments/EXP-023';C.O=E;A.O=E;CFG=json.loads((E/'config.json').read_text());OLD=json.loads((ROOT/'experiments/EXP-020/data_split.json').read_text());K=json.loads((ROOT/'experiments/EXP-017/lambda_derivation.json').read_text())['K_actual_batch2_and3'][0];SCHED=list(csv.DictReader((ROOT/'experiments/EXP-017/learning_rate_schedule.csv').open()));RUN=E/'runs/EXP-023-RUN-001';RUN.mkdir(parents=True,exist_ok=False);CODE=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();START=time.time();CPU=C.cpu_seconds();L=CFG['lambda_weight'];status={'status':'RUNNING','run_id':'EXP-023-RUN-001','code_sha':CODE,'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),'start_utc':C.now(),'phase':'preflight','completed':{'W0':100,'W1':100,'W2':100}}
def guard(reserve=0):
 wall=time.time()-START;cpu=C.cpu_seconds()-CPU
 if wall>=2700-reserve or cpu>=5400-reserve:raise TimeoutError('cumulative resource budget reserve')
 status.update(wall_seconds=wall,cpu_seconds=cpu,last_utc=C.now());return wall,cpu
C.guard=guard

def progress():guard();C.dump(E/'status.json',status);C.dump(RUN/'status.json',status)
def stop(sig,frame):raise TimeoutError('watchdog signal '+str(sig))
signal.signal(signal.SIGTERM,stop)
def parent():
 p=ROOT/CFG['parent_path'];assert C.sha(p)==CFG['parent_sha256'];z=torch.load(p,map_location='cpu',weights_only=False);assert z['epoch']==100 and z['group']=='U2' and z['lambda_weight']==z['lambda_neuron']==L;assert all(k in z for k in ['model','optimizer','scheduler','scaler','rng','sampling']);return z
def restore():
 ck=parent();m=C.model().train();m.load_state_dict(ck['model']);o=C.optim(m);sch=C.scheduler(o);sc=torch.amp.GradScaler('cuda');o.load_state_dict(ck['optimizer']);sch.load_state_dict(ck['scheduler']);sc.load_state_dict(ck['scaler']);C.set_rng(ck['rng']);assert abs(o.param_groups[0]['lr']-float(SCHED[100]['learning_rate']))<1e-12;return ck,m,o,sch,sc

def coefficients(rows):
 p={x['layer']:x['p'] for x in rows};names=[n for n in K if K[n]>0];assert set(names)==set(p);ks=np.array([K[n] for n in names],dtype=np.float64);pv=np.array([p[n] for n in names],dtype=np.float64);assert np.isfinite(pv).all() and (pv>=0).all() and (pv<=1).all();mean=np.dot(ks,pv)/ks.sum();assert mean>0,'all zero exposures: degenerate'
 gen=torch.Generator(device='cpu').manual_seed(20261128);perm=torch.randperm(len(names),generator=gen).tolist();a={'W0':ks,'W1':ks*(.75+.25*pv/mean)};perm_p=pv[perm];mean2=np.dot(ks,perm_p)/ks.sum();assert mean2>0;a['W2']=ks*(.75+.25*perm_p/mean2)
 assert not np.array_equal(pv,perm_p),'fixed permutation degenerate';assert all(np.isfinite(x).all() and (x>=0).all() and abs(x.sum()-ks.sum())<1e-10 for x in a.values());assert np.array_equal(ks*(1+0*pv),ks) and np.array_equal(ks*(.75+.25*np.ones_like(pv)),ks)
 C.dump(E/'permutation.json',{'seed':20261128,'names':names,'permutation':perm,'original_p':pv.tolist(),'permuted_p':perm_p.tolist(),'K_weighted_mean_W1':mean,'K_weighted_mean_W2':mean2});records=[];result={group:dict(K) for group in ['W0','W1','W2']}
 for group,vals in a.items():
  for i,n in enumerate(names):result[group][n]=float(vals[i]);records.append({'group':group,'layer':n,'original_K':float(ks[i]),'assigned_p':float(pv[i] if group!='W2' else perm_p[i]),'Khat':float(vals[i]),'alpha':0 if group=='W0' else .25,'sum_Khat':float(vals.sum())})
 C.append(E/'weight_penalty_coefficients.csv',records);return result

def test_preflight(cf,train_ids):
 # Same exact epoch101 sampler and augmentation sequence for four independent parent restores.
 results=[];gen=torch.Generator().manual_seed(12551);perm=torch.randperm(len(train_ids),generator=gen).tolist();worker=torch.Generator().manual_seed(99912551);data=C.TrainFileOnly(train_ids,True);loader=DataLoader(Subset(data,perm[:256]),batch_size=256,shuffle=False,num_workers=4,pin_memory=True,drop_last=True,generator=worker);x,y,ids=next(iter(loader));x=x.cuda();y=y.cuda();bnref=None;zref=None;models={}
 for group in ['AUTHOR','W0','W1','W2']:
  guard(150);ck,m,o,sch,sc=restore();lr=o.param_groups[0]['lr'];o.zero_grad(set_to_none=True)
  with torch.autocast('cuda'):z=m(x);loss=F.cross_entropy(z.mean(0),y)
  assert torch.isfinite(loss);val=z.detach().float().cpu();assert zref is None or torch.equal(val,zref);zref=val
  sc.scale(loss).backward();wgrad=next(z.weight.grad for n,z in m.named_modules() if n=='conv1.conv');assert wgrad is not None and torch.isfinite(wgrad).all() and torch.count_nonzero(wgrad)>0;functional.reset_net(m);old=sc.get_scale();sc.step(o);sc.update();skipped=sc.get_scale()<old;o.zero_grad(set_to_none=True);calc_penalty(m)
  bn={n:(z.penalty_gamma.detach().cpu().clone(),z.penalty_beta.detach().cpu().clone()) for n,z in m.named_modules() if isinstance(z,C.SparseBN)}
  if bnref is None:bnref=bn
  else:assert all(torch.equal(bn[n][0],bnref[n][0]) and torch.equal(bn[n][1],bnref[n][1]) for n in bn)
  for n,z in m.named_modules():
   if isinstance(z,C.SparseConv2d):
    assert z.penalty.item()==K[n]
    if group!='AUTHOR':z.penalty.fill_(cf[group][n])
  assert all(torch.equal(z.penalty_gamma.cpu(),bn[n][0]) and torch.equal(z.penalty_beta.cpu(),bn[n][1]) for n,z in m.named_modules() if isinstance(z,C.SparseBN))
  before={n:z.threshold.item() for n,z in m.named_modules() if isinstance(z,C.SparseConv2d)};penalize(m,L*lr,L*lr);after={n:z.threshold.item() for n,z in m.named_modules() if isinstance(z,C.SparseConv2d)};clear_penalty(m);models[group]={'model_sha256':C.digest(m),'thresholds':after,'BN':{n:[z.weight.detach().cpu().tolist(),z.bias.detach().cpu().tolist()] for n,z in m.named_modules() if isinstance(z,C.SparseBN)}};results.append({'group':group,'loss':loss.item(),'lr':lr,'AMP_skipped':skipped,'pre_update_forward_equal':True,'BN_penalty_equal':True,'nonzero_raw_weight_gradient':True,'actual_threshold_delta':{n:after[n]-before[n] for n in after}});del ck,m,o,sch,sc,z,loss;gc.collect();torch.cuda.empty_cache()
 assert models['AUTHOR']==models['W0'];assert any(models['W1']['thresholds'][n]!=models['W0']['thresholds'][n] for n in K if K[n]>0);assert any(models['W2']['thresholds'][n]!=models['W0']['thresholds'][n] for n in K if K[n]>0);assert models['W0']['BN']==models['W1']['BN']==models['W2']['BN'];C.dump(E/'preflight_tests.json',{'status':'PASSED','groups':results,'W0_matches_author':True,'same_batch_forward':True,'BN_penalty_and_update_unchanged':True,'raw_weight_grad':True,'independent_parent_restore':True,'scheduler_cycle':300,'initial_lr':results[0]['lr']})

def record_checkpoint(group,epoch,path):
 p=Path(path);ck=torch.load(p,map_location='cpu',weights_only=False);assert ck['epoch']==epoch and all(n in ck for n in ['model','optimizer','scheduler','scaler','rng','sampling']);C.append(E/'checkpoint_manifest.csv',[{'group':group,'epoch':epoch,'path':str(p),'sha256':C.sha(p),'bytes':p.stat().st_size,'load_verified':True,'code_sha':CODE}]);del ck

def topology(m,group,epoch,parent_masks):
 rows=[]
 for n,z in m.named_modules():
  if isinstance(z,C.SparseConv2d):
   cur=z.sparse_mask().detach().bool().cpu();old=parent_masks[n];pruned=int((old&~cur).sum().item());regrown=int((~old&cur).sum().item());original=int(old.sum().item());now=int(cur.sum().item());assert now==original-pruned+regrown;rows.append({'group':group,'epoch':epoch,'layer':n,'parent_nonzero':original,'current_nonzero':now,'pruned_count':pruned,'regrown_count':regrown,'identity_verified':True})
 C.append(E/'topology_transition.csv',rows)

def train_group(group,cf,s,train,val_loader,diag_loader,parent_masks):
 ck,m,opt,sch,scaler=restore();assert ck['epoch']==100;success=int(ck['metrics']['cumulative_successful_steps']);skips=int(ck['metrics']['cumulative_amp_skips']);manual=int(ck['metrics']['cumulative_manual_updates']);del ck;d=RUN/group;d.mkdir(exist_ok=True);zero={n:0 for n in K if K[n]>0}
 for epoch in range(101,111):
  guard(180);status.update(phase='training',group=group,epoch=epoch);progress();t0=time.monotonic();lr=opt.param_groups[0]['lr'];assert abs(lr-float(SCHED[epoch-1]['learning_rate']))<1e-12
  gen=torch.Generator().manual_seed(12450+epoch);perm=torch.randperm(len(train),generator=gen).tolist();used=perm[:175*256];dropped=[s['train'][i] for i in perm[175*256:]];assert len(dropped)==200;wg=torch.Generator().manual_seed(99912450+epoch);dl=DataLoader(Subset(train,used),batch_size=256,shuffle=False,num_workers=4,pin_memory=True,drop_last=True,generator=wg);order=hashlib.sha256(json.dumps([s['train'][i] for i in used],separators=(',',':')).encode()).hexdigest();existing=list(csv.DictReader((E/'training_sample_audit.csv').open())) if (E/'training_sample_audit.csv').exists() else []
  for prev in existing:
   if int(prev['epoch'])==epoch:assert prev['sample_order_sha256']==order and json.loads(prev['dropped_ids'])==dropped
  C.append(E/'training_sample_audit.csv',[{'group':group,'epoch':epoch,'sample_order_sha256':order,'dropped_count':200,'dropped_ids':json.dumps(dropped),'sampler_seed':12450+epoch,'worker_seed':99912450+epoch}]);m.train();count=correct=0;lsum=0.;es=ek=em=0;threshold_start={n:z.threshold.item() for n,z in m.named_modules() if isinstance(z,C.SparseConv2d)}
  for j,(x,y,ids) in enumerate(dl):
   if j%10==0:guard(150)
   x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda'):z=m(x);loss=F.cross_entropy(z.mean(0),y)
   if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
   scaler.scale(loss).backward();functional.reset_net(m);scale=scaler.get_scale();scaler.step(opt);scaler.update();skip=scaler.get_scale()<scale;opt.zero_grad(set_to_none=True);calc_penalty(m)
   for n,mod in m.named_modules():
    if isinstance(mod,C.SparseConv2d):
     if mod.penalty.item()!=K[n]:raise RuntimeError('author K is not fixed '+n)
     mod.penalty.fill_(cf[group][n])
   penalize(m,L*lr,L*lr);clear_penalty(m);em+=1;es+=int(not skip);ek+=int(skip);n=len(y);count+=n;lsum+=loss.item()*n;correct+=int((z.mean(0).argmax(1)==y).sum().item())
   if j%20==0:
    for name,mod in m.named_modules():
     if isinstance(mod,C.SparseConv2d) and not torch.isfinite(mod.threshold).all():raise FloatingPointError('nonfinite threshold '+name)
    wall,cpu=guard(150);C.dump(E/'heartbeat.json',{'group':group,'epoch':epoch,'batch':j,'wall_seconds':wall,'cpu_seconds':cpu,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
  assert j==174 and count==44800;success+=es;skips+=ek;manual+=em;train_time=time.monotonic()-t0;start=time.monotonic();v=A.evaluate(m,val_loader,group,epoch);val_time=time.monotonic()-start;layer=A.layer_audit(m,group,epoch)
  for n,z in m.named_modules():
   if isinstance(z,C.SparseConv2d) and K[n]>0:
    zero[n]=zero[n]+1 if z.density()==0 else 0
    if zero[n]>=1:raise RuntimeError('penalized layer exhausted '+n)
  for name,param in m.named_parameters():
   if not torch.isfinite(param).all():raise FloatingPointError('nonfinite parameter '+name)
  threshold_checks=[]
  for n,z in m.named_modules():
   if isinstance(z,C.SparseConv2d):
    before=threshold_start[n];after=z.threshold.item();expected_ideal=before+175*L*lr*cf[group][n];threshold_checks.append({'group':group,'epoch':epoch,'layer':n,'threshold_start':before,'threshold_end':after,'K_original':K[n],'K_applied':cf[group][n],'ideal_delta':175*L*lr*cf[group][n],'actual_delta':after-before,'float32_accumulation_error':after-expected_ideal})
  C.append(E/'threshold_increment_audit.csv',threshold_checks);sch.step(epoch);row={'group':group,'epoch':epoch,'learning_rate':lr,'train_loss':lsum/count,'train_accuracy':correct/count,'train_samples':count,**v,**layer,'epoch_successful_steps':es,'epoch_amp_skips':ek,'epoch_manual_updates':em,'cumulative_successful_steps':success,'cumulative_amp_skips':skips,'cumulative_manual_updates':manual,'train_seconds':train_time,'validation_seconds':val_time,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
  tmp=d/'latest.pth.tmp';path=d/'latest.pth';torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'group':group,'lambda_weight':L,'lambda_neuron':L,'rng':C.rng(),'sampling':{'method':'epoch-specific torch randperm and DataLoader worker seed','seed_base':12450,'next_epoch':epoch+1},'config':CFG,'code_sha':CODE,'parent_sha256':CFG['parent_sha256'],'K_applied':cf[group],'metrics':row},tmp);os.replace(tmp,path)
  if epoch in [105,110]:
   fixed=d/f'epoch-{epoch:03d}.pth';os.link(path,fixed);record_checkpoint(group,epoch,fixed);topology(m,group,epoch,parent_masks)
   rr,aa=exposure.measure(m,diag_loader,'D_grad_'+group,epoch,K,guard);C.append(E/'exposure_statistics.csv',rr);C.append(E/'sop_reconstruction_audit.csv',aa['checks'])
  C.append(E/'epoch_metrics.csv',[row]);status['completed'][group]=epoch;progress();print(json.dumps({'event':'epoch_complete',**row}),flush=True)
 del m,opt,sch,scaler;gc.collect();torch.cuda.empty_cache()

def main():
 guard(180);assert torch.cuda.device_count()==1 and '4090' in torch.cuda.get_device_name();torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.cuda.set_per_process_memory_fraction(.90);s=C.check();assert C.sha(ROOT/CFG['parent_path'])==CFG['parent_sha256'];manifest=json.loads((ROOT/'experiments/EXP-018/checkpoint_manifest.json').read_text());assert any(x['group']=='U2' and x['path']==str(ROOT/CFG['parent_path']) and x['sha256']==CFG['parent_sha256'] for x in manifest)
 for n,h in [('D_grad','93c4c834c3fd3c8b92a64caa88da8a671e27be23dc2df11999c8e5f3dfbe47d8'),('D_cal','b671cbd803b13cc76b02f5db4bad311ef9492276622051df01af56a931301baa')]:assert hashlib.sha256(json.dumps(OLD[n],separators=(',',':')).encode()).hexdigest()==h
 assert not set(OLD['D_grad'])&set(OLD['D_cal']) and not (set(OLD['D_grad'])|set(OLD['D_cal']))&(set(s['validation'])|set(s['diagnostic']));C.dump(E/'exposure_formula_tests.json',exposure.unit_tests());C.dump(E/'parent_checkpoint_audit.json',{'before':C.sha(ROOT/CFG['parent_path']),'manifest_match':True,'expected':CFG['parent_sha256']});ck,m,opt,sch,sc=restore();C.dump(E/'recovery_audit.json',{'status':'PASSED','epoch':ck['epoch'],'group':ck['group'],'optimizer_state':True,'momentum_buffers':len(opt.state),'scheduler_state':sch.state_dict(),'scaler_state':sc.state_dict(),'next_epoch_lr':opt.param_groups[0]['lr'],'rng_restored':True});del ck,opt,sch,sc;m.eval();buf={n:v.detach().clone() for n,v in m.state_dict().items()};status['phase']='exposure';progress();ds={}
 for label in ['D_grad','D_cal']:
  dl=DataLoader(C.TrainFileOnly(OLD[label],False),batch_size=64,shuffle=False,num_workers=4,pin_memory=True,drop_last=False);ds[label],audit=exposure.measure(m,dl,label,100,K,guard);C.append(E/'exposure_statistics.csv',ds[label]);C.append(E/'sop_reconstruction_audit.csv',audit['checks']);assert audit['status']=='PASSED';del dl
 assert all(torch.equal(v,buf[n]) for n,v in m.state_dict().items());del buf
 primary=['conv1.conv','conv2','conv3.conv'];details={}
 for label in ['D_grad','D_cal']:
  p={r['layer']:r['p'] for r in ds[label]};assert all(K[n]==1024 for n in primary);vals=[p[n] for n in primary];mean=sum(vals)/3;assert mean>0;details[label]={'p':{n:p[n] for n in primary},'A_p':(max(vals)-min(vals))/mean,'min_layer':min(primary,key=lambda n:p[n]),'max_layer':max(primary,key=lambda n:p[n])}
 passed=all(details[x]['A_p']>=.2 for x in details) and details['D_grad']['min_layer']==details['D_cal']['min_layer'] and details['D_grad']['max_layer']==details['D_cal']['max_layer'];C.dump(E/'exposure_stability.json',details);C.dump(E/'sop_reconstruction_audit.json',{'status':'PASSED','input_binary_verified':True,'model_state_unchanged':True,'capture_output_equal':True,'details_file':'sop_reconstruction_audit.csv'});C.dump(E/'hypothesis_gate_results.json',{'H022_direction_gate_passed':passed,'details':details,'H022_scientific_status':'UNTESTED_pending_theory_review','H023_scientific_status':'UNTESTED'});print(json.dumps({'event':'H022_gate','passed':passed,'details':details}),flush=True)
 if not passed:return 'PARTIAL_GATE_NOT_MET'
 cf=coefficients(ds['D_grad']);test_preflight(cf,s['train']);parent_masks={n:z.sparse_mask().detach().bool().cpu() for n,z in m.named_modules() if isinstance(z,C.SparseConv2d)};del m;gc.collect();torch.cuda.empty_cache()
 (E/'exposure_reference_epoch100.csv').write_bytes((E/'exposure_statistics.csv').read_bytes())
 lock={'parent_sha256':CFG['parent_sha256'],'alpha':.25,'D_grad_index_sha256':OLD['D_grad_sha256'],'D_cal_index_sha256':OLD['D_cal_sha256'],'coefficients_sha256':C.sha(E/'weight_penalty_coefficients.csv'),'permutation_sha256':C.sha(E/'permutation.json'),'exposure_sha256':C.sha(E/'exposure_reference_epoch100.csv'),'H022_gate':details,'groups':['W0','W1','W2'],'epochs':[101,110]};C.dump(E/'selection_lock.json',lock)
 subprocess.run(['git','add','experiments/EXP-023/exposure_reference_epoch100.csv','experiments/EXP-023/exposure_statistics.csv','experiments/EXP-023/sop_reconstruction_audit.csv','experiments/EXP-023/exposure_stability.json','experiments/EXP-023/hypothesis_gate_results.json','experiments/EXP-023/weight_penalty_coefficients.csv','experiments/EXP-023/permutation.json','experiments/EXP-023/selection_lock.json','experiments/EXP-023/preflight_tests.json'],check=True);subprocess.run(['git','commit','-m','lock(EXP-023): admit fixed exposure coefficients'],check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);locksha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();C.dump(E/'lock_commit.json',{'sha':locksha,'before_training':True});print('coefficient lock committed',locksha,flush=True)
 train=C.TrainFileOnly(s['train'],True);val=C.TrainFileOnly(s['validation'],False);vl=DataLoader(val,batch_size=256,shuffle=False,num_workers=4,pin_memory=True,drop_last=False);diag=DataLoader(C.TrainFileOnly(OLD['D_grad'],False),batch_size=64,shuffle=False,num_workers=4,pin_memory=True,drop_last=False)
 for group in ['W0','W1','W2']:train_group(group,cf,s,train,vl,diag,parent_masks)
 return 'COMPLETED'
try:status['status']=main()
except BaseException as e:status.update(status='PARTIAL',error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
finally:
 try:
  after=C.sha(ROOT/CFG['parent_path']);C.dump(E/'parent_checkpoint_audit.json',{'before':CFG['parent_sha256'],'after':after,'expected':CFG['parent_sha256'],'match':after==CFG['parent_sha256']})
 except BaseException:pass
 status.update(end_utc=C.now(),wall_seconds=time.time()-START,cpu_seconds=C.cpu_seconds()-CPU,peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None);C.dump(E/'status.json',status);C.dump(RUN/'status.json',status);print(json.dumps(status),flush=True)
if status['status'] not in ['COMPLETED','PARTIAL_GATE_NOT_MET']:sys.exit(1)

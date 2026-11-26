"""One-shot tmux supervisor for approved recovery, training and final validation."""
import os,sys,subprocess,json,time,datetime,signal,hashlib,csv,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];RUN=ROOT/'experiments/EXP-004/runs/EXP-004-RUN-003';CONTROL=ROOT/'experiments/EXP-004/round005/SUPERVISOR-001';CONTROL.mkdir(parents=True,exist_ok=False)
UUID='GPU-c0ba2c71-aee0-5675-4047-210764508ba5';start=datetime.datetime.now(datetime.timezone.utc);deadline=start+datetime.timedelta(hours=7);end=deadline.timestamp();sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,PYTHONUNBUFFERED='1',GIT_TERMINAL_PROMPT='0');child=None
state={'status':'RUNNING','supervisor_pid':os.getpid(),'started_utc':start.isoformat(),'deadline_utc':deadline.isoformat(),'gpu_uuid':UUID,'code_sha':sha,'parent_run':'EXP-004-RUN-002','run_id':'EXP-004-RUN-003','events':[]}
def update(event=None):
 if event:state['events'].append({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**event})
 state['heartbeat_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=CONTROL/'status.json.tmp';tmp.write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n');os.replace(tmp,CONTROL/'status.json');print(json.dumps(state['events'][-1] if event else {'heartbeat_utc':state['heartbeat_utc']}),flush=True)
def stop_signal(sig,frame):
 state['received_signal']=sig
 if child and child.poll() is None:child.send_signal(sig)
 raise RuntimeError(f'supervisor received signal {sig}')
signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
def execute(name,cmd,max_seconds=None):
 global child
 log=CONTROL/(name+'.log');started=time.monotonic()
 with log.open('w') as f:
  child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  update({'stage':name,'child_pid':child.pid,'command':cmd})
  while child.poll() is None:
   if time.time()>=end-30 or (max_seconds is not None and time.monotonic()-started>max_seconds):
    child.send_signal(signal.SIGTERM)
    try:child.wait(timeout=20)
    except subprocess.TimeoutExpired:child.kill();child.wait()
    update({'stage':name,'reason':'deadline or stage timeout','exit_code':child.returncode});return child.returncode
   time.sleep(15);update()
  code=child.returncode
 update({'stage':name,'exit_code':code,'wall_seconds':time.monotonic()-started});return code

def checksum(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def deliver():
 current=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()
 if current!=sha or dirty:update({'stage':'delivery','error':'HEAD changed or tracked files modified; no automatic commit'});return
 if RUN.exists() and (RUN/'status.json').exists():run=json.loads((RUN/'status.json').read_text())
 else:run={'status':'BLOCKED','completed_epochs':37,'best_epoch':34,'best_validation_accuracy':.814,'reason':'training not started'}
 final=json.loads((RUN/'final_summary.json').read_text()) if (RUN/'final_summary.json').exists() else None
 checkpoints=[]
 for name in ('checkpoint_latest.pth','checkpoint_best_validation.pth'):
  p=RUN/name
  if p.exists():checkpoints.append({'name':name,'path':str(p),'size_bytes':p.stat().st_size,'sha256':checksum(p)})
 state.update(run_status=run,final_evaluation_available=final is not None,checkpoints=checkpoints,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 # COMPLETE only after 200 epochs and final evaluation, otherwise retain truthful partial state.
 result_status='COMPLETED' if run.get('status')=='COMPLETED' and final else ('BLOCKED' if run.get('status')=='BLOCKED' else 'PARTIAL')
 state['status']=result_status;update({'stage':'delivery','execution_status':result_status})
 (CONTROL/'delivery.json').write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n')
 text=f"\n\n## ROUND-005 恢复运行\n\n父运行 EXP-004-RUN-002 保留 PARTIAL；恢复段 EXP-004-RUN-003 状态 {result_status}，最近完成 Epoch {run.get('completed_epochs')}，全程最佳 Epoch {run.get('best_epoch')}，最佳验证准确率 {run.get('best_validation_accuracy')}。执行代码 `{sha}`；详细退出记录、GPU 预算及检查点哈希见 round005/SUPERVISOR-001/delivery.json；正式运行数据见 runs/EXP-004-RUN-003/。最终固定验证 {'已产生 final_summary.json' if final else '未完成或未产生'}。不使用正式测试集，H-001/H-002 保持 UNTESTED。\n"
 for path in (ROOT/'research/RESEARCH_STATE.md',ROOT/'experiments/EXP-004/REPORT.md'):
  path.write_text(path.read_text()+text)
 (ROOT/'experiments/EXP-004/status.json').write_text(json.dumps({'execution_status':result_status,'scientific_status':'UNTESTED','parent_run_id':'EXP-004-RUN-002','run_id':'EXP-004-RUN-003','completed_epochs':run.get('completed_epochs'),'best_epoch':run.get('best_epoch'),'best_validation_accuracy':run.get('best_validation_accuracy'),'final_evaluation_available':final is not None},indent=2)+'\n')
 reg=ROOT/'research/EXPERIMENT_REGISTRY.csv'
 with reg.open(newline='') as f:r=csv.DictReader(f);fields=r.fieldnames;rows=list(r)
 rows.append({'experiment_id':'EXP-004','run_id':'EXP-004-RUN-003','round':'ROUND-005','stage':'resume-training','hypothesis_ids':'H-001;H-002','theory_version':'v0.1.1','branch':'research/round-005-baseline-resume','code_sha':sha,'seed':'42','execution_status':result_status,'scientific_status':'UNTESTED','config_path':'experiments/EXP-004/config.yaml','metrics_path':'experiments/EXP-004/runs/EXP-004-RUN-003/epochs.csv' if (RUN/'epochs.csv').exists() else '', 'report_path':'experiments/EXP-004/REPORT.md','start_time_utc':run.get('start_utc',''),'end_time_utc':run.get('end_utc',''),'notes':'resumed parent RUN-002; checkpoint info in supervisor delivery'})
 with reg.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
 paths=['research/RESEARCH_STATE.md','research/EXPERIMENT_REGISTRY.csv','experiments/EXP-004/REPORT.md','experiments/EXP-004/status.json','experiments/EXP-004/round005/RECOVERY-CHECK-001','experiments/EXP-004/round005/SUPERVISOR-001']
 if RUN.exists():paths.append('experiments/EXP-004/runs/EXP-004-RUN-003')
 for path in paths:
  if (ROOT/path).exists():subprocess.run(['git','add',path],cwd=ROOT,check=True)
 subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True);subprocess.run(['git','commit','-m','results(EXP-004): record checkpoint resume outcome'],cwd=ROOT,check=True)
 result_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();push=subprocess.run(['git','push','origin','research/round-005-baseline-resume'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
 (CONTROL/'push-local.json').write_text(json.dumps({'result_sha':result_sha,'push_exit_code':push.returncode,'stdout':push.stdout,'stderr':push.stderr},indent=2)+'\n');print('result_sha',result_sha,'push_exit',push.returncode,push.stderr,flush=True)
try:
 update({'stage':'preflight','code_sha':sha})
 if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():raise RuntimeError('tracked worktree must be clean before start')
 before=subprocess.check_output(['nvidia-smi','-i',UUID,'--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader'],text=True);state['gpu_before']=before.strip();update({'stage':'gpu_observation','value':before.strip()})
 if execute('restore_check',[sys.executable,'-u',str(ROOT/'experiments/EXP-004/round005/restore_check.py')],max_seconds=600)!=0:raise RuntimeError('restore consistency check failed')
 check=json.loads((ROOT/'experiments/EXP-004/round005/RECOVERY-CHECK-001/metrics.json').read_text())
 if check['status']!='COMPLETED':raise RuntimeError('restore check did not pass')
 code=execute('training',[sys.executable,'-u',str(ROOT/'experiments/EXP-004/round005/resume.py'),'--run-id','EXP-004-RUN-003','--deadline-utc',deadline.isoformat(),'--approved-gpu-uuid',UUID,'--code-sha',sha])
 if RUN.exists() and (CONTROL/'training.log').exists():
  import shutil;shutil.copyfile(CONTROL/'training.log',RUN/'run.log')
 if code==0 and time.time()<end-120:execute('final_validation',[sys.executable,'-u',str(ROOT/'experiments/EXP-004/round005/finish.py'),'EXP-004-RUN-003'])
except BaseException as e:state.update(supervisor_error=repr(e),traceback=traceback.format_exc());update({'stage':'exception','error':repr(e)})
finally:
 try:deliver()
 except BaseException as e:
  state.update(status='DELIVERY_FAILED',delivery_error=repr(e),delivery_traceback=traceback.format_exc());update({'stage':'delivery_exception','error':repr(e)})

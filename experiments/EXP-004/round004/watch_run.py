"""Watch one authorized long-running job and publish its actual outcomes."""
import os,sys,time,json,datetime,subprocess,signal,hashlib,csv,shutil,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];RUN=ROOT/'experiments/EXP-004/runs/EXP-004-RUN-002';PID=int(sys.argv[1]);CODE='365da5f0c371b2f9f010f44b8327a0e489184a17';UUID='GPU-1118b368-bc7d-543a-1b80-f79a13730e24';DEADLINE=datetime.datetime.fromisoformat('2026-11-25T14:32:46.377541+00:00').timestamp()
def proc_start():
 try:return Path(f'/proc/{PID}/stat').read_text().split()[21]
 except (FileNotFoundError,ProcessLookupError):return None
initial=proc_start()
if not initial:sys.exit('training process already absent; manual audit required')
def alive():return proc_start()==initial
print(f'watching pid {PID}, proc-start {initial}',flush=True)
terminated=False
while alive():
 if time.time()>=DEADLINE:
  os.kill(PID,signal.SIGTERM);terminated=True;print('budget deadline: sent SIGTERM to matching training PID',flush=True);break
 time.sleep(20)
while alive():time.sleep(1)
time.sleep(3)
log=Path('/tmp/EXP-004-RUN-002.log')
if log.exists():shutil.copyfile(log,RUN/'run.log')
statusfile=RUN/'status.json';st=json.loads(statusfile.read_text()) if statusfile.exists() else {'run_id':RUN.name,'completed_epochs':0}
if st.get('status')=='RUNNING':
 st.update(status='PARTIAL' if st.get('completed_epochs',0) else 'FAILED',error='process exited without final status'+(' after budget signal' if terminated else ''),end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());statusfile.write_text(json.dumps(st,indent=2)+'\n')
summary=None
if st.get('status')=='COMPLETED' and time.time()<DEADLINE-90:
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID);left=max(1,int(DEADLINE-time.time()-30))
 try:
  with (RUN/'final_summary.log').open('w') as f:
   ret=subprocess.run([sys.executable,'-u',str(ROOT/'experiments/EXP-004/round004/summarize.py'),RUN.name],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=left)
  if ret.returncode==0:summary=json.loads((RUN/'final_summary.json').read_text())
  else:print('final summary failed, code',ret.returncode,flush=True)
 except Exception as e:print('final summary error',repr(e),flush=True)
checks=[]
for name in ['checkpoint_latest.pth','checkpoint_best_validation.pth']:
 p=RUN/name
 if p.exists():
  h=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
  checks.append({'path':str(p),'sha256':h.hexdigest(),'size_bytes':p.stat().st_size})
record={'training_status':st,'final_summary_available':summary is not None,'checkpoints':checks,'watcher_finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deadline_termination':terminated}
(RUN/'delivery.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()
if head!=CODE or dirty:
 print('not updating tracked research files or committing: HEAD changed or tracked files modified',flush=True);sys.exit(2)
state=ROOT/'research/RESEARCH_STATE.md';report=ROOT/'experiments/EXP-004/REPORT.md';status=ROOT/'experiments/EXP-004/status.json'
epoch=st.get('completed_epochs',0);best=st.get('best_epoch');acc=st.get('best_validation_accuracy')
text=f"\n\n## ROUND-004 共享 GPU 正式运行结果\n\nEXP-004-RUN-002 状态 {st.get('status')}，实际完成 {epoch}/200 Epoch；最佳验证 Epoch {best}，最佳验证 accuracy {acc}。实际运行代码 `{CODE}`；单 GPU UUID `{UUID}`。训练完整记录、逐层指标、检查点校验及终止原因见 runs/EXP-004-RUN-002/status.json、epochs.csv、layers.csv、delivery.json；最终验证 {'已完成' if summary else '未完成或未生成，见 final_summary.log'}。正式测试集未用于选模；H-001/H-002 维持 UNTESTED。\n"
for p in [state,report]:p.write_text(p.read_text()+text)
status.write_text(json.dumps({'execution_status':st.get('status'),'scientific_status':'UNTESTED','run_id':RUN.name,'completed_epochs':epoch,'best_epoch':best,'best_validation_accuracy':acc,'final_summary_available':summary is not None},indent=2)+'\n')
reg=ROOT/'research/EXPERIMENT_REGISTRY.csv'
with reg.open(newline='') as f:r=csv.DictReader(f);fields=r.fieldnames;rows=list(r)
rows.append({'experiment_id':'EXP-004','run_id':RUN.name,'round':'ROUND-004','stage':'controlled-training','hypothesis_ids':'H-001;H-002','theory_version':'v0.1.1','branch':'research/round-004-baseline-training','code_sha':CODE,'seed':'42','execution_status':st.get('status'),'scientific_status':'UNTESTED','config_path':'experiments/EXP-004/config.yaml','metrics_path':'experiments/EXP-004/runs/EXP-004-RUN-002/epochs.csv' if epoch else '', 'report_path':'experiments/EXP-004/REPORT.md','start_time_utc':st.get('start_utc',''),'end_time_utc':st.get('end_utc',''),'notes':'shared GPU; validation only; full checkpoint SHA in delivery.json'})
with reg.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
paths=['experiments/EXP-004/runs/EXP-004-RUN-002','experiments/EXP-004/round004/watch_run.py','experiments/EXP-004/REPORT.md','experiments/EXP-004/status.json','research/RESEARCH_STATE.md','research/EXPERIMENT_REGISTRY.csv']
subprocess.run(['git','add',*paths],cwd=ROOT,check=True);subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True);subprocess.run(['git','commit','-m','results(EXP-004): record authorized controlled training'],cwd=ROOT,check=True)
sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();print('result commit',sha,flush=True)
env=dict(os.environ,GIT_TERMINAL_PROMPT='0');res=subprocess.run(['git','push','origin','research/round-004-baseline-training'],cwd=ROOT,env=env,text=True,capture_output=True,timeout=180);print('push exit',res.returncode,res.stdout,res.stderr,flush=True)
(RUN/'delivery-local.json').write_text(json.dumps({'result_commit_sha':sha,'push_exit_code':res.returncode,'push_output':res.stdout+res.stderr},indent=2)+'\n')

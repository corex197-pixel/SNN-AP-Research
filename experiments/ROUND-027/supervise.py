"""One phase per invocation, cumulative resource limits with persistent ledger."""
import os,sys,time,json,subprocess,resource,signal
from pathlib import Path
root=Path(__file__).resolve().parents[2];os.chdir(root);E=root/'experiments/EXP-029';phase=sys.argv[1];assert phase in ('preflight','train','endpoint_admission','evaluate');p=E/'resource_ledger.json';ledger=json.loads(p.read_text()) if p.exists() else {'gpu_process_wall_seconds':0.,'cpu_accounted_seconds':120.,'cpu_initial_reserve_seconds':120.,'stages':[]};assert ledger['gpu_process_wall_seconds']<1740 and ledger['cpu_accounted_seconds']<5340
rundir=E/'runs'/f'{phase}-{len(ledger["stages"])+1:03d}';rundir.mkdir(parents=True);start=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN);env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',ROUND027_STAGE_ID=rundir.name);env['ROUND027_REMAIN_WALL']=str(1800-ledger['gpu_process_wall_seconds']);env['ROUND027_REMAIN_CPU']=str(5400-ledger['cpu_accounted_seconds']);log=(rundir/'run.log').open('w');child=subprocess.Popen([str(root/'.local/round014-venv/bin/python'),'-u',str(root/f'experiments/ROUND-027/{phase}.py')],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True);term=None
while child.poll() is None:
 elapsed=time.monotonic()-start
 if elapsed>=1800-ledger['gpu_process_wall_seconds']-45:
  term='wall_budget_reserve';os.killpg(child.pid,signal.SIGTERM);break
 try:
  import psutil
  proc=psutil.Process(child.pid);live=[proc]+proc.children(recursive=True);cpu=sum(t.user+t.system for t in [q.cpu_times() for q in live if q.is_running()])
  if cpu>=5400-ledger['cpu_accounted_seconds']-60:term='cpu_budget_reserve';os.killpg(child.pid,signal.SIGTERM);break
 except Exception:pass
 time.sleep(.5)
try:code=child.wait(timeout=15)
except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
log.close();after=resource.getrusage(resource.RUSAGE_CHILDREN);wall=time.monotonic()-start;cpu=after.ru_utime-before.ru_utime+after.ru_stime-before.ru_stime;ledger['gpu_process_wall_seconds']+=wall;ledger['cpu_accounted_seconds']+=cpu;ledger['stages'].append({'stage':phase,'run_id':rundir.name,'exit_code':code,'wall_seconds':wall,'cpu_process_tree_wait4_seconds':cpu,'termination':term});tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(ledger,indent=2)+'\n');os.replace(tmp,p);print(json.dumps(ledger['stages'][-1]),flush=True);sys.exit(0 if code==0 else 1)

"""Stop this run at endpoint preflight so a missing load-before/after check can be repaired."""
import time,json,os,signal,sys
from pathlib import Path
p=Path('experiments/EXP-028/status.json');pid=int(sys.argv[1]);deadline=time.monotonic()+1650
while time.monotonic()<deadline:
 try:
  s=json.loads(p.read_text())
  if s.get('phase')=='endpoint_preflight':
   os.kill(pid,signal.SIGTERM);print(json.dumps({'action':'SIGTERM_owned_process_before_endpoint_lock','pid':pid,'status_snapshot':s,'reason':'endpoint check covers only repeated post-load forward; repair before evaluation'}),flush=True);break
  if s.get('status')!='RUNNING':print('process already ended',flush=True);break
 except (FileNotFoundError,json.JSONDecodeError):pass
 except ProcessLookupError:print('process already gone',flush=True);break
 time.sleep(.2)

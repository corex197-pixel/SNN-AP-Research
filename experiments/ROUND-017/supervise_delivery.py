"""Watch the authorized detached continuation and deliver its eventual results."""
import os,sys,time,json,subprocess,datetime
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'experiments/EXP-018';pid=int(sys.argv[1])
def alive():
 try:
  p=Path(f'/proc/{pid}/stat').read_text().split();return p[2]!='Z'
 except (FileNotFoundError,ProcessLookupError):return False
while alive():time.sleep(15)
time.sleep(3)
p=O/'status.json'
if not p.exists():
 status={'status':'PARTIAL','run_id':'EXP-018-CONTINUE-001','code_sha':'UNKNOWN','completed':{'U0':20,'U2':20},'error':'continuation exited before status creation'}
else:
 status=json.loads(p.read_text())
 if status['status']=='RUNNING':status.update(status='PARTIAL',error='continuation process exited without final status; see run.log')
p.write_text(json.dumps(status,indent=2)+'\n')
try:
 subprocess.run(['python','experiments/ROUND-017/finalize.py'],cwd=R,check=True,timeout=1200)
 subprocess.run(['git','add','experiments/EXP-018','experiments/EXP-019','research/RESEARCH_STATE.md','research/UNISPARSE_BASELINE.md','research/EXPERIMENT_REGISTRY.csv'],cwd=R,check=True,timeout=300)
 q=subprocess.run(['git','diff','--cached','--quiet'],cwd=R)
 if q.returncode==1:subprocess.run(['git','commit','-m','results(EXP-018,019): deliver staged continuation and cost'],cwd=R,check=True,timeout=300)
 subprocess.run(['git','push'],cwd=R,check=True,timeout=600)
except BaseException as e:
 print('delivery failed:',repr(e),flush=True);raise

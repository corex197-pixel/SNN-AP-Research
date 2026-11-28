"""Persistently deliver one finished experiment, including partial failures."""
import sys,time,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[2];E=R/'experiments/EXP-024';pid=int(sys.argv[1])
def alive():
 try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
 except (FileNotFoundError,ProcessLookupError):return False
while alive():time.sleep(10)
time.sleep(3);p=E/'status.json';s=json.loads(p.read_text()) if p.exists() else {'status':'PARTIAL','error':'missing status'}
if s.get('status')=='RUNNING':s.update(status='PARTIAL',error='experiment exited without terminal status; inspect run.log');p.write_text(json.dumps(s,indent=2)+'\n')
try:
 subprocess.run([str(R/'.local/round014-venv/bin/python'),'experiments/ROUND-022/finalize.py'],cwd=R,check=True,timeout=600)
 subprocess.run(['git','add','experiments/EXP-024','research/RESEARCH_STATE.md','research/EXPERIMENT_REGISTRY.csv'],cwd=R,check=True,timeout=300)
 if subprocess.run(['git','diff','--cached','--quiet'],cwd=R).returncode==1:subprocess.run(['git','commit','-m','results(EXP-024): deliver exact structured pruning pilot'],cwd=R,check=True,timeout=300)
 subprocess.run(['git','push'],cwd=R,check=True,timeout=600)
except BaseException as e:print('delivery failed:',repr(e),flush=True);raise

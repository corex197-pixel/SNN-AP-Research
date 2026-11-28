"""Fail-closed actual entry, using mandatory object-level tests and remote lock identity."""
import copy
NAMES=('K0','K1','K2','K3','M6')
TRAIN={'temporal_real','weight_nondegenerate','formula_independent','data_split','teacher_P0_protected','resource_benchmark','entry_negative_tests'}|{'candidate_'+x for x in NAMES}
EVAL={'training_complete','entry_negative_tests'}|{'endpoint_'+x for x in NAMES}|{'forward_reload_'+x for x in NAMES}
def verify(lock,kind,identity,config_sha,remote_sha,lock_commit,expected_samples):
 need=TRAIN if kind=='training' else EVAL;errors=[]
 if lock.get('kind')!=kind:errors.append('kind')
 if lock.get('identity')!=identity:errors.append('identity')
 if lock.get('config_sha256')!=config_sha:errors.append('config')
 if not lock_commit or remote_sha!=lock_commit:errors.append('remote_lock')
 if set(lock.get('required_tests',[]))!=need:errors.append('required_set')
 rec=lock.get('tests',[]);by={}
 for r in rec:
  k=r.get('id')
  if k in by:errors.append('duplicate')
  by[k]=r
 for k in need:
  r=by.get(k)
  if not r:errors.append('missing:'+k);continue
  if r.get('status')!='PASSED' or r.get('actual_exit_code')!=0:errors.append('failed:'+k)
  if r.get('actual_sample_count')!=expected_samples[k] or r.get('expected_sample_count')!=expected_samples[k]:errors.append('sample:'+k)
  if not r.get('assertion') or not isinstance(r.get('observed_result'),dict):errors.append('evidence:'+k)
  if k.startswith('candidate_') and (r.get('target_group')!=k[10:] or r.get('optimizer_names')!=['fc2.bias','fc2.weight']):errors.append('scope:'+k)
  if (k.startswith('endpoint_') or k.startswith('forward_reload_')) and r.get('target_group')!=k.split('_')[-1]:errors.append('target:'+k)
 if set(lock.get('passed_tests',[]))!={r['id'] for r in rec if r.get('status')=='PASSED' and r.get('actual_exit_code')==0}:errors.append('passed_set')
 return errors

def stage_entry(lock,kind,identity,config_sha,remote_sha,lock_commit,expected_samples,loader_factory,action):
 errors=verify(lock,kind,identity,config_sha,remote_sha,lock_commit,expected_samples)
 if errors:raise RuntimeError('admission rejected: '+','.join(errors))
 return action(loader_factory())

def negatives(train,eval_lock,ids,eval_ids,cfg,commit,train_samples,eval_samples):
 out=[]
 def trial(label,kind,mut=None,identity=None,config=None,remote=None):
  lk=copy.deepcopy(train if kind=='training' else eval_lock);ident=copy.deepcopy(ids if kind=='training' else eval_ids);counts={'loader':0,'action':0}
  if mut:mut(lk)
  if identity:identity(ident)
  def loader():counts['loader']+=1;return None
  def action(x):counts['action']+=1
  err=None
  try:stage_entry(lk,kind,ident,config or cfg,remote if remote is not None else commit,commit,train_samples if kind=='training' else eval_samples,loader,action)
  except RuntimeError as e:err=str(e)
  out.append({'name':label,'status':'PASSED' if err and counts=={'loader':0,'action':0} else 'FAILED','actual_exit_code':0 if err and counts=={'loader':0,'action':0} else 1,'rejection':err,'loader_constructions':counts['loader'],'evaluation_or_training_calls':counts['action']})
 trial('missing_group_preflight','training',lambda l:l.update(tests=[x for x in l['tests'] if x['id']!='candidate_K3']))
 trial('teacher_or_P0_hash_changed','training',identity=lambda d:d.update(teacher='changed'))
 trial('optimizer_scope_wrong','training',lambda l:next(x for x in l['tests'] if x['id']=='candidate_K0').update(optimizer_names=['wrong']))
 trial('kd_configuration_changed','training',config='changed')
 trial('training_remote_not_synced','training',remote='different')
 trial('missing_endpoint_identity','evaluation',lambda l:l.update(tests=[x for x in l['tests'] if x['id']!='endpoint_M6']))
 trial('missing_forward_equivalence','evaluation',lambda l:l.update(tests=[x for x in l['tests'] if x['id']!='forward_reload_M6']))
 trial('endpoint_hash_changed','evaluation',identity=lambda d:d.update(M6='changed'))
 trial('training_incomplete','evaluation',lambda l:next(x for x in l['tests'] if x['id']=='training_complete').update(status='FAILED',actual_exit_code=1))
 trial('evaluation_remote_not_synced','evaluation',remote='different')
 return out

"""Fail-closed validation for separate training and endpoint admission locks."""
import copy
TRAIN={'identity_and_structure','data_and_seed','learning_rate_schedule','candidate_F0_initial','candidate_F1_initial','F0_real_amp_update','F1_real_amp_update','F0_optimizer_scope','F1_optimizer_scope','F0_frozen_protection','F1_gradient_update','both_thresholds_fixed','checkpoint_resume','fp32_eval_reset_tail'}
EVAL={'F0_endpoint_identity','F1_endpoint_identity','F0_endpoint_integrity','F1_endpoint_integrity','paired_epoch_and_sample_protocol','fixed_reference_identity'}
def validate(lock,required,identity,config_sha,expectations):
 errors=[]
 if lock.get('kind')!=expectations['kind']:errors.append('wrong lock kind')
 if lock.get('identities')!=identity:errors.append('identity mismatch')
 if lock.get('config_sha256')!=config_sha:errors.append('configuration changed')
 if set(lock.get('required_tests',[]))!=required:errors.append('required coverage mismatch')
 rec=lock.get('tests',[]);by={}
 for r in rec:
  if r.get('test_id') in by:errors.append('duplicate test')
  by[r.get('test_id')]=r
 for name in required:
  r=by.get(name)
  if not r:errors.append('missing '+name);continue
  spec=expectations['tests'][name]
  for k,v in spec.items():
   if r.get(k)!=v:errors.append(name+' '+k+' mismatch')
  if r.get('status')!='PASSED' or r.get('actual_exit_code')!=0:errors.append(name+' failed')
  if not r.get('assertion') or not isinstance(r.get('observed_result'),dict):errors.append(name+' evidence missing')
 if set(lock.get('passed_tests',[]))!={r['test_id'] for r in rec if r.get('status')=='PASSED' and r.get('actual_exit_code')==0}:errors.append('passed set mismatch')
 return errors

def negatives(lock,required,identity,config_sha,expected):
 out=[]
 def run(name,mut,ids=None,conf=None):
  ll=copy.deepcopy(lock);ii=copy.deepcopy(identity);cc=config_sha;mut(ll)
  if ids:ids(ii)
  if conf:cc='changed'
  err=validate(ll,required,ii,cc,expected);out.append({'test_id':name,'status':'PASSED' if err else 'FAILED','actual_exit_code':0 if err else 1,'observed_rejection_reasons':err})
 run('missing_candidate',lambda l:l.update(tests=[x for x in l['tests'] if x['test_id']!='candidate_F1_initial']))
 run('identity_changed',lambda l:None,lambda i:i.update({'parent_sha256':'changed'}))
 run('wrong_optimizer_scope',lambda l:next(x for x in l['tests'] if x['test_id']=='F0_optimizer_scope').update(target_group='F1'))
 run('insufficient_samples',lambda l:next(x for x in l['tests'] if x['test_id']=='F1_real_amp_update').update(actual_sample_count=1))
 run('failed_assertion',lambda l:next(x for x in l['tests'] if x['test_id']=='F0_real_amp_update').update(status='FAILED',actual_exit_code=1))
 run('config_changed',lambda l:None,conf=True)
 return out

"""Candidate-specific, fail-closed validation used by the evaluation entry."""
import copy

def validate(lock, expected, current_hashes):
 errors=[]
 if lock.get('model_hashes')!=expected['model_hashes']:errors.append('locked model identity mismatch')
 if current_hashes!=expected['model_hashes']:errors.append('current model identity mismatch')
 if lock.get('historical_lock_commit')!=expected['historical_lock_commit']:errors.append('historical lock mismatch')
 if lock.get('data_hashes')!=expected['data_hashes']:errors.append('data identity mismatch')
 if set(lock.get('required_tests',[]))!=set(expected['requirements']):errors.append('required coverage mismatch')
 records=lock.get('tests',[]);by={}
 for x in records:
  key=x.get('test_id')
  if key in by:errors.append('duplicate test '+str(key))
  by[key]=x
 for key,spec in expected['requirements'].items():
  x=by.get(key)
  if x is None:errors.append('missing '+key);continue
  for field,value in spec.items():
   if x.get(field)!=value:errors.append(key+' mismatch '+field)
  if x.get('status')!='PASSED' or x.get('actual_exit_code')!=0:errors.append(key+' failed or abnormal exit')
  if not x.get('assertion') or not isinstance(x.get('observed_result'),dict):errors.append(key+' missing evidence')
 if set(lock.get('passed_tests',[]))!={x['test_id'] for x in records if x.get('status')=='PASSED' and x.get('actual_exit_code')==0}:errors.append('passed set differs from records')
 return errors

def negative_tests(lock,expected):
 out=[]
 def trial(name,edit,hash_edit=None):
  z=copy.deepcopy(lock);h=copy.deepcopy(expected['model_hashes']);edit(z)
  if hash_edit:hash_edit(h)
  errors=validate(z,expected,h);out.append({'test_id':name,'expected':'reject','observed_errors':errors,'status':'PASSED' if errors else 'FAILED','actual_exit_code':0 if errors else 1})
 trial('missing_50_candidate',lambda z:z.update(tests=[x for x in z['tests'] if x['test_id']!='pair_spikes_50']))
 trial('missing_75_candidate',lambda z:z.update(tests=[x for x in z['tests'] if x['test_id']!='pair_spikes_75']))
 trial('generic_only',lambda z:z.update(tests=[x for x in z['tests'] if x['test_id'].startswith('generic_')]))
 trial('candidate_model_hash_mismatch',lambda z:z['tests'][next(i for i,x in enumerate(z['tests']) if x['test_id']=='pair_spikes_50')].update(target_model_sha256='bad'))
 trial('sample_coverage_insufficient',lambda z:z['tests'][next(i for i,x in enumerate(z['tests']) if x['test_id']=='pair_spikes_50')].update(actual_sample_count=511))
 trial('failed_assertion_or_exit',lambda z:z['tests'][next(i for i,x in enumerate(z['tests']) if x['test_id']=='pair_spikes_75')].update(status='FAILED',actual_exit_code=1))
 trial('candidate_changed_after_lock',lambda z:None,lambda h:h.update({'50-R3':'changed'}))
 return out

"""Hand-computable R127-129 test; independent of UniSparse and GPU."""
import numpy as np

def check():
 # Two timesteps, one sample, three hidden neurons, two classes.
 A=np.array([[2.,-1.,.5],[1.,3.,-.5]],dtype=np.float64)
 b=np.array([.25,-.75],dtype=np.float64)
 spikes=np.array([[[1.,0.,1.]],[[0.,1.,1.]]],dtype=np.float64)
 expected_teacher=np.array([[[2.75,-.25]],[[-.25,1.75]]],dtype=np.float64)
 expected_p0=np.array([[[2.75,-.25]],[[.75,-1.25]]],dtype=np.float64)
 expected_deleted=np.array([[[0.,0.]],[[-1.,3.]]],dtype=np.float64)
 expected_margin=np.array([[0.],[-4.]],dtype=np.float64)
 teacher=spikes@A.T+b;pruned=spikes[:,:,[0,2]]@A[:,[0,2]].T+b;deleted=spikes[:,:,[1]]@A[:,[1]].T
 margin_diff=(teacher[:,:,0]-teacher[:,:,1])-(pruned[:,:,0]-pruned[:,:,1]);via_deleted=deleted[:,:,0]-deleted[:,:,1]
 errs={'R127_teacher':float(np.max(np.abs(teacher-expected_teacher))),'R128_pruned':float(np.max(np.abs(pruned-expected_p0))),'R128_deleted':float(np.max(np.abs(deleted-expected_deleted))),'R128_difference':float(np.max(np.abs(teacher-pruned-deleted))),'R129_expected':float(np.max(np.abs(margin_diff-expected_margin))),'R129_deleted':float(np.max(np.abs(margin_diff-via_deleted)))}
 return {'inputs':{'A':A.tolist(),'b':b.tolist(),'spikes':spikes.tolist(),'retained':[0,2],'deleted':[1],'true_class':0,'teacher_competitor':1},'expected':{'teacher':expected_teacher.tolist(),'pruned':expected_p0.tolist(),'deleted_contribution':expected_deleted.tolist(),'margin_difference':expected_margin.tolist()},'actual':{'teacher':teacher.tolist(),'pruned':pruned.tolist(),'deleted_contribution':deleted.tolist(),'margin_difference':margin_diff.tolist()},'absolute_errors':errs,'tolerance':1e-10,'status':'PASSED' if max(errs.values())<=1e-10 else 'FAILED'}

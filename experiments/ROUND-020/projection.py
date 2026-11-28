"""R86-R89 half-space projection, float64, without arbitrary epsilons."""
import torch

def project(t,g):
 t=t.detach().double();g=g.detach().double();n=torch.linalg.vector_norm(t)
 if not bool(torch.isfinite(t).all() and torch.isfinite(g).all()):raise ValueError('nonfinite input')
 if n==0:return g.clone()
 u=t/n;return g-torch.minimum(torch.dot(u,g),torch.zeros((),dtype=torch.float64,device=g.device))*u

def normalized(v,g0):
 n0=torch.linalg.vector_norm(g0);n=torch.linalg.vector_norm(v)
 if n0==0:return torch.zeros_like(v)
 if not torch.isfinite(n) or n/n0<=1e-8:raise ValueError('projection degenerate')
 return v*(n0/n)

def unit_tests():
 def t(x):return torch.tensor(x,dtype=torch.float64)
 a=t([1,0]);assert torch.equal(project(a,t([1,2])),t([1,2]));v=project(a,t([-2,3]));assert torch.equal(v,t([0,3]));assert torch.equal(project(t([0,0]),t([-2,3])),t([-2,3]));assert torch.equal(project(a,t([0,0])),t([0,0]));assert torch.equal(project(a,-a),t([0,0]))
 try:normalized(project(a,-a),a);raise AssertionError('expected degeneracy')
 except ValueError:pass
 tiny=t([1e-100,0]);assert torch.equal(project(tiny,t([-2,3])),t([0,3]))
 # Independent feasible competitors are no closer than the boundary projection.
 g=t([-2,3]);dist=torch.linalg.vector_norm(v-g)
 for x in [0,.1,1,10]:
  for y in [-4,0,2,3,7]:assert torch.linalg.vector_norm(t([x,y])-g)>=dist-1e-12
 assert torch.dot(a,normalized(v,t([1,1])))>=0
 return {'status':'PASSED','nonconflict':True,'orthogonal_boundary':True,'zero_task':True,'zero_activity':True,'antiparallel_degeneracy':True,'small_nonzero_norm':True,'minimum_distance_competitors':True,'positive_scale_preserves_sign':True}
if __name__=='__main__':print(unit_tests())

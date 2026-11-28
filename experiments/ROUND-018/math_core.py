"""ROUND-018 fixed formulas. No optimizer or model-state mutation."""
import torch

def coefficients(d,rho):
 d=d.detach().to(torch.float64);mean=d.mean()
 if not bool(torch.isfinite(d).all()) or bool((d<0).any()):raise ValueError('invalid fanout')
 return torch.ones_like(d) if mean==0 else (1-rho)+rho*d/mean

def match(g,reference):
 n=torch.linalg.vector_norm(g);n0=torch.linalg.vector_norm(reference)
 if n0==0:return torch.zeros_like(g)
 if not torch.isfinite(n) or n==0:raise ValueError('zero or nonfinite gradient')
 return g*(n0/n)

def tests():
 d=torch.tensor([0.,1.,7.,2.],dtype=torch.float64); a=coefficients(d,.25)
 assert torch.equal(coefficients(d,0),torch.ones_like(d))
 assert torch.equal(coefficients(torch.ones(4),.25),torch.ones(4,dtype=torch.float64))
 assert torch.equal(coefficients(torch.zeros(4),.25),torch.ones(4,dtype=torch.float64))
 assert abs(a.mean().item()-1)<1e-14 and bool(torch.isfinite(a).all()) and bool((a>=0).all())
 p=torch.randperm(len(d),generator=torch.Generator().manual_seed(20261128));assert torch.equal(a.sort().values,a[p].sort().values)
 x=torch.tensor([1.,2.,3.],dtype=torch.float64,requires_grad=True);r=torch.sigmoid(x);dd=torch.tensor([0.,1.,5.],dtype=torch.float64);d0=3.;aa=coefficients(dd,.25)
 g0=torch.autograd.grad(d0*r.sum(),x,retain_graph=True)[0];h=torch.autograd.grad((dd*r).sum(),x,retain_graph=True)[0];g1=torch.autograd.grad(d0*(aa*r).sum(),x)[0]
 assert torch.allclose(g1,.75*g0+.25*d0/dd.mean()*h,atol=1e-12,rtol=1e-12)
 assert abs(torch.linalg.vector_norm(match(g1,g0)).item()-torch.linalg.vector_norm(g0).item())<1e-12
 return {'status':'PASSED','rho0':True,'uniform':True,'zero_mean_fallback':True,'finite_nonnegative_mean_one':True,'permutation_multiset':True,'R78':True,'R79':True}
if __name__=='__main__':print(tests())

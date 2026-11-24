"""Synthetic mechanism checks; no network performance claims."""
import csv, itertools, json, sys, time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Activity-Pruning-SNN-main'))
from models.submodules.layers import ATLIF, Surrogate
OUT=ROOT/'experiments/EXP-001/runs/EXP-001-RUN-001'

def simulate(x,tau,theta,u0,ge=True):
    u=u0.clone(); X=torch.zeros_like(u); O=X.clone(); leak=X.clone(); reset=X.clone(); ms=[]; us=[]; ss=[]
    for z in x:
        leak=leak+(1-tau)*u; m=tau*u+z
        s=(m>=theta if ge else m>theta).to(x.dtype)
        X=X+z; O=O+theta*s; reset=reset+s*(m-theta); u=(1-s)*m
        ms.append(m.clone()); us.append(u.clone()); ss.append(s.clone())
    residual=u-u0; err=X-O-leak-reset-residual
    norm=1+X.abs()+O.abs()+leak.abs()+reset.abs()+residual.abs()
    return dict(X=X,O=O,leak=leak,reset=reset,residual=residual,error=err.abs()/norm,m=torch.stack(ms),u=torch.stack(us),s=torch.stack(ss))

def main():
    start=time.time(); OUT.mkdir(parents=True,exist_ok=False)
    rows=[]; errors=[]; r2errors=[]; real_checks=0; real_max=0.; real_r1=[]; failures=[]
    for T,tau,theta,kind,uinit,seed,shape,ge in itertools.product([1,4,8,16,64],[0.,.5,.9,1.],[.5,1.,2.],['constant','impulse','nonnegative','signed'],[0.,-.35],[0,1,2],[(1,),(2,3)], [True,False]):
        g=torch.Generator().manual_seed(seed); sz=(T,)+shape
        if kind=='constant': x=torch.full(sz,.6,dtype=torch.float64)
        elif kind=='impulse':
            x=torch.zeros(sz,dtype=torch.float64); x[0]=2.5
        elif kind=='nonnegative': x=torch.rand(sz,generator=g,dtype=torch.float64)*2
        else: x=torch.randn(sz,generator=g,dtype=torch.float64)
        u0=torch.full(shape,uinit,dtype=torch.float64)
        a=simulate(x,tau,theta,u0,ge); b=simulate(x,tau,theta*1.5,u0,ge)
        rr=(b['O']-a['O'])+(b['leak']-a['leak'])+(b['reset']-a['reset'])+(b['residual']-a['residual'])
        denom=1+(b['O']-a['O']).abs()+(b['leak']-a['leak']).abs()+(b['reset']-a['reset']).abs()+(b['residual']-a['residual']).abs()
        e=a['error'].flatten().tolist(); e2=(rr.abs()/denom).flatten().tolist(); errors.extend(e);r2errors.extend(e2)
        rows.append([T,tau,theta,kind,uinit,seed,str(shape),'>=' if ge else '>',max(e),sum(e)/len(e),max(e2)])
        if max(e)>1e-10 or max(e2)>1e-10: failures.append(f'R1/R2 {rows[-1]}')
        # Baseline has zero initial state, >= boundary and internally loops over T.
        if ge and uinit==0 and seed==0:
            xf=x.float(); ref=simulate(xf,tau,theta,torch.zeros(shape),True)
            module=ATLIF(T=T,thresh=theta,tau=tau,sp=0).eval(); seen=[]; original=Surrogate.forward
            def capture(ctx,input,thre,sp):
                seen.append(input.detach().clone()); return original(ctx,input,thre,sp)
            Surrogate.forward=staticmethod(capture)
            try:
                with torch.no_grad(): out=module(xf); out2=module(xf)
            finally: Surrogate.forward=staticmethod(original)
            m=torch.stack(seen[:T]); m2=torch.stack(seen[T:]); s=(out/theta); u=(1-s)*m
            dif=max((out-ref['s']*theta).abs().max().item(),(m-ref['m']).abs().max().item(),(u-ref['u']).abs().max().item(),(out-out2).abs().max().item(),(m-m2).abs().max().item())
            real_max=max(real_max,dif); real_checks+=1
            prev=torch.cat([torch.zeros_like(u[:1]),u[:-1]])
            X=xf.sum(0); O=out.sum(0); L=((1-tau)*prev).sum(0); D=(s*(m-theta)).sum(0); R=u[-1]
            err=(X-O-L-D-R).abs()/(1+X.abs()+O.abs()+L.abs()+D.abs()+R.abs()); real_r1.extend(err.flatten().tolist())
            if dif>1e-5 or err.max().item()>1e-5: failures.append(f'baseline mismatch {rows[-1]} diff={dif}')
    # Exact threshold edge distinguishes >= from >; both must preserve the identity.
    edge=torch.tensor([[1.]],dtype=torch.float64)
    assert simulate(edge,1.,1.,torch.zeros(1,dtype=torch.float64),True)['s'].item()==1
    assert simulate(edge,1.,1.,torch.zeros(1,dtype=torch.float64),False)['s'].item()==0
    # Synthetic constant input; all values, not just selected successes, are preserved.
    curve=[]
    for theta in np.linspace(.25,3.,56):
        x=torch.full((64,1),.6,dtype=torch.float64); a=simulate(x,.9,float(theta),torch.zeros(1,dtype=torch.float64))
        curve.append([theta,a['s'].mean().item(),a['O'].item(),a['leak'].item(),a['reset'].item(),a['residual'].item()])
    def writecsv(path,header,data):
        with path.open('w',newline='') as f: w=csv.writer(f);w.writerow(header);w.writerows(data)
    writecsv(OUT/'metrics.csv',['T','tau','theta','input','u0','seed','shape','boundary','r1_max','r1_mean','r2_max'],rows)
    writecsv(OUT/'curve.csv',['theta','FR','O','D_leak','D_reset','D_res'],curve)
    figures=ROOT/'experiments/EXP-001/figures'; figures.mkdir(exist_ok=True)
    z=np.array(curve)
    for name,cols,ylabel in [('threshold_fr',[1],'FR'),('threshold_output',[2],'累计输出'),('decomposition',[3,4,5],'动力学项')]:
        plt.figure(); [plt.plot(z[:,0],z[:,i],label=['theta','FR','O','D_leak','D_reset','D_res'][i]) for i in cols];plt.xlabel('theta');plt.ylabel(ylabel if False else name);plt.legend();plt.tight_layout();plt.savefig(figures/(name+'.png'));plt.close()
    plt.figure(); plt.hist(np.log10(np.maximum(errors,1e-20)),bins=40);plt.xlabel('log10 R1 normalized error (zeros clipped at 1e-20)');plt.ylabel('neuron cases');plt.tight_layout();plt.savefig(figures/'identity_error.png');plt.close()
    def summary(v,tol): return {'count':len(v),'max':max(v),'mean':float(np.mean(v)),'failure_fraction':float(np.mean(np.array(v)>tol))}
    result={'run_id':'EXP-001-RUN-001','synthetic_only':True,'cases':len(rows),'r1_float64':summary(errors,1e-10),'r2_float64':summary(r2errors,1e-10),'baseline_cases':real_checks,'baseline_max_trajectory_difference':real_max,'baseline_r1_float32':summary(real_r1,1e-5),'exact_boundary_ge_spikes':True,'exact_boundary_gt_spikes':False,'failures':failures,'wall_seconds':time.time()-start}
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2));
    if failures: sys.exit(1)
if __name__=='__main__': main()

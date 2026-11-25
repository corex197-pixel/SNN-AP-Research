"""Synthetic mechanism checks; no network performance claims."""
import csv, itertools, json, sys, time, argparse, subprocess, platform, contextlib, traceback, re
from datetime import datetime, timezone
import yaml
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Activity-Pruning-SNN-main'))
from models.submodules.layers import ATLIF, Surrogate
OUT=None

@contextlib.contextmanager
def capture_membrane():
    descriptor=Surrogate.__dict__['forward']
    original=Surrogate.forward
    seen=[]
    def capture(ctx, input, thre, sp):
        seen.append(input.detach().clone())
        return original(ctx,input,thre,sp)
    Surrogate.forward=staticmethod(capture)
    try:
        yield seen
    finally:
        Surrogate.forward=descriptor

def control(x,T,tau,theta,sp=0.002):
    # Separate identical modules; eval does not suppress baseline update_value accumulation.
    modules=[ATLIF(T=T,thresh=theta,tau=tau,sp=sp).eval() for _ in range(3)]
    with torch.no_grad():
        o0=modules[0](x)
        with capture_membrane() as seen:
            o1=modules[1](x)
        o2=modules[2](x)
    outputs=[o0,o1,o2]
    states=[(m.update_value,m.r,m.s,m.act_value.item(),m.thresh.item()) for m in modules]
    diff=max((o0-o1).abs().max().item(),(o0-o2).abs().max().item())
    pulse=all(torch.equal(o0/theta,o/theta) for o in outputs[1:])
    same=all(states[0]==st for st in states[1:])
    return diff,pulse,same,states[0][0],len(seen)


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

def main(cfg):
    start=time.time()
    rows=[]; errors=[]; r2errors=[]; real_checks=0; real_max=0.; real_r1=[]; real_r2=[]; failures=[]; real_rows=[]; control_rows=[]
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
        rows.append([T,tau,theta,kind,uinit,seed,str(shape),'>=' if ge else '>',max(e),sum(e)/len(e),max(e2),sum(e2)/len(e2),len(e),str(x.dtype),'PASS' if max(e)<=1e-10 and max(e2)<=1e-10 else 'FAILED'])
        if max(e)>1e-10 or max(e2)>1e-10: failures.append(f'R1/R2 {rows[-1]}')
        # Baseline has zero initial state, >= boundary and internally loops over T.
        if ge and uinit==0 and seed==0:
            xf=x.float(); ctrl=control(xf,T,tau,theta); control_rows.append([T,tau,theta,kind,seed,str(shape),xf.numel(),str(xf.dtype),*ctrl]);
            if ctrl[0]>1e-5 or not ctrl[1] or not ctrl[2] or ctrl[4]!=T: failures.append(f'control failure {control_rows[-1]}')
            ref=simulate(xf,tau,theta,torch.zeros(shape),True)
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
            # Independent actual second-threshold forward, with captured pre-reset states.
            module2=ATLIF(T=T,thresh=theta*1.5,tau=tau,sp=0).eval(); seen2=[]
            def capture2(ctx,input,thre,sp):
                seen2.append(input.detach().clone()); return original(ctx,input,thre,sp)
            Surrogate.forward=staticmethod(capture2)
            try:
                with torch.no_grad(): out_hi=module2(xf)
            finally: Surrogate.forward=staticmethod(original)
            mh=torch.stack(seen2); sh=out_hi/(theta*1.5); uh=(1-sh)*mh
            ph=torch.cat([torch.zeros_like(uh[:1]),uh[:-1]])
            Oh=out_hi.sum(0); Lh=((1-tau)*ph).sum(0); Dh=(sh*(mh-theta*1.5)).sum(0); Rh=uh[-1]
            dO=Oh-O; dL=Lh-L; dD=Dh-D; dR=Rh-R
            e2=(dO+dL+dD+dR).abs()/(1+dO.abs()+dL.abs()+dD.abs()+dR.abs());real_r2.extend(e2.flatten().tolist())
            real_rows.append([T,tau,theta,kind,seed,str(shape),err.numel(),str(xf.dtype),err.max().item(),err.mean().item(),e2.max().item(),e2.mean().item(),dif,'PASS' if dif<=1e-5 and err.max().item()<=1e-5 and e2.max().item()<=1e-5 else 'FAILED'])
            if e2.max().item()>1e-5: failures.append(f'baseline R2 mismatch {rows[-1]}')
            if dif>1e-5 or err.max().item()>1e-5: failures.append(f'baseline mismatch {rows[-1]} diff={dif}')
    # Exact threshold edge distinguishes >= from >; both must preserve the identity.
    edge=torch.tensor([[1.]],dtype=torch.float64)
    assert simulate(edge,1.,1.,torch.zeros(1,dtype=torch.float64),True)['s'].item()==1
    assert simulate(edge,1.,1.,torch.zeros(1,dtype=torch.float64),False)['s'].item()==0
    # Explicit actual boundary, fully silent and firing every step, including 4-D states.
    for T in [1,4,8,16,64]:
        for kind,value in [('boundary',1.),('silent',-1.),('every_step',2.)]:
            x=torch.full((T,2,2,2,2),value,dtype=torch.float32)
            c=control(x,T,.9,1.); control_rows.append([T,.9,1.,kind,0,str(tuple(x.shape[1:])),x.numel(),str(x.dtype),*c])
            if c[0]>1e-5 or not c[1] or not c[2] or c[4]!=T: failures.append(f'edge control failure {control_rows[-1]}')
    # Restore even on an exception.
    descriptor=Surrogate.__dict__['forward']
    try:
        with capture_membrane(): raise RuntimeError('intentional restoration test')
    except RuntimeError: pass
    exception_restored=Surrogate.__dict__['forward'] is descriptor
    if not exception_restored: failures.append('capture exception restoration failed')
    # Synthetic constant input; all values, not just selected successes, are preserved.
    curve=[]
    for theta in np.linspace(.25,3.,56):
        x=torch.full((64,1),.6,dtype=torch.float64); a=simulate(x,.9,float(theta),torch.zeros(1,dtype=torch.float64))
        curve.append([theta,a['s'].mean().item(),a['O'].item(),a['leak'].item(),a['reset'].item(),a['residual'].item()])
    def writecsv(path,header,data):
        with path.open('w',newline='') as f: w=csv.writer(f);w.writerow(header);w.writerows(data)
    writecsv(OUT/'metrics.csv',['T','tau','theta','input','u0','seed','shape','boundary','r1_max','r1_mean','r2_max','r2_mean','elements','dtype','status'],rows)
    writecsv(OUT/'baseline_metrics.csv',['T','tau','theta','input','seed','shape','elements','dtype','r1_max','r1_mean','r2_max','r2_mean','trajectory_max','status'],real_rows)
    writecsv(OUT/'control.csv',['T','tau','theta','input','seed','shape','input_elements','dtype','max_output_difference','spikes_equal','states_equal','update_value','captured_steps'],control_rows)
    writecsv(OUT/'curve.csv',['theta','FR','O','D_leak','D_reset','D_res'],curve)
    figures=OUT/'figures'; figures.mkdir(exist_ok=True)
    z=np.array(curve)
    for name,cols,ylabel in [('threshold_fr',[1],'FR'),('threshold_output',[2],'累计输出'),('decomposition',[3,4,5],'动力学项')]:
        plt.figure(); [plt.plot(z[:,0],z[:,i],label=['theta','FR','O','D_leak','D_reset','D_res'][i]) for i in cols];plt.xlabel('theta');plt.ylabel(ylabel if False else name);plt.legend();plt.tight_layout();plt.savefig(figures/(name+'.png'));plt.close()
    plt.figure(); plt.hist(np.log10(np.maximum(errors,1e-20)),bins=40);plt.xlabel('log10 R1 normalized error (zeros clipped at 1e-20)');plt.ylabel('neuron cases');plt.tight_layout();plt.savefig(figures/'identity_error.png');plt.close()
    def summary(v,tol): return {'count':len(v),'max':max(v),'mean':float(np.mean(v)),'failure_fraction':float(np.mean(np.array(v)>tol))}
    result={'run_id':OUT.name,'synthetic_only':True,'cases':len(rows),'r1_float64':summary(errors,1e-10),'r2_float64':summary(r2errors,1e-10),'baseline_cases':real_checks,'baseline_max_trajectory_difference':real_max,'baseline_r1_float32':summary(real_r1,1e-5),'baseline_r2_float32':summary(real_r2,1e-5),'exact_boundary_ge_spikes':True,'exact_boundary_gt_spikes':False,'control_cases':len(control_rows),'control_max_output_difference':max(r[8] for r in control_rows),'control_all_spikes_equal':all(r[9] for r in control_rows),'control_all_states_equal':all(r[10] for r in control_rows),'capture_exception_restored':exception_restored,'failures':failures,'wall_seconds':time.time()-start}
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2));
    writecsv(OUT/'summary.csv',['metric','count','max','mean','failure_fraction'],[[name, *[result[name][k] for k in ('count','max','mean','failure_fraction')]] for name in ['r1_float64','r2_float64','baseline_r1_float32','baseline_r2_float32']])
    if failures: sys.exit(1)
if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--run-id',required=True); parser.add_argument('--config',required=True); args=parser.parse_args()
    if not re.fullmatch(r'EXP-001-RUN-[0-9]{3,}',args.run_id): parser.error('invalid run id')
    cfg=yaml.safe_load(Path(args.config).read_text())
    expected={'round':'ROUND-001','experiment':'EXP-001','theory_version':'v0.1.0','T':[1,4,8,16,64],'tau':[0,0.5,0.9,1],'theta':[0.5,1,2],'seed':[0,1,2],'float64_tolerance':1e-10,'float32_tolerance':1e-5}
    if cfg != expected: parser.error('config differs from fixed protocol; refusing to misreport effective config')
    OUT=ROOT/'experiments/EXP-001/runs'/args.run_id
    OUT.mkdir(parents=True,exist_ok=False)
    (OUT/'effective_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True)
    if dirty: (OUT/'tracked_changes.patch').write_text(subprocess.check_output(['git','diff','HEAD','--binary'],cwd=ROOT,text=True))
    meta={'command':sys.argv,'code_sha':sha,'tracked_changes_present':bool(dirty),'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'numpy':np.__version__,'start_time_utc':datetime.now(timezone.utc).isoformat(),'status':'RUNNING'}
    cpu=time.process_time(); wall=time.perf_counter()
    with (OUT/'run.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            main(cfg); meta['status']='COMPLETED'
        except BaseException:
            meta['status']='FAILED'; traceback.print_exc(); raise
        finally:
            meta.update(end_time_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-wall,process_cpu_seconds=time.process_time()-cpu)
            (OUT/'run_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))

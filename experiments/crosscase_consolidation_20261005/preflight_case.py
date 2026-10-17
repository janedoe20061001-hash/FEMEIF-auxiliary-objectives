"""Numerical/scientific preflight only; no fitted trial optimizations."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[k]='2'
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
from scipy.integrate import quad

case=Path(sys.argv[1]).resolve();sys.path.insert(0,str(case));os.chdir(case)
from trend_kernel import kernel,score
import legacy_objectives as legacy
from length_metrics import GlobalObjectives as OldLength
from global_metrics import GlobalObjectives,GROUPS,LENGTH,TREND,CUMULATIVE
cfg=json.loads((case/'config.json').read_text())
if case.name=='case1':
    f=np.geomspace(100,1e6,180);z=legacy.impedance(f,legacy.TARGET)[0]
    imp=legacy.impedance;lo=legacy.LOWER;hi=legacy.UPPER
else:
    from case2 import load_target,impedance,LOWER,UPPER
    f,z=load_target(cfg['target_data']);imp=impedance;lo=LOWER;hi=UPPER
old=OldLength(f,aux=(LENGTH,),target_z=z);u=old.u
x=np.random.default_rng(2000).uniform(lo,hi,(cfg['population'],len(lo)))
spectra=imp(f,10**x)
r=np.stack([20*np.log10(abs(spectra))-old.a,np.angle(spectra)-old.phase],axis=-1)/old.scale
checks=[];previous=None;chosen=None
for order in [4,8,16,32]:
    t=time.perf_counter();K=kernel(u,order);val=score(r,K)
    delta=None if previous is None else float(np.max(abs(val-previous)/np.maximum(val,1e-14)))
    checks.append(dict(order=order,seconds=time.perf_counter()-t,max_relative_change=delta,
                       min_eigenvalue=float(np.linalg.eigvalsh(K)[0])))
    print(json.dumps(dict(case=case.name,quadrature=checks[-1])),flush=True)
    np.testing.assert_allclose(score(np.ones((len(u),1)),K),1,rtol=1e-12)
    np.testing.assert_allclose(score((u-.5)[:,None],K)**2,1/24,rtol=1e-12)
    if order>=16 and delta<1e-6:
        chosen=order;break
    previous=val
if chosen is None:
    raise RuntimeError('T-01 did not meet predeclared relative convergence tolerance 1e-6')
np.save(case/'trend_kernel.npy',K)

# Independent adaptive direct integration on a deliberately nonuniform small grid.
us=np.array([0.,.07,.35,.6,1.]);rs=np.array([.3,-.2,.4,-.1,.25])
hs=np.diff(us);As=np.r_[0,np.cumsum(hs*(rs[:-1]+rs[1:])/2)]
def antiderivative(t):
    i=min(np.searchsorted(us,t,side='right')-1,len(hs)-1)
    s=t-us[i]
    return As[i]+s*rs[i]+s*s*(rs[i+1]-rs[i])/(2*hs[i])
def integrand(b,a):
    if b==a:return float(np.interp(a,us,rs))**2
    return ((antiderivative(b)-antiderivative(a))/(b-a))**2
direct=0.
for i in range(len(hs)):
    for j in range(i,len(hs)):
        inner=lambda a:quad(lambda b:integrand(b,a),max(a,us[j]),us[j+1],epsabs=2e-11,epsrel=2e-10)[0]
        direct+=2*quad(inner,us[i],us[i+1],epsabs=2e-10,epsrel=2e-9)[0]
small=score(rs[:,None],kernel(us,32))**2
np.testing.assert_allclose(small,direct,rtol=1e-6,atol=1e-10)
reverse=score(rs[::-1,None],kernel(1-us[::-1],32))
np.testing.assert_allclose(np.sqrt(small),reverse,rtol=1e-12)
split=np.sort(np.r_[us,(us[:-1]+us[1:])/2])
sub=score(np.interp(split,us,rs)[:,None],kernel(split,32))
np.testing.assert_allclose(sub,np.sqrt(small),rtol=1e-6)

bench={}
for group in GROUPS:
    obj=GlobalObjectives(f,aux=group,target_z=z)
    values=obj.evaluate_spectra(spectra)
    np.testing.assert_array_equal(values[:,:2],old.evaluate_spectra(spectra)[:,:2])
    if len(group)==2:
        np.testing.assert_array_equal(values,legacy.Objectives(f,aux=group,target_z=z).evaluate_spectra(spectra))
    elif group==(LENGTH,):
        np.testing.assert_array_equal(values,old.evaluate_spectra(spectra))
    np.testing.assert_allclose(obj.evaluate_spectra(z),0,atol=1e-14)
    t=time.perf_counter()
    for _ in range(5):obj.evaluate_spectra(spectra)
    bench['+'.join(group) or 'baseline']=(time.perf_counter()-t)/5
record=dict(status='passed',case=case.name,no_trial_optimization=True,chosen_quadrature_order=chosen,
    quadrature_convergence=checks,tolerance=1e-6,independent_integral=direct,kernel_integral=float(small),
    exact_constant_and_linear=True,nonuniform_grid_and_reversal_and_subdivision=True,
    unchanged_controls_bitwise=True,unchanged_primary_bitwise=True,zero_loss=True,
    kernel_sha256=hashlib.sha256((case/'trend_kernel.npy').read_bytes()).hexdigest(),
    full_population_evaluation_seconds=bench,target=GlobalObjectives(f,target_z=z).target_summary())
(case/'preflight.json').write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(record,ensure_ascii=True),flush=True)

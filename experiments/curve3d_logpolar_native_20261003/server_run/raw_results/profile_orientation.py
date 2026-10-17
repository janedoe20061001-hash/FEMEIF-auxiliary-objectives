"""Calculation-only profiling; no optimization or reduced-budget fit."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,time,json
import numpy as np
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root/'case1'))
from objectives import Objectives,impedance
f=np.geomspace(100,1e6,180);obj=Objectives(f,aux=('orientation_logpolar',))
file=root/'case1/runs/orientation_logpolar/seed_000/initial_log_parameters.npy'
if file.exists():x=np.load(file)
else:
    from legacy_objectives import LOWER,UPPER
    x=np.random.default_rng(0).uniform(LOWER,UPPER,(200,18))
p=obj.path(impedance(f,10**x))
start=time.perf_counter();old=obj.orientation_loss(p);old_seconds=time.perf_counter()-start

def scalar(p,chunk_size=32):
    result=[];d0=obj.u[obj.j]-obj.u[obj.i];d0sq=d0*d0
    target=obj.directions
    for start in range(0,len(p),chunk_size):
        a=p[start:start+chunk_size]
        d1=a[:,obj.j,1]-a[:,obj.i,1];d2=a[:,obj.j,2]-a[:,obj.i,2]
        norm=np.sqrt(d0sq+d1*d1+d2*d2)
        e0=d0/norm-target[:,0];e1=d1/norm-target[:,1];e2=d2/norm-target[:,2]
        squared=e0*e0+e1*e1+e2*e2
        result.extend(np.sum(squared*obj.pair_weights,axis=-1)/obj.pair_weights.sum())
    return np.asarray(result)

start=time.perf_counter();new=scalar(p);new_seconds=time.perf_counter()-start
np.testing.assert_allclose(new,old,rtol=1e-14,atol=1e-14)
print(json.dumps(dict(old_seconds=old_seconds,new_seconds=new_seconds,speedup=old_seconds/new_seconds,
    max_abs_difference=float(abs(new-old).max()),bitwise_equal=bool(np.array_equal(new,old)),candidates=len(p))),flush=True)

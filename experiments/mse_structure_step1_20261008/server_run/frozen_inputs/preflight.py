from pathlib import Path
import sys,json,time
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE/'frozen'))
from model import Model
from adaptive import Adaptive,spectra
sys.path.insert(0,str(HERE/'toolkit'))
from refined_grid import improved_dense_grid
def main():
    start=time.perf_counter();m=Model('case1');r=m.record
    with np.load(HERE/'target_grid.npz') as d:base=d['frequency']
    a=Adaptive(r,base);params=[];labels=[]
    for seed in range(2300,2400):
        with np.load(HERE/'old300'/f'{seed}.npz') as d:
            for k in ['selected_index','balanced_index']:params.append(d['log_parameters'][int(d[k])]);labels.append(f'{seed}:{k}')
    rng=np.random.default_rng(2026100802)
    for i,x in enumerate(rng.uniform(m.lower,m.upper,(32,len(m.lower)))):params.append(x);labels.append(f'random:{i}')
    for label,x in [('target',np.log10(r['target_parameters'])),('lower',m.lower),('upper',m.upper)]:params.append(x);labels.append(label)
    x=np.asarray(params);np.testing.assert_allclose(spectra(m.f,10**x),m.network(m.f,10**x),rtol=1e-9,atol=1e-9)
    batch=a.score(x);np.testing.assert_allclose(a.score(x[:3]),batch[:3],rtol=1e-12,atol=1e-10)
    np.testing.assert_allclose(a.score_python(x),batch,rtol=1e-8,atol=1e-8)
    np.testing.assert_allclose(batch[-3],0,atol=1e-20)
    errors=[];checks=[]
    for i,(p,label) in enumerate(zip(x,labels)):
        refs=[a.reference(p,improved_dense_grid(r,10**np.atleast_2d(p),lev))[0] for lev in [2,4]]
        tolerance=np.maximum([.01,np.deg2rad(.5)**2],.005*np.maximum(refs[0],refs[1]));delta=np.maximum(abs(batch[i]-refs[0]),abs(batch[i]-refs[1]));ok=bool(np.all(delta<=tolerance))
        stable=bool(np.all(abs(refs[0]-refs[1])<=tolerance));checks.append(dict(label=label,adaptive=batch[i].tolist(),reference2=refs[0].tolist(),reference4=refs[1].tolist(),max_abs_delta=delta.tolist(),tolerance=tolerance.tolist(),passed=ok,reference_stable=stable))
        if not ok or not stable:errors.append(label)
    result=dict(passed=not errors,candidates=len(x),errors=errors,queries=a.queries,max_points=a.max_points,max_layers=a.max_layers,seconds=time.perf_counter()-start,checks=checks)
    (HERE/'PREFLIGHT.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k!='checks'}));assert not errors
if __name__=='__main__':main()

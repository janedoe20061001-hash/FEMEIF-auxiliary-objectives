"""Regression failures, independent integrals, exact truth and derivative checks."""
from core import *
from features import Features,rule
from checks import polygon,agrees
from concurrent.futures import ProcessPoolExecutor,as_completed
import time,traceback,platform
from importlib.metadata import version

def regression(record):
    st=time.perf_counter();d=target(record['target']);g=Features(d);lp=np.array(record['log_parameters'])
    names=('J0','D','H','A','ARX')
    train=g.one(lp,names);fine=g.fixed(lp,names,32,2)
    poly=polygon(g,lp,names,128);poly2=polygon(g,lp,names,256)
    global_train=g.one(lp,('G0',));global_fine=g.fixed(lp,('G0',),32,2);global_poly=polygon(g,lp,('G0',),256)
    global_ok=agrees(global_train,global_fine.mean(axis=0)) and agrees(global_fine,global_poly)
    ok=agrees(train,fine.mean(axis=0)) and agrees(fine,poly2) and agrees(poly,poly2) and global_ok
    result=dict(**record,passed=ok,train=train.tolist(),fine=fine.mean(axis=0).tolist(),polygon=poly2.mean(axis=0).tolist(),
        global_passed=global_ok,global_scores=[float(global_train[0]),float(global_fine.mean()),float(global_poly.mean())],train_fine_agrees=agrees(train,fine.mean(axis=0)),polygon_agrees=agrees(fine,poly2),polygon_converged=agrees(poly,poly2),seconds=time.perf_counter()-st)
    dump(OUT/'preflight_rows'/f"{record['target'].replace('/','_')}_{record['method']}_{record['seed']}.json",result)
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True);checks=[]
    # Integration matrix must reproduce exact polynomial prefix integrals.
    for n in (8,16,32,64):
        x,w,b=rule(n)
        for k in range(6):np.testing.assert_allclose(b@(x**k),x**(k+1)/(k+1),atol=1e-13,rtol=1e-12)
    records=json.loads((INPUTS/'regression.json').read_text())
    for ident in target_ids():
        d=target(ident);g=Features(d);p=d['circuit'].p;lp=np.log10(p)
        truth=g.one(lp,('J0','D','H','A','ARX'));assert max(truth)<1e-16,(ident,truth)
        xx=np.linspace(g.lo,g.hi,101);z,dz=g.c.response(10**xx);h=1e-5
        a,_=g.c.response(10**xx*np.exp(h));b,_=g.c.response(10**xx*np.exp(-h))
        mask=abs(dz)>1e-6
        # Central differences can be biased on narrow features; use sufficiently small h.
        h=1e-7;a,_=g.c.response(10**xx*np.exp(h));b,_=g.c.response(10**xx*np.exp(-h))
        np.testing.assert_allclose((a-b)[mask]/(2*h),dz[mask],rtol=3e-5,atol=1e-5)
        checks.append(dict(target=ident,truth=truth.tolist(),derivative_passed=True))
        rng=np.random.default_rng(20261014)
        for i in range(8):
            records.append(dict(target=ident,method='random',seed=i,log_parameters=rng.uniform(d['lower'],d['upper']).tolist()))
    rows=[]
    with ProcessPoolExecutor(max_workers=allocated_cpus()) as pool:
        fs=[pool.submit(regression,r) for r in records]
        for f in as_completed(fs):
            r=f.result();rows.append(r);print(json.dumps(dict(phase='preflight',done=len(rows),total=len(records),target=r['target'],passed=r['passed'])),flush=True)
            dump(OUT/'STATUS.json',dict(phase='preflight',completed_phase=len(rows),total_phase=len(records),failed_phase=sum(not r['passed'] for r in rows),updated_unix=time.time()))
    result=dict(passed=all(r['passed'] for r in rows),cases=len(rows),failed=sum(not r['passed'] for r in rows),checks=checks,hashes=hashes(),
        versions={n:version(n) for n in ['numpy','scipy','pymoo','moocore','mpmath']},timings_seconds_per_job={},scope='88 historical candidates plus24 seeded random candidates')
    dump(OUT/'PREFLIGHT.json',result)
    assert result['passed'],'Independent numerical regression failed; formal optimization blocked'

if __name__=='__main__':
    try:main()
    except Exception:
        dump(OUT/'PREFLIGHT_FAILED.json',dict(traceback=traceback.format_exc(),updated_unix=time.time()));raise

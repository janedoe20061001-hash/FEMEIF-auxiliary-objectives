"""Numerical equivalence, pathological regressions, scale audit, actual speed gate."""
from core import *
from features import Features
from reference_features import Features as Reference,rule
from checks import polygon,agrees,target_lengths
from concurrent.futures import ProcessPoolExecutor,as_completed
from importlib.metadata import version
import time,traceback,argparse

NAMES=('J0','J0_W','ARX_L','ARX_S_L')

def regression(item):
    i,record=item;st=time.perf_counter();d=target(record['target']);g=Features(d);old=Reference(d);lp=np.array(record['log_parameters'])
    p=10**lp
    # Exact same panels/order: implementation engineering must preserve old area.
    same=g.fixed(lp,('ARX_L',),16,0)[:,0];same_old=old.fixed(lp,('ARX',),16,0)[:,0]
    engine_ok=bool(np.all(abs(same-same_old)<=1e-7+1e-7*abs(same_old)))
    train=g.one(lp,NAMES);fine=g.fixed(lp,NAMES,32,2)
    poly=polygon(g,lp,NAMES,128);poly2=polygon(g,lp,NAMES,256)
    glob=g.one(lp,('G0',));glob_fine=g.fixed(lp,('G0',),32,2);glob_poly=polygon(g,lp,('G0',),256)
    global_ok=agrees(glob,glob_fine.mean(axis=0)) and agrees(glob_fine,glob_poly)
    # Native and independent Python impedances/derivatives on candidate anchors.
    xx=np.unique(np.r_[np.linspace(g.lo,g.hi,97),mode_points_for_check(g,p)])
    z,dz=g.response(10**xx,p);ref,dr=g.c.response(10**xx,p)
    model_ok=bool(np.all(abs(z-ref)<=1e-8+1e-8*abs(ref)) and np.all(abs(dz-dr)<=1e-7+1e-7*abs(dr)))
    # Warm both score paths; same candidate and same numerical convergence requirements.
    g.one(lp,('ARX_L',));old.one(lp,('ARX',))
    light_times=[];old_times=[]
    for rep in range(3):
        if rep%2:
            t=time.perf_counter();old_score=float(old.one(lp,('ARX',))[0]);old_times.append(time.perf_counter()-t)
            t=time.perf_counter();light_score=float(g.one(lp,('ARX_L',))[0]);light_times.append(time.perf_counter()-t)
        else:
            t=time.perf_counter();light_score=float(g.one(lp,('ARX_L',))[0]);light_times.append(time.perf_counter()-t)
            t=time.perf_counter();old_score=float(old.one(lp,('ARX',))[0]);old_times.append(time.perf_counter()-t)
    fine_converged=agrees(g.fixed(lp,NAMES,32,1),fine)
    numerical=agrees(train,fine.mean(axis=0)) and agrees(fine,poly2) and agrees(poly,poly2) and global_ok and fine_converged
    numerical=numerical and agrees(light_score,old_score,strict=True)
    result=dict(**record,index=i,passed=bool(engine_ok and model_ok and numerical),engine_passed=engine_ok,model_passed=model_ok,
        numerical_passed=numerical,train=train.tolist(),fine=fine.mean(axis=0).tolist(),polygon=poly2.mean(axis=0).tolist(),
        fine_converged=fine_converged,polygon_converged=agrees(poly,poly2),independent_agrees=agrees(fine,poly2),
        window_scores=fine.tolist(),window_labels=[w['block']['label'] for w in g.windows],
        old_arx=old_score,light_arx=light_score,old_seconds=float(np.median(old_times)),light_seconds=float(np.median(light_times)),
        auxiliary_speedup=float(np.median(old_times)/np.median(light_times)),seconds=time.perf_counter()-st)
    dump(OUT/'preflight_rows'/f'{i:04}.json',result);return result

def mode_points_for_check(g,p):
    from reference_features import mode_points
    return mode_points(g.modes(p)+g.target_modes,g.lo,g.hi)

def main(pilot=False):
    OUT.mkdir(parents=True,exist_ok=True);checks=[]
    for n in (8,16,32,64):
        x,w,b=rule(n)
        for k in range(6):np.testing.assert_allclose(b@(x**k),x**(k+1)/(k+1),atol=1e-13,rtol=1e-12)
    records=json.loads((INPUTS/'development_candidates.json').read_text(encoding='utf-8'))
    for ident in target_ids():
        d=target(ident);g=Features(d);lp=np.log10(d['circuit'].p)
        truth=g.one(lp,NAMES);assert max(truth)<1e-16,(ident,truth)
        a=target_lengths(g,128);b=target_lengths(g,256);native=g.raw(g.c.p,[],64,2)[:,2]
        assert np.all(abs(a-b)<1e-7+1e-4*abs(b)) and np.all(abs(native-b)<1e-7+1e-4*abs(b))
        checks.append(dict(target=ident,truth=truth.tolist(),scales=g.description(),target_arclength_passed=True))
        rng=np.random.default_rng(20261015)
        for i in range(8):records.append(dict(target=ident,method='random',seed=i,log_parameters=rng.uniform(d['lower'],d['upper']).tolist(),source='fixed seed random bounds'))
    if pilot:
        records=[next(r for r in records if r['target']==t and r['method']==m) for t in target_ids() for m in ['ARX','random']]
    rows=[]
    with ProcessPoolExecutor(max_workers=allocated_cpus()) as pool:
        fs=[pool.submit(regression,(i,r)) for i,r in enumerate(records)]
        for f in as_completed(fs):
            r=f.result();rows.append(r)
            print(json.dumps(dict(phase='preflight',done=len(rows),total=len(records),target=r['target'],passed=r['passed'],speedup=r['auxiliary_speedup'])),flush=True)
            dump(OUT/'STATUS.json',dict(phase='preflight',completed_phase=len(rows),total_phase=len(records),failed_phase=sum(not x['passed'] for x in rows),updated_unix=time.time()))
    speeds={t:float(np.median([r['auxiliary_speedup'] for r in rows if r['target']==t])) for t in target_ids()}
    result=dict(passed=all(r['passed'] for r in rows) and min(speeds.values())>=CFG['light'].get('minimum_speedup',3),numerical_passed=all(r['passed'] for r in rows),cases=len(rows),failed=sum(not r['passed'] for r in rows),checks=checks,hashes=hashes(),median_speedup_by_target=speeds,
        versions={n:version(n) for n in ['numpy','scipy','pymoo','moocore','mpmath']},scope='Fixed historical counterexamples, 192 official finals, 8 diagnostic alternatives, 24 seeded random candidates',pilot=pilot)
    dump(OUT/('PILOT.json' if pilot else 'PREFLIGHT.json'),result)
    assert result['passed'],'Numerical or speed preflight blocked formal optimization'

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');arg=parser.parse_args()
    try:main(arg.pilot)
    except Exception:
        dump(OUT/'PREFLIGHT_FAILED.json',dict(traceback=traceback.format_exc(),updated_unix=time.time()));raise

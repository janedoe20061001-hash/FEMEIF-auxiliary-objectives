from common import *
import time
cfg=json.loads((HERE/'CONFIG.json').read_text());rows=[];start=time.perf_counter()
for s in cfg['run_scales']:
    m=Target(s);a=evaluator(m);rng=np.random.default_rng(2026100901)
    x=rng.uniform(m.lower,m.upper,(32,18))
    old=[]
    for seed in range(2300,2316):
        with np.load(OLD/f'server_run/retrieved/runs/adaptive/seed_{seed}/generation_1000.npz') as d:old.append(d['log_parameters'][int(d['selected_index'])])
    x=np.vstack([x,old,m.lower,m.upper,np.log10(m.record['target_parameters'])]);F=a.score(x)
    np.testing.assert_allclose(m.network(m.f,10**x),spectra(m.f,10**x),rtol=1e-9,atol=1e-9)
    single=np.array([a.score(xx)[0] for xx in x]);np.testing.assert_allclose(F,single,rtol=1e-10,atol=1e-9)
    for i,xx in enumerate(x):
        refs=[]
        for lev in [2,4]:
            f=improved_dense_grid(m.record,10**xx[None,:],lev);refs.append(a.reference(xx,f)[0])
        tol=np.maximum([.01,np.deg2rad(.5)**2],.005*refs[-1]);err=np.abs(F[i]-refs[-1]);stable=np.all(abs(refs[0]-refs[1])<=tol)
        rows.append(dict(scale=s,index=i,score=F[i].tolist(),reference=refs[-1].tolist(),absolute_error=err.tolist(),tolerance=tol.tolist(),passed=bool(np.all(err<=tol) and stable),reference_stable=bool(stable)))
    print(f'preflight {s}: {sum(r["passed"] for r in rows if r["scale"]==s)}/51',flush=True)
dump(HERE/'PREFLIGHT.json',dict(passed=all(r['passed'] for r in rows),seconds=time.perf_counter()-start,rows=rows))
assert all(r['passed'] for r in rows),'Numerical scoring preflight failed'

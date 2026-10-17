from run import *

def preflight():
    m=model('case2','upto30m');full=model('case2');mask=full.f<=3e7
    np.testing.assert_array_equal(m.f,full.f[mask]);np.testing.assert_array_equal(m.target,full.target[mask])
    assert len(m.f)==183 and m.f[0]==40 and m.f[-1]==29e6 and np.all(m.f<=3e7)
    cfg=json.loads((BASE/'frozen/case2/config.json').read_text());assert sha(ROOT/'targetImpedance_case2.csv')==cfg['target_data']['sha256']
    data=np.genfromtxt(ROOT/'targetImpedance_case2.csv',delimiter=',',names=True,encoding='utf-8-sig');keep=data['frequency']<=3e7
    np.testing.assert_array_equal(m.f,data['frequency'][keep]);np.testing.assert_allclose(m.target,data['mag'][keep]*np.exp(1j*np.deg2rad(data['pha'][keep])),rtol=1e-12)
    x=np.random.default_rng(20261011).uniform(m.lower,m.upper,(5,len(m.lower)))
    np.testing.assert_allclose(m.score(x),metrics(m.spectrum(x),m.target,m.alpha),rtol=1e-12,atol=1e-9)
    f=m.f[np.unique(np.linspace(0,len(m.f)-1,13,dtype=int))];z,c=nodal(m,10**x,f);ref=m.network(f,10**x)
    np.testing.assert_allclose(z,ref,rtol=2e-7,atol=1e-6)
    dump(OUT/'PREFLIGHT.json',dict(passed=True,frequency_cap_hz=30000000,points=183,actual_range_hz=[40,29000000],removed_above_cap=18,
         nodal_max_relative_error=float((abs(z-ref)/abs(ref)).max()),independent_loss=True,target_csv_sha256=sha(ROOT/'targetImpedance_case2.csv')))

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=28);p.add_argument('--preflight',action='store_true');a=p.parse_args();preflight()
    if a.preflight:return
    jobs=[('upto30m','case2',method,s) for method in ['scalar_nsga','scalar_ga','dual_nsga'] for s in range(2300,2400)]
    t=time.time();done=0;failed=[]
    from importlib.metadata import version
    dump(OUT/'MANIFEST.json',dict(source_sha256={p.name:sha(p) for p in HERE.glob('*') if p.suffix in ['.py','.md']},frozen_model_sha256=sha(BASE/'model.py'),
         planned_runs=300,planned_evaluations=45000000,workers=a.workers,python=platform.python_version(),versions={k:version(k) for k in ['numpy','scipy','pymoo','moocore']}))
    def status(state):dump(OUT/'STATUS.json',dict(status=state,total=300,completed=done,failed=failed,seconds=time.time()-t))
    status('running')
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(one,j) for j in jobs]):
            r=future.result()
            if r['status']=='complete':done+=1
            else:failed.append(r)
            status('running')
            if (done+len(failed))%50==0:print(json.dumps(dict(completed=done,failed=len(failed),total=300)),flush=True)
    status('failed' if failed else 'complete');assert not failed

if __name__=='__main__':main()

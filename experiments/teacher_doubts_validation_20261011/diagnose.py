from core import *
import csv,time
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import least_squares

def csvwrite(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def dense_grid(m,parameters,refine=1):
    if m.record['id']=='case2':return m.f
    logs=[np.linspace(np.log10(m.f[0]),np.log10(m.f[-1]),8192*refine+1),np.log10(m.f)]
    for p in [np.asarray(m.record['target_parameters']),*parameters]:
        for r,l,c in p.reshape(6,3):
            fc=1/(2*np.pi*np.sqrt(l*c));q=r*np.sqrt(c/l);dx=min(.5,max(6/q,1e-12))/np.log(10);center=np.log10(fc)
            if not m.f[0]<=fc<=m.f[-1]:continue
            logs.append(np.linspace(center-dx,center+dx,384*refine+1))
            offsets=np.geomspace(dx,max(dx,.5),192*refine+1);logs.extend([center-offsets,center+offsets])
    x=np.unique(np.concatenate(logs));x=x[(x>=np.log10(m.f[0]))&(x<=np.log10(m.f[-1]))];return 10**x

def curve_metrics(m,x):
    p=10**np.asarray(x);f=dense_grid(m,[p]);z=m.network(f,p)[0]
    target=m.network(f,np.asarray(m.record['target_parameters']))[0] if m.record['id']=='case1' else m.target
    da=20*np.log10(abs(z)/abs(target));dp=np.rad2deg(np.angle(z)-np.angle(target))
    result=dict(curve_points=len(f),max_amplitude_db=float(abs(da).max()),max_phase_deg=float(abs(dp).max()),
                max_amplitude_hz=float(f[np.argmax(abs(da))]),max_phase_hz=float(f[np.argmax(abs(dp))]))
    if m.record['id']=='case1':
        # Independent doubled density guards against missing narrow candidate peaks.
        ff=dense_grid(m,[p],2);zz=m.network(ff,p)[0];tt=m.network(ff,np.asarray(m.record['target_parameters']))[0]
        result.update(refined_max_amplitude_db=float(abs(20*np.log10(abs(zz)/abs(tt))).max()),refined_max_phase_deg=float(abs(np.rad2deg(np.angle(zz)-np.angle(tt))).max()))
    return result

def polish(job):
    condition,case,method,generation,seed,x=job;m=model(case,condition);queries=0;t=time.perf_counter()
    def residual(parameters):
        nonlocal queries
        queries+=1;z=m.spectrum(parameters)[0]
        return np.concatenate([20*np.log10(abs(z))-m.amp,np.sqrt(m.alpha)*(np.angle(z)-m.phase)])/np.sqrt(len(m.f))
    start=m.score(x)[0];result=least_squares(residual,x,bounds=(m.lower,m.upper),max_nfev=2000,loss='linear')
    final=m.score(result.x)[0]
    np.testing.assert_allclose(np.sum(residual(result.x)**2),final[2],rtol=1e-12,atol=1e-9)
    name=f'{condition}_{case}_{method}_{generation}';d=OUT/'polish';d.mkdir(exist_ok=True)
    np.savez_compressed(d/f'{name}.npz',initial_log_parameters=x,log_parameters=result.x,initial_metrics=start,metrics=final)
    r=dict(condition=condition,case=case,method=method,generation=generation,seed=seed,status=int(result.status),
           success=bool(result.success),message=result.message,nfev=int(result.nfev),actual_model_queries=queries,
           seconds=time.perf_counter()-t,initial_wsmse=float(start[2]),final_metrics=final.tolist(),
           optimality=float(result.optimality),active_bounds=int(np.count_nonzero(result.active_mask)),
           initial_curve=curve_metrics(m,x),final_curve=curve_metrics(m,result.x))
    dump(d/f'{name}.json',r);return r

def verify_and_select():
    status=json.loads((OUT/'STATUS.json').read_text());assert status['status']=='complete' and status['completed']==800 and not status['failed']
    rows=[];best={};total=0;individuals=0;replay=[];initial_hashes={};jobs=[];mismatches=[]
    for d in sorted((OUT/'runs').glob('*/*/*/seed_*')):
        condition,case,method,seedname=d.parts[-4:];seed=int(seedname[5:]);m=model(case,condition)
        record=json.loads((d/'summary.json').read_text());assert record['status']=='complete' and record['evaluations']==m.record['population']*1000;total+=record['evaluations']
        x0=np.load(d/'initial_log_parameters.npy');expected=np.random.default_rng(seed).uniform(m.lower,m.upper,x0.shape);np.testing.assert_array_equal(x0,expected)
        assert x0.shape==(m.record['population'],len(m.lower))
        h=hashlib.sha256(x0.tobytes()).hexdigest();assert h==record['initial_sha256']
        key=(case,seed)
        if key in initial_hashes:assert initial_hashes[key]==h
        initial_hashes[key]=h
        if method in ['scalar_nsga','dual_nsga'] and condition=='full':
            assert record['original300_replay'] is not None;replay.append(record['original300_replay'])
            if record['original300_replay'].get('matched') is False:mismatches.append(dict(case=case,method=method,seed=seed))
        with np.load(d/'trajectory.npz') as a:
            generations=a['generations'];x=a['log_parameters'];v=a['metrics'];reasons=a['knee_reasons']
        assert np.array_equal(generations,np.arange(20,1001,20))
        independently=metrics(m.spectrum(x.reshape(-1,len(m.lower))),m.target,m.alpha).reshape(v.shape)
        np.testing.assert_allclose(v,independently,rtol=1e-11,atol=1e-9);individuals+=len(x)*2
        assert np.all(x>=m.lower) and np.all(x<=m.upper)
        band_values=model('case2','paper_band').score(x.reshape(-1,len(m.lower))).reshape(v.shape) if case=='case2' else None
        for i,g in enumerate(generations):
            for which,selection in enumerate(['knee' if method=='dual_nsga' else 'minimum','minimum'][:2 if method=='dual_nsga' else 1]):
                r=dict(condition=condition,case=case,method=method,selection=selection,seed=seed,generation=int(g),reason=str(reasons[i]) if which==0 else 'minimum_wsmse',**dict(zip(METRICS,v[i,which].tolist())))
                r['paper_band_wsmse']=float(band_values[i,which,2]) if case=='case2' else None
                rows.append(r);k=(condition,case,method,selection,int(g))
                if k not in best or (r['wsmse'],seed)<(best[k][0]['wsmse'],best[k][0]['seed']):best[k]=(r,x[i,which].copy())
        hist=np.genfromtxt(d/'history.csv',delimiter=',',names=True,encoding='utf-8-sig');assert len(hist)==1000
        if method!='dual_nsga':assert np.all(np.diff(hist['wsmse'])<=1e-8)
        for g in [300,1000]:
            with np.load(d/f'generation_{g:04}.npz') as a:fx=a['log_parameters'];fv=a['metrics'];fo=a['optimization_objective']
            np.testing.assert_allclose(fv,metrics(m.spectrum(fx),m.target,m.alpha),rtol=1e-11,atol=1e-9)
            np.testing.assert_allclose(fo,fv[:,:2] if method=='dual_nsga' else fv[:,2:3],rtol=1e-11,atol=1e-9)
            assert fx.shape==x0.shape and np.all(fx>=m.lower) and np.all(fx<=m.upper)
            ki,reason=knee(fv[:,:2]) if method=='dual_nsga' else (int(np.argmin(fv[:,2])),'scalar_minimum')
            i=list(generations).index(g);np.testing.assert_array_equal(fx[[ki,int(np.argmin(fv[:,2]))]],x[i]);assert reason==str(reasons[i]);individuals+=len(fx)
    assert total==135000000 and len(replay)==400
    assert sorted(r['seed'] for r in mismatches)==[2334,2365]
    csvwrite(OUT/'all_seed_stage_metrics.csv',rows)
    summary=[]
    for k,(r,x) in best.items():
        if k[-1] not in [260,280,300,600,1000]:continue
        group=[v for v in rows if (v['condition'],v['case'],v['method'],v['selection'],v['generation'])==k]
        assert len(group)==100
        vals=np.array([[v[n] for n in METRICS] for v in group]);b=dict(r);b['log_parameters']=x.tolist();b['curve']=curve_metrics(model(r['case'],r['condition']),x)
        summary.append(dict(condition=k[0],case=k[1],method=k[2],selection=k[3],generation=k[4],
                            mean_metrics=vals.mean(axis=0).tolist(),median_metrics=np.median(vals,axis=0).tolist(),
                            best=b,knee_fallbacks=sum(v['reason']!='convex_knee' for v in group) if k[3]=='knee' else None,
                            mean_paper_band_wsmse=float(np.mean([v['paper_band_wsmse'] for v in group])) if k[1]=='case2' else None))
        if k[-1] in [300,1000] and k[3]!=('minimum' if k[2]=='dual_nsga' else 'excluded'):
            jobs.append((r['condition'],r['case'],r['method'],r['generation'],r['seed'],x))
    assert len(jobs)==16
    dump(OUT/'SUMMARY.json',summary)
    dump(OUT/'VALIDATION.json',dict(passed=True,runs=800,total_evaluations=total,individuals_recomputed=individuals,
                                  initialization_matches=True,original300_replays=len(replay),original300_matched=398,
                                  original300_mismatches=mismatches,failed_attempt_extra_evaluations=90000,
                                  replay_probe_evaluations=360000,
                                  maximum_matched_replay_parameter_difference=max(r['max_parameter_difference'] for r in replay if r.get('matched',True)),
                                  maximum_matched_replay_metric_difference=max(r['max_metric_difference'] for r in replay if r.get('matched',True))))
    return jobs

def main():
    jobs=verify_and_select();print('800 runs verified, starting 16 bounded local refinements on server',flush=True)
    with ProcessPoolExecutor(max_workers=8) as pool:results=list(pool.map(polish,jobs))
    dump(OUT/'POLISH_SUMMARY.json',results);print('diagnosis complete',flush=True)

if __name__=='__main__':main()

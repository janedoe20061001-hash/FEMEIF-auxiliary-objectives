from common import *
import time,argparse,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
def one(job):
    s,seed=job;m=Target(s);cat=target_catalog(m);a=evaluator(m);d=HERE/'batch'/label(s)/f'seed_{seed}';base=dict(scale=s,seed=seed)
    rows=[];ers=[];regions=[];front=[];front_events=[];sensitivity=[];validation=[]
    rec=json.loads((d/'summary.json').read_text());assert rec['status']=='complete' and rec['evaluations']==60000
    x0=np.load(d/'initial_log_parameters.npy');np.testing.assert_array_equal(x0,np.random.default_rng(seed).uniform(m.lower,m.upper,(200,18)))
    with np.load(d/'trajectory.npz') as trajectory:
        assert trajectory['generations'].tolist()==list(range(20,301,20))
        for g,xs,fs in zip(trajectory['generations'],trajectory['log_parameters'],trajectory['training_metrics']):
            np.testing.assert_allclose(a.score(xs),fs,rtol=1e-10,atol=1e-8)
            for j,rule in enumerate(['min_wsmse','balanced_minmax']):
                bb=dict(**base,generation=int(g),rule=rule);p=10**xs[j]
                diag,ee,bands=diagnostic(m,p,cat);ev={e['event_id']:e for e in ee}
                rows.append(dict(**bb,train_mag=float(fs[j,0]),train_phase=float(fs[j,1]),joint_complete=ev['P3']['complete'] and ev['V2']['complete'],P3_matched=ev['P3']['matched'],V2_matched=ev['V2']['matched'],P3_complete=ev['P3']['complete'],V2_complete=ev['V2']['complete'],**diag))
                ers.extend(dict(**bb,**e) for e in ee)
                for r in regional(m,p,cat,4):regions.append(dict(**bb,**r))
                if g==300:
                    for shift,height,width in [(.01,2,np.log2(1.5)),(.02,3,1),(.03,4,np.log2(3))]:
                        complete={e:ev[e]['matched'] and ev[e]['width_log2_ratio'] is not None and abs(ev[e]['shift_decades'])<=shift and abs(ev[e]['height_delta_db'])<=height and abs(ev[e]['width_log2_ratio'])<=width for e in ['P3','V2']}
                        sensitivity.append(dict(**bb,shift_limit=shift,height_limit_db=height,width_factor_limit=float(2**width),P3_complete=complete['P3'],V2_complete=complete['V2'],joint_complete=all(complete.values())))
    with np.load(d/'generation_300.npz') as data:
        xx=data['log_parameters'];F=data['optimization_objective'];np.testing.assert_allclose(a.score(xx),F,rtol=1e-10,atol=1e-8)
        assert xx.shape==(200,18) and np.all(xx>=m.lower) and np.all(xx<=m.upper)
        np.testing.assert_allclose(m.network(m.f,10**xx),spectra(m.f,10**xx),rtol=1e-9,atol=1e-9)
        dom=np.all(F[:,None,:]<=F[None,:,:],axis=2)&np.any(F[:,None,:]<F[None,:,:],axis=2);fi=np.flatnonzero(~dom.any(axis=0));np.testing.assert_array_equal(fi,np.sort(data['front_indices']))
        ix=int(data['selected_index']);bx=int(data['balanced_index']);assert ix==int(fi[np.argmin(F[fi,0]+m.alpha*F[fi,1])]);assert bx==int(fi[np.argmin(np.maximum(F[fi,0],m.alpha*F[fi,1]))])
        # Independent dense quadrature, including every retained final candidate.
        for index,x in enumerate(xx):
            refs=[]
            for lev in [2,4]:refs.append(a.reference(x,improved_dense_grid(m.record,10**x[None,:],lev))[0])
            tol=np.maximum([.01,np.deg2rad(.5)**2],.005*refs[-1]);diff=abs(F[index]-refs[-1]);stable=bool(np.all(abs(refs[0]-refs[1])<=tol))
            validation.append(dict(**base,index=index,mag_error=float(diff[0]),phase_error=float(diff[1]),mag_tolerance=float(tol[0]),phase_tolerance=float(tol[1]),reference_stable=stable,passed=bool(stable and np.all(diff<=tol))))
        for index in fi:
            bb=dict(**base,generation=300,index=int(index),is_primary=bool(index==ix),is_balanced=bool(index==bx));diag,ee,_=diagnostic(m,10**xx[index],cat);ev={e['event_id']:e for e in ee}
            front.append(dict(**bb,train_mag=float(F[index,0]),train_phase=float(F[index,1]),joint_complete=ev['P3']['complete'] and ev['V2']['complete'],P3_complete=ev['P3']['complete'],V2_complete=ev['V2']['complete'],**diag));front_events.extend(dict(**bb,**e) for e in ee)
    result=dict(summary=rec,representatives=rows,events=ers,regions=regions,front=front,front_events=front_events,sensitivity=sensitivity,validation=validation)
    dump(OUT/'analysis/chunks'/f'{label(s)}_{seed}.json',result);return s,seed
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);opt=parser.parse_args();cfg=json.loads((HERE/'CONFIG.json').read_text());start=time.perf_counter()
    jobs=[(s,seed) for s in cfg['run_scales'] for seed in cfg['seeds'] if not (OUT/'analysis/chunks'/f'{label(s)}_{seed}.json').exists()]
    with ProcessPoolExecutor(max_workers=opt.workers) as pool:
        futures={};pending=jobs[:]
        while pending or futures:
            available=[j for j in pending if (HERE/'batch'/label(j[0])/f'seed_{j[1]}'/'summary.json').exists()]
            for j in available[:max(0,opt.workers-len(futures))]:futures[pool.submit(one,j)]=j;pending.remove(j)
            for f in list(futures):
                if f.done():s,seed=f.result();futures.pop(f);print(f'Analyzed {s}, {seed}',flush=True)
            status=HERE/'batch/status.json'
            if status.exists() and json.loads(status.read_text())['status']=='failed':raise RuntimeError('Training failed; cannot omit failed runs')
            if pending or futures:time.sleep(5)
    data=[json.loads((OUT/'analysis/chunks'/f'{label(s)}_{seed}.json').read_text()) for s in cfg['run_scales'] for seed in cfg['seeds']]
    for key,name in [('representatives','REPRESENTATIVES.csv'),('events','EVENTS.csv'),('regions','REGIONAL_ERRORS.csv'),('front','FRONT.csv'),('front_events','FRONT_EVENTS.csv'),('sensitivity','THRESHOLD_SENSITIVITY.csv'),('validation','QUADRATURE.csv')]:csvout(OUT/'analysis'/name,[r for d in data for r in d[key]])
    reps=[r for d in data for r in d['representatives']];ff=[r for d in data for r in d['front']];vv=[r for d in data for r in d['validation']]
    valid=dict(runs=len(data),evaluations=sum(d['summary']['evaluations'] for d in data),representatives=len(reps),front_candidates=len(ff),independent_final_candidates=len(vv),quadrature_failures=sum(not v['passed'] for v in vv),unstable_representatives=sum(not r['grid_stable'] for r in reps),unstable_front=sum(not r['grid_stable'] for r in ff),seconds=time.perf_counter()-start)
    dump(OUT/'analysis/VALIDATION.json',valid);assert not valid['quadrature_failures'] and not valid['unstable_representatives'] and not valid['unstable_front']
    print(json.dumps(valid),flush=True)
if __name__=='__main__':main()

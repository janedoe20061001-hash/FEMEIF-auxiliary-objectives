from common import *
import time,argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
RUNS=OLD/'server_run/retrieved/runs'
ANALYSIS=OLD/'server_run/retrieved/analysis'
def regional_job(job):
    arm,seed=job;m=Target(1);cat=target_catalog(m);rows=[];errors=[]
    with np.load(RUNS/arm/f'seed_{seed}'/'trajectory.npz') as d:
        for g,xs in zip(d['generations'],d['log_parameters']):
            if g not in [20,100,300,1000]:continue
            for i,rule in enumerate(['min_wsmse','balanced_minmax']):
                r2=regional(m,10**xs[i],cat,2);r4=regional(m,10**xs[i],cat,4)
                for a,b in zip(r2,r4):
                    stable=all(abs(a[k]-b[k])<=max(floor,.01*abs(b[k])) for k,floor in [('mag_contribution',.0001),('phase_contribution',np.deg2rad(.05)**2),('global_mag',.01),('global_phase',np.deg2rad(.5)**2)])
                    rows.append(dict(arm=arm,seed=seed,generation=int(g),rule=rule,integral_stable=stable,**b))
                    if not stable:errors.append(dict(arm=arm,seed=seed,generation=int(g),rule=rule,level2=a,level4=b))
    return rows,errors
def readcsv(name):
    with (ANALYSIS/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def candidate_pairs():
    diag=readcsv('front_diagnostics.csv');events=readcsv('front_events.csv');by={}
    for r in diag:
        if int(r['generation'])!=1000:continue
        key=(r['arm'],int(r['seed']));by.setdefault(key,{})[int(r['index'])]=dict(mag=float(r['train_mag']),phase=float(r['train_phase']),events={})
    for r in events:
        if int(r['generation'])==1000 and r['event_id'] in ['P3','V2']:
            key=(r['arm'],int(r['seed']));by[key][int(r['index'])]['events'][r['event_id']]=r
    counts=[];examples=[];floor=np.array([.01,np.deg2rad(.5)**2])
    for (arm,seed),dd in by.items():
        with np.load(RUNS/arm/f'seed_{seed}'/'generation_1000.npz') as snap:
            ix=np.array(sorted(dd));xx=snap['log_parameters'][ix]
            # Exact cloned parameter vectors add no diagnostic evidence.
            _,unique=np.unique(xx,axis=0,return_index=True);ix=ix[np.sort(unique)]
        ff=np.array([[dd[i]['mag'],dd[i]['phase']] for i in ix]);shape=np.zeros((len(ix),len(ix)),bool);joint=[]
        for i in ix:joint.append(all(dd[i]['events'][e]['complete']=='True' for e in ['P3','V2']))
        for ii,i in enumerate(ix):
            for jj in range(ii+1,len(ix)):
                j=ix[jj];strong=False
                for e in ['P3','V2']:
                    a=dd[i]['events'][e];b=dd[j]['events'][e]
                    strong|=a['matched']!=b['matched']
                    if a['matched']==b['matched']=='True':
                        strong|=abs(float(a['height_delta_db'])-float(b['height_delta_db']))>=6
                        if a['width_log2_ratio'] and b['width_log2_ratio']:strong|=abs(float(a['width_log2_ratio'])-float(b['width_log2_ratio']))>=1
                shape[ii,jj]=strong
        relative=abs(ff[:,None,:]-ff[None,:,:])/np.maximum(np.minimum(ff[:,None,:],ff[None,:,:]),floor)
        maxrel=relative.max(axis=2);upper=np.triu(np.ones_like(shape),1).astype(bool)
        split=np.asarray(joint)[:,None]!=np.asarray(joint)[None,:]
        for threshold in [.02,.05,.10]:
            close=(maxrel<=threshold)&upper;strong=close&shape;complete_split=strong&split
            counts.append(dict(arm=arm,seed=seed,generation=1000,threshold=threshold,unique_front_candidates=len(ix),has_close_pair=bool(close.any()),has_meaningful_shape_pair=bool(strong.any()),has_joint_complete_vs_incomplete_meaningful_pair=bool(complete_split.any()),close_pairs=int(close.sum()),meaningful_pairs=int(strong.sum())))
            if threshold==.05 and strong.any():
                ii,jj=np.unravel_index(np.where(strong,maxrel,np.inf).argmin(),strong.shape)
                examples.append(dict(arm=arm,seed=seed,index_a=int(ix[ii]),index_b=int(ix[jj]),max_relative_score_difference=float(maxrel[ii,jj]),a=dd[ix[ii]],b=dd[ix[jj]]))
    return counts,examples
def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);opt=p.parse_args();start=time.perf_counter()
    d=OUT/'offline';d.mkdir(parents=True,exist_ok=True)
    counts,examples=candidate_pairs();csvout(d/'CANDIDATE_PAIRS.csv',counts);dump(d/'PAIR_EXAMPLES.json',examples)
    summary=[]
    for arm in ['original','adaptive']:
        for threshold in [.02,.05,.1]:
            rows=[r for r in counts if r['arm']==arm and r['threshold']==threshold]
            summary.append(dict(arm=arm,threshold=threshold,seeds=len(rows),close_seeds=sum(r['has_close_pair'] for r in rows),meaningful_shape_seeds=sum(r['has_meaningful_shape_pair'] for r in rows),joint_complete_split_seeds=sum(r['has_joint_complete_vs_incomplete_meaningful_pair'] for r in rows)))
    dump(d/'PAIR_SUMMARY.json',summary);print(json.dumps(summary),flush=True)
    rows=[];errors=[]
    with ProcessPoolExecutor(max_workers=opt.workers) as pool:
        futures=[pool.submit(regional_job,(arm,s)) for arm in ['original','adaptive'] for s in range(2300,2400)]
        for n,future in enumerate(as_completed(futures),1):
            rr,ee=future.result();rows.extend(rr);errors.extend(ee)
            if n%20==0:print(f'regional checked {n}/200',flush=True)
    rows.sort(key=lambda r:(r['arm'],r['seed'],r['generation'],r['rule'],r['weighting'],r['window']));csvout(d/'REGIONAL_ERRORS.csv',rows)
    dump(d/'VALIDATION.json',dict(regional_rows=len(rows),curve_count=1600,unstable_rows=len(errors),seconds=time.perf_counter()-start,failures=errors))
    # Preserve failures for review instead of silently accepting unstable integration.
    assert not errors,'Regional integrals need extra refinement before interpretation'
    print('Offline analysis complete',flush=True)
if __name__=='__main__':main()

"""Alternating, warmed, same-window score timing; audit excluded for both."""
from metric import *
from features import Features

def bench():
    tasks=json.loads((OUT/'TASKS.json').read_text())['tasks'];selected=[]
    for tid in ['case1','mixed_resonators/base','mixed_resonators/lc_reciprocal_8p0']:
        selected += [t for t in tasks if t[1]==tid and t[0].startswith('held_')][:4]
    rows=[];models={};oldmodels={}
    for task in selected:
        ident,tid,p,labels,meta=task;p=np.array(p);label=labels[0]
        if tid not in models:models[tid]=Variation(tid);oldmodels[tid]=Features(target(tid))
        m=models[tid];old=oldmodels[tid];m.scales(label);sc=old.scales();j=next(j for j,w in enumerate(old.windows) if w['block']['label']==label)
        def vscore():
            previous=None
            for n in [64,128,256,512,1024]:
                now=m.polygon(p,label,n)
                if n>=256 and agree(previous,now):return now['JV']
                previous=now
            raise RuntimeError('JV timing candidate not converged')
        def ascore():
            modes=old.modes(p);previous=None
            for ref in range(5):
                lo,span,scale,offset,edges=old.grid(modes,ref)
                e=edges[offset[j]:offset[j+1]]
                grid=(lo[j:j+1],span[j:j+1],scale[j:j+1],np.array([0,len(e)],np.int32),e)
                for n in [16,32,64]:
                    now=old.raw(p,modes,n,ref,grid=grid,flags=1)[0,0]/sc[j]**2
                    if previous is not None and abs(now-previous)<=1e-4+1e-3*abs(now):return float(now)
                    previous=now
            raise RuntimeError('ARX timing candidate not converged')
        cold={}
        for name,fn in [('JV',vscore),('ARX_S_L',ascore)]:
            st=time.perf_counter();value=fn();cold[name]=dict(seconds=time.perf_counter()-st,value=value)
        runs=[]
        for repeat in range(5):
            values={}
            for name,fn in ([('JV',vscore),('ARX_S_L',ascore)] if repeat%2==0 else [('ARX_S_L',ascore),('JV',vscore)]):
                st=time.perf_counter();value=fn();values[name]=dict(seconds=time.perf_counter()-st,value=value)
            runs.append(values)
        row=dict(id=ident,target=tid,label=label,cold_first_call_after_models=cold,runs=runs)
        row['median_JV']=float(np.median([x['JV']['seconds'] for x in runs]));row['median_ARX']=float(np.median([x['ARX_S_L']['seconds'] for x in runs]));row['ratio']=row['median_JV']/row['median_ARX'];rows.append(row)
    ratios=[r['ratio'] for r in rows]
    result=dict(scope='12 fixed real candidates, same local window and tolerance; Python vectorized polygon versus existing compiled ARX; not optimized implementation equivalence',rows=rows,median_paired_ratio=float(np.median(ratios)),p95_paired_ratio=float(np.percentile(ratios,95)),gate_passed=bool(np.median(ratios)<=1.25),audit_in_timing=False,model_startup_excluded=True)
    dump(OUT/'BENCHMARK.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
if __name__=='__main__':bench()

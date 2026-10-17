"""Evaluate existing curves only; never changes optimization or target data."""
from pathlib import Path
import sys,json,csv,hashlib,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'output/sampling_feasibility_20261008'
sys.path.insert(0,str(ROOT/'experiments/optimized_curve_structure_20261007'))
from model import Model,response
from refined_grid import improved_dense_grid

def values(r,p,x):
    z=response(r['tree'],r['parameter_names'],10**np.asarray(x),p)[0]
    return np.column_stack([20*np.log10(abs(z)),np.rad2deg(np.angle(z))])

def midpoint(r,p,x):
    known={float(t):v for t,v in zip(x,values(r,p,x))}
    active=list(zip(x[:-1],x[1:]));limit=False;depth=False
    for layer in range(24):
        if not active:break
        room=3000-len(known)
        if room<len(active):
            limit=True
            active=[active[i] for i in np.unique(np.linspace(0,len(active)-1,room,dtype=int))] if room else []
        mids=np.asarray([(a+b)/2 for a,b in active])
        if not len(mids):break
        v=values(r,p,mids);nxt=[]
        for (a,b),m,y in zip(active,mids,v):
            known[float(m)]=y
            err=abs(y-(known[float(a)]+known[float(b)])/2)
            if err[0]>.1 or err[1]>.5:nxt.extend([(a,m),(m,b)])
        active=nxt
        if limit:break
    else:depth=bool(active)
    xx=np.asarray(sorted(known));yy=np.asarray([known[t] for t in xx])
    return xx,yy,limit,depth

def gap(x,y,rx,ry):
    return np.max(abs(ry-np.column_stack([np.interp(rx,x,y[:,j]) for j in range(2)])),axis=0)

def writecsv(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    start=time.perf_counter();m=Model('case1');r=m.record;x=np.log10(m.f)
    with np.load(ROOT/'experiments/sampling_refinement_20261007/grids/case1.npz') as d:fixed=np.log10(d['frequency'])
    samples=[('target',-1,np.asarray(r['target_parameters']))]
    sources={}
    runs=ROOT/'experiments/dual_trajectory_20261008/server_run/retrieved/runs/case1'
    for seed in range(2300,2400):
        path=runs/f'seed_{seed}/generation_300.npz';sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as d:
            for rule,key in [('min_wsmse','selected_index'),('balanced_minmax','balanced_index')]:
                samples.append((rule,seed,10**d['log_parameters'][int(d[key])]))
    rows=[]
    for i,(rule,seed,p) in enumerate(samples):
        # Independent series/parallel evaluator agrees with frozen case1 routine.
        zz=response(r['tree'],r['parameter_names'],m.f,p)[0]
        np.testing.assert_allclose(zz,m.network(m.f,np.atleast_2d(p))[0],rtol=1e-9,atol=1e-9)
        references=[]
        for level in [2,4]:
            f=improved_dense_grid(r,np.atleast_2d(p),level);rx=np.log10(f);references.append((rx,values(r,p,rx)))
        ax,ay,limit,depth=midpoint(r,p,x)
        mode=np.log10(improved_dense_grid(r,np.atleast_2d(p),1))
        for method,xx,yy in [('original_180',x,values(r,p,x)),('target_fixed_731',fixed,values(r,p,fixed)),('value_midpoint',ax,ay),('model_informed',mode,values(r,p,mode))]:
            g2,g4=[gap(xx,yy,rx,ry) for rx,ry in references]
            stable=bool(abs(g2[0]-g4[0])<=.02 and abs(g2[1]-g4[1])<=.1)
            rows.append(dict(rule=rule,seed=seed,method=method,points=len(xx),amp_gap_db=float(max(g2[0],g4[0])),phase_gap_deg=float(max(g2[1],g4[1])),pass_both=bool(max(g2[0],g4[0])<=.2 and max(g2[1],g4[1])<=1),reference_stable=stable,amp_ref_delta_db=float(abs(g2[0]-g4[0])),phase_ref_delta_deg=float(abs(g2[1]-g4[1])),budget_stop=limit if method=='value_midpoint' else False,depth_stop=depth if method=='value_midpoint' else False))
        if i%25==0:print(f'Checked {i+1}/{len(samples)} curves',flush=True)
    writecsv(OUT/'per_curve_sampling.csv',rows)
    groups=[]
    for rule in ['target','min_wsmse','balanced_minmax']:
        for method in ['original_180','target_fixed_731','value_midpoint','model_informed']:
            rr=[a for a in rows if a['rule']==rule and a['method']==method]
            groups.append(dict(rule=rule,method=method,curves=len(rr),passed=sum(a['pass_both'] for a in rr),stable=sum(a['reference_stable'] for a in rr),points_median=float(np.median([a['points'] for a in rr])),points_max=max(a['points'] for a in rr),amp_gap_median=float(np.median([a['amp_gap_db'] for a in rr])),amp_gap_max=max(a['amp_gap_db'] for a in rr),phase_gap_median=float(np.median([a['phase_gap_deg'] for a in rr])),phase_gap_max=max(a['phase_gap_deg'] for a in rr),budget_stops=sum(a['budget_stop'] for a in rr),depth_stops=sum(a['depth_stop'] for a in rr)))
    result={'curves':len(samples),'rows':len(rows),'groups':groups,'seconds':time.perf_counter()-start,'optimization_runs':0,'sources':sources,'protocol_sha256':hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest(),'all_reference_stable':all(a['reference_stable'] for a in rows)}
    (OUT/'SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');writecsv(OUT/'group_summary.csv',groups)
    print(json.dumps({k:v for k,v in result.items() if k not in ['sources','groups']}),flush=True)
if __name__=='__main__':main()

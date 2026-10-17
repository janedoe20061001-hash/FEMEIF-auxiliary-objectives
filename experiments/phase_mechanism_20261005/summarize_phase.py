"""Summarize paired-seed diagnostics; exploratory intervals, no new fit claims."""
from pathlib import Path
import json,csv
import numpy as np
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parent
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,(np.integer,np.bool_)):return x.item()
    return x
def rho(a,b):
    if np.ptp(a)==0 or np.ptp(b)==0:return np.nan
    return np.corrcoef(rankdata(a),rankdata(b))[0,1]
def partial(a,b,c):
    if any(np.ptp(v)==0 for v in [a,b,c]):return np.nan
    a,b,c=map(rankdata,[a,b,c]);X=np.c_[np.ones(len(c)),c]
    ar=a-X@np.linalg.lstsq(X,a,rcond=None)[0];br=b-X@np.linalg.lstsq(X,b,rcond=None)[0]
    return np.corrcoef(ar,br)[0,1] if min(np.std(ar),np.std(br))>1e-12 else np.nan
def ci(a,b):
    rng=np.random.default_rng(20261005);ix=rng.integers(0,len(a),(10000,len(a)))
    return np.quantile((1-a[ix].mean(axis=1)/b[ix].mean(axis=1))*100,[.025,.975])
def col(rows,k):return np.array([float(r[k]) for r in rows])
metrics=['I03','II01','II02','R05','angular_rate','complex_tangent','phase_sign','cross_position']
result={}
for case in ['case1','case2']:
    rows=read(ROOT/case/'selected.csv');front=read(ROOT/case/'front.csv');sens=read(ROOT/case/'sensitivity.csv')
    cs=read(ROOT/case/'correlations.csv');cf=read(ROOT/case/'conflicts.csv')
    groups=list(dict.fromkeys(r['group'] for r in rows));b=sorted([r for r in rows if r['group']=='baseline'],key=lambda r:int(r['seed']))
    bm=col(b,'magnitude');bp=col(b,'phase');bc={k:col(b,k) for k in metrics}
    summary=[]
    for g in groups:
        r=sorted([r for r in rows if r['group']==g],key=lambda r:int(r['seed']));p=col(r,'phase');m=col(r,'magnitude')
        z=col(r,'I03')==0
        fr=sorted([v for v in front if v['group']==g],key=lambda r:int(r['seed']))
        cap=col(fr,'phase_at_cap');bcap=col(fr,'baseline_phase_at_cap');valid=np.isfinite(cap)
        sr=dict(group=g,mean_magnitude=m.mean(),mean_phase=p.mean(),phase_reduction_pct=(1-p.mean()/bp.mean())*100,
            phase_reduction_ci95=ci(p,bp),phase_better=int(sum(p<bp)),both_better=int(sum((p<bp)&(m<bm))),
            phase_delta_median=np.median(p-bp),phase_delta_q25_q75=np.quantile(p-bp,[.25,.75]),
            zero_cross_count=int(z.sum()),zero_cross_phase_median=np.median(p[z]) if z.any() else None,
            zero_cross_phase_range=[p[z].min(),p[z].max()] if z.any() else None,
            phase_wrong_sign_fraction=col(r,'phase_wrong_sign').mean()/p.mean(),
            real_offset_explained_median=np.median(col(r,'real_offset_explained')),
            derivative_direction_share_median=np.median(col(r,'derivative_direction_share')),
            full_front_fraction=col(r,'full_front_fraction').mean(),primary_front_fraction=col(r,'primary_front_fraction').mean(),
            boundary_fraction=col(r,'boundary_fraction').mean(),cross_zero_population=col(r,'cross_zero_population').mean(),
            cap_feasible=int(valid.sum()),cap_phase_reduction_pct=(1-cap[valid].mean()/bcap[valid].mean())*100 if valid.any() else None,
            cap_phase_reduction_ci95=ci(cap[valid],bcap[valid]) if valid.any() else None,
            cap_phase_better=int(sum(cap[valid]<bcap[valid])),
            phase_min_reduction_pct=(1-col(fr,'phase_min').mean()/col(fr,'baseline_phase_min').mean())*100,
            dominating_available=sum(v['dominates_baseline_available']=='True' for v in fr),
            selected_worse_despite_dominating_available=sum(v['selected_worse_despite_dominating_available']=='True' for v in fr),
            metric_analysis=[],weight_sensitivity=[])
        for k in metrics:
            a=col(r,k);valid=np.isfinite(a);db=a-bc[k];vd=valid&np.isfinite(bc[k])
            sr['metric_analysis'].append(dict(metric=k,n=int(valid.sum()),mean=np.mean(a[valid]) if valid.any() else None,
                phase_rho=rho(a[valid],p[valid]) if valid.sum()>2 else None,
                magnitude_rho=rho(a[valid],m[valid]) if valid.sum()>2 else None,
                partial_phase_rho_controlling_magnitude=partial(a[valid],p[valid],m[valid]) if valid.sum()>3 else None,
                auxiliary_better=int(sum(db[vd]<0)),auxiliary_better_phase_worse=int(sum((db[vd]<0)&(p[vd]>bp[vd]))),
                auxiliary_delta_phase_delta_rho=rho(db[vd],p[vd]-bp[vd]) if vd.sum()>2 else None))
        for factor in [.5,1.,2.,4.]:
            rr=[v for v in sens if v['group']==g and float(v['factor'])==factor]
            sr['weight_sensitivity'].append(dict(factor=factor,magnitude_reduction_pct=(1-col(rr,'magnitude').mean()/col(rr,'baseline_magnitude').mean())*100,
                phase_reduction_pct=(1-col(rr,'phase').mean()/col(rr,'baseline_phase').mean())*100))
        summary.append(sr)
    correlations=[]
    for context in ['initial','own_final']:
        for g in (['baseline'] if context=='initial' else groups):
            for k in metrics:
                vals=col([r for r in cs if r['context']==context and r['group']==g and r['metric']==k],'phase_rho');vals=vals[np.isfinite(vals)]
                correlations.append(dict(context=context,group=g,metric=k,valid_seeds=len(vals),median=np.median(vals) if len(vals) else None))
    conflict=[]
    for k in metrics:
        v=col([r for r in cf if r['metric']==k],'conflict_fraction');v=v[np.isfinite(v)]
        conflict.append(dict(metric=k,valid_seeds=len(v),mean=v.mean() if len(v) else None,median=np.median(v) if len(v) else None))
    result[case]=dict(groups=summary,correlations=correlations,initial_primary_dominance_conflicts=conflict,
        counterexample=json.loads((ROOT/case/'real_shift_counterexample.json').read_text()),
        target_geometry=json.loads((ROOT/case/'target_geometry.json').read_text()))
(ROOT/'SUMMARY.json').write_text(json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
for case,data in result.items():
    for r in data['groups']:
        print(json.dumps(clean(dict(case=case,**{k:r[k] for k in ['group','phase_reduction_pct','phase_better','zero_cross_count','cap_feasible','cap_phase_reduction_pct','phase_min_reduction_pct','selected_worse_despite_dominating_available']}))))

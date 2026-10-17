"""Read-only mechanism diagnostics on archived populations; no optimization."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,sys,json,csv,hashlib,time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

def csvout(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(np.integer,np.bool_)):return x.item()
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x
def dump(p,x):p.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def rho(a,b):
    if np.ptp(a)==0 or np.ptp(b)==0:return np.nan
    return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])
def partial(a,b,c):
    if any(np.ptp(v)==0 for v in [a,b,c]):return np.nan
    a,b,c=map(rankdata,[a,b,c]);X=np.column_stack([np.ones(len(c)),c])
    ar=a-X@np.linalg.lstsq(X,a,rcond=None)[0];br=b-X@np.linalg.lstsq(X,b,rcond=None)[0]
    return float(np.corrcoef(ar,br)[0,1]) if min(np.std(ar),np.std(br))>1e-12 else np.nan

ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
source=Path(args.source).resolve();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
os.chdir(source);sys.path.insert(0,str(source))
import experiment as exp
from legacy_objectives import Objectives,derivative,crossings,region_matches
from length_metrics import GlobalObjectives as Length
cfg,f=exp.config(source/'config.json')
if source.name=='case1':
    from legacy_objectives import impedance,TARGET
    target=impedance(f,TARGET)[0]
else:
    from case2 import impedance,load_target
    _,target=load_target(cfg['target_data'])
names=['reactance_crossing_direction_consistency','characteristic_frequency_shift_error_decades','transition_speed_relative_error']
legacy=Objectives(f,aux=names,target_z=target);length=Length(f,aux=('cumulative_length_logpolar',),target_z=target)
groups=[exp.group_name(g) for g in exp.groups(cfg)];x=np.log10(f);alpha=cfg['final_selection']['phase_weight']
phase=np.angle(target);gt=derivative(target,x);qt=gt/target
angular_scale=np.sqrt(np.mean(qt.imag**2));vector_scale=abs(gt)+legacy.ev
metric_names=['I03','II01','II02','R05','angular_rate','complex_tangent','phase_sign','cross_position']

def metrics(z):
    z=np.atleast_2d(z);base=legacy.evaluate_spectra(z);gv=derivative(z,x,axis=1)
    q=gv/z;pe=(np.angle(z)-phase)**2
    bad=np.signbit(z.imag)!=np.signbit(target.imag)
    angular=np.sqrt(np.mean((q.imag-qt.imag)**2,axis=1))/angular_scale
    vector=np.sqrt(np.mean(abs(gv-gt)**2/vector_scale**2,axis=1))
    scalar=np.mean((abs(gv)-abs(gt))**2/vector_scale**2,axis=1)
    orient=np.maximum(vector**2-scalar,0)
    direction_share=np.divide(orient,vector**2,out=np.zeros_like(orient),where=vector>0)
    res=z-target;c=res.real.mean(axis=1);den=np.mean(abs(res)**2,axis=1)
    offset_share=np.divide(c*c,den,out=np.zeros_like(c),where=den>0)
    cp=[]
    for zz in z:
        events,types=crossings(zz.imag,legacy.u);cost=[]
        for ev,typ,mask in zip(legacy.cross_events,legacy.cross_types,region_matches(events,legacy.cross_edges)):
            match=events[mask&(types==typ)]
            if not len(match):cost=[];break
            cost.append(float(np.min(abs(match-ev))))
        cp.append(np.mean(cost) if cost else np.nan)
    return np.column_stack([base[:,2:],length.evaluate_spectra(z)[:,2],angular,vector,bad.mean(axis=1),cp]),dict(
        magnitude=base[:,0],phase=base[:,1],phase_wrong_sign=np.mean(pe*bad,axis=1),
        phase_correct_sign=np.mean(pe*(~bad),axis=1),derivative_direction_share=direction_share,
        real_offset_explained=offset_share,real_offset=c)

selected=[];front=[];sensitivity=[];corr=[];conflicts=[];profiles=[];examples=[];raw_hashes={}
start=time.time()
seeds=list(range(2000,2100))
for seed in seeds:
    populations=[]
    for name in groups:
        path=source/'runs'/name/f'seed_{seed:03d}'/'final_population.npz'
        with np.load(path) as d:F=d['objectives'].copy();pars=d['log_parameters'].copy();idx=int(d['selected_index'])
        raw_hashes[str(path.relative_to(source))]=hashlib.sha256(path.read_bytes()).hexdigest()
        z=impedance(f,10**pars);m,diag=metrics(z)
        np.testing.assert_allclose(np.c_[diag['magnitude'],diag['phase']],F[:,:2],rtol=1e-11,atol=1e-11)
        assert idx==int(np.argmin(F[:,0]+alpha*F[:,1]))
        r=dict(case=source.name,seed=seed,group=name)
        r.update({k:float(v[idx]) for k,v in diag.items()});r.update({k:float(m[idx,j]) for j,k in enumerate(metric_names)})
        r.update(cross_zero_population=float(np.mean(m[:,0]==0)),full_front_fraction=len(exp.nd_indices(F))/len(F),
            primary_front_fraction=len(exp.nd_indices(F[:,:2]))/len(F),
            boundary_fraction=float(np.mean(np.isclose(pars[idx],exp.LOWER,atol=1e-6)|np.isclose(pars[idx],exp.UPPER,atol=1e-6))))
        selected.append(r);populations.append((F,idx,z,m))
        for j,k in enumerate(metric_names):
            valid=np.isfinite(m[:,j])
            corr.append(dict(case=source.name,seed=seed,group=name,context='own_final',metric=k,n=int(valid.sum()),
                             phase_rho=rho(m[valid,j],F[valid,1]) if valid.sum()>2 else np.nan))
    bF,bi,_,_=populations[0];cap=bF[bi,0];basephase=bF[bi,1]
    bp=float(bF[bF[:,0]<=cap,1].min())
    for name,(F,idx,z,m) in zip(groups,populations):
        feasible=F[:,0]<=cap;dom=(F[:,0]<=cap)&(F[:,1]<=basephase)&((F[:,0]<cap)|(F[:,1]<basephase))
        cp=float(F[feasible,1].min()) if feasible.any() else np.nan
        front.append(dict(case=source.name,seed=seed,group=name,baseline_magnitude_cap=cap,
            feasible=bool(feasible.any()),phase_at_cap=cp,baseline_phase_at_cap=bp,
            phase_min=float(F[:,1].min()),baseline_phase_min=float(bF[:,1].min()),
            dominates_baseline_available=bool(dom.any()),selected_phase_worse=bool(F[idx,1]>basephase),
            selected_worse_despite_dominating_available=bool(dom.any() and F[idx,1]>basephase)))
        for factor in [.5,1.,2.,4.]:
            ix=int(np.argmin(F[:,0]+factor*alpha*F[:,1]));bb=int(np.argmin(bF[:,0]+factor*alpha*bF[:,1]))
            sensitivity.append(dict(case=source.name,seed=seed,group=name,factor=factor,magnitude=float(F[ix,0]),phase=float(F[ix,1]),
                                    baseline_magnitude=float(bF[bb,0]),baseline_phase=float(bF[bb,1])))
        if seed==seeds[0]:profiles.append([])
    # Same initial populations isolate metric ordering from group-specific evolution.
    init=exp.initial_population(cfg,seed);z0=impedance(f,10**init);im,di=metrics(z0);P=np.c_[di['magnitude'],di['phase']]
    better=(P[:,None,:]<=P[None,:,:]).all(axis=2)&(P[:,None,:]<P[None,:,:]).any(axis=2)
    for j,k in enumerate(metric_names):
        valid=np.isfinite(im[:,j]);pairs=better&valid[:,None]&valid[None,:]
        conflict=(im[:,j,None]>im[None,:,j])&pairs
        conflicts.append(dict(case=source.name,seed=seed,metric=k,pairs=int(pairs.sum()),
                              conflict_fraction=float(conflict.sum()/pairs.sum()) if pairs.any() else np.nan))
        corr.append(dict(case=source.name,seed=seed,group='baseline',context='initial',metric=k,n=int(valid.sum()),
                         phase_rho=rho(im[valid,j],P[valid,1]) if valid.sum()>2 else np.nan))
    if (seed-1999)%20==0:print(json.dumps(dict(case=source.name,completed=seed-1999,seconds=time.time()-start)),flush=True)

for name,rows in [('selected',selected),('front',front),('sensitivity',sensitivity),('correlations',corr),('conflicts',conflicts)]:csvout(out/(name+'.csv'),rows)
# Analytical common-null-direction illustration: spectra only, never treated as circuit fits.
null=[]
for multiplier in [0.,.1,1.,10.]:
    c=multiplier*np.median(abs(target));m,d=metrics(target+c)
    null.append(dict(multiplier=multiplier,real_shift_ohm=c,**{k:v[0] for k,v in d.items()},**{k:m[0,j] for j,k in enumerate(metric_names)}))
dump(out/'real_shift_counterexample.json',null)
dump(out/'target_geometry.json',dict(case=source.name,angular_rate_scale=angular_scale,
     target_angular_share_median=np.median(qt.imag**2/(abs(qt)**2)),target_angular_share_mean=np.mean(qt.imag**2/(abs(qt)**2)),
     derivative_identity_error=np.max(abs(abs(gt)**2-abs(target)**2*(qt.real**2+qt.imag**2))),
     target_crossings=len(legacy.cross_events),node_gap_decades=np.diff(x).tolist()))
dump(out/'AUDIT.json',dict(status='complete',case=source.name,runs=600,seeds=100,no_optimization=True,
    verified_primary_objectives=True,raw_sha256=raw_hashes,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seconds=time.time()-start))

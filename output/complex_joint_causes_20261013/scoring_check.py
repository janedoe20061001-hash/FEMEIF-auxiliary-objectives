"""Fixed-candidate scoring, quadrature diagnosis and saved-population checks.
No new parameter search, no changes to historical scores or gates.
"""
from investigate import *
from scipy.stats import spearmanr
import time

def rank_scores():
    rows=[];correlations=[];contrib=[]
    for ident in ['mixed_resonators/base','mixed_resonators/lc_reciprocal_8p0']:
        d=target(ident);c=d['circuit'];features=Features(d)
        pool=[];labels=[]
        for method in CFG['methods']:
            for seed in CFG['seeds']:
                p,F,a=load(ident,method,seed);pool.append(p);labels.append((method,seed,a))
        pool=np.array(pool)
        # Level 8 same window/target scale and nodes for all frozen chosen candidates.
        scores,details=features.components(np.log10(pool),names=('D','H','A'),level=8)
        j0=[];amp=[];pha=[]
        for w in features.windows:
            xx=np.linspace(w['x'][0],w['x'][-1],1025);tt=np.linspace(0,1,len(xx));ref,_=c.response(10**xx)
            zz,_=c.response(10**xx,pool)
            val=average(abs(zz-ref)**2/w['scale']**2,tt);j0.append(val)
            aa=average((20*np.log10(abs(zz)/abs(ref)))**2,tt);pp=average((np.angle(zz)-np.angle(ref))**2,tt)
            amp.append(aa);pha.append(pp)
            for i,(method,seed,a) in enumerate(labels):
                if w['block']['label'] in ('P2','V3'):
                    di=next(r for r in details if r['label']==w['block']['label'])
                    contrib.append(dict(target=ident,method=method,seed=seed,block=w['block']['label'],
                        log_span=float(xx[-1]-xx[0]),J0=float(val[i]),local_mag=float(aa[i]),local_pha=float(pp[i]),
                        D=float(di['scores']['D'][i]),A=float(di['scores']['A'][i]),num_windows=len(features.windows)))
        j0=np.mean(j0,axis=0);amp=np.mean(amp,axis=0);pha=np.mean(pha,axis=0)
        for i,(method,seed,a) in enumerate(labels):
            relevant=[r for r in a['amplitude_events'] if r['event_id'] in ('P2','V3')]
            rows.append(dict(target=ident,method=method,seed=seed,accepted=a['accepted'],J0=j0[i],D=scores['D'][i],A=scores['A'][i],H=scores['H'][i],
                equal_window_mag=amp[i],equal_window_pha=pha[i],mag=a['objective'][0],pha=a['objective'][1],pair_failed=sum(not r['complete'] for r in relevant)))
        for score in ['J0','D','A','H','mag','pha','equal_window_mag','equal_window_pha']:
            rr=[r for r in rows if r['target']==ident]
            for other in ['pair_failed','J0']:
                rho=spearmanr([r[score] for r in rr],[r[other] for r in rr]).statistic
                correlations.append(dict(target=ident,score=score,against=other,n=len(rr),spearman_rho=rho,note='Descriptive fixed selected pool, dependent methods/seeds; no inferential p-value'))
        print('fixed candidate scoring',ident,flush=True)
    save('FIXED_CANDIDATE_SCORES.csv',rows);save('SCORE_CORRELATIONS.csv',correlations);save('WINDOW_SCORE.csv',contrib)

def population_check():
    rows=[]
    for ident,new in [('mixed_resonators/base','D'),('mixed_resonators/lc_reciprocal_8p0','A')]:
        d=target(ident);c=d['circuit'];events={e['event_id']:e for e in d['record']['events']};bounds=(4.65,4.78) if ident.endswith('/base') else (4.58,4.66)
        x=np.linspace(*bounds,2049);t,dt=c.response(10**x)
        for seed in CFG['seeds']:
            for method in ['MAG_PHA',new]:
                dd=DATA/ident/method/f'seed_{seed}'
                with np.load(dd/'generation_0300.npz') as a: X=a['X'];F=a['F'];ix=int(a['selected_index'])
                zz,_=c.response(10**x,10**X)
                # Dense local complex error, not event threshold success.
                local=np.trapezoid(abs(zz-t)**2,x=x,axis=1)/(x[-1]-x[0])/max(abs(t))**2
                near=(F[:,0]<=1.1*F[ix,0])&(F[:,1]<=1.1*F[ix,1])
                best=int(np.argmin(local));nearbest=int(np.argmin(np.where(near,local,np.inf)))
                rows.append(dict(target=ident,method=method,seed=seed,selected_local=local[ix],best_population_local=local[best],
                    best_local_mag=F[best,0],best_local_pha=F[best,1],selected_mag=F[ix,0],selected_pha=F[ix,1],
                    best_within_10pct_both_local=local[nearbest],near_candidates=int(sum(near)),
                    selected_index=ix,best_local_index=best,near_best_index=nearbest))
    save('POPULATION_LOCAL.csv',rows)

def case1():
    ident='case1';d=target(ident);c=d['circuit'];features=Features(d);rows=[];branch=[]
    initial=OUT/'CASE1_GRID_CHECK.csv'
    if initial.exists() and not (OUT/'CASE1_INITIAL_DEEP_D.csv').exists():
        (OUT/'CASE1_INITIAL_DEEP_D.csv').write_bytes(initial.read_bytes())
    deep={('D',45000),('A',45000),('A',45006),('H_A',45000)}
    # All 56 newly proposed metric combinations; retain failed gates.
    for method in ['D','H','A','D_H','D_A','H_A','D_H_A']:
        for seed in CFG['seeds']:
            p,F,a=load(ident,method,seed);levels={}
            for level in (1,8,16,32) if (method,seed) in deep else (1,):
                sc=features.score(np.log10(p),method,level=level)[0];levels[level]=float(sc)
            ag=features.anchored_geometry(np.log10(p),level=32) if (method,seed) in deep else None
            anch=float(np.mean([ag[n][0] for n in method.split('_')])) if ag else None
            coarse=np.unique(np.concatenate([w['x'] for w in features.windows]));zc,_=c.response(10**coarse,p)
            v=p.reshape(6,3);fc=1/(2*np.pi*np.sqrt(v[:,1]*v[:,2]));q=v[:,0]*np.sqrt(v[:,2]/v[:,1])
            valid=(np.log10(fc)>=coarse[0])&(np.log10(fc)<=coarse[-1]);zfc,_=c.response(fc[valid],p)
            zfmax=float(max(abs(zfc))) if len(zfc) else float('nan')
            zcmax=float(max(abs(zc)));gap=float(20*np.log10(zfmax/zcmax))
            for k,(freq,Q) in enumerate(zip(fc,q)):
                xx=np.log10(freq)
                ww=next((w for w in features.windows if w['x'][0]<=xx<=w['x'][-1]),None)
                if ww is None:continue
                interval=(ww['x'][-1]-ww['x'][0])/128
                approximate_width=1/(Q*np.log(10))
                branch.append(dict(method=method,seed=seed,branch=k+1,fc_hz=freq,Q=Q,
                    approximate_branch_3db_width_decades=approximate_width,coarse_step_decades=interval,
                    width_over_coarse_step=approximate_width/interval,nearest_coarse_distance_over_width=float(min(abs(ww['x']-xx))/approximate_width),
                    R_ohm=v[k,0],window_scale_ohm=ww['scale'],R_over_window_scale=v[k,0]/ww['scale'],window=ww['block']['label']))
            rows.append(dict(method=method,seed=seed,old_accepted=a['accepted'],old_converged=a['converged'],
                old_aux_agrees=a['aux_training_agrees'],score_1=levels[1],score_8=levels.get(8),score_16=levels.get(16),score_32=levels.get(32),anchored_32=anch,
                old_anchored=float(np.mean([a['anchored_geometry'][n] for n in method.split('_')])),deep_recomputed=(method,seed) in deep,
                coarse_max_ohm=zcmax,candidate_branch_center_max_ohm=zfmax,center_vs_coarse_db=gap,
                relative_16_32=abs(levels[32]-levels[16])/max(abs(levels[32]),1e-12) if ag else None,relative_32_anchor=abs(levels[32]-anch)/max(abs(anch),1e-12) if ag else None,
                note='New diagnosis only; original acceptance not changed; anchors are not automatically truth'))
        print('case1 scoring',method,flush=True)
        save('CASE1_GRID_CHECK.csv',rows);save('CASE1_BRANCH_GRID.csv',branch)

if __name__=='__main__':
    if '--case1' not in sys.argv:rank_scores();population_check()
    case1()

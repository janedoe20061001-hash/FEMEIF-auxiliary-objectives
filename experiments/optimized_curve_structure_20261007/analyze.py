"""Validate the added arm, then diagnose every paired per-seed selected curve."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse,csv,hashlib,json,time
import numpy as np
from model import Model
from diagnostics import dense_grid,diagnose,interaction,loss_location
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
COLS=['magnitude_mse_db2','phase_mse_rad2','wsmse','mean_distance_ohm']
GROUPS=['wsmse','mean_distance','dual_mse','dual_balanced']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [clean(v) for v in x]
        if isinstance(x,np.ndarray):return clean(x.tolist())
        if isinstance(x,np.integer):return int(x)
        if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
        if isinstance(x,np.bool_):return bool(x)
        return x
    Path(p).write_text(json.dumps(clean(v),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def csvout(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def mean(vals):
    a=[float(v) for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(a)) if a else None
def check_dual(r,m,runs,scalar_runs):
    params=[];metrics=[];front_metrics=[];front_size=[];checked=0;selectrows=[]
    for seed in range(r['seed_start'],r['seed_start']+r['seed_count']):
        d=runs/r['id']/f'seed_{seed}';s=json.loads((d/'summary.json').read_text())
        assert s['status']=='complete' and s['n_obj']==2 and s['generations']==300 and s['evaluations']==r['population']*300
        x0=np.load(d/'initial_log_parameters.npy'); old=np.load(scalar_runs/r['id']/'wsmse'/f'seed_{seed}'/'initial_log_parameters.npy')
        np.testing.assert_array_equal(x0,old);np.testing.assert_array_equal(x0,np.random.default_rng(seed).uniform(m.lower,m.upper,x0.shape))
        assert hashlib.sha256(x0.tobytes()).hexdigest()==s['initial_population_sha256']
        with np.load(d/'final_population.npz') as p:
            x=p['log_parameters'];assert x.shape==(r['population'],len(m.lower)) and np.isfinite(x).all() and np.all(x>=m.lower) and np.all(x<=m.upper)
            v=m.score(x);np.testing.assert_allclose(v,p['cross_metrics'],rtol=1e-10,atol=1e-8);np.testing.assert_allclose(v[:,:2],p['optimization_objective'],rtol=1e-10,atol=1e-8)
            F=p['optimization_objective'];dominates=np.all(F[:,None,:]<=F[None,:,:],axis=2)&np.any(F[:,None,:]<F[None,:,:],axis=2)
            front=np.flatnonzero(~dominates.any(axis=0));np.testing.assert_array_equal(np.sort(p['front_indices']),front)
            scores={'min_wsmse':v[:,2],'balanced_minmax':np.maximum(v[:,0],m.alpha*v[:,1]),'min_distance':v[:,3],'min_magnitude':v[:,0],'min_phase':v[:,1]}
            selected={name:int(front[np.argmin(a[front])]) for name,a in scores.items()}
            assert selected==s['selection_indices'];assert int(p['selected_index'])==selected['min_wsmse']
            for rule,ix in selected.items():selectrows.append(dict(circuit=r['id'],seed=seed,rule=rule,front_size=len(front),**{k:float(b) for k,b in zip(COLS,v[ix])}))
            params.append([x[selected['min_wsmse']],x[selected['balanced_minmax']]]);metrics.append([v[selected['min_wsmse']],v[selected['balanced_minmax']]])
            front_metrics.append(v[front]);front_size.append(len(front));checked+=len(x)
            np.testing.assert_allclose(v[selected['min_wsmse']],p['selected_metrics'],rtol=1e-10,atol=1e-8)
        hs=list(csv.DictReader((d/'history.csv').open(encoding='utf-8-sig')));assert len(hs)==300
        assert [int(h['evaluations']) for h in hs]==[r['population']*g for g in range(1,301)]
        np.testing.assert_allclose([float(hs[-1][k]) for k in COLS],metrics[-1][0],rtol=1e-10,atol=1e-8)
    return np.array(params),np.array(metrics),checked,selectrows
def summarize(rows):
    exclude=['circuit','family','condition','seed','group','grid','dense_points','grid_stable','worst_amplitude_band','worst_phase_band']
    keys=[k for k in rows[0] if k not in exclude];means=[];pairs=[];rng=np.random.default_rng(20261007)
    for ident in dict.fromkeys(r['circuit'] for r in rows):
        for group in GROUPS:
            rr=[r for r in rows if r['circuit']==ident and r['group']==group]
            means.append(dict(circuit=ident,family=rr[0]['family'],condition=rr[0]['condition'],group=group,seeds=len(rr),grid_stable_seeds=sum(r['grid_stable'] for r in rr),**{k:mean([r.get(k) for r in rr]) for k in keys}))
        a={r['seed']:r for r in rows if r['circuit']==ident and r['group']=='wsmse'}
        for group in GROUPS[1:]:
            b={r['seed']:r for r in rows if r['circuit']==ident and r['group']==group}
            bootstrap_cache={}
            for key in keys:
                seeds=[s for s in a if a[s].get(key) is not None and b[s].get(key) is not None]
                if not seeds:continue
                delta=np.array([b[s][key]-a[s][key] for s in seeds],float);n=len(delta)
                cache_key=tuple(seeds)
                if cache_key not in bootstrap_cache:bootstrap_cache[cache_key]=rng.integers(0,n,(10000,n))
                boots=bootstrap_cache[cache_key]; ci=np.quantile(delta[boots].mean(axis=1),[.025,.975])
                pairs.append(dict(circuit=ident,group=group,metric=key,paired_seeds=n,mean_difference=float(delta.mean()),ci95_low=float(ci[0]),ci95_high=float(ci[1]),group_lower_seeds=int(np.sum(delta<0))))
    return means,pairs
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dual-runs',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--pilot',action='store_true');ap.add_argument('--circuit');args=ap.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False);(out/'selected').mkdir();(out/'profiles').mkdir()
    records=json.loads((HERE/'circuits.json').read_text(encoding='utf8'));manifest=json.loads((args.dual_runs/'manifest.json').read_text());status=json.loads((args.dual_runs/'status.json').read_text())
    assert status['status']=='complete' and status['completed']==1160 and status['failed']==0
    for name,value in manifest['source_sha256'].items():assert sha(HERE/name)==value,(name,'source changed')
    assert manifest['n_obj']==2 and manifest['planned_evaluations']==45060000
    scalar=ROOT/'output/primary_loss_comparison_20261007';scalar_runs=ROOT/'experiments/primary_loss_comparison_20261007/server_run/retrieved/runs'
    scalar_manifest=json.loads((scalar_runs/'manifest.json').read_text());assert manifest['versions']=={k:scalar_manifest['versions'][k] for k in manifest['versions']}
    assert json.loads((scalar/'VALIDATION.json').read_text())['passed']
    if args.pilot:records=[r for r in records if r['id'] in ['case1','case2','rc_series_c1','rlc_bridge_c2']]
    if args.circuit:
        records=[r for r in records if r['id']==args.circuit];assert len(records)==1
    rows=[];bands=[];refinements=[];selectors=[];matches=[];checked=0;seed_count=0;start=time.time()
    for r in records:
        m=Model(r['id']);dx,dm,count,sr=check_dual(r,m,args.dual_runs,scalar_runs);checked+=count;selectors+=sr
        with np.load(scalar/'selected'/f'{r["id"]}.npz') as old:
            f=old['frequency'];target=old['target'];seeds=old['seeds'];np.testing.assert_array_equal(f,m.f);np.testing.assert_array_equal(target,m.target)
            np.testing.assert_array_equal(seeds,np.arange(r['seed_start'],r['seed_start']+r['seed_count']))
            x=np.concatenate([old['log_parameters'],dx],axis=1);metrics=np.concatenate([old['metrics'],dm],axis=1);curves=m.spectrum(x.reshape(-1,len(m.lower))).reshape(len(seeds),4,len(f))
            np.testing.assert_allclose(curves[:,:2],old['curves'],rtol=1e-10,atol=1e-8);np.testing.assert_allclose(m.score_spectra(curves.reshape(-1,len(f))).reshape(len(seeds),4,4),metrics,rtol=1e-10,atol=1e-8)
        np.savez_compressed(out/'selected'/f'{r["id"]}.npz',frequency=f,target=target,seeds=seeds,groups=np.array(GROUPS),log_parameters=x,curves=curves,metrics=metrics)
        for si,seed in enumerate(seeds):
            seed_count+=1
            if args.pilot and si>=3:continue
            dense=target_grid=target_dense=z_dense=None
            if r['target_parameters'] is not None:
                dense=dense_grid(r,10**x[si],1); fine=dense_grid(r,10**x[si],2)
                target_dense=m.network(dense,np.array(r['target_parameters']))[0];target_fine=m.network(fine,np.array(r['target_parameters']))[0]
                z_dense=m.network(dense,10**x[si]);z_fine=m.network(fine,10**x[si])
            for gi,group in enumerate(GROUPS):
                base=dict(circuit=r['id'],family=r['family'],condition=r['condition'],seed=int(seed),group=group)
                sampled,_,_=diagnose(f,target,curves[si,gi]);loc=loss_location(target,curves[si,gi]);inter=interaction(m,x[si,gi]) if gi<3 else {'interaction_eta':None,'additive_cancellation':None,'interaction_rms_ohm':None,'total_rms_ohm':None,'additive_rms_ohm':None,'parameter_log10_rms':None}
                stable=True
                if dense is not None:
                    coarse,_,_=diagnose(dense,target_dense,z_dense[gi]);diag,bb,matched=diagnose(fine,target_fine,z_fine[gi]);grid='analytic_dense'
                    changes={k:abs(diag[k]-coarse[k]) for k in ['amplitude_rmse_db','phase_rmse_deg','mean_distance_logweighted_ohm','max_amplitude_error_db','max_phase_error_deg']}
                    dc=max(changes['amplitude_rmse_db']/max(diag['amplitude_rmse_db'],.1),changes['phase_rmse_deg']/max(diag['phase_rmse_deg'],.5),changes['mean_distance_logweighted_ohm']/max(diag['mean_distance_logweighted_ohm'],1e-8))
                    event_diff=sum(abs(diag[k]-coarse[k]) for k in ['amp_missed','amp_extra','phase_missed','phase_extra','cross_missed','cross_extra'])
                    stable=dc<=.02 and event_diff==0
                    refinements.append(dict(**base,coarse_points=len(dense),fine_points=len(fine),max_relative_integral_change_with_floors=dc,event_count_discrepancy=event_diff,stable=stable,**changes))
                    diag.update(dense_minus_sampled_amplitude_rmse_db=diag['amplitude_rmse_db']-sampled['amplitude_rmse_db'],dense_minus_sampled_phase_rmse_deg=diag['phase_rmse_deg']-sampled['phase_rmse_deg'],dense_extra_fit_amp_events=diag['amp_fit_events']-sampled['amp_fit_events'],dense_extra_target_amp_events=diag['amp_target_events']-sampled['amp_target_events'])
                    points=len(fine)
                else:
                    diag,bb,matched=diagnose(f,target,curves[si,gi]);grid='measured_points';points=len(f)
                    diag.update(dense_minus_sampled_amplitude_rmse_db=None,dense_minus_sampled_phase_rmse_deg=None,dense_extra_fit_amp_events=None,dense_extra_target_amp_events=None)
                row=dict(**base,grid=grid,dense_points=points,grid_stable=stable,**{k:float(v) for k,v in zip(COLS,metrics[si,gi])},**diag,**loc,**inter)
                rows.append(row);bands.extend(dict(**base,grid=grid,**b) for b in bb);matches.extend(dict(**base,**match) for match in matched)
            if si==0 or (args.pilot and si==2):print(json.dumps(dict(circuit=r['id'],seed=int(seed),elapsed_seconds=time.time()-start)),flush=True)
        print(json.dumps(dict(circuit_finished=r['id'],rows=len(rows),elapsed_seconds=time.time()-start)),flush=True)
    csvout(out/'per_seed_diagnostics.csv',rows);csvout(out/'band_diagnostics.csv',bands);csvout(out/'grid_refinement.csv',refinements);csvout(out/'dual_selection_sensitivity.csv',selectors);csvout(out/'feature_matches.csv',matches)
    means,pairs=summarize(rows);csvout(out/'group_means.csv',means);csvout(out/'paired_comparisons.csv',pairs)
    dump(out/'SUMMARY.json',dict(means=means,paired_comparisons=pairs,rows=len(rows),groups=GROUPS,dual_final_individuals_recomputed=checked,paired_seed_runs=seed_count,grid_unstable_rows=sum(not r['stable'] for r in refinements),grid_refinement_rows=len(refinements),pilot=args.pilot))
    if not args.pilot:
        expected=sum(r['seed_count'] for r in records)
        assert checked==sum(r['seed_count']*r['population'] for r in records) and len(rows)==4*expected and seed_count==expected
    dump(out/'VALIDATION.json',dict(passed=True,pilot=args.pilot,dual_final_individuals_recomputed=checked,primary_selected_curves=len(rows)*3//4,balanced_selected_curves=len(rows)//4,source_hashes=True,paired_initial_populations=True,independent_pareto_front_check=True,all_selection_rules_recomputed=True,selected_spectra_metrics_recomputed=True,generations_300=True,added_evaluations=45060000,case2_all_points_retained=True,grid_unstable_rows=sum(not r['stable'] for r in refinements),analysis_sha256=sha(__file__),diagnostics_sha256=sha(HERE/'diagnostics.py')))
    print(json.dumps(dict(passed=True,diagnostic_rows=len(rows),dual_individuals=checked,seconds=time.time()-start)))
if __name__=='__main__':main()

"""Verify new objective against two frozen paired baselines; diagnose original curves."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,argparse,json,csv,hashlib,importlib.util
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE/'frozen_scalar'))
from model import Model
spec=importlib.util.spec_from_file_location('previous_seed_independent',ROOT/'experiments/seed_replication_20261007/analyze.py');ind=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ind;spec.loader.exec_module(ind)
sys.path.insert(0,str(ROOT/'experiments/optimized_curve_structure_20261007'))
from diagnostics import diagnose,weights
from refined_grid import improved_dense_grid
COLS=['magnitude_mse_db2','phase_mse_rad2','wsmse','mean_distance_ohm','mean_logradius_distance'];GROUPS=['wsmse','mean_distance','logradius_distance'];CASES=['case1','case2'];SEEDS=np.arange(2026100700,2026100800)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csvout(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8-sig',newline='') as h:w=csv.DictWriter(h,fieldnames=keys);w.writeheader();w.writerows(rows)
def logscore(z,target):
    z=np.atleast_2d(z);rho=np.hypot(z.real,z.imag);rho0=np.hypot(target.real,target.imag)
    a=20*np.log10(rho);a0=20*np.log10(rho0)
    x=a*z.imag/rho;y=a*z.real/rho;tx=a0*target.imag/rho0;ty=a0*target.real/rho0
    return np.mean(np.hypot(x-tx,y-ty),axis=1)
def allscores(z,target,alpha):
    z=np.atleast_2d(z)
    return np.column_stack((ind.independent_metrics(z,target,alpha),logscore(z,target)))
def special(target,z):
    a0=20*np.log10(abs(target));a=20*np.log10(abs(z));dp=np.rad2deg(np.angle(z)-np.angle(target));near=abs(a0)<=1.;flip=np.signbit(a)!=np.signbit(a0)
    return dict(target_near_zero_db_points=int(near.sum()),near_zero_db_phase_rmse_deg=float(np.sqrt(np.mean(dp[near]**2))) if near.any() else None,near_zero_db_amplitude_rmse_db=float(np.sqrt(np.mean((a[near]-a0[near])**2))) if near.any() else None,opposite_logradius_sign_points=int(flip.sum()),opposite_sign_phase_rmse_deg=float(np.sqrt(np.mean(dp[flip]**2))) if flip.any() else None)
def diagnostic_job(job):
    ident,seed,r,params=job;f=np.asarray(r['frequency_hz']);target=np.asarray(r['target_real'])+1j*np.asarray(r['target_imag']);z=ind.independent_spectra(ident,f,10**params);results=[];bands=[];grid=[]
    for gi,g in enumerate(GROUPS):
        d,b,_=diagnose(f,target,z[gi]);d.update(special(target,z[gi]));results.append(dict(circuit=ident,seed=seed,group=g,grid='original',**d))
        bands.extend(dict(circuit=ident,seed=seed,group=g,grid='original',**v) for v in b)
    if ident=='case1':
        levels=[]
        for ref in [1,2]:
            ff=improved_dense_grid(r,10**params,ref);tt=ind.independent_spectra(ident,ff,r['target_parameters'])[0];zz=ind.independent_spectra(ident,ff,10**params);xx=np.log10(ff);level=[]
            for gi,g in enumerate(GROUPS):
                d,b,_=diagnose(ff,tt,zz[gi]);a=20*np.log10(abs(zz[gi]));a0=20*np.log10(abs(tt));u=np.stack((a*np.sin(np.angle(zz[gi])),a*np.cos(np.angle(zz[gi]))),axis=-1);t=np.stack((a0*np.sin(np.angle(tt)),a0*np.cos(np.angle(tt))),axis=-1)
                d['mean_logradius_distance_logweighted']=float(np.trapezoid(np.hypot((u-t)[:,0],(u-t)[:,1]),xx)/(xx[-1]-xx[0]));level.append(d)
            levels.append(level)
        for gi,g in enumerate(GROUPS):
            first,second=levels[0][gi],levels[1][gi];keys=['amplitude_rmse_db','phase_rmse_deg','mean_distance_logweighted_ohm','mean_logradius_distance_logweighted']
            changes={k:abs(first[k]-second[k])/max(abs(second[k]),1e-6) for k in keys}
            integral_stable=all(v<=.01 for v in changes.values());events_stable=all(first[k]==second[k] for k in ['amp_missed','amp_extra','phase_missed','phase_extra'])
            stable=integral_stable and events_stable
            grid.append(dict(circuit=ident,seed=seed,group=g,integral_stable=integral_stable,event_counts_stable=events_stable,grid_stable=stable,**{k+'_relative_change':v for k,v in changes.items()}))
            results.append(dict(circuit=ident,seed=seed,group=g,grid='analytic_dense',grid_stable=stable,**second))
    return results,bands,grid
def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--workers',type=int,default=8);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True);(out/'selected').mkdir(exist_ok=True)
    manifest=json.loads((a.runs/'manifest.json').read_text());status=json.loads((a.runs/'status.json').read_text());assert status['status']=='complete' and status['completed']==200 and status['failed']==0
    assert manifest['reference_ohm']==1 and manifest['seeds']==SEEDS.tolist()
    for name,h in manifest['source_sha256'].items():assert sha(HERE/name)==h
    oldmanifest=json.loads((ROOT/'experiments/seed_replication_20261007/server_run/retrieved/runs/manifest.json').read_text())
    for name,h in oldmanifest['source_sha256'].items():
        if name.startswith('frozen_scalar/'):assert sha(HERE/name)==h
    rows=[];jobs=[];data=np.zeros((100,2,3,5));checked=budget=0;records=[]
    for ci,c in enumerate(CASES):
        m=Model(c);records.append(m.record)
        with np.load(ROOT/'output/seed_replication_20261007/selected'/f'{c}.npz') as base:
            np.testing.assert_array_equal(base['seeds'],SEEDS);np.testing.assert_array_equal(base['frequency'],m.f);np.testing.assert_array_equal(base['target'],m.target)
            params=np.zeros((100,3,len(m.lower)));curves=np.zeros((100,3,len(m.f)),complex);params[:,:2]=base['log_parameters'];curves[:,:2]=base['curves']
            for si,seed in enumerate(SEEDS):
                for gi,g in enumerate(GROUPS[:2]):
                    zz=ind.independent_spectra(c,m.f,10**params[si,gi])[0];np.testing.assert_allclose(zz,curves[si,gi],rtol=1e-9,atol=1e-8);metrics=allscores(zz,m.target,m.alpha)[0];np.testing.assert_allclose(metrics[:4],base['metrics'][si,gi],rtol=1e-9,atol=1e-8);data[si,ci,gi]=metrics
                d=a.runs/c/f'seed_{seed}';sm=json.loads((d/'summary.json').read_text());assert sm['n_obj']==1 and sm['objective_column']==4 and sm['seed']==int(seed) and sm['evaluations']==m.record['population']*300;budget+=sm['evaluations']
                x0=np.load(d/'initial_log_parameters.npy');np.testing.assert_array_equal(x0,np.random.default_rng(int(seed)).uniform(m.lower,m.upper,(m.record['population'],len(m.lower))))
                for g in GROUPS[:2]:np.testing.assert_array_equal(x0,np.load(ROOT/'experiments/seed_replication_20261007/server_run/retrieved/runs'/c/g/f'seed_{seed}'/'initial_log_parameters.npy'))
                assert hashlib.sha256(x0.tobytes()).hexdigest()==sm['initial_population_sha256']
                with np.load(d/'final_population.npz') as pop:
                    x=pop['log_parameters'];assert x.shape==(m.record['population'],len(m.lower));assert np.isfinite(x).all() and np.all(x>=m.lower) and np.all(x<=m.upper)
                    z=ind.independent_spectra(c,m.f,10**x);metrics=allscores(z,m.target,m.alpha);np.testing.assert_allclose(metrics,pop['cross_metrics'],rtol=1e-9,atol=1e-8);np.testing.assert_allclose(metrics[:,4],pop['optimization_objective'][:,0],rtol=1e-9,atol=1e-8)
                    ix=int(pop['selected_index']);assert ix==sm['selected_index'];assert metrics[ix,4]<=metrics[:,4].min()+1e-8
                    # Verify exact original trigonometric objective selection as well.
                    amp=20*np.log10(abs(m.spectrum(x)));theta=np.angle(m.spectrum(x));ta=20*np.log10(abs(m.target));tp=np.angle(m.target)
                    trig=np.hypot(amp*np.sin(theta)-ta*np.sin(tp),amp*np.cos(theta)-ta*np.cos(tp)).mean(axis=1);assert ix==int(np.argmin(trig))
                    np.testing.assert_array_equal(pop['selected_log_parameters'],x[ix]);np.testing.assert_allclose(pop['selected_metrics'],metrics[ix],rtol=1e-9,atol=1e-8);np.testing.assert_allclose([sm[k] for k in COLS],metrics[ix],rtol=1e-9,atol=1e-8)
                    hs=list(csv.DictReader((d/'history.csv').open(encoding='utf-8-sig')));assert len(hs)==300;hv=np.asarray([float(h[COLS[4]]) for h in hs]);assert np.all(np.diff(hv)<=1e-10*np.maximum(1,hv[:-1]));np.testing.assert_allclose(hv.min(),metrics[ix,4],rtol=1e-9,atol=1e-8)
                    checked+=len(x);params[si,2]=x[ix];curves[si,2]=z[ix];data[si,ci,2]=metrics[ix]
                for gi,g in enumerate(GROUPS):rows.append(dict(circuit=c,seed=int(seed),group=g,**dict(zip(COLS,data[si,ci,gi].tolist()))))
                jobs.append((c,int(seed),m.record,params[si]))
            np.savez_compressed(out/'selected'/f'{c}.npz',frequency=m.f,target=m.target,seeds=SEEDS,groups=np.asarray(GROUPS),log_parameters=params,curves=curves,metrics=data[:,ci])
        print(f'Verified {c}: 100 new runs and both baseline selections',flush=True)
    assert checked==35000 and budget==10500000
    diagnostics=[];bands=[];grid=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        done=0
        for f in as_completed([pool.submit(diagnostic_job,j) for j in jobs]):
            dd,bb,gg=f.result();diagnostics.extend(dd);bands.extend(bb);grid.extend(gg);done+=1
            if done%25==0:print(f'Diagnosed {done}/200 paired seeds',flush=True)
    diagnostics.sort(key=lambda r:(r['circuit'],r['seed'],r['grid'],r['group']));bands.sort(key=lambda r:(r['circuit'],r['seed'],r['group'],r['band']));grid.sort(key=lambda r:(r['seed'],r['group']))
    rng=np.random.default_rng(20261007);draws=rng.integers(0,100,(10000,100));boot=data[draws].mean(axis=1);means=[];comparisons=[];structmeans=[];structcomp=[]
    for ci,c in enumerate(CASES):
        for gi,g in enumerate(GROUPS):means.append(dict(circuit=c,group=g,seeds=100,**dict(zip(COLS,data[:,ci,gi].mean(axis=0).tolist()))))
        for bi,baseline in enumerate(GROUPS[:2]):
            for k,name in enumerate(COLS):
                delta=data[:,ci,2,k]-data[:,ci,bi,k];interval=np.quantile(boot[:,ci,2,k]-boot[:,ci,bi,k],[.025,.975]).tolist()
                comparisons.append(dict(circuit=c,baseline=baseline,metric=name,new_minus_baseline=float(delta.mean()),ci95=interval,new_lower_seeds=int(np.sum(delta<0)),new_over_baseline=float(data[:,ci,2,k].mean()/data[:,ci,bi,k].mean())))
        for level in (['original','analytic_dense'] if c=='case1' else ['original']):
            lookup={(r['seed'],r['group']):r for r in diagnostics if r['circuit']==c and r['grid']==level};metrics=['amplitude_rmse_db','phase_rmse_deg','amp_missed','amp_extra','phase_missed','phase_extra','worst_band_phase_rmse_deg','mean_distance_logweighted_ohm']
            if level=='analytic_dense':metrics.append('mean_logradius_distance_logweighted')
            vals={g:np.asarray([[lookup[(int(s),g)][k] for k in metrics] for s in SEEDS]) for g in GROUPS}
            unstable=sum(not r['grid_stable'] for r in grid if r['circuit']==c) if level=='analytic_dense' else 0
            for g in GROUPS:structmeans.append(dict(circuit=c,grid=level,group=g,seeds=100,unstable_rows_across_groups=unstable,**dict(zip(metrics,vals[g].mean(axis=0).tolist()))))
            for baseline in GROUPS[:2]:
                delta=vals['logradius_distance']-vals[baseline];bb=delta[draws].mean(axis=1)
                for k,name in enumerate(metrics):structcomp.append(dict(circuit=c,grid=level,baseline=baseline,metric=name,new_minus_baseline=float(delta[:,k].mean()),ci95=np.quantile(bb[:,k],[.025,.975]).tolist(),numerically_stable=(unstable==0)))
    near=[]
    for c in CASES:
        for g in GROUPS:
            rr=[r for r in diagnostics if r['circuit']==c and r['group']==g and r['grid']=='original'];keys=['target_near_zero_db_points','near_zero_db_phase_rmse_deg','near_zero_db_amplitude_rmse_db','opposite_logradius_sign_points','opposite_sign_phase_rmse_deg'];near.append(dict(circuit=c,group=g,**{k:float(np.mean([r[k] for r in rr if r[k] is not None])) if any(r[k] is not None for r in rr) else None for k in keys}))
    csvout(out/'selected_metrics.csv',rows);csvout(out/'group_means.csv',means);csvout(out/'per_seed_structure.csv',diagnostics);csvout(out/'band_diagnostics.csv',bands);csvout(out/'grid_refinement.csv',grid);csvout(out/'special_regions.csv',near)
    result=dict(means=means,paired_comparisons=comparisons,structure_means=structmeans,structure_comparisons=structcomp,special_regions=near,new_runs=200,baseline_runs_reused=400,curves_compared=600,unstable_dense_rows=sum(not r['grid_stable'] for r in grid),bootstrap_replicates=10000,bootstrap_rng=20261007,reference_ohm=1.)
    dump(out/'SUMMARY.json',result);dump(out/'VALIDATION.json',dict(passed=True,new_runs=200,final_individuals_recomputed=checked,evaluations=budget,source_hashes=True,paired_initialization_with_both_baselines=True,baseline_metrics_and_selected_parameters_unchanged=True,independent_trig_and_component_formula=True,parameter_bounds=True,own_objective_final_and_history_minimum=True,dense_rows_checked=len(grid),dense_rows_unstable=result['unstable_dense_rows'],analysis_sha256=sha(Path(__file__))))
    print(json.dumps(dict(means=means,paired_comparisons=comparisons,structure_means=structmeans,special_regions=near,unstable_dense_rows=result['unstable_dense_rows']),ensure_ascii=False,indent=2))
if __name__=='__main__':main()

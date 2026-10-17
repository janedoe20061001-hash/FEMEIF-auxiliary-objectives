"""Independent audit and common-grid per-seed comparison of sampling arms."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,csv,hashlib,argparse,time,importlib.util,shutil
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE/'frozen_scalar'))
from model import Model
spec=importlib.util.spec_from_file_location('sampling_previous_independent',ROOT/'experiments/seed_replication_20261007/analyze.py');ind=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ind;spec.loader.exec_module(ind)
sys.path.insert(0,str(ROOT/'experiments/optimized_curve_structure_20261007'))
from diagnostics import diagnose
from refined_grid import improved_dense_grid
ARMS=['original','dense_interpolated','dense_true'];GROUPS=['wsmse','mean_distance','dual_mse'];COLS=['magnitude_mse_db2','phase_mse_rad2','wsmse','mean_distance_ohm']
KEYS=['amplitude_rmse_db','phase_rmse_deg','mean_distance_logweighted_ohm','amp_missed','amp_extra','phase_missed','phase_extra','worst_band_phase_rmse_deg','amplitude_bad_span_fraction','phase_bad_span_fraction','amp_mean_abs_shift_decades','amp_mean_abs_width_log2_ratio','phase_mean_abs_shift_decades','phase_mean_abs_width_log2_ratio','fit_sampling_gap_max_db']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csvout(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8-sig',newline='') as h:w=csv.DictWriter(h,fieldnames=keys);w.writeheader();w.writerows(rows)
def spectra(r,f,parameters):
    p=np.atleast_2d(parameters)
    if r['id']=='case1':return ind.independent_spectra('case1',f,p)
    ss=2j*np.pi*np.asarray(f)[None,:];values={name:p[:,i,None] for i,name in enumerate(r['parameter_names'])}
    def ev(node):
        if isinstance(node,str):
            v=values[node];return v+np.zeros_like(ss) if node[0]=='R' else (ss*v if node[0]=='L' else 1/(ss*v))
        zz=[ev(n) for n in node[1:]]
        if node[0]=='S':return np.add.reduce(zz)
        if node[0]=='P':
            numerator=np.multiply.reduce(zz);denominator=np.zeros_like(numerator)
            for j in range(len(zz)):denominator+=np.multiply.reduce([v for i,v in enumerate(zz) if i!=j])
            return numerator/denominator
        # Kirchhoff's two interior-node voltages, with one volt across the port.
        y=[1/v for v in zz];aa=y[0]+y[1]+y[4];bb=y[2]+y[3]+y[4];det=(y[0]+y[1])*(y[2]+y[3])+y[4]*sum(y[:4])
        vx=(bb*y[0]+y[4]*y[2])/det;vy=(y[4]*y[0]+aa*y[2])/det
        return 1/(y[1]*vx+y[3]*vy)
    return ev(r['tree'])
def scores(z,target,alpha,w=None):
    z=np.atleast_2d(z);a=20*np.log10(np.hypot(z.real,z.imag)/np.hypot(target.real,target.imag));p=np.arctan2(z.imag,z.real)-np.arctan2(target.imag,target.real);d=np.hypot(z.real-target.real,z.imag-target.imag)
    if w is None:ma=np.mean(a*a,axis=1);mp=np.mean(p*p,axis=1);md=np.mean(d,axis=1)
    else:ma=np.einsum('ij,j->i',a*a,w);mp=np.einsum('ij,j->i',p*p,w);md=np.einsum('ij,j->i',d,w)
    return np.column_stack((ma,mp,ma+alpha*mp,md))
def weighted(m,x,f,t,w):
    z=m.network(f,10**x);a=np.square(20*np.log10(abs(z))-20*np.log10(abs(t)))@w;p=np.square(np.angle(z)-np.angle(t))@w;d=abs(z-t)@w
    return np.column_stack((a,p,a+m.alpha*p,d))
def diagnostic_job(job):
    r,seed,parameters=job;levels=[];grid_results=[];final_bands=[];last_ref=None;stable_flags=None
    for refinement in [1,2,4,8]:
        f=improved_dense_grid(r,10**parameters,refinement);target=spectra(r,f,r['target_parameters'])[0];zz=spectra(r,f,10**parameters);current=[];all_bands=[]
        for i in range(len(parameters)):
            d,b,_=diagnose(f,target,zz[i]);current.append(d);all_bands.append(b)
        if levels:
            stable_flags=[];grid_results=[]
            for i,(coarse,fine) in enumerate(zip(levels[-1],current)):
                changes={k:abs(coarse[k]-fine[k])/max(abs(fine[k]),floor) for k,floor in [('amplitude_rmse_db',.1),('phase_rmse_deg',.5),('mean_distance_logweighted_ohm',1e-8)]}
                ec=all(coarse[k]==fine[k] for k in ['amp_missed','amp_extra','phase_missed','phase_extra']);stable=ec and max(changes.values())<=.01;stable_flags.append(stable)
                grid_results.append(dict(circuit=r['id'],seed=seed,selection=i,coarse_refinement=last_ref,fine_refinement=refinement,points=len(f),grid_stable=stable,event_counts_stable=ec,**{k+'_relative_change':v for k,v in changes.items()}))
            if all(stable_flags):final_bands=all_bands;levels.append(current);break
        levels.append(current);last_ref=refinement;final_bands=all_bands
    output=[];bands=[]
    with np.load(HERE/'grids'/f'{r["id"]}.npz') as gr:dense_training_f=gr['frequency']
    for i,d in enumerate(levels[-1]):
        if i<9:arm=ARMS[i//3];group=GROUPS[i%3];rule='primary'
        else:arm=ARMS[i-9];group='dual_mse';rule='balanced'
        training_f=np.asarray(r['frequency_hz']) if arm=='original' else dense_training_f
        training_fit=spectra(r,training_f,10**parameters[i])[0]
        d['fit_sampling_gap_max_db']=float(np.max(abs(20*np.log10(abs(zz[i]))-np.interp(np.log10(f),np.log10(training_f),20*np.log10(abs(training_fit))))))
        base=dict(circuit=r['id'],family=r['family'],seed=seed,arm=arm,group=group,rule=rule,grid='analytic_common',grid_stable=stable_flags[i])
        output.append(dict(**base,**d));bands.extend(dict(**base,**b) for b in final_bands[i])
    return output,bands,grid_results
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--workers',type=int,default=8);ap.add_argument('--circuit');ap.add_argument('--reuse-chunks',type=Path);a=ap.parse_args();out=a.output;out.mkdir(exist_ok=True,parents=True);(out/'selected').mkdir(exist_ok=True)
    config=json.loads((HERE/'CONFIG.json').read_text());manifest=json.loads((a.runs/'manifest.json').read_text());status=json.loads((a.runs/'status.json').read_text());assert status['failed']==0
    if a.circuit:
        assert status['status'] in ['running','complete'];config['records']=[r for r in config['records'] if r['circuit']==a.circuit];assert len(config['records'])==1
        config['planned_runs']=sum(len(r['seeds'])*6 for r in config['records']);config['planned_evaluations']=sum(len(r['seeds'])*6*r['population']*300 for r in config['records'])
    else:assert status['status']=='complete' and status['completed']==config['planned_runs']
    for name,h in manifest['source_sha256'].items():assert sha(HERE/name)==h,name
    rows=[];jobs=[];checked=budget=runs_checked=0;start=time.time();timings=[];records=[];baselines=[];cached_diagnostics=[];cached_bands=[];cached_grid=[];reused_chunks=[]
    helper_paths=[Path(__file__),ROOT/'experiments/seed_replication_20261007/analyze.py',ROOT/'experiments/optimized_curve_structure_20261007/diagnostics.py',ROOT/'experiments/optimized_curve_structure_20261007/refined_grid.py']
    helper_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in helper_paths}
    for cfg in config['records']:
        c=cfg['circuit'];m=Model(c);r=m.record;records.append(r);seeds=np.asarray(cfg['seeds']);nv=len(m.lower);params=np.zeros((len(seeds),3,3,nv));original_metrics=np.zeros((len(seeds),3,3,4));training_metrics=np.zeros_like(original_metrics);balanced=np.zeros((len(seeds),3,nv))
        baseline_path=ROOT/'output/optimized_curve_structure_20261007/selected'/f'{c}.npz';baselines.append(baseline_path)
        cache_path=a.reuse_chunks/c/'CHUNK_CACHE.json' if a.reuse_chunks else None
        if cache_path and cache_path.exists():
            cache=json.loads(cache_path.read_text(encoding='utf-8'));assert cache['circuit']==c and cache['helper_hashes']==helper_hashes and cache['baseline_sha256']==sha(baseline_path)
            for name,h in cache['run_input_sha256'].items():assert sha(a.runs/name)==h,name
            selected=cache_path.parent/'selected'/f'{c}.npz';assert sha(selected)==cache['selected_sha256'];shutil.copyfile(selected,out/'selected'/f'{c}.npz')
            rows.extend(cache['rows']);timings.extend(cache['timings']);cached_diagnostics.extend(cache['diagnostics']);cached_bands.extend(cache['bands']);cached_grid.extend(cache['grid']);checked+=cache['checked'];budget+=cache['budget'];runs_checked+=cache['runs_checked'];reused_chunks.append(c)
            print(f'Reused fully verified immutable chunk: {c}',flush=True);continue
        with np.load(baseline_path) as old:
            np.testing.assert_array_equal(old['seeds'],seeds);np.testing.assert_array_equal(old['frequency'],m.f);np.testing.assert_array_equal(old['target'],m.target);params[:,0]=old['log_parameters'][:,:3];balanced[:,0]=old['log_parameters'][:,3];original_metrics[:,0]=old['metrics'][:,:3];training_metrics[:,0]=old['metrics'][:,:3]
            vv=scores(spectra(r,m.f,10**params[:,0].reshape(-1,nv)),m.target,m.alpha).reshape(len(seeds),3,4);np.testing.assert_allclose(vv,original_metrics[:,0],rtol=1e-7,atol=1e-8)
        with np.load(HERE/'grids'/f'{c}.npz') as gr:f=gr['frequency'];w=gr['weights'];targets=[gr['target_interpolated'],gr['target_true']]
        for si,seed in enumerate(seeds):
            x0=np.random.default_rng(int(seed)).uniform(m.lower,m.upper,(r['population'],nv))
            for g in GROUPS[:2]:np.testing.assert_array_equal(x0,np.load(ROOT/'experiments/primary_loss_comparison_20261007/server_run/retrieved/runs'/c/g/f'seed_{seed}'/'initial_log_parameters.npy'))
            np.testing.assert_array_equal(x0,np.load(ROOT/'experiments/optimized_curve_structure_20261007/server_run/retrieved/runs'/c/f'seed_{seed}'/'initial_log_parameters.npy'))
            for ai,arm in enumerate(ARMS[1:],1):
                target=targets[ai-1]
                for gi,g in enumerate(GROUPS):
                    d=a.runs/c/arm/g/f'seed_{seed}';sm=json.loads((d/'summary.json').read_text());assert sm['status']=='complete' and sm['seed']==int(seed) and sm['arm']==arm and sm['group']==g
                    assert sm['generations']==300 and sm['evaluations']==r['population']*300;budget+=sm['evaluations'];runs_checked+=1;timings.append(dict(circuit=c,arm=arm,group=g,seconds=sm['elapsed_seconds']))
                    np.testing.assert_array_equal(np.load(d/'initial_log_parameters.npy'),x0);assert hashlib.sha256(x0.tobytes()).hexdigest()==sm['initial_population_sha256']
                    with np.load(d/'final_population.npz') as pop:
                        x=pop['log_parameters'];assert x.shape==(r['population'],nv);assert np.isfinite(x).all() and np.all(x>=m.lower) and np.all(x<=m.upper)
                        z=spectra(r,f,10**x);v=scores(z,target,m.alpha,w);vm=weighted(m,x,f,target,w);np.testing.assert_allclose(v,vm,rtol=1e-7,atol=1e-8);np.testing.assert_allclose(v,pop['training_metrics'],rtol=1e-7,atol=1e-8)
                        vo=scores(spectra(r,m.f,10**x),m.target,m.alpha);np.testing.assert_allclose(vo,pop['original_metrics'],rtol=1e-7,atol=1e-8)
                        F=pop['optimization_objective'];np.testing.assert_allclose(F,v[:,:2] if g=='dual_mse' else v[:,2 if g=='wsmse' else 3,None],rtol=1e-7,atol=1e-8)
                        # Independent nondominated front from stored objective values.
                        dom=np.all(F[:,None,:]<=F[None,:,:],axis=2)&np.any(F[:,None,:]<F[None,:,:],axis=2);front=np.flatnonzero(~dom.any(axis=0));np.testing.assert_array_equal(np.sort(pop['front_indices']),front)
                        ix=int(pop['selected_index']);bx=int(pop['balanced_index']);assert ix==sm['selected_index'] and bx==sm['balanced_index'];assert ix in front
                        if g=='dual_mse':
                            assert ix==int(front[np.argmin(pop['training_metrics'][front,2])]);assert bx==int(front[np.argmin(np.maximum(pop['training_metrics'][front,0],m.alpha*pop['training_metrics'][front,1]))]);balanced[si,ai]=x[bx]
                        else:assert ix==int(np.argmin(F[:,0]))
                        np.testing.assert_array_equal(pop['selected_log_parameters'],x[ix]);np.testing.assert_allclose(pop['selected_metrics'],v[ix],rtol=1e-7,atol=1e-8)
                        np.testing.assert_allclose([sm['training_metrics'][k] for k in COLS],v[ix],rtol=1e-7,atol=1e-8);np.testing.assert_allclose([sm['original_metrics'][k] for k in COLS],vo[ix],rtol=1e-7,atol=1e-8)
                        params[si,ai,gi]=x[ix];original_metrics[si,ai,gi]=vo[ix];training_metrics[si,ai,gi]=v[ix];checked+=len(x)
                    hs=list(csv.DictReader((d/'history.csv').open(encoding='utf-8-sig')));assert len(hs)==300;assert [int(h['evaluations']) for h in hs]==[r['population']*i for i in range(1,301)]
                    np.testing.assert_allclose([float(hs[-1][k]) for k in COLS],v[ix],rtol=1e-7,atol=1e-8)
                    if g!='dual_mse':
                        hv=np.asarray([float(h[COLS[2 if g=='wsmse' else 3]]) for h in hs]);assert np.all(np.diff(hv)<=1e-10*np.maximum(1,hv[:-1]))
            for ai,arm in enumerate(ARMS):
                for gi,g in enumerate(GROUPS):rows.append(dict(circuit=c,seed=int(seed),arm=arm,group=g,**dict(zip(COLS,original_metrics[si,ai,gi].tolist())),**{'training_'+k:v for k,v in zip(COLS,training_metrics[si,ai,gi].tolist())}))
            jobs.append((r,int(seed),np.r_[params[si].reshape(9,nv),balanced[si]]))
        curves=spectra(r,m.f,10**params.reshape(-1,nv)).reshape(len(seeds),3,3,len(m.f));np.savez_compressed(out/'selected'/f'{c}.npz',frequency=m.f,target=m.target,seeds=seeds,arms=np.asarray(ARMS),groups=np.asarray(GROUPS),log_parameters=params,curves=curves,original_metrics=original_metrics,training_metrics=training_metrics,balanced_log_parameters=balanced)
        print(f'Verified {c}; {runs_checked}/{config["planned_runs"]} new runs',flush=True)
    assert checked*300==budget==config['planned_evaluations'] and runs_checked==config['planned_runs']
    csvout(out/'selected_metrics.csv',rows);dump(out/'AUDIT.json',dict(passed=True,new_runs=runs_checked,final_individuals_recomputed=checked,evaluations=budget,baseline_sources_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in baselines},analysis_sha256=sha(Path(__file__))))
    diagnostics=cached_diagnostics;bands=cached_bands;grid=cached_grid
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        done=0
        for f in as_completed([pool.submit(diagnostic_job,j) for j in jobs]):
            dd,bb,gg=f.result();diagnostics.extend(dd);bands.extend(bb);grid.extend(gg);done+=1
            dump(out/'status.json',dict(stage='diagnostics',completed=done,total=len(jobs),elapsed_seconds=time.time()-start))
            if done%20==0:print(f'Diagnosed {done}/{len(jobs)} paired seeds',flush=True)
    diagnostics.sort(key=lambda r:(r['circuit'],r['seed'],r['arm'],r['group'],r['rule']));bands.sort(key=lambda r:(r['circuit'],r['seed'],r['arm'],r['group'],r['rule'],r['band']));grid.sort(key=lambda r:(r['circuit'],r['seed'],r['selection']))
    csvout(out/'per_seed_diagnostics.csv',diagnostics);csvout(out/'band_diagnostics.csv',bands);csvout(out/'grid_refinement.csv',grid)
    rng=np.random.default_rng(2026100711);means=[];pairs=[];boots_for_aggregate={};unstable=sum(not r['grid_stable'] for r in grid)
    for cfg in config['records']:
        c=cfg['circuit'];seeds=cfg['seeds'];draws=rng.integers(0,len(seeds),(10000,len(seeds)));lookup={(r['seed'],r['arm'],r['group'],r['rule']):r for r in diagnostics if r['circuit']==c}
        for rule in ['primary','balanced']:
            for g in GROUPS if rule=='primary' else ['dual_mse']:
                for arm in ARMS:
                    rr=[lookup[(s,arm,g,rule)] for s in seeds];means.append(dict(circuit=c,arm=arm,group=g,rule=rule,seeds=len(rr),stable_seeds=sum(r['grid_stable'] for r in rr),**{k:float(np.mean([r[k] for r in rr if r[k] is not None])) if any(r[k] is not None for r in rr) else None for k in KEYS}))
                for base in ARMS[:2]:
                    aa=[lookup[(s,'dense_true',g,rule)] for s in seeds];bb=[lookup[(s,base,g,rule)] for s in seeds]
                    for k in KEYS:
                        valid=np.asarray([x[k] is not None and y[k] is not None for x,y in zip(aa,bb)]);ss=np.flatnonzero(valid)
                        if not len(ss):continue
                        delta=np.asarray([aa[i][k]-bb[i][k] for i in ss]);dd=draws if len(ss)==len(seeds) else rng.integers(0,len(ss),(10000,len(ss)));boot=delta[dd].mean(axis=1)
                        row=dict(circuit=c,group=g,rule=rule,baseline=base,metric=k,paired_seeds=len(ss),new_minus_baseline=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),new_lower_seeds=int(np.sum(delta<0)),numerically_stable=all(aa[i]['grid_stable'] and bb[i]['grid_stable'] for i in ss));pairs.append(row)
                        if rule=='primary' and len(ss)==len(seeds):boots_for_aggregate[(c,g,base,k)]=(float(delta.mean()),delta)
    aggregate=[];shared_draws=rng.integers(0,20,(10000,20));case1_extra_draws=rng.integers(0,80,(10000,80))
    for cohort in ['gap8','controls2']:
        ids=[r['circuit'] for r in config['records'] if (r['circuit'] not in ['rc_series_c1','rlc_bridge_c2'])==(cohort=='gap8')]
        if not ids:continue
        for g in GROUPS:
            for base in ARMS[:2]:
                for k in KEYS:
                    keys=[(c,g,base,k) for c in ids]
                    if not all(key in boots_for_aggregate for key in keys):continue
                    values=[boots_for_aggregate[key] for key in keys];bb=[]
                    for c,value in zip(ids,values):
                        delta=value[1]
                        if c=='case1':assert len(delta)==100;bb.append((delta[:20][shared_draws].sum(axis=1)+delta[20:][case1_extra_draws].sum(axis=1))/100)
                        else:assert len(delta)==20;bb.append(delta[shared_draws].mean(axis=1))
                    boot=np.mean(bb,axis=0);aggregate.append(dict(cohort=cohort,circuits=len(ids),group=g,baseline=base,metric=k,equal_circuit_mean_difference=float(np.mean([v[0] for v in values])),ci95=np.quantile(boot,[.025,.975]).tolist(),improved_circuit_means=sum(v[0]<0 for v in values)))
    om=[]
    for cfg in config['records']:
        c=cfg['circuit']
        for arm in ARMS:
            for g in GROUPS:
                rr=[r for r in rows if r['circuit']==c and r['arm']==arm and r['group']==g];om.append(dict(circuit=c,arm=arm,group=g,seeds=len(rr),**{k:float(np.mean([r[k] for r in rr])) for k in COLS}))
    csvout(out/'original_group_means.csv',om);csvout(out/'structure_means.csv',means);csvout(out/'paired_comparisons.csv',[dict(r,ci95=str(r['ci95'])) for r in pairs]);csvout(out/'aggregate_comparisons.csv',[dict(r,ci95=str(r['ci95'])) for r in aggregate]);csvout(out/'run_timings.csv',timings)
    blocks=sum(len(r['seeds']) for r in config['records']);result=dict(structure_means=means,paired_comparisons=pairs,aggregate_comparisons=aggregate,original_group_means=om,new_runs=runs_checked,baseline_runs_reused=blocks*3,paired_seed_blocks=blocks,primary_curves_compared=blocks*9,balanced_curves_compared=blocks*3,grid_rows=len(grid),grid_unstable_rows=unstable,bootstrap_replicates=10000,bootstrap_rng=2026100711,aggregate_bootstrap='shared 20 seeds jointly; Case1 remaining 80 seeds separately; fixed circuit set',reused_verified_chunks=reused_chunks,scope_circuit=a.circuit,entire_experiment_complete=(a.circuit is None),elapsed_analysis_seconds=time.time()-start)
    dump(out/'SUMMARY.json',result);dump(out/'VALIDATION.json',dict(passed=True,optimization_audit=True,independent_networks_and_losses=True,paired_initialization_with_all_three_baselines=True,exact_stored_objective_selection_and_independent_front=True,baseline_parameters_unchanged=True,final_individuals_recomputed=checked,evaluations=budget,grid_rows_checked=len(grid),grid_unstable_rows=unstable,analysis_sha256=sha(Path(__file__))))
    if a.circuit:
        inputs=[p for p in (a.runs/a.circuit).rglob('*') if p.is_file()];dump(out/'CHUNK_CACHE.json',dict(circuit=a.circuit,helper_hashes=helper_hashes,baseline_sha256=sha(baselines[0]),run_input_sha256={p.relative_to(a.runs).as_posix():sha(p) for p in inputs},selected_sha256=sha(out/'selected'/f'{a.circuit}.npz'),checked=checked,budget=budget,runs_checked=runs_checked,rows=rows,timings=timings,diagnostics=diagnostics,bands=bands,grid=grid))
    dump(out/'status.json',dict(stage='complete',completed=blocks,total=blocks,scope_circuit=a.circuit,entire_experiment_complete=(a.circuit is None),elapsed_seconds=time.time()-start));print(json.dumps(dict(passed=True,new_runs=runs_checked,individuals_checked=checked,unstable=unstable,seconds=time.time()-start)))
if __name__=='__main__':main()

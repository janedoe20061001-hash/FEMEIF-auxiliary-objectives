"""Independent final-population checks and preregistered seed-block inference."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse,sys,json,csv,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE/'frozen_scalar'))
from model import Model
COLS=['magnitude_mse_db2','phase_mse_rad2','wsmse','mean_distance_ohm'];CASES=['case1','case2'];GROUPS=['wsmse','mean_distance']
SEEDS=np.arange(2026100700,2026100800)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvout(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def independent_spectra(ident,f,p):
    p=np.atleast_2d(p);s=2j*np.pi*np.asarray(f)[None,:]
    if ident=='case1':
        t=p.reshape(-1,6,3);r=t[:,:,0,None];l=t[:,:,1,None];c=t[:,:,2,None]
        return np.sum(r*s[:,None,:]*l/(r+s[:,None,:]*l+r*s[:,None,:]**2*l*c),axis=1)
    r=p[:,:7,None];l=p[:,7:11,None];c=p[:,11:15,None];a=r[:,0:6:2]+s[:,None,:]*l[:,:3];b=r[:,1:6:2]
    return np.sum(a*b/(a+b+s[:,None,:]*c[:,:3]*a*b),axis=1)+r[:,6]+s*l[:,3]+1/(s*c[:,3])
def independent_metrics(z,target,alpha):
    rho=np.hypot(z.real,z.imag);rho0=np.hypot(target.real,target.imag);da=20*np.log10(rho/rho0)
    dp=np.arctan2(z.imag,z.real)-np.arctan2(target.imag,target.real)
    a=np.mean(da**2,axis=1);p=np.mean(dp**2,axis=1);d=np.mean(np.hypot(z.real-target.real,z.imag-target.imag),axis=1)
    return np.column_stack([a,p,a+alpha*p,d])
def ratios(mean,g):
    a=mean[...,0,g,0]/mean[...,1,g,0];p=mean[...,0,g,1]/mean[...,1,g,1]
    return {'R_A':a,'R_phi':p,'Q':a/p}
def interval(v):return np.quantile(v,[.025,.975]).tolist()
def main():
    arg=argparse.ArgumentParser();arg.add_argument('--runs',required=True,type=Path);arg.add_argument('--output',required=True,type=Path);opt=arg.parse_args();out=opt.output;out.mkdir(parents=True,exist_ok=True);(out/'selected').mkdir(exist_ok=True)
    manifest=json.loads((opt.runs/'manifest.json').read_text());status=json.loads((opt.runs/'status.json').read_text())
    assert status['status']=='complete' and status['completed']==400 and status['failed']==0
    assert manifest['planned_runs']==400 and manifest['planned_evaluations']==21000000 and manifest['seeds']==SEEDS.tolist()
    for name,h in manifest['source_sha256'].items():assert sha(HERE/name)==h
    old_manifest=json.loads((ROOT/'experiments/primary_loss_comparison_20261007/server_run/retrieved/runs/manifest.json').read_text())
    for name,h in old_manifest['source_sha256'].items():
        if (HERE/'frozen_scalar'/name).exists():assert sha(HERE/'frozen_scalar'/name)==h
    data=np.zeros((100,2,2,4));rows=[];seed_ratios=[];band_rows=[];checked=budget=0
    for ci,ident in enumerate(CASES):
        m=Model(ident);curves=[];params=[]
        for si,seed in enumerate(SEEDS):
            pp=[];zz=[];initial=[]
            for gi,group in enumerate(GROUPS):
                d=opt.runs/ident/group/f'seed_{seed}';sm=json.loads((d/'summary.json').read_text());col=2+gi
                assert (sm['circuit'],sm['seed'],sm['group'])==(ident,int(seed),group) and sm['n_obj']==1 and sm['generations']==300 and sm['evaluations']==m.record['population']*300
                budget+=sm['evaluations'];x0=np.load(d/'initial_log_parameters.npy');initial.append(x0)
                np.testing.assert_array_equal(x0,np.random.default_rng(int(seed)).uniform(m.lower,m.upper,(m.record['population'],len(m.lower))))
                assert hashlib.sha256(x0.tobytes()).hexdigest()==sm['initial_population_sha256']
                with np.load(d/'final_population.npz') as saved:
                    x=saved['log_parameters'];assert x.shape==(m.record['population'],len(m.lower));assert np.all(x>=m.lower) and np.all(x<=m.upper) and np.isfinite(x).all()
                    z=m.spectrum(x);z2=independent_spectra(ident,m.f,10**x);np.testing.assert_allclose(z,z2,rtol=1e-9,atol=1e-8)
                    metrics=independent_metrics(z2,m.target,m.alpha);np.testing.assert_allclose(metrics,saved['cross_metrics'],rtol=1e-9,atol=1e-8)
                    np.testing.assert_allclose(saved['optimization_objective'][:,0],metrics[:,col],rtol=1e-9,atol=1e-8)
                    ix=int(saved['selected_index']);assert ix==sm['selected_index'];assert metrics[ix,col]<=np.min(metrics[:,col])+1e-8+1e-9*float(np.min(metrics[:,col]))
                    exact=m.score_spectra(z);assert ix==int(np.argmin(exact[:,col]));np.testing.assert_array_equal(saved['selected_log_parameters'],x[ix]);np.testing.assert_allclose(saved['selected_metrics'],metrics[ix],rtol=1e-9,atol=1e-8)
                    np.testing.assert_allclose([sm[k] for k in COLS],metrics[ix],rtol=1e-9,atol=1e-8)
                    hs=list(csv.DictReader((d/'history.csv').open(encoding='utf-8-sig')));assert len(hs)==300
                    vals=np.array([[float(h[k]) for k in COLS] for h in hs]);np.testing.assert_allclose(vals[-1],metrics[ix],rtol=1e-9,atol=1e-8)
                    assert np.all(np.diff(vals[:,col])<=1e-10*np.maximum(1,vals[:-1,col]));np.testing.assert_allclose(metrics[ix,col],vals[:,col].min(),rtol=1e-9,atol=1e-8)
                    checked+=len(x);data[si,ci,gi]=metrics[ix];pp.append(x[ix]);zz.append(z[ix])
                    rows.append(dict(batch='new',circuit=ident,seed=int(seed),group=group,alpha=m.alpha,**dict(zip(COLS,metrics[ix].tolist()))))
                    seed_ratios.append(dict(circuit=ident,seed=int(seed),group=group,amplitude_over_phase_numeric_ratio=float(metrics[ix,0]/metrics[ix,1])))
                    da=20*np.log10(abs(z[ix])/abs(m.target));dp=np.angle(z[ix])-np.angle(m.target);edges=np.linspace(np.log10(m.f[0]),np.log10(m.f[-1]),9)
                    for k,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
                        mask=(np.log10(m.f)>=lo)&(np.log10(m.f)<hi if k<7 else np.log10(m.f)<=hi)
                        band_rows.append(dict(circuit=ident,seed=int(seed),group=group,band=k+1,lo_hz=float(10**lo),hi_hz=float(10**hi),points=int(mask.sum()),amplitude_contribution=float(np.sum(da[mask]**2)/len(m.f)),phase_contribution=float(np.sum(dp[mask]**2)/len(m.f))))
                
            np.testing.assert_array_equal(initial[0],initial[1]);params.append(pp);curves.append(zz)
        np.savez_compressed(out/'selected'/f'{ident}.npz',frequency=m.f,target=m.target,seeds=SEEDS,groups=np.asarray(GROUPS),log_parameters=np.asarray(params),curves=np.asarray(curves),metrics=data[:,ci])
        print(f'Validated {ident}: 200 runs',flush=True)
    assert checked==70000 and budget==21000000
    old_seeds=np.arange(2300,2400);assert not set(old_seeds)&set(SEEDS);old=np.zeros_like(data);seen=set()
    for row in csv.DictReader((ROOT/'output/primary_loss_comparison_20261007/selected_metrics.csv').open(encoding='utf-8-sig')):
        if row['circuit'] not in CASES:continue
        s=int(row['seed'])-2300;ci=CASES.index(row['circuit']);gi=GROUPS.index(row['group']);assert (s,ci,gi) not in seen;seen.add((s,ci,gi));old[s,ci,gi]=[float(row[k]) for k in COLS]
    assert len(seen)==400
    # The old batch is independently re-audited already; selected arrays must
    # match the original published seed table before old/new comparison.
    for ci,c in enumerate(CASES):
        with np.load(ROOT/'output/primary_loss_comparison_20261007/selected'/f'{c}.npz') as sel:
            np.testing.assert_array_equal(sel['seeds'],old_seeds);np.testing.assert_allclose(sel['metrics'],old[:,ci],rtol=1e-12,atol=1e-10)
    rng=np.random.default_rng(20261007);newboot=data[rng.integers(0,100,(10000,100))].mean(axis=1);oldboot=old[rng.integers(0,100,(10000,100))].mean(axis=1)
    mean=data.mean(axis=0);oldmean=old.mean(axis=0);ratio_results=[];means=[];component_comparisons=[];batch_comparisons=[];statistics=[]
    for gi,g in enumerate(GROUPS):
        rr=ratios(mean,gi);r0=ratios(oldmean,gi);rb=ratios(newboot,gi);ob=ratios(oldboot,gi)
        for name in ['R_A','R_phi','Q']:
            ratio_results.append(dict(group=g,ratio=name,old=float(r0[name]),old_ci95=interval(ob[name]),new=float(rr[name]),new_ci95=interval(rb[name]),new_over_old=float(rr[name]/r0[name]),new_over_old_ci95=interval(rb[name]/ob[name]),log_new_over_old=float(np.log(rr[name]/r0[name])),log_new_over_old_ci95=interval(np.log(rb[name]/ob[name]))))
        for ci,c in enumerate(CASES):
            for batch,arr in [('old',old),('new',data)]:
                vals=arr[:,ci,gi];means.append(dict(batch=batch,circuit=c,group=g,seeds=100,**dict(zip(COLS,vals.mean(axis=0).tolist()))))
                ratio=vals[:,0]/vals[:,1];statistics.append(dict(batch=batch,circuit=c,group=g,amp_phase_pearson=float(np.corrcoef(vals[:,0],vals[:,1])[0,1]),amp_over_phase_numeric_ratio=float(vals[:,0].mean()/vals[:,1].mean()),seed_ratio_p10=float(np.quantile(ratio,.1)),seed_ratio_p50=float(np.median(ratio)),seed_ratio_p90=float(np.quantile(ratio,.9))))
            for k,name in enumerate(COLS):batch_comparisons.append(dict(circuit=c,group=g,metric=name,new_minus_old=float(mean[ci,gi,k]-oldmean[ci,gi,k]),ci95=interval(newboot[:,ci,gi,k]-oldboot[:,ci,gi,k])))
    for ci,c in enumerate(CASES):
        for k,name in enumerate(COLS):component_comparisons.append(dict(circuit=c,metric=name,distance_minus_wsmse=float(mean[ci,1,k]-mean[ci,0,k]),ci95=interval(newboot[:,ci,1,k]-newboot[:,ci,0,k]),distance_over_wsmse=float(mean[ci,1,k]/mean[ci,0,k]),distance_lower_seeds=int(np.sum(data[:,ci,1,k]<data[:,ci,0,k]))))
    q=next(r for r in ratio_results if r['group']=='mean_distance' and r['ratio']=='Q');lo,hi=q['new_ci95']
    decision='supports_stable_closeness_within_5pct' if .95<=lo and hi<=1.05 else ('does_not_replicate_5pct_closeness' if hi<.95 or lo>1.05 else 'inconclusive_for_5pct_closeness')
    result=dict(means=means,ratios=ratio_results,component_comparisons=component_comparisons,batch_comparisons=batch_comparisons,amp_phase_statistics=statistics,primary_group='mean_distance',primary_Q=q,closeness_interval=[.95,1.05],decision=decision,bootstrap_replicates=10000,bootstrap_rng=20261007,bootstrap_unit='shared seed block across cases and objectives; old/new independently resampled')
    dump(out/'SUMMARY.json',result);csvout(out/'selected_metrics.csv',rows);csvout(out/'group_means.csv',means);csvout(out/'seed_ratios.csv',seed_ratios);csvout(out/'band_contributions.csv',band_rows)
    csvout(out/'ratio_comparison.csv',[{**{k:v for k,v in r.items() if not isinstance(v,list)},**{k+'_low':v[0] for k,v in r.items() if isinstance(v,list)},**{k+'_high':v[1] for k,v in r.items() if isinstance(v,list)}} for r in ratio_results])
    dump(out/'VALIDATION.json',dict(passed=True,new_runs=400,final_individuals_recomputed=checked,evaluations=budget,source_hashes_match_original=True,new_seeds_disjoint=True,paired_initialization=True,bounds=True,independent_circuit_and_metric_arithmetic=True,own_objective_final_and_history_minimum=True,old_metrics_match=True,bootstrap_shared_seed_blocks=True,analysis_sha256=sha(Path(__file__))))
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

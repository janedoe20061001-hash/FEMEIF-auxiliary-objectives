"""Recompute every final individual and compare means of per-seed own-loss minima."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import Model,response
HERE=Path(__file__).resolve().parent
COLS=['magnitude_mse_db2','phase_mse_rad2','wsmse','mean_distance_ohm'];GROUPS=['wsmse','mean_distance']
LABELS=['Optimize WSMSE','Optimize mean |Z-Z*|'];COLORS=['#0284c7','#db6b24']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csvout(p,rows):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True);(out/'selected').mkdir(exist_ok=True)
    manifest=json.loads((a.runs/'manifest.json').read_text());assert manifest['n_obj']==1
    for name,digest in manifest['source_sha256'].items():assert sha(HERE/name)==digest
    status=json.loads((a.runs/'status.json').read_text());assert status['status']=='complete' and status['completed']==2320 and status['failed']==0
    records=json.loads((HERE/'circuits.json').read_text(encoding='utf-8'));rows=[];means=[];comparisons=[];checked=0;budget=0
    rng=np.random.default_rng(20261007)
    for ri,r in enumerate(records):
        ident=r['id'];m=Model(ident);n=r['seed_count'];pars=[];curves=[];history=[];data=np.zeros((n,2,4))
        for si,seed in enumerate(range(r['seed_start'],r['seed_start']+n)):
            initial=[];sp=[];sc=[];sh=[]
            for gi,group in enumerate(GROUPS):
                d=a.runs/ident/group/f'seed_{seed}';summary=json.loads((d/'summary.json').read_text());col=2+gi
                assert summary['status']=='complete' and (summary['circuit'],summary['seed'],summary['group'])==(ident,seed,group)
                assert summary['n_obj']==1 and summary['generations']==300 and summary['evaluations']==r['population']*300
                budget+=summary['evaluations'];x0=np.load(d/'initial_log_parameters.npy');initial.append(x0)
                np.testing.assert_array_equal(x0,np.random.default_rng(seed).uniform(m.lower,m.upper,(r['population'],len(m.lower))))
                assert hashlib.sha256(x0.tobytes()).hexdigest()==summary['initial_population_sha256']
                with np.load(d/'final_population.npz') as saved:
                    x=saved['log_parameters'];assert x.shape==(r['population'],len(m.lower))
                    assert np.isfinite(x).all() and np.all(x>=m.lower) and np.all(x<=m.upper)
                    z=m.spectrum(x)
                    # Independent formula route for D (complex modulus) and WSMSE.
                    amp=20*np.log10(abs(z));phase=np.angle(z);ma=((amp-m.amp)**2).mean(axis=1);mp=((phase-m.phase)**2).mean(axis=1)
                    metrics=np.column_stack([ma,mp,ma+m.alpha*mp,abs(z-m.target).mean(axis=1)])
                    np.testing.assert_allclose(saved['cross_metrics'],metrics,rtol=1e-10,atol=1e-8)
                    np.testing.assert_allclose(saved['optimization_objective'][:,0],metrics[:,col],rtol=1e-10,atol=1e-8)
                    assert saved['optimization_objective'].shape==(len(x),1)
                    online=m.score_spectra(z)
                    ix=int(np.argmin(online[:,col]));assert ix==int(saved['selected_index'])==summary['selected_index']
                    assert metrics[ix,col]<=np.min(metrics[:,col])+1e-12*max(1,float(np.min(metrics[:,col])))
                    np.testing.assert_array_equal(saved['selected_log_parameters'],x[ix]);np.testing.assert_array_equal(saved['parameters'],10**x)
                    np.testing.assert_allclose(saved['selected_metrics'],metrics[ix],rtol=1e-10,atol=1e-8)
                    np.testing.assert_allclose([summary[k] for k in COLS],metrics[ix],rtol=1e-10,atol=1e-8)
                    sp.append(x[ix]);sc.append(z[ix]);data[si,gi]=metrics[ix];checked+=len(x)
                    # Polar identity exposes the weighting of absolute impedance distance.
                    rho=abs(z[ix]);rho0=abs(m.target);delta=np.angle(z[ix])-np.angle(m.target)
                    np.testing.assert_allclose(abs(z[ix]-m.target)**2,(rho-rho0)**2+2*rho*rho0*(1-np.cos(delta)),rtol=1e-7,atol=1e-7*np.max(rho0)**2)
                    rows.append(dict(circuit=ident,family=r['family'],condition=r['condition'],seed=seed,group=group,alpha=m.alpha,
                                     **{k:float(v) for k,v in zip(COLS,metrics[ix])}))
                hs=list(csv.DictReader((d/'history.csv').open(encoding='utf-8-sig')))
                assert len(hs)==300 and [int(h['generation']) for h in hs]==list(range(1,301))
                assert [int(h['evaluations']) for h in hs]==[r['population']*i for i in range(1,301)]
                hvalues=np.array([[float(h[k]) for k in COLS] for h in hs]);np.testing.assert_allclose(hvalues[-1],metrics[ix],rtol=1e-10,atol=1e-8)
                # Scalar elitist survival must retain the best value each generation.
                assert np.all(np.diff(hvalues[:,col])<=1e-10*np.maximum(1,hvalues[:-1,col]))
                sh.append(hvalues)
            np.testing.assert_array_equal(initial[0],initial[1]);pars.append(sp);curves.append(sc);history.append(sh)
        for gi,group in enumerate(GROUPS):
            means.append(dict(circuit=ident,title=r['title'],family=r['family'],condition=r['condition'],group=group,seeds=n,alpha=m.alpha,
                              **{k:float(v) for k,v in zip(COLS,data[:,gi].mean(axis=0))}))
        boots=rng.integers(0,n,(10000,n))
        for col,metric in enumerate(COLS):
            delta=data[:,1,col]-data[:,0,col];a0=float(data[:,0,col].mean());a1=float(data[:,1,col].mean())
            comparisons.append(dict(circuit=ident,family=r['family'],condition=r['condition'],metric=metric,seeds=n,wsmse_group_mean=a0,distance_group_mean=a1,
                distance_minus_wsmse_mean=float(delta.mean()),bootstrap95=np.quantile(delta[boots].mean(axis=1),[.025,.975]).tolist(),
                relative_change_pct=100*(a1/a0-1) if a0 else None,distance_lower_seeds=int(np.sum(delta<0)),wsmse_lower_seeds=int(np.sum(delta>0)),ties=int(np.sum(delta==0))))
        np.savez_compressed(out/'selected'/f'{ident}.npz',frequency=m.f,target=m.target,curves=np.asarray(curves),log_parameters=np.asarray(pars),metrics=data,
            groups=np.asarray(GROUPS),seeds=np.arange(r['seed_start'],r['seed_start']+n),history=np.asarray(history))
        fig,ax=plt.subplots(1,4,figsize=(14,3.7))
        for col,(axis,label) in enumerate(zip(ax,['Magnitude MSE (dB²)','Phase MSE (rad²)','WSMSE','Mean distance (Ω)'])):
            for v in data[:,:,col]:axis.plot([0,1],v,color='#d5dde5',alpha=.6,lw=.5)
            b=axis.boxplot([data[:,gi,col] for gi in (0,1)],positions=[0,1],patch_artist=True,tick_labels=['WSMSE','Distance'])
            for patch,color in zip(b['boxes'],COLORS):patch.set_facecolor(color);patch.set_alpha(.35)
            axis.set_ylabel(label);axis.grid(axis='y',alpha=.18)
        fig.suptitle(f'{ident}: own-loss final minimum, {n} paired seeds');fig.tight_layout();fig.savefig(out/'figures'/f'{ident}_paired.png',dpi=145);plt.close(fig)
        fig,ax=plt.subplots(2,1,figsize=(10,6),sharex=True)
        ax[0].semilogx(m.f,m.amp,'k--',lw=1.2,label='Target');ax[1].semilogx(m.f,np.rad2deg(m.phase),'k--',lw=1.2,label='Target')
        for gi,(label,color) in enumerate(zip(LABELS,COLORS)):
            ax[0].semilogx(m.f,20*np.log10(abs(curves[0][gi])),label=label,color=color)
            ax[1].semilogx(m.f,np.rad2deg(np.angle(curves[0][gi])),label=label,color=color)
        ax[0].set_ylabel('Magnitude (dBΩ)');ax[1].set_ylabel('Phase (degree)');ax[1].set_xlabel('Frequency (Hz)')
        for axis in ax:axis.legend(fontsize=8);axis.grid(alpha=.18)
        fig.suptitle(f'{ident}: fixed representative seed {r["seed_start"]}');fig.tight_layout();fig.savefig(out/'figures'/f'{ident}_curves.png',dpi=145);plt.close(fig)
        if ident in ('case1','case2'):
            fig,ax=plt.subplots(1,4,figsize=(14,3.7));h=np.asarray(history)
            for col,(axis,label) in enumerate(zip(ax,['Magnitude MSE','Phase MSE','WSMSE','Mean distance'])):
                for gi,(name,color) in enumerate(zip(LABELS,COLORS)):
                    q=np.quantile(h[:,gi,:,col],[.25,.5,.75],axis=0);axis.plot(range(1,301),q[1],label=name,color=color);axis.fill_between(range(1,301),q[0],q[2],color=color,alpha=.12)
                axis.set_title(label);axis.set_xlabel('Generation');axis.grid(alpha=.18)
            ax[0].legend(fontsize=7);fig.suptitle(ident+': median and IQR of own-loss selected values');fig.tight_layout();fig.savefig(out/'figures'/f'{ident}_history.png',dpi=145);plt.close(fig)
        if (ri+1)%5==0:print(f'Validated and plotted {ri+1}/50 circuits',flush=True)
    assert checked==300400 and budget==90120000 and len(rows)==2320
    csvout(out/'selected_metrics.csv',rows);csvout(out/'group_means.csv',means)
    csvout(out/'paired_comparisons.csv',[{**{k:v for k,v in c.items() if k!='bootstrap95'},'ci95_low':c['bootstrap95'][0],'ci95_high':c['bootstrap95'][1]} for c in comparisons])
    synth=[c for c in comparisons if c['circuit'] not in ('case1','case2')]
    tally={metric:dict(distance_lower_mean=sum(c['distance_group_mean']<c['wsmse_group_mean'] for c in synth if c['metric']==metric),
                       wsmse_lower_mean=sum(c['distance_group_mean']>c['wsmse_group_mean'] for c in synth if c['metric']==metric)) for metric in COLS}
    both=sum(all(next(c for c in synth if c['circuit']==r['id'] and c['metric']==metric)['distance_minus_wsmse_mean']<0 for metric in COLS[:2]) for r in records[2:])
    oldboth=sum(all(next(c for c in synth if c['circuit']==r['id'] and c['metric']==metric)['distance_minus_wsmse_mean']>0 for metric in COLS[:2]) for r in records[2:])
    family=[]
    for name in dict.fromkeys(r['family'] for r in records[2:]):
        for metric in COLS[:2]:
            cc=[c for c in synth if c['family']==name and c['metric']==metric]
            # Geometric ratios describe relative direction without averaging different absolute circuit scales.
            ratios=np.array([c['distance_group_mean']/c['wsmse_group_mean'] for c in cc])
            family.append(dict(family=name,metric=metric,circuits=3,distance_lower_circuits=int(np.sum(ratios<1)),geometric_mean_error_ratio=float(np.exp(np.log(ratios).mean()))))
    csvout(out/'topology_summary.csv',family)
    dump(out/'SUMMARY.json',dict(means=means,paired_comparisons=comparisons,synthetic_tally=tally,synthetic_distance_both_components_lower=both,
        synthetic_wsmse_both_components_lower=oldboth,topology_summary=family,runs=2320,synthetic_circuits=48,synthetic_topologies=16))
    dump(out/'VALIDATION.json',dict(passed=True,runs=2320,final_individuals_recomputed=checked,evaluations=budget,scalar_objectives=True,
        own_loss_minimum_selection=True,all_cross_metrics_same_individual=True,paired_initialization=True,seed_reproduction=True,
        parameter_bounds=True,history_budget_and_monotone_own_loss=True,source_hashes=True,polar_distance_identity=True,analysis_sha256=sha(__file__)))
    print(json.dumps(dict(passed=True,runs=2320,final_individuals=checked,evaluations=budget)))
if __name__=='__main__':main()

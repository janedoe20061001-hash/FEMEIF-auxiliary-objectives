"""Independent arithmetic audit of frozen scalar runs; no optimization rerun."""
from pathlib import Path
import csv, json, hashlib, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'experiments/primary_loss_comparison_20261007'
RUNS=SOURCE/'server_run/retrieved/runs'
OUT=ROOT/'output/primary_loss_comparison_20261007'
AUDIT=OUT/'audit'
AUDIT.mkdir(exist_ok=True)
COLS=['magnitude_mse_db2','phase_mse_rad2','wsmse','mean_distance_ohm']
GROUPS=['wsmse','mean_distance']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csvout(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def independent_network(record,f,p):
    p=np.atleast_2d(p);s=2j*np.pi*np.asarray(f)[None,:]
    if record['id']=='case1':
        t=p.reshape(-1,6,3);r=t[:,:,0,None];l=t[:,:,1,None];c=t[:,:,2,None]
        return np.sum(r*s[:,None,:]*l/(r+s[:,None,:]*l+r*s[:,None,:]**2*l*c),axis=1)
    if record['id']=='case2':
        r=p[:,:7,None];l=p[:,7:11,None];c=p[:,11:15,None];a=r[:,0:6:2]+s[:,None,:]*l[:,:3];b=r[:,1:6:2]
        return np.sum(a*b/(a+b+s[:,None,:]*c[:,:3]*a*b),axis=1)+r[:,6]+s*l[:,3]+1/(s*c[:,3])
    values=dict(zip(record['parameter_names'],p.T))
    def rec(t):
        if isinstance(t,str):
            v=values[t][:,None]
            return np.broadcast_to(v,(len(p),len(f))) if t[0]=='R' else (s*v if t[0]=='L' else 1/(s*v))
        z=[rec(x) for x in t[1:]]
        if t[0]=='S':return np.add.reduce(z)
        if t[0]=='P':
            a=z[0]
            for b in z[1:]:a=a*b/(a+b)
            return a
        # Wheatstone bridge: solve two internal-node Kirchhoff equations,
        # with terminal voltages 1 and 0, independently of determinant formulas.
        y=[1/a for a in z];aa=y[0]+y[1]+y[4];bb=y[2]+y[3]+y[4];dd=aa*bb-y[4]**2
        v1=(y[0]*bb+y[4]*y[2])/dd;v2=(y[2]*aa+y[4]*y[0])/dd
        return 1/(y[0]*(1-v1)+y[2]*(1-v2))
    return rec(record['tree'])
def independent_metrics(z,target,alpha):
    # Compute D from Cartesian coordinates, phase from atan2, magnitude from hypot.
    rho=np.hypot(z.real,z.imag);rho0=np.hypot(target.real,target.imag)
    da=20*np.log10(rho/rho0);dp=np.arctan2(z.imag,z.real)-np.arctan2(target.imag,target.real)
    ma=np.mean(da*da,axis=1);mp=np.mean(dp*dp,axis=1)
    d=np.mean(np.hypot(z.real-target.real,z.imag-target.imag),axis=1)
    return np.column_stack((ma,mp,ma+alpha*mp,d))
records=json.loads((SOURCE/'circuits.json').read_text(encoding='utf-8'))
manifest=json.loads((RUNS/'manifest.json').read_text())
for name,h in manifest['source_sha256'].items():assert sha(SOURCE/name)==h
published={(r['circuit'],int(r['seed']),r['group']):r for r in csv.DictReader((OUT/'selected_metrics.csv').open(encoding='utf-8-sig'))}
means=[];per_seed=[];points=[];bands=[];stats=[];count=0;max_relative=0.;max_curve_relative=0.
for ri,r in enumerate(records):
    ident=r['id'];f=np.asarray(r['frequency_hz']);target=np.asarray(r['target_real'])+1j*np.asarray(r['target_imag']);mat=[]
    with np.load(OUT/'selected'/f'{ident}.npz') as selected:
        np.testing.assert_array_equal(selected['frequency'],f);np.testing.assert_array_equal(selected['target'],target)
        for si,seed in enumerate(selected['seeds']):
            pair=[]
            for gi,group in enumerate(GROUPS):
                d=RUNS/ident/group/f'seed_{seed}'
                sm=json.loads((d/'summary.json').read_text())
                with np.load(d/'final_population.npz') as pop:
                    z=independent_network(r,f,10**pop['log_parameters']);metrics=independent_metrics(z,target,r['alpha'])
                    np.testing.assert_allclose(metrics,pop['cross_metrics'],rtol=2e-7,atol=1e-8)
                    max_relative=max(max_relative,float(np.max(abs(metrics-pop['cross_metrics'])/np.maximum(abs(pop['cross_metrics']),1e-8))))
                    col=2+gi;ix=int(np.argmin(metrics[:,col]));stored_ix=int(pop['selected_index'])
                    # At machine-level ties the numeric route may change argmin, so
                    # verify the recorded selection attains the recomputed minimum.
                    assert abs(metrics[stored_ix,col]-metrics[ix,col])<=1e-8+2e-7*abs(metrics[ix,col])
                    np.testing.assert_array_equal(pop['selected_log_parameters'],selected['log_parameters'][si,gi])
                    m=metrics[stored_ix];zs=z[stored_ix]
                    np.testing.assert_allclose(zs,selected['curves'][si,gi],rtol=2e-7,atol=1e-7)
                    max_curve_relative=max(max_curve_relative,float(np.max(abs(zs-selected['curves'][si,gi])/np.maximum(abs(zs),1e-8))))
                    np.testing.assert_allclose(m,selected['metrics'][si,gi],rtol=2e-7,atol=1e-8)
                    np.testing.assert_allclose(m,[float(published[(ident,int(seed),group)][c]) for c in COLS],rtol=2e-7,atol=1e-8)
                    np.testing.assert_allclose(m,[sm[c] for c in COLS],rtol=2e-7,atol=1e-8)
                    history=list(csv.DictReader((d/'history.csv').open(encoding='utf-8-sig')))
                    assert len(history)==300
                    hmin=min(float(h[COLS[col]]) for h in history)
                    np.testing.assert_allclose(m[col],hmin,rtol=2e-7,atol=1e-8)
                    count+=len(z);pair.append(m)
                    per_seed.append(dict(circuit=ident,seed=int(seed),group=group,**dict(zip(COLS,m.tolist()))))
            mat.append(pair)
        mat=np.asarray(mat)
        for gi,group in enumerate(GROUPS):means.append(dict(circuit=ident,group=group,**dict(zip(COLS,mat[:,gi].mean(axis=0).tolist()))))
        if ident in ['case1','case2']:
            for gi,group in enumerate(GROUPS):
                amp=mat[:,gi,0];phase=mat[:,gi,1]
                stats.append(dict(circuit=ident,group=group,amp_phase_pearson=float(np.corrcoef(amp,phase)[0,1]),amp_over_phase_ratio=float(amp.mean()/phase.mean()),seed_ratio_p10=float(np.quantile(amp/phase,.1)),seed_ratio_p50=float(np.median(amp/phase)),seed_ratio_p90=float(np.quantile(amp/phase,.9))))
        if ident=='case1':
            # Original screenshot is specifically seed 2300, not a mean curve.
            zs=selected['curves'][0];assert int(selected['seeds'][0])==2300
            dist=np.hypot((zs-target).real,(zs-target).imag)
            for i,fi in enumerate(f):
                points.append(dict(index=i,frequency_hz=float(fi),target_real_ohm=float(target[i].real),target_imag_ohm=float(target[i].imag),target_magnitude_ohm=float(abs(target[i])),wsmse_fit_real_ohm=float(zs[0,i].real),wsmse_fit_imag_ohm=float(zs[0,i].imag),distance_fit_real_ohm=float(zs[1,i].real),distance_fit_imag_ohm=float(zs[1,i].imag),wsmse_distance_ohm=float(dist[0,i]),distance_distance_ohm=float(dist[1,i]),wsmse_contribution_to_mean_ohm=float(dist[0,i]/len(f)),distance_contribution_to_mean_ohm=float(dist[1,i]/len(f)),distance_minus_wsmse_ohm=float(dist[1,i]-dist[0,i])))
            edges=[100.,1e3,1e4,1e5,1e6]
            for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
                mask=(f>=lo)&(f<hi if j<3 else f<=hi)
                row=dict(lo_hz=lo,hi_hz=hi,points=int(mask.sum()))
                for gi,group in enumerate(GROUPS):
                    row[group+'_contribution_to_overall_mean_ohm']=float(dist[gi,mask].sum()/len(f));row[group+'_share_of_total_distance']=float(dist[gi,mask].sum()/dist[gi].sum())
                bands.append(row)
            np.testing.assert_allclose(dist.mean(axis=1),selected['metrics'][0,:,3],rtol=1e-12)
            fig,ax=plt.subplots(2,1,figsize=(10,6),sharex=True)
            for gi,(name,color) in enumerate(zip(['WSMSE-selected fit','Distance-selected fit'],['#0284c7','#db6b24'])):
                ax[0].semilogx(f,dist[gi],color=color,label=name);ax[1].semilogx(f,np.cumsum(dist[gi])/len(f),color=color,label=name)
            ax[0].set_ylabel('Pointwise |Z - target| (ohm)');ax[1].set_ylabel('Cumulative sum / 180 (ohm)');ax[1].set_xlabel('Frequency (Hz)')
            for a in ax:a.grid(alpha=.2);a.legend()
            fig.suptitle('Case1 / seed 2300: original 180 points, Cartesian distance audit');fig.tight_layout();fig.savefig(AUDIT/'case1_seed2300_distance_audit.png',dpi=160);plt.close(fig)
    if (ri+1)%10==0:print(f'Independent audit {ri+1}/50 circuits',flush=True)
lookup={(m['circuit'],m['group']):m for m in means}
original=json.loads((OUT/'SUMMARY.json').read_text(encoding='utf-8'))
for row in original['means']:
    np.testing.assert_allclose([lookup[(row['circuit'],row['group'])][k] for k in COLS],[row[k] for k in COLS],rtol=2e-7,atol=1e-8)
csvout(AUDIT/'recomputed_seed_metrics.csv',per_seed);csvout(AUDIT/'recomputed_means.csv',means);csvout(AUDIT/'case1_seed2300_points.csv',points);csvout(AUDIT/'case1_seed2300_bands.csv',bands);csvout(AUDIT/'amp_phase_ratios.csv',stats)
case_ratios={g:{k:lookup[('case1',g)][k]/lookup[('case2',g)][k] for k in COLS} for g in GROUPS}
improvement_ratios={c:{k:lookup[(c,'mean_distance')][k]/lookup[(c,'wsmse')][k] for k in COLS} for c in ['case1','case2']}
worst=sorted(points,key=lambda p:p['distance_minus_wsmse_ohm'])[:10]
result=dict(passed=True,runs=len(per_seed),final_individuals=count,max_metric_relative_discrepancy=max_relative,max_selected_curve_relative_discrepancy=max_curve_relative,source_manifest_hashes=True,independent_network_arithmetic=True,selected_individual_consistency=True,own_objective_final_and_history_minimum=True,all_published_seed_metrics_and_means_reproduced=True,same_frequency_distance=True,phase_radians=True,amplitude_db=True,case1_over_case2_ratios=case_ratios,distance_over_wsmse_ratios=improvement_ratios,amp_phase_statistics=stats,seed2300_bands=bands,seed2300_largest_distance_advantages=worst,seed2300_distance_group_lower_points=sum(p['distance_minus_wsmse_ohm']<0 for p in points),audit_code_sha256=sha(__file__))
assert result['runs']==2320 and count==300400
dump(AUDIT/'AUDIT.json',result)
print(json.dumps({k:result[k] for k in ['passed','runs','final_individuals','max_metric_relative_discrepancy','max_selected_curve_relative_discrepancy','case1_over_case2_ratios','distance_over_wsmse_ratios','amp_phase_statistics','seed2300_bands','seed2300_distance_group_lower_points']},ensure_ascii=False,indent=2))

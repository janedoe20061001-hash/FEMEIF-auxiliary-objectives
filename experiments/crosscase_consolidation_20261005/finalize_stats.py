"""Fixed 60-test family and correlation summaries; no fitting or tuning."""
from pathlib import Path
import csv,json,itertools,hashlib
import numpy as np
from scipy.stats import wilcoxon,rankdata

root=Path(__file__).resolve().parent
out=root/'analysis';out.mkdir(exist_ok=True)
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def save(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def corr(a,b):
    if np.ptp(a)==0 or np.ptp(b)==0:return None,None
    return float(np.corrcoef(a,b)[0,1]),float(np.corrcoef(rankdata(a),rankdata(b))[0,1])

comparisons=[];groupstats=[];cs=[];ss=[];frequency_rows=[];band_rows=[];diagnostics=[]
for case in ['case1','case2']:
    d=root/case;assert json.loads((d/'analysis/audit.json').read_text())['status']=='passed'
    cfg=json.loads((d/'config.json').read_text());rows=read(d/'analysis/rows.csv')
    groups=['+'.join(g) if g else 'baseline' for g in cfg['groups']]
    values={g:np.array([[float(r[k]) for k in ['magnitude','phase','wsmse']] for r in rows if r['group']==g]) for g in groups}
    for g in groups:
        rr=[r for r in rows if r['group']==g];v=values[g]
        groupstats.append(dict(case=case,group=g,mean=v.mean(axis=0).tolist(),median=np.median(v,axis=0).tolist(),
            std=v.std(axis=0,ddof=1).tolist(),mean_seconds=float(np.mean([float(r['seconds']) for r in rr])),
            full_front_fraction=float(np.mean([float(r['full_front_fraction']) for r in rr])),
            primary_front_fraction=float(np.mean([float(r['primary_front_fraction']) for r in rr]))))
    ix=np.random.default_rng(20261005).integers(0,100,(10000,100))
    for control,test in itertools.combinations(groups,2):
        a=values[test];b=values[control]
        pct=100*(1-a.mean(axis=0)/b.mean(axis=0))
        boots=100*(1-a[ix].mean(axis=1)/b[ix].mean(axis=1))
        p=[float(wilcoxon(a[:,j]-b[:,j],alternative='two-sided',method='approx').pvalue) if np.any(a[:,j]!=b[:,j]) else 1. for j in range(2)]
        comparisons.append(dict(case=case,group=test,comparator=control,reduction_pct=pct.tolist(),
            bootstrap95=np.quantile(boots,[.025,.975],axis=0).T.tolist(),p_raw=p,
            both_lower_seeds=int(np.all(a[:,:2]<b[:,:2],axis=1).sum())))
    raw=read(d/'analysis/correlations.csv')
    for key in sorted({(r['context'],r['id'],r['metric'],r['primary']) for r in raw}):
        rr=[r for r in raw if (r['context'],r['id'],r['metric'],r['primary'])==key]
        row=dict(case=case,context=key[0],id=key[1],metric=key[2],primary=key[3],runs=100)
        for kind in ['pearson','spearman']:
            v=np.array([float(r[kind]) for r in rr if r[kind]!=''])
            row[kind+'_valid']=len(v)
            for q,tag in [(.25,'q25'),(.5,'median'),(.75,'q75')]:row[kind+'_'+tag]=float(np.quantile(v,q)) if len(v) else None
        cs.append(row)
    raw=read(d/'analysis/selected.csv')
    for mid in sorted({r['id'] for r in raw}):
        rr=[r for r in raw if r['id']==mid];x=np.array([float(r['loss']) for r in rr])
        for primary in ['magnitude','phase']:
            p,s=corr(x,np.array([float(r[primary]) for r in rr]));ss.append(dict(case=case,id=mid,metric=rr[0]['metric'],primary=primary,n=100,pearson=p,spearman=s))
    mf=json.loads((d/'runs/manifest.json').read_text());f=np.array(mf['frequency_hz'])
    if case=='case1':
        import importlib.util
        spec=importlib.util.spec_from_file_location('target_model',d/'legacy_objectives.py');model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)
        z=model.impedance(f,model.TARGET)[0]
    else:z=np.array(mf['target_real'])+1j*np.array(mf['target_imag'])
    data=np.load(d/'analysis/selected_curves.npz');curves=data['curves']
    errors=np.stack([(20*np.log10(abs(curves))-20*np.log10(abs(z)))**2,(np.angle(curves)-np.angle(z))**2],axis=-1)
    means=errors.mean(axis=0);u=(np.log10(f)-np.log10(f[0]))/np.log10(f[-1]/f[0])
    for k,g in enumerate(groups):
        np.testing.assert_allclose(means[k].mean(axis=0),values[g].mean(axis=0)[:2],rtol=1e-11,atol=1e-12)
        for j in range(len(f)):frequency_rows.append(dict(case=case,group=g,frequency_hz=float(f[j]),magnitude_mse=float(means[k,j,0]),phase_mse=float(means[k,j,1])))
        for band in range(3):
            mask=(u>=band/3)&((u<(band+1)/3) if band<2 else (u<=1))
            avg=means[k,mask].mean(axis=0);total=means[k,mask].sum(axis=0)/len(f)
            band_rows.append(dict(case=case,group=g,band=band+1,points=int(mask.sum()),frequency_min=float(f[mask][0]),frequency_max=float(f[mask][-1]),
                magnitude_mse=float(avg[0]),phase_mse=float(avg[1]),magnitude_contribution=float(total[0]),phase_contribution=float(total[1])))
    jumps=np.abs(np.diff(np.angle(z)));j=int(np.argmax(jumps));outside=abs(np.angle(z))>np.pi/2
    diagnostics.append(dict(case=case,points=len(f),frequency_range=[float(f[0]),float(f[-1])],
        magnitude_range=[float(abs(z).min()),float(abs(z).max())],log_spacing_min=float(np.diff(np.log10(f)).min()),
        log_spacing_max=float(np.diff(np.log10(f)).max()),max_principal_phase_jump_deg=float(jumps[j]*180/np.pi),
        jump_interval=f[j:j+2].tolist(),outside_positive_real_phase_count=int(outside.sum()),
        outside_phase_contribution_by_group={g:dict(absolute=float(means[k,outside,1].sum()/len(f)),
           fraction=float(means[k,outside,1].sum()/means[k,:,1].sum())) for k,g in enumerate(groups)},
        preflight=json.loads((d/'preflight.json').read_text())['target']))

p=np.array([p for r in comparisons for p in r['p_raw']]);assert len(p)==60
order=np.argsort(p);adjusted=np.empty(60);adjusted[order]=np.minimum(1,np.maximum.accumulate((60-np.arange(60))*p[order]))
for i,r in enumerate(comparisons):r['holm60']=adjusted[2*i:2*i+2].tolist()
dump(out/'comparisons.json',comparisons);dump(out/'groups.json',groupstats);dump(out/'target_diagnostics.json',diagnostics)
save(out/'correlation_summary.csv',cs);save(out/'selected_correlations.csv',ss)
save(out/'frequency_error_profiles.csv',frequency_rows);save(out/'band_errors.csv',band_rows)
flat=[]
for r in comparisons:
    for j,name in enumerate(['magnitude','phase']):flat.append(dict(case=r['case'],group=r['group'],comparator=r['comparator'],primary=name,
        reduction_pct=r['reduction_pct'][j],ci_low=r['bootstrap95'][j][0],ci_high=r['bootstrap95'][j][1],p_raw=r['p_raw'][j],holm60=r['holm60'][j],both_lower_seeds=r['both_lower_seeds']))
save(out/'comparisons.csv',flat)
dump(out/'COMPLETE.json',dict(status='passed',runs=1200,evaluations=63000000,primary_tests=60,
    per_case=[json.loads((root/c/'analysis/audit.json').read_text()) for c in ['case1','case2']],
    source_hashes={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*case.py')},
    correlation_note='run-level correlations; constant variables undefined; selected correlations across independent seeds'))
print(json.dumps({'status':'complete','comparisons':len(comparisons),'correlation_rows':len(cs)}))

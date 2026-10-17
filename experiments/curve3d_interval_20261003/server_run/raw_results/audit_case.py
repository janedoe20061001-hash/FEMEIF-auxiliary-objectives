"""Independent completion checks, full-population recalculation, paired analysis."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import sys
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from scipy.stats import wilcoxon

case=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(case))
import experiment as e
from objectives import CATALOG
from global_metrics import GlobalObjectives

def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

cfg,freq=e.config(case/'config.json');root=case/'runs'
CATALOG=[row for row in CATALOG if row[1] in {e.group_name(g) for g in e.groups(cfg)}]
mf=json.loads((root/'manifest.json').read_text())
assert e.manifest(cfg,freq)['fingerprint']==mf['fingerprint']
if case.name=='case1':
    from legacy_objectives import impedance,TARGET
    target=impedance(freq,TARGET)[0]
else:
    from case2 import impedance,load_target
    _,target=load_target(cfg['target_data'])
records={};fronts={};initial_hashes={};histories=0
for seed in range(1100,1200):
    expected_x=e.initial_population(cfg,seed)
    for group in e.groups(cfg):
        name=e.group_name(group);folder=root/name/f'seed_{seed:03d}'
        record=json.loads((folder/'summary.json').read_text())
        assert record['status']=='complete' and record['fingerprint']==mf['fingerprint']
        assert record['seed']==seed and record['group']==name
        assert record['generations']==300 and record['evaluations']==cfg['population']*300
        initial=np.load(folder/'initial_log_parameters.npy')
        np.testing.assert_array_equal(initial,expected_x)
        assert hashlib.sha256(initial.tobytes()).hexdigest()==record['initial_population_sha256']
        with np.load(folder/'final_population.npz') as saved:
            x=saved['log_parameters'];f=saved['objectives']
            assert x.shape==expected_x.shape and f.shape==(cfg['population'],2+len(group))
            assert (x>=e.LOWER-1e-13).all() and (x<=e.UPPER+1e-13).all()
            recalculated=GlobalObjectives(freq,aux=group,target_z=target).evaluate_spectra(impedance(freq,10**x))
            np.testing.assert_allclose(f,recalculated,rtol=1e-11,atol=1e-12)
            selected=e.select_wsmse(f,cfg['final_selection']['phase_weight'])
            assert selected['wsmse_selected_index']==int(saved['selected_index'])
            for key,value in selected.items():
                np.testing.assert_allclose(record[key],value,rtol=1e-12,atol=1e-13)
            fronts[(name,seed)]=f[e.nd_indices(f[:,:2]),:2]
        with (folder/'history.csv').open(encoding='utf-8-sig',newline='') as fh:
            history=list(csv.DictReader(fh))
        assert len(history)==300 and int(history[-1]['evaluations'])==cfg['population']*300
        assert [int(h['generation']) for h in history]==list(range(1,301))
        records[(name,seed)]=record;histories+=1

def coverage(a,b):
    """Fraction of b primary-front points strictly dominated by at least one a."""
    return float(np.any(np.all(a[:,None,:]<=b[None,:,:],axis=2)&
                        np.any(a[:,None,:]<b[None,:,:],axis=2),axis=0).mean())

summary=[];rng=np.random.default_rng(20261003)
bootstrap=rng.integers(0,100,(10000,100))
fields=['wsmse_selected_magnitude_mse','wsmse_selected_phase_mse','wsmse_selected_value']
baseline=np.array([[records[('baseline',s)][k] for k in fields] for s in range(1100,1200)])
for metric_id,name,label in CATALOG:
    a=np.array([[records[(name,s)][k] for k in fields] for s in range(1100,1200)])
    av=a.mean(axis=0);bv=baseline.mean(axis=0)
    pct=100*(1-av/bv)
    bootpct=100*(1-a[bootstrap].mean(axis=1)/baseline[bootstrap].mean(axis=1))
    cis=np.quantile(bootpct,[.025,.975],axis=0).T
    p=[]
    for i in range(2):
        d=a[:,i]-baseline[:,i]
        p.append(float(wilcoxon(d,alternative='two-sided',method='approx').pvalue) if np.any(d!=0) else 1.)
    item=dict(case=case.name,id=metric_id,group=name,label=label,paired_seeds=100,
              baseline_mean=bv.tolist(),auxiliary_mean=av.tolist(),reduction_pct=pct.tolist(),
              reduction_pct_bootstrap95=cis.tolist(),wilcoxon_two_sided_p=p,
              baseline_median=np.median(baseline,axis=0).tolist(),auxiliary_median=np.median(a,axis=0).tolist(),
              both_lower_seeds=int(np.all(a[:,:2]<baseline[:,:2],axis=1).sum()),
              wsmse_lower_seeds=int((a[:,2]<baseline[:,2]).sum()),
              aux_dominates_baseline_front_mean=float(np.mean([coverage(fronts[(name,s)],fronts[('baseline',s)]) for s in range(1100,1200)])),
              baseline_dominates_aux_front_mean=float(np.mean([coverage(fronts[('baseline',s)],fronts[(name,s)]) for s in range(1100,1200)])))
    summary.append(item)
dump(case/'analysis.json',summary)
dump(case/'audit.json',dict(status='passed',runs=histories,paired_seeds=100,
     evaluations=histories*cfg['population']*300,fingerprint=mf['fingerprint'],
     final_population_objectives_recomputed=True,all_initial_populations_verified=True,
     all_histories_verified=True,selection_verified=True,
     bootstrap_repetitions=10000,bootstrap_seed=20261003,
     interval='paired-seed percentile bootstrap of ratio-of-means reduction'))
print(json.dumps({'case':case.name,'audit':'passed','runs':histories,'results':summary},ensure_ascii=True))

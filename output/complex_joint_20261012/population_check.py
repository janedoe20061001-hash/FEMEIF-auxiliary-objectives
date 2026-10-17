"""Read final populations without changing the frozen selected results."""
from pathlib import Path
import csv,json,numpy as np
OUT=Path(__file__).resolve().parent;CODE=OUT/'retrieved/experiments/complex_joint_20261012';DATA=OUT/'retrieved/output/complex_joint_20261012'
cfg=json.loads((CODE/'protocol.json').read_text());ids=json.loads((CODE/'inputs/manifest.json').read_text())['targets'];rows=[]
def load(ident,method,seed):
    with np.load(DATA/'jobs'/ident/method/f'seed_{seed}'/'generation_0300.npz') as a:
        f=a['F'][:,:2];sel=f[int(a['selected_index'])];return f,sel
def count(f,p):return int(np.sum(np.all(f<=p,axis=1)&np.any(f<p,axis=1)))
for ident in ids:
    for seed in cfg['seeds']:
        base,b=load(ident,'MAG_PHA',seed)
        for method in cfg['methods']:
            f,s=load(ident,method,seed)
            rows.append(dict(target=ident,method=method,seed=seed,selected_mag=float(s[0]),selected_pha=float(s[1]),
                baseline_mag=float(b[0]),baseline_pha=float(b[1]),new_population_dominates_baseline_selection=count(f,b),
                baseline_population_dominates_new_selection=count(base,s),
                new_selected_dominates_baseline=bool(np.all(s<=b)&np.any(s<b)),
                baseline_selected_dominates_new=bool(np.all(b<=s)&np.any(b<s))))
with (OUT/'analysis/POPULATION_PAIR.csv').open('w',encoding='utf-8-sig',newline='') as h:
    w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
stage=[]
for ident,method in [('mixed_resonators/base','D'),('mixed_resonators/lc_reciprocal_8p0','A'),('case2','H'),('case1','D_H_A')]:
    for m in ['MAG_PHA',method]:
        for seed in [45006,45002,45004]:
            for generation in range(20,301,20):
                with np.load(DATA/'jobs'/ident/m/f'seed_{seed}'/f'generation_{generation:04}.npz') as a:
                    f=a['F'];s=f[int(a['selected_index'])]
                stage.append(dict(target=ident,method=m,seed=seed,generation=generation,mag=float(s[0]),pha=float(s[1]),
                    minimum_mag=float(f[:,0].min()),minimum_pha=float(f[:,1].min())))
with (OUT/'analysis/STAGE_SELECTION.csv').open('w',encoding='utf-8-sig',newline='') as h:
    w=csv.DictWriter(h,fieldnames=list(stage[0]));w.writeheader();w.writerows(stage)
print('Population pairs:',len(rows),'stage observations:',len(stage))

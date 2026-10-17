from review_analysis import *

with (AN/'REVIEWED_SEEDS.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
lookup={(v['target'],v['method'],v['seed']):v for v in rows};rng=np.random.default_rng(20261014);out=[]
for ident in target_ids():
    for method in ['G0','J0','D','H','A','MAG_PHA_ISLANDS']:
        delta=np.array([float(lookup[ident,'ARX',str(s)]['local_complex_error'])-float(lookup[ident,method,str(s)]['local_complex_error']) for s in CFG['seeds']])
        boot=np.mean(delta[rng.integers(0,8,(32768,8))],axis=1)
        out.append(dict(target=ident,comparison='ARX_vs_'+method,local_better=int(sum(delta<0)),mean_delta=float(delta.mean()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975))))
csvout('AREA_ABLATION_PAIRED.csv',out)

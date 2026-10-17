"""Full archive structure checks and independent Windows numerical replay."""
from review_analysis import *

full=OUT/'retrieved';root=full/'output/complex_area_20261014';records=[];st=time.time()
assert read(OUT/'RECOVERY.json')['verified']
manifest=read(root/'MANIFEST.json')
for name,h in manifest['hashes'].items():
    assert sha(full/name)==h and sha(SNAP/name)==h
for ident in target_ids():
    data=target(ident);g=Features(data);q=data['quadrature']
    for method in CFG['methods']:
        for seed in CFG['seeds']:
            d=root/'jobs'/ident/method/f'seed_{seed}';s=read(d/'summary.json');baseline=method.startswith('MAG_PHA')
            for gen in range(20,301,20):
                with np.load(d/f'generation_{gen:04}.npz') as a:
                    X=a['X'];F=a['F'];ix,selection=knee(F[:,:2])
                    assert X.shape==(200,len(data['lower'])) and F.shape==(200,2 if baseline else 3)
                    assert np.isfinite(X).all() and np.isfinite(F).all()
                    assert np.all(X>=data['lower']-1e-12) and np.all(X<=data['upper']+1e-12)
                    assert ix==int(a['selected_index'])
                    if gen==300:lp=X[ix];training=F[ix]
            base=q.score(lp[None,:])[0]
            base_delta=abs(base-training[:2]);base_ok=bool(np.all(base_delta<=np.array(CFG['audit']['base_atol'])+CFG['audit']['base_rtol']*abs(base)))
            aux=None if baseline else float(g.score(lp[None,:],method)[0]);aux_delta=None if baseline else abs(aux-training[2])
            aux_ok=baseline or bool(aux_delta<=CFG['audit']['feature_atol']+CFG['audit']['feature_rtol']*abs(aux))
            assert base_ok and aux_ok,(ident,method,seed,base,training,aux)
            records.append(dict(target=ident,method=method,seed=seed,primary_replay=base.tolist(),primary_delta=base_delta.tolist(),auxiliary_replay=aux,auxiliary_delta=aux_delta,
                passed_original_objective_tolerances=base_ok and aux_ok))
    print('local replay condition done',ident,flush=True)
dump(OUT/'LOCAL_NUMERICAL_RECHECK.json',dict(tasks=len(records),saved_populations_checked=15*len(records),all_passed=True,
    thresholds='Original audit primary/aux tolerances unchanged; replay is supplementary, not replacement audit',
    implementation='Compact frozen source identical to full archive; Case1 uses frozen pure-Python fallback on Windows, Linux native.so preserved',
    max_primary_absolute_delta=np.max([r['primary_delta'] for r in records],axis=0).tolist(),
    max_auxiliary_relative_delta=max(r['auxiliary_delta']/max(abs(r['auxiliary_replay']),1e-12) for r in records if r['auxiliary_replay'] is not None),
    records=records,seconds=time.time()-st))

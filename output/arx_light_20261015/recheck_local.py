"""Read-only portable replay of frozen primary and auxiliary definitions."""
from review_analysis import *
from reference_features import Features as Reference

class Portable(Reference):
    def __init__(self,data):
        super().__init__(data)
        rec=next(x for x in read(OUT/'PREFLIGHT.json')['checks'] if x['target']==data['id'])
        self.new=np.array([s['new_area_den'] for s in rec['scales']])
    def fixed(self,lp,names,n=32,refinement=1):
        if names==('G0',):return super().fixed(lp,names,n,refinement)
        orig=('J0','ARX');v=super().fixed(lp,orig,n,refinement)
        old=np.array([w['area_den'][2] for w in self.windows]);out=[]
        for name in names:
            if name=='J0':out.append(v[:,0])
            elif name=='J0_W':out.append(v[:,0]*(old/self.new)**2)
            elif name=='ARX_L':out.append(v[:,1])
            elif name=='ARX_S_L':out.append(v[:,1]*(old/self.new)**2)
            else:raise KeyError(name)
        return np.column_stack(out)

def check_job(job):
    ident,method,seed=job;d=DATA/'jobs'/ident/method/f'seed_{seed}';data=target(ident)
    g=Portable(data);q=data['quadrature'];baseline=method=='MAG_PHA'
    for gen in range(20,301,20):
        with np.load(d/f'generation_{gen:04}.npz') as a:
            X,F=a['X'],a['F'];ix,_=knee(F[:,:2])
            assert X.shape==(200,len(data['lower'])) and F.shape==(200,2 if baseline else 3)
            assert np.isfinite(X).all() and np.isfinite(F).all()
            assert np.all(X>=data['lower']-1e-12) and np.all(X<=data['upper']+1e-12)
            assert ix==int(a['selected_index'])
            if gen==300:lp=X[ix];training=F[ix]
    a=read(d/'audit.json');assert sha(d/'audited_curve.npz')==a['curve_sha256']
    with np.load(d/'audited_curve.npz') as curve:
        z,_=data['circuit'].response(10**curve['x'],10**lp)
        delta=abs(z-curve['z']);curve_relative=float(np.max(delta/np.maximum(abs(curve['z']),1e-30)))
        tight=bool(np.all(delta<=1e-9+1e-11*abs(curve['z'])))
        if not tight:
            indices=np.flatnonzero(delta>1e-9+1e-11*abs(curve['z']))
            record=dict(target=ident,method=method,seed=seed,extra_tight_probe_passed=False,
                max_relative_delta=curve_relative,indices=indices.tolist(),
                log_parameters=lp.tolist(),x=curve['x'][indices].tolist(),
                original_nodal_gate_passed=bool(np.all(delta<=CFG['audit']['nodal_atol']+CFG['audit']['nodal_rtol']*abs(curve['z']))))
            dump(OUT/f"curve_probe_{ident.replace('/','_')}_{method}_{seed}.json",record)
        # Original frozen 1e-8/1e-9 nodal gate, not the unplanned 1e-11 probe.
        np.testing.assert_allclose(z,curve['z'],rtol=CFG['audit']['nodal_rtol'],atol=CFG['audit']['nodal_atol'])
    base=q.score(lp[None,:])[0];bd=abs(base-training[:2])
    base_ok=bool(np.all(bd<=np.array(CFG['audit']['base_atol'])+CFG['audit']['base_rtol']*abs(base)))
    aux=None if baseline else float(g.score(lp[None,:],method)[0]);ad=None if baseline else abs(aux-training[2])
    aux_ok=baseline or bool(ad<=CFG['audit']['feature_atol']+CFG['audit']['feature_rtol']*abs(aux))
    assert base_ok and aux_ok,(ident,method,seed,base,training,aux)
    # Stronger supplementary relative-error report, without changing original acceptance.
    return dict(target=ident,method=method,seed=seed,primary_replay=base.tolist(),primary_delta=bd.tolist(),auxiliary_replay=aux,
        auxiliary_delta=ad,auxiliary_relative_delta=None if baseline else ad/max(abs(aux),1e-12),curve_relative_delta=curve_relative,
        extra_tight_curve_probe_passed=tight,all_original_gates_passed=True)

def main():
    st=time.time();jobs=[(t,m,s) for t in target_ids() for m in CFG['methods'] for s in CFG['seeds']];rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for r in pool.map(check_job,jobs):
            rows.append(r)
            if len(rows)%12==0:print('portable numerical replay',len(rows),'/',len(jobs),flush=True)
    dump(OUT/'LOCAL_NUMERICAL_RECHECK.json',dict(tasks=len(rows),saved_populations_checked=15*len(rows),all_original_gates_passed=True,
        implementation='Original Python circuit/primary fallback and Python Gauss area; same frozen target scales; Linux native binaries retained, not loaded on Windows',
        max_primary_absolute_delta=np.max([r['primary_delta'] for r in rows],axis=0).tolist(),
        max_auxiliary_relative_delta=max(r['auxiliary_relative_delta'] for r in rows if r['auxiliary_relative_delta'] is not None),
        max_curve_relative_delta=max(r['curve_relative_delta'] for r in rows),
        extra_tight_curve_probe_failures=sum(not r['extra_tight_curve_probe_passed'] for r in rows),
        records=rows,seconds=time.time()-st))

if __name__=='__main__':main()

"""Recompute selected curves, objective drift, nodal model and full-band defects."""
from core import *
from features import Features,areas
from run import path_for,verify
import traceback

def one(job):
    ident,method,seed,root,smoke=job;d=path_for(root,ident,method,seed)
    try:
        summary=verify(d);data=target(ident);q=data['quadrature'];c=data['circuit'];g=Features(data)
        with np.load(d/'generation_0300.npz') as a:
            x=a['X'];f=a['F'];index,selection=knee(f[:,:2]);lp=x[index];training=f[index]
            assert index==int(a['selected_index'])
        x0=np.load(d/'initial.npy');expected=np.random.default_rng(seed).uniform(data['lower'],data['upper'],size=x0.shape)
        np.testing.assert_array_equal(x0,expected)
        history=json.loads((d/'history.json').read_text());assert [h['evaluations'] for h in history]==list(range(200,60001,200))
        for generation in range(20,301,20):
            with np.load(d/f'generation_{generation:04}.npz') as a:
                assert a['X'].shape==(200,len(c.p)) and a['F'].shape==(200,2 if method=='MAG_PHA' else 3)
                assert np.isfinite(a['F']).all()
        fresh=q.score(lp[None,:])[0]
        np.testing.assert_allclose(fresh,training[:2],rtol=1e-10,atol=1e-9)
        if method!='MAG_PHA':np.testing.assert_allclose(g.score(lp[None,:],method)[0],training[2],rtol=1e-10,atol=1e-9)
        levels=[];previous=None;stable=False
        for level in CFG['audit']['levels']:
            primary,xx,zz,tt=q.reference(lp,level)
            aux=None if method=='MAG_PHA' else float(g.score(lp[None,:],method,level)[0])
            geometry,details=g.components(lp[None,:],level=level)
            row=dict(level=level,primary=primary.tolist(),auxiliary=aux,geometry={k:float(v[0]) for k,v in geometry.items()})
            levels.append(row)
            if previous is not None:
                stable=np.all(abs(primary-np.array(previous['primary']))<=np.array(CFG['audit']['base_atol'])+CFG['audit']['base_rtol']*abs(primary))
                if aux is not None:stable=bool(stable and abs(aux-previous['auxiliary'])<=CFG['audit']['feature_atol']+CFG['audit']['feature_rtol']*abs(aux))
                if stable:break
            previous=row
        base_drift=bool(np.all(abs(primary-training[:2])<=np.array(CFG['audit']['base_atol'])+CFG['audit']['base_rtol']*abs(primary)))
        aux_drift=True if aux is None else bool(abs(aux-training[2])<=CFG['audit']['feature_atol']+CFG['audit']['feature_rtol']*abs(aux))
        anchored=g.anchored_geometry(lp)
        geometry_anchor_agrees={k:bool(abs(v[0]-geometry[k][0])<=CFG['audit']['feature_atol']+CFG['audit']['feature_rtol']*abs(v[0])) for k,v in anchored.items()}
        if method not in ('MAG_PHA','PTC_ref','C_ref'):
            anchored_aux=float(np.mean([anchored[n][0] for n in method.split('_')]))
            aux_drift=bool(aux_drift and abs(anchored_aux-training[2])<=CFG['audit']['feature_atol']+CFG['audit']['feature_rtol']*abs(anchored_aux))
        else:anchored_aux=None
        # Independent circuit laws on actual final curve points, especially sharp target/candidate features.
        sample=np.unique(np.r_[np.linspace(xx[0],xx[-1],25),[e['x'] for e in data['record']['events']],
            xx[np.argsort(abs(zz))[-8:]]])
        nodal_check=checked_nodal(c,10**sample,10**lp)
        fit=diag.events(xx,zz,BASECFG);events,extras=diag.match(data['record']['events'],fit,BASECFG)
        if data['kind']=='measured':
            phase=[];extra_phase=[];diagnostic_stable=True;local=[];bands=[]
        else:
            diagnostic,(xx,zz,tt)=diag.diagnostic(q,lp,data['record']['events'],data['record']['phase_events'])
            events=diagnostic['events'];extras=diagnostic['extra_events'];phase=diagnostic['phase_events'];extra_phase=diagnostic['extra_phase_events']
            diagnostic_stable=diagnostic['stable'];local=diagnostic['local'];bands=diagnostic['bands']
        np.savez_compressed(d/'audited_curve.npz',x=xx,z=zz,target=tt,log_parameters=lp)
        rec=dict(status='audited',target=ident,method=method,seed=seed,selection=selection,training=training.tolist(),
            objective=primary.tolist(),converged=bool(stable),base_training_agrees=base_drift,aux_training_agrees=aux_drift,
            diagnostic_stable=diagnostic_stable,accepted=bool(stable and base_drift and aux_drift and diagnostic_stable),
            anchored_geometry={k:float(v[0]) for k,v in anchored.items()},geometry_anchor_agrees=geometry_anchor_agrees,anchored_auxiliary=anchored_aux,
            nodal_passed=True,nodal_check=nodal_check,levels=levels,geometry={k:float(v[0]) for k,v in geometry.items()},blocks=details,
            amplitude_events=events,extra_amplitude_events=extras,phase_events=phase,extra_phase_events=extra_phase,
            local=local,bands=bands,model_limit=data['kind']=='measured',
            endpoint_error_ohm=abs(zz[[0,-1]]-tt[[0,-1]]).tolist(),seconds=summary['seconds'],cpu_seconds=summary['cpu_seconds'],
            base_frequency_queries=summary['base_frequency_queries'],auxiliary_frequency_queries=summary['auxiliary_frequency_queries'],
            curve_sha256=sha(d/'audited_curve.npz'))
        dump(d/'audit.json',rec);return rec
    except Exception:
        rec=dict(status='audit_failed',target=ident,method=method,seed=seed,traceback=traceback.format_exc())
        dump(d/'AUDIT_FAILED.json',rec);return rec

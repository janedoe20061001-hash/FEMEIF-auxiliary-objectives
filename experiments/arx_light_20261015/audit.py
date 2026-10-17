"""Independent final selection, budget, quadrature, nodal and structure audit."""
from core import *
from features import Features
from checks import polygon,agrees
from run import path_for,verify
import traceback

def one(job):
    ident,method,seed,root,smoke=job;d=path_for(root,ident,method,seed)
    try:
        summary=verify(d);data=target(ident);q=data['quadrature'];c=data['circuit'];g=Features(data)
        baseline=method in ('MAG_PHA','MAG_PHA_ISLANDS')
        x0=np.load(d/'initial.npy');np.testing.assert_array_equal(x0,np.random.default_rng(seed).uniform(data['lower'],data['upper'],size=x0.shape))
        history=json.loads((d/'history.json').read_text());assert [h['evaluations'] for h in history]==list(range(200,60001,200))
        for gen in range(20,301,20):
            with np.load(d/f'generation_{gen:04}.npz') as a:
                assert a['X'].shape==(200,len(c.p)) and a['F'].shape==(200,2 if baseline else 3)
                assert np.isfinite(a['F']).all();idx,selection=knee(a['F'][:,:2]);assert idx==int(a['selected_index'])
                if gen==300:lp=a['X'][idx];training=a['F'][idx]
        np.testing.assert_allclose(q.score(lp[None,:])[0],training[:2],rtol=1e-10,atol=1e-9)
        levels=[];prev=None;stable=False
        for level in [1,2,4,8]:
            primary,xx,zz,tt=q.reference(lp,level);levels.append(dict(level=level,primary=primary.tolist()))
            if prev is not None:
                stable=bool(np.all(abs(primary-prev)<=np.array(CFG['audit']['base_atol'])+CFG['audit']['base_rtol']*abs(primary)))
                if stable:break
            prev=primary
        base_drift=bool(np.all(abs(primary-training[:2])<=np.array(CFG['audit']['base_atol'])+CFG['audit']['base_rtol']*abs(primary)))
        aux_record=None;aux_ok=True
        if not baseline:
            fresh=g.score(lp[None,:],method)[0];np.testing.assert_allclose(fresh,training[2],rtol=1e-10,atol=1e-9)
            a=g.fixed(lp,(method,),32,1);b=g.fixed(lp,(method,),32,2)
            poly=polygon(g,lp,(method,),128);poly2=polygon(g,lp,(method,),256)
            aux_ok=agrees(a,b) and agrees(b,poly2) and agrees(poly,poly2) and agrees(training[2],b.mean())
            aux_record=dict(fresh=fresh,fine=float(b.mean()),polygon=float(poly2.mean()),accepted=aux_ok,window_scores=b[:,0].tolist(),
                fine_converged=agrees(a,b),polygon_converged=agrees(poly,poly2),independent_agrees=agrees(b,poly2))
        sample=np.unique(np.r_[np.linspace(xx[0],xx[-1],25),[e['x'] for e in data['record']['events']],xx[np.argsort(abs(zz))[-8:]]])
        nodal=checked_nodal(c,10**sample,10**lp)
        diagnostic,(xx,zz,tt)=diag.diagnostic(q,lp,data['record']['events'],data['record']['phase_events'])
        np.savez_compressed(d/'audited_curve.npz',x=xx,z=zz,target=tt,log_parameters=lp)
        rec=dict(status='audited',target=ident,method=method,seed=seed,selection=selection,training=training.tolist(),objective=primary.tolist(),
            converged=stable,base_training_agrees=base_drift,auxiliary=aux_record,nodal_check=nodal,nodal_passed=True,
            accepted=bool(stable and base_drift and aux_ok and diagnostic['stable']),diagnostic_stable=diagnostic['stable'],levels=levels,
            amplitude_events=diagnostic['events'],extra_amplitude_events=diagnostic['extra_events'],phase_events=diagnostic['phase_events'],
            extra_phase_events=diagnostic['extra_phase_events'],local=diagnostic['local'],bands=diagnostic['bands'],
            cpu_seconds=summary['cpu_seconds'],seconds=summary['seconds'],curve_sha256=sha(d/'audited_curve.npz'))
        dump(d/'audit.json',rec);return rec
    except Exception:
        rec=dict(status='audit_failed',target=ident,method=method,seed=seed,traceback=traceback.format_exc());dump(d/'AUDIT_FAILED.json',rec);return rec

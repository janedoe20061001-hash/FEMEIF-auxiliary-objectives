"""No fitting: identities, blind-spot examples and full-population control equivalence."""
from pathlib import Path
import importlib.util,json,sys
import numpy as np
from global_metrics import GlobalObjectives,GROUPS,AUX,cumulative_length,variations,integral_squared

u=np.linspace(0,1,23);h=np.diff(u)
p=np.column_stack([u,2*u,u]);swapped=np.column_stack([u,u,2*u]);reflected=np.column_stack([u,-2*u,-u])
np.testing.assert_allclose(cumulative_length(p),cumulative_length(swapped),atol=1e-14)
pos,neg=variations(p);sp,sn=variations(swapped);rp,rn=variations(reflected)
assert integral_squared(pos+neg-sp-sn,h)>0
np.testing.assert_array_equal(pos+neg,rp+rn)
assert integral_squared(np.concatenate([pos-rp,neg-rn],axis=-1),h)>0
np.testing.assert_allclose(pos-neg,p[:,1:]-p[:1,1:],atol=1e-14)
e=u**3;back=e[-1]-e
sym=.5*(integral_squared(e[:,None],h)+integral_squared(back[:,None],h))
np.testing.assert_allclose(sym,.5*(integral_squared(back[::-1,None],h)+integral_squared(e[::-1,None],h)),atol=1e-14)
# An isolated final-segment length error is not confined to the final interval
# in the reverse cumulative profile. This is a software counterexample, not data treatment.
late=np.zeros_like(u);late[-1]=1
assert integral_squared((late[-1]-late)[:,None],h)>integral_squared(late[:,None],h)
f=np.geomspace(10,1e6,len(u));z=10**(1+u*u)*np.exp(1j*(-.6+u))
candidates=np.array([z*2,z*np.exp(.2j),z[::-1],10**(1+.3*u)*np.exp(.1j*u)])
primary=GlobalObjectives(f,target_z=z).evaluate_spectra(candidates)
for group in GROUPS:
    obj=GlobalObjectives(f,aux=group,target_z=z)
    np.testing.assert_allclose(obj.evaluate_spectra(z),0,atol=1e-14)
    vals=obj.evaluate_spectra(candidates);np.testing.assert_array_equal(vals[:,:2],primary)
    assert vals.shape==(4,2+len(group)) and np.isfinite(vals).all()
obj=GlobalObjectives(f,target_z=z);path=obj.path(candidates);plus,minus=variations(path)
vp=(plus+minus-obj.tv)/obj.axis_total
dp=((path[...,1:]-path[...,:1,1:])-(obj.p[:,1:]-obj.p[:1,1:]))/obj.axis_total
direction=np.concatenate([(plus-obj.pos)/obj.axis_total,(minus-obj.neg)/obj.axis_total],axis=-1)
np.testing.assert_allclose(integral_squared(direction,h)/2,
    .5*integral_squared(vp,h)/2+.5*integral_squared(dp,h)/2,rtol=1e-13,atol=1e-14)
g=np.array([.5-.5/np.sqrt(3),.5+.5/np.sqrt(3)])
samples=(1-g[:,None])*direction[:,:-1,None,:]+g[:,None]*direction[:,1:,None,:]
np.testing.assert_allclose(np.sum(np.sum(samples*samples,axis=-1)*h[:,None]/2,axis=(1,2)),
                           integral_squared(direction,h),rtol=1e-13,atol=1e-14)
root=Path(__file__).resolve().parent;old=root.parent/'curve3d_refinement_20261003'
if not old.exists():old=Path('/root/autodl-tmp/femeif_curve3d_refinement_20261003')
sys.path.insert(0,str(old))
spec=importlib.util.spec_from_file_location('old_metrics',old/'global_metrics.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
records=[];diagnostics=[]
for case in ['case1','case2']:
    folder=root/case;sys.path.insert(0,str(folder));cfg=json.loads((folder/'config.json').read_text(encoding='utf-8'))
    if case=='case1':
        from legacy_objectives import impedance,TARGET,LOWER,UPPER
        freq=np.geomspace(100,1e6,180);target=impedance(freq,TARGET)[0];nvar=18
    else:
        from case2 import impedance,load_target,LOWER,UPPER,N_VAR
        freq,target=load_target({**cfg['target_data'],'file':str(folder/cfg['target_data']['file'])});nvar=N_VAR
    x=np.random.default_rng(200).uniform(LOWER,UPPER,(cfg['population'],nvar));spectra=impedance(freq,10**x)
    control=GlobalObjectives(freq,aux=AUX[:2],target_z=target)
    new=control.evaluate_spectra(spectra);previous=module.GlobalObjectives(freq,aux=AUX[:2],target_z=target).evaluate_spectra(spectra)
    np.testing.assert_array_equal(new,previous)
    for group in GROUPS:
        value=GlobalObjectives(freq,aux=group,target_z=target).evaluate_spectra(spectra)
        assert value.shape==(cfg['population'],2+len(group)) and np.isfinite(value).all()
    records.append(dict(case=case,full_initial_population=cfg['population'],frequency_points=len(freq),
                        old_controls_bitwise_equal=True,max_abs_difference=float(abs(new-previous).max())))
    diagnostics.append(dict(case=case,**control.target_summary()))
result=dict(status='passed',no_optimization_performed=True,zero_error=True,axis_exchange_counterexample=True,
    direction_counterexample=True,positive_negative_identity=True,reversal_identity=True,
    independent_gaussian_integrals=True,control_equivalence=records,target_only_diagnostics=diagnostics)
(root/'software_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

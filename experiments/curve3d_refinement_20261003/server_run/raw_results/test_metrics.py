"""Formula checks only; no optimization or reduced-budget experiment."""
from pathlib import Path
import importlib.util,sys,json
import numpy as np
from global_metrics import GlobalObjectives,AUX,cumulative_area,cumulative_length,squared_linear_integral
from curve_math import area

f=np.geomspace(10,1e6,19);u=np.linspace(0,1,len(f))
z=10**(1+u*u)*np.exp(1j*(-.6+u))
obj=GlobalObjectives(f,aux=AUX,target_z=z)
np.testing.assert_allclose(obj.evaluate_spectra(z),0,atol=1e-14)
candidates=np.array([z*2,z*np.exp(.2j),z[::-1],10**(1+.3*u)*np.exp(.1j*u)])
val=obj.evaluate_spectra(candidates)
np.testing.assert_array_equal(val[:,:2],GlobalObjectives(f,target_z=z).evaluate_spectra(candidates))
np.testing.assert_allclose(val[:2,4],0,atol=1e-14) # area is translation invariant
np.testing.assert_allclose(val[:2,5],0,atol=1e-14) # length is translation invariant
assert (val[:2,:2].sum(axis=1)>0).all()
straight=np.column_stack([u,u*0,u*0]);wiggle=np.column_stack([u,np.sin(2*np.pi*u),u*0])
np.testing.assert_allclose(area(wiggle),area(straight),atol=1e-14)
assert squared_linear_integral(cumulative_area(wiggle)-cumulative_area(straight),np.diff(u))>0
assert cumulative_length(wiggle)[-1]>cumulative_length(straight)[-1]
# Exact quadratic integration: independent two-point Gaussian evaluation.
g=np.array([.5-.5/np.sqrt(3),.5+.5/np.sqrt(3)])
for coord in ['logpolar','stdpolar']:
    p=obj.path(candidates,coord);ref=obj.references[coord]
    np.testing.assert_allclose(cumulative_area(p)[:,-1],area(p),atol=1e-14)
    errors=cumulative_area(p)-ref['cumarea']
    q=(1-g[:,None])*errors[:,:-1,None,:]+g[:,None]*errors[:,1:,None,:]
    gauss=np.sum(np.sum(q*q,axis=-1)*obj.du[:,None]/2,axis=(1,2))
    np.testing.assert_allclose(gauss,squared_linear_integral(errors,obj.du),rtol=1e-13,atol=1e-14)
ref=obj.references['stdpolar'];xy=np.column_stack([obj.a,obj.phase])
samples=(1-g[:,None])*xy[:-1,None,:]+g[:,None]*xy[1:,None,:]
mu=np.sum(samples*obj.du[:,None,None]/2,axis=(0,1))
std=np.sqrt(np.sum((samples-mu)**2*obj.du[:,None,None]/2,axis=(0,1)))
np.testing.assert_allclose(ref['offset'],mu,rtol=1e-14,atol=1e-14)
np.testing.assert_allclose(ref['scale'],std,rtol=1e-14,atol=1e-14)
root=Path(__file__).resolve().parent
old=root.parent/'curve3d_logpolar_native_20261003'
if not old.exists():old=Path('/root/autodl-tmp/femeif_curve3d_logpolar_native_20261003')
sys.path.insert(0,str(old))
spec=importlib.util.spec_from_file_location('old_metrics',old/'global_metrics.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
records=[]
for case in ['case1','case2']:
    case_root=root/case;sys.path.insert(0,str(case_root))
    cfg=json.loads((case_root/'config.json').read_text(encoding='utf-8'))
    if case=='case1':
        from legacy_objectives import impedance,TARGET,LOWER,UPPER
        freq=np.geomspace(100,1e6,180);target=impedance(freq,TARGET)[0];nvar=18
    else:
        from case2 import impedance,load_target,LOWER,UPPER,N_VAR
        freq,target=load_target({**cfg['target_data'],'file':str(case_root/cfg['target_data']['file'])});nvar=N_VAR
    parameters=np.random.default_rng(100).uniform(LOWER,UPPER,(cfg['population'],nvar))
    spectra=impedance(freq,10**parameters)
    new=GlobalObjectives(freq,aux=AUX[:2],target_z=target).evaluate_spectra(spectra)
    previous=module.GlobalObjectives(freq,aux=AUX[:2],target_z=target).evaluate_spectra(spectra)
    np.testing.assert_array_equal(new,previous)
    all_values=GlobalObjectives(freq,aux=AUX,target_z=target).evaluate_spectra(spectra)
    assert all_values.shape==(cfg['population'],8) and np.isfinite(all_values).all()
    records.append(dict(case=case,full_initial_population=cfg['population'],frequency_points=len(freq),
                        old_controls_bitwise_equal=True,max_abs_difference=float(abs(new-previous).max())))
record=dict(status='passed',zero_error=True,cancellation_counterexample=True,length_information_check=True,
    independent_gaussian_integrals=True,no_optimization_performed=True,control_equivalence=records)
(root/'software_checks.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record))

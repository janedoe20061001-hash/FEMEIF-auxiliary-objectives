"""Deterministic target generation and prior-model cross-checks, before optimization."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import tarfile
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.append(str(ROOT/'tmp/curve_refinement_deps'))
from model import module,response
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
    prior=ROOT/'experiments/crosscase_consolidation_20261005/server_run/review_results'
    for case in ('case1','case2'):
        d=HERE/'frozen'/case;d.mkdir(parents=True,exist_ok=True)
        for name in ['legacy_objectives.py','config.json','requirements-lock.txt']+(['case2.py'] if case=='case2' else []):shutil.copy2(prior/case/name,d/name)
        if case=='case2':shutil.copytree(prior/case/'data',d/'data',dirs_exist_ok=True)
    source=ROOT/'experiments/parameter_curve_population_20261006/run_population.py'
    shutil.copy2(source,HERE/'frozen/population_source.py')
    pop=module('scalar_population_source',HERE/'frozen/population_source.py')
    records=[];checks=[]
    for case in ('case1','case2'):
        cfg=json.loads((prior/case/'config.json').read_text())
        if case=='case1':
            n=module('prepare_case1',prior/case/'legacy_objectives.py');f=np.geomspace(100,1e6,180);target=n.impedance(f,n.TARGET)[0]
            names=[f'{kind}{i}' for i in range(1,7) for kind in ('R','L','C')];true=n.TARGET.flatten();tree=['S',*[['P',f'R{i}',f'L{i}',f'C{i}'] for i in range(1,7)]]
        else:
            n=module('prepare_case2',prior/case/'case2.py');f,target=n.load_target(cfg['target_data']);names=list(n.PARAMETER_NAMES);true=None
            tree=['S',*[['P',f'C{i}',['S',f'L{i}',f'R{2*i-1}'],f'R{2*i}'] for i in range(1,4)],'C4','R7','L4']
        records.append(dict(id=case,title=case,family=case,condition='paper',tree=tree,parameter_names=names,
            target_parameters=None if true is None else true.tolist(),frequency_hz=f.tolist(),target_real=target.real.tolist(),target_imag=target.imag.tolist(),
            lower_log10=n.LOWER.tolist(),upper_log10=n.UPPER.tolist(),alpha=cfg['final_selection']['phase_weight'],population=cfg['population'],
            mutation_variable_probability=cfg['mutation_variable_probability'],seed_start=2300,seed_count=100,generations=300))
        x=np.random.default_rng(20261007).uniform(n.LOWER,n.UPPER,(7,len(names)))
        np.testing.assert_allclose(response(tree,names,f,10**x),n.impedance(f,10**x),rtol=1e-12,atol=1e-9)
        checks.append(dict(id=case,prior_network_equality=True))
    f=np.geomspace(100,1e8,240)
    for fi,(family,title,tree) in enumerate(pop.FAMILIES):
        for condition in range(3):
            rng=np.random.default_rng(20261007+1000*fi+condition)
            circuit=pop.sample_network(family,title,tree,'NEW',condition,condition,0,pop.lhs(rng,1,16)[0])
            target=response(tree,circuit.names,f,circuit.p)[0]
            x=np.log10(circuit.p);low=x-2;high=x+2
            trial=10**rng.uniform(low,high,(7,len(x)))
            np.testing.assert_allclose(response(tree,circuit.names,f,trial),circuit.response(f,trial)[0],rtol=1e-10,atol=1e-8)
            ar=float(np.ptp(20*np.log10(abs(target))));pr=float(np.ptp(np.angle(target)));assert ar>0 and pr>0
            ident=f'{family}_c{condition+1}'
            records.append(dict(id=ident,title=title,family=family,condition=['lowQ_close','midQ_mid','highQ_wide'][condition],tree=tree,
                parameter_names=circuit.names,target_parameters=circuit.p.tolist(),frequency_hz=f.tolist(),target_real=target.real.tolist(),target_imag=target.imag.tolist(),
                lower_log10=low.tolist(),upper_log10=high.tolist(),alpha=(ar/pr)**2,alpha_rule='target_range_squared_ratio',target_magnitude_range_db=ar,target_phase_range_rad=pr,
                population=120,mutation_variable_probability=1/len(x),seed_start=2300,seed_count=20,generations=300,
                nominal_Q=circuit.design['nominal_unit_Q'].tolist(),spacing_ratio=float(circuit.design['spacing_ratio'])))
            checks.append(dict(id=ident,prior_network_equality=True))
    dump(HERE/'circuits.json',records)
    from model import Model
    for r in records:
        m=Model(r['id']);np.testing.assert_allclose(m.score_spectra(m.target),0,atol=1e-12)
        z=m.target[None]*np.array([[1.1]])
        np.testing.assert_allclose(m.score_spectra(z)[:,3],np.mean(abs(z-m.target),axis=1),atol=1e-10,rtol=1e-12)
    # Nonconstant errors distinguish mean distance from root mean squared distance.
    residual=np.array([0+0j,3+4j]);assert abs(residual).mean()==2.5 and np.sqrt(np.mean(abs(residual)**2))!=2.5
    # Bridge expression independently compared to nodal voltage solve.
    m=Model('rlc_bridge_c2');pars=np.array(m.record['target_parameters']);v=dict(zip(m.record['parameter_names'],pars));w=2*np.pi*m.f
    y0=np.full(len(w),1/v['R1']);y1=1/(1j*w*v['L1']);y2=1j*w*v['C1'];y3=np.full(len(w),1/v['R2']);y4=np.full(len(w),1/v['R3'])
    a=np.array([[y0+y1+y4,-y4],[-y4,y2+y3+y4]]).transpose(2,0,1)
    uv=np.linalg.solve(a,np.column_stack([y0,y2])[...,None])[...,0]
    nodal=1/(y0*(1-uv[:,0])+y2*(1-uv[:,1]))
    np.testing.assert_allclose(m.target,nodal,rtol=1e-10,atol=1e-8)
    assert len(records)==50
    total=sum(r['seed_count']*2*r['population']*r['generations'] for r in records);assert total==90120000
    dump(HERE/'PREPARATION_CHECKS.json',dict(passed=True,circuits=50,synthetic_circuits=48,topologies=16,planned_runs=2320,planned_evaluations=total,
        network_checks=checks,target_zero=True,distance_abs_equivalence=True,distance_not_rms=True,bridge_nodal_equality=True,
        source_population_sha256=sha(source),source_paper_sha256=sha(ROOT/'Feature-Engineering_Enabled_Multi-Objective_Evolutionary_Impedance_Fitting_Technique.pdf')))
    print(json.dumps(dict(circuits=50,runs=2320,evaluations=total,preparation_passed=True)))
if __name__=='__main__':main()

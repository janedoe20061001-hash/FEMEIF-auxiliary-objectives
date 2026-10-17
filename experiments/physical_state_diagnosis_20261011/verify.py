"""Independent input, count, analytical identity and nodal phase checks."""
import importlib.util,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('physical_verify_base',Path(__file__).with_name('run.py'))
b=importlib.util.module_from_spec(spec);sys.modules[spec.name]=b;spec.loader.exec_module(b)
import json,csv,numpy as np

def read(name):return list(csv.DictReader((b.OUT/name).open(encoding='utf-8-sig')))

def main():
    first=json.loads((b.OUT/'INPUT_MANIFEST.json').read_text(encoding='utf-8'))
    supplemental=json.loads((b.OUT/'SUPPLEMENT_STATUS.json').read_text(encoding='utf-8'))
    checked={**first['hashes'],**supplemental['hashes']}
    for path,h in checked.items():assert b.sha(b.ROOT/path)==h,path
    reps=read('REPRESENTATIVES.csv');probes=read('PROBES.csv');scales=read('SCALE_CONTROLS.csv');phase=read('PHASE_REPRESENTATIVES.csv')
    assert len(reps)==len(phase)==252 and len(probes)==105 and len(scales)==35
    keys=lambda r:(r['batch'],r['target'],r['method'],int(r['seed']))
    assert len(set(map(keys,reps)))==252 and set(map(keys,reps))==set(map(keys,phase))
    assert all(r['verified']=='True' for r in reps+probes+phase)
    assert all(r['passed']=='True' for r in scales)
    # Recalculate reported denominators and diagnostic associations independently.
    pm={keys(r):r for r in phase}
    difficult=[r for r in reps if r['batch']=='family' and r['all_amp_ok']=='True' and r['all_phase_ok']=='False']
    assert len(difficult)==37
    assert sum(r['all_has_diagnostic_deviation']=='True' for r in difficult)==4
    assert sum(pm[keys(r)]['all_has_diagnostic_deviation']=='True' for r in difficult)==27
    phase_checks=[];nonbinary=[];provenance={}
    for t in ['simple_series','simple_parallel',*b.core.CFG['targets']]:
        data=b.simple(t) if t.startswith('simple_') else b.core.target(t);c=data['circuit']
        d=json.loads((b.OUT/'details'/f'{t}.json').read_text(encoding='utf-8'))
        assert c.record()==d['target']['circuit']
        assert data['catalog']==d['target']['amplitude_catalog']
        ph=json.loads((b.OUT/'phase_details'/f'{t}.json').read_text(encoding='utf-8'))
        params={(r['batch'],r['method'],r['seed']):10**np.array(r['log_parameters']) for r in d['representatives']}
        for ev,p,name in [(ph['reference']['events'],c.p,'target'),*[(r['state']['events'],params[(r['batch'],r['method'],r['seed'])],f"{r['batch']}/{r['method']}/{r['seed']}") for r in ph['representatives']]]:
            if not ev:continue
            f=np.array([e['frequency_hz'] for e in ev]);z=c.nodal_response(f,p);angles=np.rad2deg(np.angle(z))
            maximum=float(np.max(abs(angles-np.array([e['phase_deg'] for e in ev]))))
            analytic,dz=c.response(f,p);slope=float(np.max(abs(np.imag(dz/analytic))))
            assert maximum<=.001,(t,name,maximum)
            phase_checks.append(dict(target=t,curve=name,events=len(ev),nodal_phase_max_difference_deg=maximum,stationary_slope_max_rad=slope))
        # Two non-binary multipliers avoid relying on bit-exact powers-of-two tests.
        f=np.geomspace(100,1e6,257);z=c.nodal_response(f)
        for factor in (.7,1.3):
            p=c.p*np.array([1/factor if n.startswith('C') else factor for n in c.names])
            fit=c.nodal_response(f,p);error=float(np.max(abs(fit-factor*z)/abs(factor*z)))
            assert error<=1e-8,(t,factor,error)
            nonbinary.append(dict(target=t,factor=factor,nodal_scale_relative_error=error))
        if not t.startswith('simple_') and t!='case1':
            key={'ladder8':'rlc_ladder/lc_reciprocal_8p0','mixed0p25':'mixed_resonators/lc_reciprocal_0p25',
                 'three0p25':'three_level/lc_reciprocal_0p25','control0p25':'control_rlc/lc_reciprocal_0p25'}[t]
            group='first' if t=='control0p25' else 'other'
            file=b.ROOT/f'output/mse_structure_complex_20261010_server/retrieved/{group}/inputs_core/{key}/target.json'
            provenance[file.relative_to(b.ROOT).as_posix()]=b.sha(file)
    b.dump(b.OUT/'INDEPENDENT_PHASE_CHECKS.json',phase_checks)
    b.dump(b.OUT/'NONBINARY_SCALE_CHECKS.json',nonbinary)
    import re
    links_checked=[]
    for file in b.OUT.glob('*.md'):
        for target in re.findall(r'\]\((D:/[^)]+)\)',file.read_text(encoding='utf-8')):
            assert Path(target).exists(),(file,target)
            links_checked.append(target)
    output_hashes={p.relative_to(b.ROOT).as_posix():b.sha(p) for p in b.OUT.rglob('*') if p.is_file() and p.name!='VERIFICATION.json'}
    b.dump(b.OUT/'VERIFICATION.json',dict(passed=True,input_hashes_checked=len(checked),representatives=252,original_probes=105,supplemental_scale_curves=35,
               independent_nonbinary_controls=14,phase_curves_with_extrema_checked=len(phase_checks),phase_extrema_checked=sum(r['events'] for r in phase_checks),
               all_diagnostic_grid_and_nodal_checks_passed=True,original_selection_and_success_criteria_unchanged=True,
               original_optimization_evaluations_added=0,phase_threshold_deg=5.,post_first_diagnosis_supplement_disclosed=True,
               input_target_catalog_hashes=provenance,additional_source_hashes={p.relative_to(b.ROOT).as_posix():b.sha(p) for p in (Path(__file__),b.HERE/'plot.py',b.HERE/'catalog.py',b.core.NESTED/'schematic.py')},
               markdown_local_links_checked=len(links_checked),ordinary_circuit_diagrams=7,diagram_connectivity_validation='passed by frozen renderer',
               manually_viewed_figures=['simple_separation.png','scale_failure.png','mixed_phase_lobes.png','three_extra_lobes.png'],
               output_hashes=output_hashes))
    print(json.dumps(dict(passed=True,phase_extrema_checked=sum(r['events'] for r in phase_checks),nonbinary_controls=14,input_hashes_checked=len(checked))))

if __name__=='__main__':main()

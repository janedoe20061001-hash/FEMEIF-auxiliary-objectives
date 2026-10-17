"""Read existing fitted curves; no optimizer, new seeds, or changed acceptance rules."""
from pathlib import Path
import sys,csv,json,hashlib
from collections import Counter
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'output/mse_structure_step3_revision_20261011'
OUT=ROOT/'output/structure_metric_family_20261011'
sys.path.insert(0,str(ROOT/'experiments/mse_structure_step3_revision_20261011'))
import core


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    input_files=[SOURCE/'analysis/EVENTS.csv',SOURCE/'analysis/REPRESENTATIVES.csv']
    events=read(input_files[0]);representatives=read(input_files[1])
    failures=[]
    for target in core.CFG['targets']:
        for method in core.CFG['methods']:
            for kind in ('peak','valley'):
                rr=[r for r in events if r['rule']=='primary' and r['target']==target and r['method']==method and r['kind']==kind]
                if not rr:continue
                matched=[r for r in rr if r['matched']=='True']
                failures.append(dict(target=target,method=method,kind=kind,events=len(rr),
                                     missing=len(rr)-len(matched),complete=sum(r['complete']=='True' for r in rr),
                                     position_bad=sum(abs(float(r['shift_decades']))>.02 for r in matched),
                                     height_bad=sum(abs(float(r['height_delta_db']))>3 for r in matched),
                                     width_bad=sum(not r['width_ratio'] or not .5<=float(r['width_ratio'])<=2 for r in matched)))
    joint=[]
    for group in ('first','other'):
        p=ROOT/f'output/mse_structure_complex_20261010_server/retrieved/{group}/analysis_{group}/representatives.csv'
        input_files.append(p)
        rr=[r for r in read(p) if r['rule']=='primary' and r['generation']=='300' and r['audit_passed']=='True']
        for target in sorted({r['target'] for r in rr}):
            subset=[r for r in rr if r['target']==target]
            joint.append(dict(target=target,n=len(subset),amplitude_all_ok=sum(r['joint_amp_complete']=='True' for r in subset),
                              amplitude_ok_phase_bad=sum(r['joint_amp_complete']=='True' and int(r['phase_incomplete'])>0 for r in subset),
                              amplitude_extra_seeds=sum(int(r['extra'])>0 for r in subset)))
    bridges=[]
    for target in core.CFG['targets']:
        data=core.target(target);geo=core.Geometry(data)
        windows=sorted(geo.windows,key=lambda w:w['lo'])
        gaps=[(a['hi'],b['lo']) for a,b in zip(windows,windows[1:]) if b['lo']>a['hi']+1e-12]
        if not gaps:continue
        for r in representatives:
            if r['target']!=target or r['rule']!='primary':continue
            p=SOURCE/f"analysis/details/{target}_{r['method']}_{r['seed']}_primary.json"
            input_files.append(p);d=json.loads(p.read_text(encoding='utf-8'));parameters=10**np.array(d['log_parameters'])
            refinements=[]
            for level,uniform in ((max(4,d['diagnostic']['refinement']),1025),(max(8,d['diagnostic']['refinement']),2049)):
                grid=data['quadrature'].reference_grid(parameters,level)
                total,span=0.,0.;angular=[]
                for lo,hi in gaps:
                    x=np.unique(np.r_[lo,grid[(grid>lo)&(grid<hi)],hi,np.linspace(lo,hi,uniform)])
                    z,_=data['circuit'].response(10**x,parameters);ref,_=data['circuit'].response(10**x)
                    residual=np.angle(z*np.conj(ref))
                    total+=np.trapezoid(residual**2,x);span+=hi-lo
                    phi,star=np.unwrap(np.angle(z)),np.unwrap(np.angle(ref))
                    angular.append(dict(frequency_low=10**lo,frequency_high=10**hi,
                                        target_net_deg=float(np.rad2deg(star[-1]-star[0])),
                                        fit_net_deg=float(np.rad2deg(phi[-1]-phi[0])),
                                        target_total_rotation_deg=float(np.rad2deg(abs(np.diff(star)).sum())),
                                        fit_total_rotation_deg=float(np.rad2deg(abs(np.diff(phi)).sum()))))
                refinements.append(dict(level=level,uniform=uniform,phase_rmse_deg=float(np.rad2deg(np.sqrt(total/span))),angular=angular))
            delta=abs(refinements[-1]['phase_rmse_deg']-refinements[-2]['phase_rmse_deg'])
            bridges.append(dict(target=target,method=r['method'],seed=int(r['seed']),
                                gap_phase_rmse_deg=float(np.rad2deg(np.sqrt(total/span))),
                                gap_log10_span=span,details=angular,refinements=refinements,
                                gap_rmse_stable=delta<=.1+.01*refinements[-1]['phase_rmse_deg'],
                                refinement_rmse_delta_deg=delta))
    flags=[]
    for r in representatives:
        if r['rule']!='primary':continue
        p=SOURCE/f"analysis/details/{r['target']}_{r['method']}_{r['seed']}_primary.json"
        input_files.append(p);d=json.loads(p.read_text(encoding='utf-8'))
        flags.append(dict(target=r['target'],method=r['method'],seed=int(r['seed']),
                          extra_amplitude=d['diagnostic']['extra_events'],extra_phase=d['diagnostic']['extra_phase_events'],
                          phase_complete=sum(a['complete'] for a in d['diagnostic']['phase_events']),
                          phase_total=len(d['diagnostic']['phase_events'])))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in set(input_files)}
    result=dict(scope='Existing 80-task revision and 168-seed nested results kept separate; gap diagnostics only, no new optimization',
                event_failure_causes_nonexclusive=failures,nested_amplitude_vs_phase=joint,
                unscored_gaps=bridges,revision_extra_and_phase_events=flags,
                original_thresholds_retained=True,input_sha256=hashes,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'INVENTORY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(failures=failures,nested=joint,gaps=bridges,extras=[r for r in flags if r['extra_amplitude']]),ensure_ascii=False,indent=2))


if __name__=='__main__':main()

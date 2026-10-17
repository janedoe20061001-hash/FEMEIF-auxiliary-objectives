"""Frozen representative diagnosis and controlled physical-parameter probes; no optimizer."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys, json, hashlib, csv, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.optimize import brentq

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'output/physical_state_diagnosis_20261011'
PRIOR=ROOT/'output/mse_structure_step3_knee_20261011'
sys.path.insert(0,str(ROOT/'experiments/mse_structure_metric_family_20261011'))
import core

FACTORS=(.25,.5,1.,2.,4.)
KINDS=('damping','lc_reciprocal','lc_joint')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def clean(v):
    if isinstance(v,dict): return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [clean(x) for x in v]
    if isinstance(v,np.ndarray): return clean(v.tolist())
    if isinstance(v,np.generic): return v.item()
    return v

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def csvout(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(clean(rows))

def simple(ident):
    tree=['S' if ident=='simple_series' else 'P','R1','L1','C1']
    c=core.Circuit(ident,ident,tree,dict(R1=10.,L1=.001,C1=1e-7),('L1','C1'),'R1','R1')
    cfg=core.read_config();q=core.Quadrature(c,cfg)
    return dict(id=ident,circuit=c,quadrature=q,catalog=[],config=cfg)

def root_features(data,p,level):
    c=data['circuit'];x=data['quadrature'].reference_grid(p,level)
    z,_=c.response(10**x,p);v=z.imag
    ix=np.flatnonzero(v[:-1]*v[1:]<0)
    roots=[]
    for i in ix:
        r=brentq(lambda u:float(c.response(np.array([10**u]),p)[0][0].imag),
                 float(x[i]),float(x[i+1]),xtol=1e-14,rtol=1e-15)
        roots.append(r)
    # Exact sampled zeros need a genuine sign change on their two sides.
    for i in np.flatnonzero(v==0):
        if 0<i<len(x)-1 and v[i-1]*v[i+1]<0: roots.append(float(x[i]))
    roots=sorted(roots);unique=[]
    for r in roots:
        if not unique or r-unique[-1]>5e-13: unique.append(r)
    result=[]
    for i,r in enumerate(unique):
        zz,dz=c.response(np.array([10**r]),p);zz,dz=zz[0],dz[0]
        k=float(np.imag(dz/zz));res=abs(zz.imag)/abs(zz)
        result.append(dict(index=i,x=r,frequency_hz=10**r,direction=1 if k>0 else -1,
                           real_ohm=float(zz.real),kappa=k,root_residual=float(res),
                           root_residual_ok=bool(res<=1e-6),
                           identity_relative_error=float(abs(k-dz.imag/zz.real)/max(abs(k),1e-8))))
    return result

def nodal_check(c,p,event):
    k=event['kappa'];u=np.log(event['frequency_hz']);h=max(1e-10,min(1e-4,1e-3/max(abs(k),1)))
    f0=np.array([event['frequency_hz']]);z,_=c.response(f0,p);ind=c.nodal_response(f0,p)
    err=float(abs(ind[0]-z[0])/abs(z[0]));slopes=[]
    for step in (h,h/2):
        f=np.exp(u+np.array([-2.,-1.,1.,2.])*step)
        node=c.nodal_response(f,p);angle=np.angle(node)
        slopes.append(float((angle[0]-8*angle[1]+8*angle[2]-angle[3])/(12*step)))
    errors=[abs(v-k) for v in slopes]
    return dict(nodal_relative_error=err,nodal_kappa=slopes,
                nodal_derivative_error=errors,nodal_step_ln_f=h,
                nodal_passed=bool(err<=1e-5 and max(errors)<=.005*abs(k)+1e-5))

def features(data,p):
    a=root_features(data,p,4);b=root_features(data,p,8)
    stable=len(a)==len(b)
    if stable:
        for aa,bb in zip(a,b):
            stable &= (aa['direction']==bb['direction'] and abs(aa['x']-bb['x'])<=1e-8
                       and abs(aa['real_ohm']/bb['real_ohm']-1)<=2e-4
                       and abs(aa['kappa']-bb['kappa'])<=2e-4*abs(bb['kappa'])+1e-8)
    for e in b: e.update(nodal_check(data['circuit'],p,e))
    return dict(stable=bool(stable),passed=bool(stable and all(e['root_residual_ok'] and e['nodal_passed'] for e in b)),
                level4=a,events=b)

def match(reference,fit):
    edges=[2.,*[(a['x']+b['x'])/2 for a,b in zip(reference,reference[1:])],6.]
    rows=[];used=set()
    for i,t in enumerate(reference):
        options=[e for e in fit if edges[i]<=e['x']<(edges[i+1] if i<len(reference)-1 else 6.+1e-12)
                 and e['direction']==t['direction']]
        f=min(options,key=lambda e:abs(e['x']-t['x'])) if options else None
        r=dict(event_index=i,target_frequency_hz=t['frequency_hz'],target_direction=t['direction'],
               target_real_ohm=t['real_ohm'],target_kappa=t['kappa'],zone_low_hz=10**edges[i],zone_high_hz=10**edges[i+1],matched=f is not None)
        if f:
            used.add(f['index']);shift=f['x']-t['x'];r.update(
                fit_frequency_hz=f['frequency_hz'],fit_real_ohm=f['real_ohm'],fit_kappa=f['kappa'],
                shift_decades=shift,shift_tangent_scales=abs(shift)*np.log(10)*abs(t['kappa']),
                real_ratio=f['real_ohm']/t['real_ohm'],real_db=20*np.log10(f['real_ohm']/t['real_ohm']),
                kappa_ratio=abs(f['kappa']/t['kappa']),position_flag=bool(abs(shift)>.02),
                real_flag=bool(abs(20*np.log10(f['real_ohm']/t['real_ohm']))>3),
                kappa_flag=bool(not .5<=abs(f['kappa']/t['kappa'])<=2))
        rows.append(r)
    return rows,[e for e in fit if e['index'] not in used]

def focus_span(data):
    if data['id'].startswith('simple_'): return (2.,6.)
    ee=[e for e in data['catalog'] if e['event_id'] in core.CFG['focus_events'][data['id']]]
    return max(2.,min(e['x']-3*e['width_3db_decades'] for e in ee)),min(6.,max(e['x']+3*e['width_3db_decades'] for e in ee))

def physical_summary(rows,lo,hi):
    focus=[r for r in rows if lo<=np.log10(r['target_frequency_hz'])<=hi]
    output={}
    for name,rr in [('all',rows),('focus',focus)]:
        output.update({name+'_events':len(rr),name+'_missing':sum(not r['matched'] for r in rr),
                       name+'_position_flags':sum(r.get('position_flag',False) for r in rr),
                       name+'_real_flags':sum(r.get('real_flag',False) for r in rr),
                       name+'_kappa_flags':sum(r.get('kappa_flag',False) for r in rr),
                       name+'_has_diagnostic_deviation':bool(rr and any(not r['matched'] or r.get('position_flag') or r.get('real_flag') or r.get('kappa_flag') for r in rr))})
    return output

def target_job(ident):
    data=simple(ident) if ident.startswith('simple_') else core.target(ident)
    c=data['circuit'];reference=features(data,c.p);lo,hi=focus_span(data)
    hashes={};representatives=[];event_rows=[];records=[]
    for path in sorted(PRIOR.glob(f'*/analysis/details/{ident}_*_knee.json')):
        d=json.loads(path.read_text(encoding='utf-8'));s=d['summary'];batch=path.relative_to(PRIOR).parts[0]
        select=PRIOR/batch/'selections'/f'{ident}_{s["method"]}_{s["seed"]}.json'
        sel=json.loads(select.read_text(encoding='utf-8'));source=ROOT/sel['source']
        assert sha(source)==sel['source_sha256'];hashes[str(source.relative_to(ROOT)).replace('\\','/')]=sha(source)
        for inp in (path,select): hashes[inp.relative_to(ROOT).as_posix()]=sha(inp)
        with np.load(source) as z:
            p=z['log_parameters'][sel['selected_index']];v=z['objectives'][sel['selected_index'],:2]
        np.testing.assert_array_equal(p,d['log_parameters']);np.testing.assert_allclose(v,d['training_objective'],rtol=1e-12,atol=1e-12)
        physical=features(data,10**p);rows,extra=match(reference['events'],physical['events'])
        key=dict(batch=batch,target=ident,method=s['method'],seed=s['seed'])
        rs=dict(**key,selection_reason=sel['reason'],verified=physical['passed'],
                focus_joint_ok=s['focus_joint_ok'],all_amp_ok=s['all_amp_ok'],all_phase_ok=s['all_phase_ok'],
                local_phase_rmse_deg=s['local_phase_rmse_deg'],mag=s['mag'],pha=s['pha'],
                extra_amp=s['extra_amp'],extra_phase=s['extra_phase'],extra_crossings=len(extra),
                **physical_summary(rows,lo,hi))
        representatives.append(rs)
        event_rows.extend(dict(**key,in_focus=lo<=np.log10(r['target_frequency_hz'])<=hi,**r) for r in rows)
        records.append(dict(**key,log_parameters=p.tolist(),physical=physical,matches=rows,extra_crossings=extra))
    probes=[];probe_events=[];probe_details=[]
    for kind in KINDS:
        for factor in FACTORS:
            p=c.p.copy()
            if kind=='damping': p[c.names.index(c.damping)]*=factor
            else:
                p[c.names.index(c.lc[0])]*=factor
                p[c.names.index(c.lc[1])]*=1/factor if kind=='lc_reciprocal' else factor
            state=features(data,p);rows,extra=match(reference['events'],state['events'])
            key=dict(target=ident,kind=kind,factor=factor)
            r=dict(**key,verified=state['passed'],extra_crossings=len(extra),**physical_summary(rows,lo,hi))
            if ident.startswith('simple_'):
                expected_f=1/(2*np.pi*np.sqrt(p[1]*p[2]));q=np.sqrt(p[1]/p[2])/p[0] if ident=='simple_series' else p[0]*np.sqrt(p[2]/p[1])
                expected_k=(2 if ident=='simple_series' else -2)*q
                assert len(state['events'])==1
                e=state['events'][0];np.testing.assert_allclose([e['frequency_hz'],e['real_ohm'],e['kappa']], [expected_f,p[0],expected_k],rtol=1e-8,atol=1e-8)
                r.update(analytic_f_hz=expected_f,analytic_Q=q,analytic_real_ohm=p[0],analytic_kappa=expected_k)
            probes.append(r);probe_events.extend(dict(**key,in_focus=lo<=np.log10(r['target_frequency_hz'])<=hi,**r) for r in rows)
            probe_details.append(dict(**key,parameters=p.tolist(),physical=state,matches=rows,extra_crossings=extra))
    target_record=dict(circuit=c.record(),damping_component=c.damping,lc_components=c.lc,focus_log10_span=[lo,hi],
                       amplitude_catalog=data['catalog'],reference=reference)
    dump(OUT/'details'/f'{ident}.json',dict(target=target_record,representatives=records,probes=probe_details))
    return dict(target=ident,reference_verified=reference['passed'],representatives=representatives,event_rows=event_rows,
                probes=probes,probe_events=probe_events,input_hashes=hashes)

def grouped(rows):
    output=[]
    for key in sorted({(r['batch'],r['target'],r['method']) for r in rows}):
        rr=[r for r in rows if (r['batch'],r['target'],r['method'])==key]
        focus_pass=[r for r in rr if r['focus_joint_ok']]
        amp_pass=[r for r in rr if r['all_amp_ok']]
        output.append(dict(batch=key[0],target=key[1],method=key[2],n=len(rr),verified=sum(r['verified'] for r in rr),
                           focus_amp_pass=len(focus_pass),all_amp_pass=len(amp_pass),all_phase_pass=sum(r['all_phase_ok'] for r in rr),
                           amp_pass_but_phase_fail=sum(not r['all_phase_ok'] for r in amp_pass),
                           amp_pass_but_crossing_deviation=sum(r['all_has_diagnostic_deviation'] for r in amp_pass),
                           focus_amp_pass_but_focus_crossing_deviation=sum(r['focus_has_diagnostic_deviation'] for r in focus_pass),
                           **{k:sum(r[k] for r in rr) for k in ('focus_events','focus_missing','focus_position_flags','focus_real_flags','focus_kappa_flags')},
                           local_phase_mean_deg=float(np.mean([r['local_phase_rmse_deg'] for r in rr]))))
    return output

def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    sources=[HERE/'run.py',HERE/'PROTOCOL.md',Path(core.__file__),core.NESTED/'circuits.py',core.NESTED/'sampling.py',core.NESTED/'config.json',
             core.NESTED/'protocol.py',core.HERE/'protocol.json',core.OLD/'adaptive.py',core.OLD/'targets/s1/record.json',core.OLD/'targets/s1/grid.npz',
             core.OLD/'analysis_toolkit/refined_grid.py',core.OLD/'analysis_toolkit/diagnostics.py']
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sources}
    jobs=['simple_series','simple_parallel',*core.CFG['targets']];results=[]
    with ProcessPoolExecutor(max_workers=min(7,os.cpu_count() or 1)) as pool:
        for future in as_completed([pool.submit(target_job,t) for t in jobs]):
            r=future.result();results.append(r);hashes.update(r['input_hashes'])
            print(json.dumps(dict(target=r['target'],representatives=len(r['representatives']),probes=len(r['probes']),reference_verified=r['reference_verified'])),flush=True)
            dump(OUT/'STATUS.json',dict(status='running',targets_done=len(results),targets_total=len(jobs),seconds=time.perf_counter()-start))
    rows=[a for r in results for a in r['representatives']];ev=[a for r in results for a in r['event_rows']]
    probes=[a for r in results for a in r['probes']];pev=[a for r in results for a in r['probe_events']]
    assert len(rows)==252 and len(probes)==105
    for path,h in hashes.items(): assert sha(ROOT/path)==h, f'Input changed: {path}'
    csvout(OUT/'REPRESENTATIVES.csv',rows);csvout(OUT/'CROSSINGS.csv',ev);csvout(OUT/'PROBES.csv',probes);csvout(OUT/'PROBE_CROSSINGS.csv',pev)
    csvout(OUT/'GROUPS.csv',grouped(rows))
    dump(OUT/'INPUT_MANIFEST.json',dict(passed=True,hashes=hashes,original_representatives_unchanged=True,
                                       original_optimization_evaluations_added=0,protocol_sha256=sha(HERE/'PROTOCOL.md')))
    failed=[r for r in rows+probes if not r['verified']]
    summary=dict(status='complete',representatives=len(rows),probes=len(probes),crossing_comparisons=len(ev),
                 reference_failures=[r['target'] for r in results if not r['reference_verified']],
                 numerical_failures=failed,seconds=time.perf_counter()-start,new_optimization_evaluations=0,
                 simple_formula_controls=30,website_published=False)
    dump(OUT/'STATUS.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__': main()

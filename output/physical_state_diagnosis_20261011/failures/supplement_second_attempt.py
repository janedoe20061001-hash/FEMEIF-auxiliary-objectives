"""Post-diagnosis physical hypotheses; separate from the original frozen probe."""
from pathlib import Path
import importlib.util, sys
_source=Path(__file__).resolve().with_name('run.py')
_spec=importlib.util.spec_from_file_location('physical_diagnosis_frozen_base',_source)
_base=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_base;_spec.loader.exec_module(_base)
for _key in ('np','os','json','csv','time','ProcessPoolExecutor','as_completed','ROOT','HERE','OUT','PRIOR','core','simple','sha','dump','csvout','FACTORS','brentq'):
    globals()[_key]=getattr(_base,_key)
from scipy.signal import find_peaks

def phase_extrema(data,p,level):
    c=data['circuit'];x=data['quadrature'].reference_grid(p,level)
    z,dz=c.response(10**x,p);angle=np.rad2deg(np.angle(z));k=np.imag(dz/z)
    out=[]
    for sign,kind in [(1,'peak'),(-1,'valley')]:
        ix,pr=find_peaks(sign*angle,prominence=5.)
        for j,i in enumerate(ix):
            if k[i-1]*k[i+1]>0: raise RuntimeError('Prominent phase extremum lacks derivative bracket')
            r=brentq(lambda u:float(np.imag(np.divide(*reversed(c.response(np.array([10**u]),p))))[0]),
                      float(x[i-1]),float(x[i+1]),xtol=1e-14,rtol=1e-15)
            zz,_=c.response(np.array([10**r]),p);value=float(np.rad2deg(np.angle(zz[0])))
            out.append(dict(kind=kind,x=r,frequency_hz=10**r,phase_deg=value,
                            reactive_fraction=float(zz[0].imag/abs(zz[0])),prominence_deg=float(pr['prominences'][j]),
                            sign=1 if value>=0 else -1))
    out.sort(key=lambda e:e['x'])
    for i,e in enumerate(out):e['index']=i
    return out

def stable_phase(data,p):
    a,b=[phase_extrema(data,p,n) for n in (4,8)]
    stable=(len(a)==len(b) and all(aa['kind']==bb['kind'] and aa['sign']==bb['sign']
                  and abs(aa['x']-bb['x'])<=1e-8 and abs(aa['phase_deg']-bb['phase_deg'])<=.001 for aa,bb in zip(a,b)))
    return dict(stable=stable,level4=a,events=b)

def phase_match(ref,fit):
    edges=[2.,*[(a['x']+b['x'])/2 for a,b in zip(ref,ref[1:])],6.]
    rows=[];used=set()
    for i,t in enumerate(ref):
        options=[e for e in fit if edges[i]<=e['x']<(edges[i+1] if i<len(ref)-1 else 6.+1e-12)
                 and e['kind']==t['kind'] and e['sign']==t['sign']]
        f=min(options,key=lambda e:abs(e['x']-t['x'])) if options else None
        r=dict(event_index=i,kind=t['kind'],target_frequency_hz=t['frequency_hz'],
               target_phase_deg=t['phase_deg'],target_reactive_fraction=t['reactive_fraction'],matched=f is not None)
        if f:
            used.add(f['index']);shift=f['x']-t['x'];delta=f['phase_deg']-t['phase_deg']
            r.update(fit_frequency_hz=f['frequency_hz'],shift_decades=shift,fit_phase_deg=f['phase_deg'],
                     phase_delta_deg=delta,fit_reactive_fraction=f['reactive_fraction'],
                     position_flag=bool(abs(shift)>.02),angle_flag=bool(abs(delta)>5))
        rows.append(r)
    return rows,[e for e in fit if e['index'] not in used]

def job(ident):
    data=simple(ident) if ident.startswith('simple_') else core.target(ident);c=data['circuit']
    prior=json.loads((OUT/'details'/f'{ident}.json').read_text(encoding='utf-8'));hashes={}
    scalings=[];x=data['quadrature'].reference_grid(c.p,8);freq=10**x;z,_=c.response(freq)
    # The comparison grid remains entirely target-defined, unchanged under scaling.
    nf=np.geomspace(100,1e6,257);nz=c.nodal_response(nf)
    for factor in FACTORS:
        p=c.p*np.array([1/factor if name.startswith('C') else factor for name in c.names])
        fit,_=c.response(freq,p);fitn=c.nodal_response(nf,p)
        error=float(np.max(abs(fit-factor*z)/np.maximum(abs(factor*z),1e-300)))
        nodalerror=float(np.max(abs(fitn-factor*nz)/np.maximum(abs(factor*nz),1e-300)))
        phaseerror=float(np.max(abs(np.angle(fit)-np.angle(z))))
        offset=20*np.log10(abs(fit)/abs(z));expected=20*np.log10(factor)
        scalar=dict(target=ident,factor=factor,complex_scale_relative_error=error,nodal_scale_relative_error=nodalerror,
                    phase_max_delta_rad=phaseerror,expected_magnitude_offset_db=expected,
                    max_magnitude_offset_error_db=float(np.max(abs(offset-expected))),
                    passed=bool(error<1e-10 and nodalerror<1e-8 and phaseerror<1e-10 and np.max(abs(offset-expected))<1e-8))
        scalings.append(scalar)
    reference=stable_phase(data,c.p);rows=[];events=[];details=[]
    for original in prior['representatives']:
        k={v:original[v] for v in ('batch','target','method','seed')};p=10**np.array(original['log_parameters'])
        state=stable_phase(data,p);matches,extra=phase_match(reference['events'],state['events'])
        lo,hi=prior['target']['focus_log10_span'];focus=[r for r in matches if lo<=np.log10(r['target_frequency_hz'])<=hi]
        row=dict(**k,verified=state['stable'],all_events=len(matches),all_missing=sum(not r['matched'] for r in matches),
                 all_position_flags=sum(r.get('position_flag',False) for r in matches),all_angle_flags=sum(r.get('angle_flag',False) for r in matches),
                 focus_events=len(focus),focus_missing=sum(not r['matched'] for r in focus),
                 focus_angle_flags=sum(r.get('angle_flag',False) for r in focus),
                 extra_phase_extrema=len(extra),
                 all_has_diagnostic_deviation=bool(any(not r['matched'] or r.get('position_flag') or r.get('angle_flag') for r in matches) or extra))
        rows.append(row);events.extend(dict(**k,**r) for r in matches)
        details.append(dict(**k,state=state,matches=matches,extra_events=extra))
    dump(OUT/'phase_details'/f'{ident}.json',dict(reference=reference,representatives=details))
    return dict(target=ident,scalings=scalings,rows=rows,events=events,reference_stable=reference['stable'])

def main():
    before=json.loads((OUT/'INPUT_MANIFEST.json').read_text(encoding='utf-8'))
    sources=[HERE/'supplement.py',HERE/'SUPPLEMENT_PROTOCOL.md',*[OUT/'details'/f'{t}.json' for t in ['simple_series','simple_parallel',*core.CFG['targets']]]]
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sources}
    start=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=min(7,os.cpu_count() or 1)) as pool:
        for future in as_completed([pool.submit(job,t) for t in ['simple_series','simple_parallel',*core.CFG['targets']]]):
            r=future.result();results.append(r);print(json.dumps(dict(target=r['target'],phase_curves=len(r['rows']))),flush=True)
    for path,h in {**before['hashes'],**hashes}.items():assert sha(ROOT/path)==h
    rows=[a for r in results for a in r['rows']];ev=[a for r in results for a in r['events']];sc=[a for r in results for a in r['scalings']]
    assert len(rows)==252 and len(sc)==35
    csvout(OUT/'PHASE_REPRESENTATIVES.csv',rows);csvout(OUT/'PHASE_EXTREMA.csv',ev);csvout(OUT/'SCALE_CONTROLS.csv',sc)
    original=list(csv.DictReader((OUT/'REPRESENTATIVES.csv').open(encoding='utf-8-sig')))
    mapping={(r['batch'],r['target'],r['method'],int(r['seed'])):r for r in rows}
    cross=[]
    for r in original:
        a=mapping[(r['batch'],r['target'],r['method'],int(r['seed']))]
        if r['batch']=='family' and r['all_amp_ok']=='True' and r['all_phase_ok']=='False':
            cross.append(dict(batch=r['batch'],target=r['target'],method=r['method'],seed=int(r['seed']),
                              zero_state_flag=r['all_has_diagnostic_deviation']=='True',
                              phase_extrema_flag=a['all_has_diagnostic_deviation'],phase_events=a['all_events'],
                              missing=a['all_missing'],angle_flags=a['all_angle_flags'],position_flags=a['all_position_flags'],extras=a['extra_phase_extrema']))
    csvout(OUT/'AMP_PASS_PHASE_FAIL.csv',cross)
    summary=dict(status='complete',phase_representatives=252,scale_controls=35,
                 failed_phase_curves=[r for r in rows if not r['verified']],
                 unstable_target_references=[r['target'] for r in results if not r['reference_stable']],
                 failed_scale_controls=[r for r in sc if not r['passed']],seconds=time.perf_counter()-start,
                 new_optimization_evaluations=0,post_first_diagnosis_exploration=True,
                 latest_amp_pass_phase_fail=len(cross),latest_zero_state_flagged=sum(r['zero_state_flag'] for r in cross),
                 latest_phase_extrema_flagged=sum(r['phase_extrema_flag'] for r in cross),hashes=hashes)
    dump(OUT/'SUPPLEMENT_STATUS.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='hashes'},ensure_ascii=False),flush=True)

if __name__=='__main__': main()

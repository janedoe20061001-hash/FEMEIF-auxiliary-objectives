"""Post-run descriptive analysis; never changes frozen optimization or audits."""
from pathlib import Path
import json, csv, collections, time
import numpy as np
OUT=Path(__file__).resolve().parent
DATA=OUT/'retrieved/output/complex_joint_20261012'
CODE=OUT/'retrieved/experiments/complex_joint_20261012'
DEST=OUT/'analysis';DEST.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def csvwrite(name,rows):
    with (DEST/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
cfg=read(CODE/'protocol.json');ids=read(CODE/'inputs/manifest.json')['targets']
audits={};records=[];gates=collections.Counter();gate_method=collections.defaultdict(collections.Counter)
selection=collections.Counter();event_rows=[];local_rows=[];phase_rows=[];front_rows=[]
for ident in ids:
    for method in cfg['methods']:
        for seed in cfg['seeds']:
            d=DATA/'jobs'/ident/method/f'seed_{seed}'
            a=read(d/'audit.json');s=read(d/'summary.json');audits[ident,method,seed]=a
            assert a['status']=='audited' and s['evaluations']==60000
            assert s['generations_saved']==list(range(20,301,20))
            selection[a['selection']['reason']]+=1
            for g in ['converged','base_training_agrees','aux_training_agrees','diagnostic_stable','nodal_passed']:
                if not a[g]:gates[g]+=1;gate_method[method][g]+=1
            ev=[e for e in a['amplitude_events'] if e['width_evaluable']]
            row=dict(target=ident,method=method,seed=seed,accepted=a['accepted'],
                mag_mse=a['objective'][0],pha_mse=a['objective'][1],
                amp_failed=sum(not e['complete'] for e in ev),amp_total=len(ev),
                amp_missing=sum(not e['matched'] for e in a['amplitude_events']),amp_extra=len(a['extra_amplitude_events']),
                phase_failed=sum(not e['complete'] for e in a['phase_events']),phase_total=len(a['phase_events']),
                phase_extra=len(a['extra_phase_events']),seconds=s['seconds'],cpu_seconds=s['cpu_seconds'],
                queries=s['base_frequency_queries']+s['auxiliary_frequency_queries'],
                D=a['anchored_geometry']['D'],H=a['anchored_geometry']['H'],A=a['anchored_geometry']['A'],
                endpoint=max(a['endpoint_error_ohm']),front_size=len(a['selection']['front_indices']),
                knee_reason=a['selection']['reason'])
            records.append(row)
            for e in a['amplitude_events']:
                event_rows.append(dict(target=ident,method=method,seed=seed,accepted=a['accepted'],**e))
            for e in a['phase_events']:
                phase_rows.append(dict(target=ident,method=method,seed=seed,accepted=a['accepted'],**e))
            for e in a['local']:
                local_rows.append(dict(target=ident,method=method,seed=seed,accepted=a['accepted'],**e))
            if (ident in ('case1','case2','three_level/base','user_nested/base','mixed_resonators/lc_reciprocal_0p25')):
                with np.load(d/'generation_0300.npz') as f:
                    obj=f['F'][:,:2]; ix=f['selected_index'];p=obj[int(ix)]
                    front_rows.append(dict(target=ident,method=method,seed=seed,accepted=a['accepted'],
                        selected_mag=float(p[0]),selected_pha=float(p[1]),population_min_mag=float(obj[:,0].min()),
                        population_min_pha=float(obj[:,1].min()),
                        population_amp_not_worse=int(np.sum(obj[:,0]<=p[0])),
                        population_phase_not_worse=int(np.sum(obj[:,1]<=p[1])),front_size=row['front_size'],reason=row['knee_reason']))
by={(r['target'],r['method'],r['seed']):r for r in records}
metrics=['mag_mse','pha_mse','amp_failed','amp_missing','amp_extra','phase_failed','phase_extra','D','H','A','seconds','queries']
means=[];paired=[];event_means=[]
for ident in ids:
    for method in cfg['methods']:
        rr=[by[ident,method,s] for s in cfg['seeds']]
        row=dict(target=ident,method=method,n=len(rr),accepted=sum(r['accepted'] for r in rr),
            amp_total=rr[0]['amp_total'],phase_total=rr[0]['phase_total'],
            front_size_mean=float(np.mean([r['front_size'] for r in rr])))
        row.update({m:float(np.mean([r[m] for r in rr])) for m in metrics});means.append(row)
        if method=='MAG_PHA':continue
        for mode in ['all','both_accepted']:
            pp=[(r,by[ident,'MAG_PHA',r['seed']]) for r in rr]
            if mode=='both_accepted':pp=[(r,b) for r,b in pp if r['accepted'] and b['accepted']]
            if not pp:continue
            for metric in metrics:
                delta=np.array([r[metric]-b[metric] for r,b in pp])
                rng=np.random.default_rng(20261012);ci=np.quantile(rng.choice(delta,(10000,len(delta))).mean(axis=1),[.025,.975])
                paired.append(dict(target=ident,method=method,mode=mode,metric=metric,n=len(pp),
                    delta=float(delta.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),
                    better=int(sum(delta<0)),worse=int(sum(delta>0)),equal=int(sum(delta==0))))
for ident in ids:
    for method in cfg['methods']:
        ee=[r for r in event_rows if r['target']==ident and r['method']==method]
        for event in sorted({e['event_id'] for e in ee}):
            er=[e for e in ee if e['event_id']==event]
            lr=[e for e in local_rows if e['target']==ident and e['method']==method and e['event_id']==event]
            row=dict(target=ident,method=method,event_id=event,n=len(er),accepted=sum(e['accepted'] for e in er),
                frequency_hz=er[0]['target_frequency_hz'],width_hz=er[0]['target_width_hz'],
                matched=sum(e['matched'] for e in er),complete=sum(e['complete'] for e in er))
            for name,source,absolute in [('shift_decades',er,True),('height_delta_db',er,True),('width_ratio',er,False),
                ('amp_fraction',lr,False),('pha_fraction',lr,False),('mass',lr,False),('phase_rmse_deg',lr,False)]:
                v=[abs(e[name]) if absolute else e[name] for e in source if e.get(name) is not None]
                row[name]=float(np.mean(v)) if v else None
            event_means.append(row)
csvwrite('METHOD_CONDITION.csv',means);csvwrite('PAIRED_VALIDITY.csv',paired)
csvwrite('EVENTS.csv',event_rows);csvwrite('EVENT_MEANS.csv',event_means)
if phase_rows:csvwrite('PHASE_EVENTS.csv',phase_rows)
if local_rows:csvwrite('LOCAL_CONTRIBUTIONS.csv',local_rows)
csvwrite('FRONT_CHECK.csv',front_rows);csvwrite('SEEDS.csv',records)
overview=dict(records=len(records),accepted=sum(r['accepted'] for r in records),gates=dict(gates),
    gate_method={k:dict(v) for k,v in gate_method.items()},selection_reasons=dict(selection),
    accepted_by_method={m:sum(r['accepted'] for r in records if r['method']==m) for m in cfg['methods']},
    cpu_seconds_total=sum(r['cpu_seconds'] for r in records),job_seconds_total=sum(r['seconds'] for r in records),
    nodal_fallbacks=sum(a['nodal_check']['mode']!='double' for a in audits.values()),
    generated_unix=time.time(),note='Post-run descriptive tables preserve both complete and failed-gate results; strict subset is not a new experiment.')
save(DEST/'AUDIT_OVERVIEW.json',overview)
print(json.dumps(overview,ensure_ascii=True))

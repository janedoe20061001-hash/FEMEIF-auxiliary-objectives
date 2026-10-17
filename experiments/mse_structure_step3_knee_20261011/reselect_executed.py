"""Recheck saved optimization populations, then audit one MAG/PHA knee per seed."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
from pathlib import Path
import sys, json, hashlib, argparse, csv, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from selector import knee, checks

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
OUT=ROOT/'output/mse_structure_step3_knee_20261011'
BATCHES={'pilot':'mse_structure_step3_20261011','revision':'mse_structure_step3_revision_20261011','family':'mse_structure_metric_family_20261011'}

def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def csvout(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def setup(batch):
    sys.path.insert(0,str(ROOT/'experiments'/BATCHES[batch]))
    import core
    analyze=core.load('knee_source_analyzer',ROOT/'experiments'/BATCHES[batch]/'analyze.py')
    destination=OUT/batch
    core.OUT=destination;analyze.OUT=destination
    return core,analyze

def audit_job(batch,job):
    core,a=setup(batch);r,ev=a.audit_curve(job)
    path=core.OUT/f'analysis/details/{job[0]}_{job[1]}_{job[2]}_knee.json'
    d=json.loads(path.read_text(encoding='utf-8'));diag=d['diagnostic'];data=core.target(job[0])
    r.update(extra_amp=len(diag['extra_events']),extra_phase=len(diag['extra_phase_events']),
             phase_pass=sum(e['complete'] for e in diag['phase_events']),phase_total=len(diag['phase_events']),
             all_phase_ok=all(e['complete'] for e in diag['phase_events']),batch=batch)
    for e in ev:
        target=next(t for t in data['catalog'] if t['event_id']==e['event_id'])
        e['shift_target_widths']=None if e['shift_decades'] is None else e['shift_decades']/target['width_3db_decades']
        e['batch']=batch
    r.pop('secondary_training_eligible',None)
    d['summary']=r;d['focus']=[{k:v for k,v in e.items() if k not in ('batch','target','method','seed','rule')} for e in ev]
    d['selection_provenance']=json.loads((core.OUT/f'selections/{job[0]}_{job[1]}_{job[2]}.json').read_text())
    d['replacement_analyzer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    supplements=[]
    if not r['geometry_stable'] or not r['geometry_training_ok']:
        cls=getattr(a,'AuditGeometry',core.Geometry)
        g=cls(data,job[1]) if batch!='pilot' else cls(data)
        for n in (2049,4097):
            vals=g.score(job[4],points=n,anchors=True,return_events=True)[0]
            supplements.append(dict(points=n,events=vals.tolist(),score=float(g.aggregate(vals)) if hasattr(g,'aggregate') else float(max(vals))))
        x,y=[np.array(s['events']) for s in supplements]
        r['supplemental_geometry_stable']=bool(np.all(abs(x-y)<=.01+.01*abs(y)))
        r['supplemental_training_ok']=bool(abs(supplements[-1]['score']-job[6])<=.01+.01*abs(supplements[-1]['score']))
        d['supplemental_geometry']=supplements;d['original_geometry_flags_retained']=True
    d['summary']=r;dump(path,d)
    return r,ev

def main(batch,workers,reuse=False):
    checks();core,a=setup(batch);cfg=core.CFG;source=ROOT/'output'/BATCHES[batch]/'batch'
    assert json.loads((source/'status.json').read_text())['status']=='complete'
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['hashes']==core.hashes(), 'Frozen optimization source hashes changed'
    if reuse:
        verified=json.loads((core.OUT/'INPUT_VERIFICATION.json').read_text())
        assert verified['passed'] and verified['optimization_source_hashes']==core.hashes()
        for name,h in verified['raw_checksums'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
        jobs=[]
        for p in sorted((core.OUT/'selections').glob('*.json')):
            s=json.loads(p.read_text());data=core.target(s['target'])
            with np.load(ROOT/s['source']) as z:
                ix=s['selected_index'];x=z['log_parameters'][ix];f=z['objectives'][ix]
                assert knee(z['objectives'][:,:2])[0]==ix
                g=core.Geometry(data,s['method']) if batch!='pilot' else core.Geometry(data)
                gg=float(f[2]) if s['method']!='MAG_PHA' else float(g.score(x)[0])
                jobs.append((s['target'],s['method'],s['seed'],'knee',x.copy(),f[:2].copy(),gg,False))
        return finish(batch,workers,core,jobs,time.perf_counter())
    checksums={};jobs=[];stages=[]; selections=[];start=time.perf_counter()
    for ident in cfg['targets']:
        data=core.target(ident)
        for seed in cfg['seeds']:
            init=np.load(source/ident/'MAG_PHA'/f'seed_{seed}/initial.npy')
            for method in cfg['methods']:
                p=source/ident/method/f'seed_{seed}'
                summary=json.loads((p/'summary.json').read_text());assert summary['evaluations']==60000
                history=json.loads((p/'history.json').read_text());assert len(history)==300
                assert [h['evaluations'] for h in history]==list(range(200,60001,200))
                np.testing.assert_array_equal(init,np.load(p/'initial.npy'))
                for f,h in summary['files_sha256'].items():
                    got=hashlib.sha256((p/f).read_bytes()).hexdigest();assert got==h
                    checksums[(p/f).relative_to(ROOT).as_posix()]=got
                files=sorted(p.glob('generation_*.npz'));assert len(files)==15
                for stage in files:
                    with np.load(stage) as z:
                        x=z['log_parameters'];v=z['objectives']
                        assert x.shape==(200,len(data['lower'])) and np.isfinite(x).all() and np.isfinite(v).all()
                        assert np.all(x>=data['lower']) and np.all(x<=data['upper'])
                        ix,info=knee(v[:,:2]);gen=int(stage.stem.split('_')[1])
                        stages.append(dict(batch=batch,target=ident,method=method,seed=seed,generation=gen,index=ix,reason=info['reason'],mag=float(v[ix,0]),pha=float(v[ix,1]),distance=info['distance']))
                        if gen!=300:continue
                        ref=data['quadrature'].score(x)
                        np.testing.assert_allclose(ref,v[:,:2],rtol=1e-10,atol=1e-10)
                        g=core.Geometry(data,method) if batch!='pilot' else core.Geometry(data)
                        gg=g.score(x)
                        if method!='MAG_PHA':np.testing.assert_allclose(gg,v[:,2],rtol=1e-10,atol=1e-10)
                        info.update(batch=batch,target=ident,method=method,seed=seed,source=stage.relative_to(ROOT).as_posix(),source_sha256=checksums[stage.relative_to(ROOT).as_posix()])
                        dump(core.OUT/f'selections/{ident}_{method}_{seed}.json',info);selections.append(info)
                        jobs.append((ident,method,seed,'knee',x[ix].copy(),v[ix,:2].copy(),float(gg[ix]),False))
    csvout(core.OUT/'STAGES.csv',stages)
    dump(core.OUT/'INPUT_VERIFICATION.json',dict(passed=True,jobs=len(jobs),stage_count=len(stages),final_population_individuals=len(jobs)*200,paired_initialization=True,evaluations=len(jobs)*60000,new_optimization_evaluations=0,optimization_source_hashes=manifest['hashes'],raw_checksums=checksums,selector_sha256=hashlib.sha256((HERE/'selector.py').read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest()))
    finish(batch,workers,core,jobs,start)

def finish(batch,workers,core,jobs,start):
    rows=[];events=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(audit_job,batch,j) for j in jobs]):
            r,e=future.result();rows.append(r);events.extend(e)
            dump(core.OUT/'STATUS.json',dict(status='running',completed=len(rows),planned=len(jobs),seconds=time.perf_counter()-start))
    rows.sort(key=lambda r:(r['target'],r['method'],r['seed']));events.sort(key=lambda r:(r['target'],r['method'],r['seed'],r['event_id']))
    csvout(core.OUT/'REPRESENTATIVES.csv',rows);csvout(core.OUT/'EVENTS.csv',events)
    failures=[r for r in rows if not(r['stable'] and r['base_score_ok']) or (not r['geometry_stable'] and not r.get('supplemental_geometry_stable',False))]
    training_flags=[r for r in rows if not r['geometry_training_ok']]
    selections=[json.loads(p.read_text()) for p in (core.OUT/'selections').glob('*.json')]
    dump(core.OUT/'AUDIT.json',dict(passed=not failures and not training_flags,independent_structure_and_scores_passed=not failures,curves=len(rows),failures=failures,original_auxiliary_flags=[r for r in rows if not r['geometry_stable'] or not r['geometry_training_ok']],training_discrepancies=training_flags,original_thresholds_unchanged=True))
    dump(core.OUT/'STATUS.json',dict(status='complete',completed=len(rows),planned=len(jobs),seconds=time.perf_counter()-start,knee_counts={reason:sum(s['reason']==reason for s in selections) for reason in sorted({s['reason'] for s in selections})}))
    print(json.dumps(dict(batch=batch,curves=len(rows),failures=len(failures),training_flags=len(training_flags))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('batch',choices=BATCHES);p.add_argument('--workers',type=int,default=12);p.add_argument('--reuse',action='store_true');o=p.parse_args();main(o.batch,o.workers,o.reuse)

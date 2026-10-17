"""Refine only flagged scores; retain all original flags and training results."""
from core import *
from concurrent.futures import ProcessPoolExecutor,as_completed
import time


def check(job):
    ident,method,p,aliases=job
    mod=load('revision_supplemental_audit',HERE/'analyze.py')
    g=mod.AuditGeometry(target(ident),method)
    values=[]
    for points in (2049,4097):
        events=g.score(np.asarray(p),points=points,anchors=True,return_events=True)[0]
        values.append(dict(points=points,events=events.tolist(),score=float(g.aggregate(events))))
    a,b=values
    tolerance=CFG['geometry_audit']['absolute_tolerance']+CFG['geometry_audit']['relative_tolerance']*abs(b['score'])
    return dict(target=ident,method=method,aliases=aliases,refinements=values,
                supplementary_stable=abs(b['score']-a['score'])<=tolerance,
                tolerance=tolerance,original_flags_retained=True)


def main():
    failures=json.loads((OUT/'analysis/AUDIT.json').read_text())['failures']
    groups={}
    for r in failures:
        d=json.loads((OUT/f"analysis/details/{r['target']}_{r['method']}_{r['seed']}_{r['rule']}.json").read_text())
        # MAX and MEAN are exactly the same definition on this one-event control.
        semantic='MAG_PHA_MAX' if r['target']=='control0p25' and r['method'] in ('MAG_PHA','MAG_PHA_MAX','MAG_PHA_MEAN') else r['method']
        key=(r['target'],semantic,tuple(d['log_parameters']))
        groups.setdefault(key,[]).append(dict(method=r['method'],seed=r['seed'],rule=r['rule'],training=r['geometry_train'],reference1025=r['geometry_reference']))
    jobs=[(ident,method,list(p),aliases) for (ident,method,p),aliases in groups.items()]
    start=time.perf_counter();rows=[]
    with ProcessPoolExecutor(max_workers=min(5,len(jobs))) as pool:
        for fut in as_completed([pool.submit(check,j) for j in jobs]):
            rows.append(fut.result());print('Supplementary score check '+str(len(rows))+'/'+str(len(jobs)),flush=True)
    dump(OUT/'analysis/SUPPLEMENTAL_GEOMETRY_AUDIT.json',dict(original_failed_records=len(failures),unique_curves=len(jobs),seconds=time.perf_counter()-start,records=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),optimizer_unchanged=True))


if __name__=='__main__':main()

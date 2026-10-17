"""Retain original numerical failures; only increase diagnostic resolution."""
from core import *
from concurrent.futures import ProcessPoolExecutor,as_completed
import time

def one(job):
    ident,method,p,aliases=job;g=Geometry(target(ident),method);values=[]
    for points in (2049,4097):
        events=g.score(np.array(p),points=points,anchors=True,return_events=True)[0]
        values.append(dict(points=points,events=events.tolist(),score=float(g.aggregate(events))))
    a,b=np.array(values[0]['events']),np.array(values[1]['events'])
    tolerance=CFG['geometry_audit']['absolute_tolerance']+CFG['geometry_audit']['relative_tolerance']*abs(b)
    return dict(target=ident,method=method,aliases=aliases,refinements=values,supplementary_components_stable=bool(np.all(abs(a-b)<=tolerance)),component_tolerances=tolerance.tolist(),original_flags_retained=True)

def main():
    failures=json.loads((OUT/'analysis/AUDIT.json').read_text(encoding='utf-8'))['failures'];groups={}
    for r in failures:
        d=json.loads((OUT/f"analysis/details/{r['target']}_{r['method']}_{r['seed']}_{r['rule']}.json").read_text(encoding='utf-8'))
        key=(r['target'],r['method'],tuple(d['log_parameters']))
        groups.setdefault(key,[]).append(dict(seed=r['seed'],rule=r['rule'],training=r['geometry_train'],reference1025=r['geometry_reference'],base_ok=r['base_score_ok'],structure_stable=r['stable']))
    jobs=[(i,m,list(p),a) for (i,m,p),a in groups.items()];rows=[];start=time.perf_counter()
    if jobs:
        with ProcessPoolExecutor(max_workers=min(32,len(jobs),os.cpu_count() or 1)) as pool:
            for future in as_completed([pool.submit(one,j) for j in jobs]):
                rows.append(future.result());print('Supplement '+str(len(rows))+'/'+str(len(jobs)),flush=True)
    dump(OUT/'analysis/SUPPLEMENTAL_GEOMETRY_AUDIT.json',dict(original_failed_records=len(failures),unique_candidate_definitions=len(jobs),seconds=time.perf_counter()-start,records=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),optimizer_unchanged=True))

if __name__=='__main__':main()

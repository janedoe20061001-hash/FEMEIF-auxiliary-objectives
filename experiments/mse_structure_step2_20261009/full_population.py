from common import *
from concurrent.futures import ProcessPoolExecutor,as_completed
import time
def job(arg):
    s,seed=arg;m=Target(s);cat=target_catalog(m);rows=[]
    with np.load(HERE/'batch'/label(s)/f'seed_{seed}'/'generation_300.npz') as data:
        fi=set(map(int,data['front_indices']))
        for index,x in enumerate(data['log_parameters']):
            previous=None;stable=False
            for lev in [2,4,8]:
                f=improved_dense_grid(m.record,10**x[None,:],lev);ee=classify_events(cat,amp_events(np.log10(f),spectra(f,10**x)[0]))
                if previous:
                    stable=all(a['matched']==b['matched'] and a['complete']==b['complete'] for a,b in zip(previous,ee))
                    if stable:break
                previous=ee
            ev={e['event_id']:e for e in ee};rows.append(dict(scale=s,seed=seed,index=index,is_front=index in fi,grid_stable=stable,refinement=lev,P3_matched=ev['P3']['matched'],V2_matched=ev['V2']['matched'],P3_complete=ev['P3']['complete'],V2_complete=ev['V2']['complete'],joint_complete=ev['P3']['complete'] and ev['V2']['complete']))
    return rows
def main():
    cfg=json.loads((HERE/'CONFIG.json').read_text());rows=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(job,(s,seed)) for s in cfg['run_scales'] for seed in cfg['seeds']]
        for i,f in enumerate(as_completed(futures),1):rows.extend(f.result())
    csvout(OUT/'analysis/FULL_POPULATION_STRUCTURE.csv',rows);assert all(r['grid_stable'] for r in rows)
    summary=[]
    for s in cfg['run_scales']:
        rr=[r for r in rows if r['scale']==s]
        summary.append(dict(scale=s,seeds=8,candidates=len(rr),any_joint_complete_seeds=len({r['seed'] for r in rr if r['joint_complete']}),any_P3_complete_seeds=len({r['seed'] for r in rr if r['P3_complete']}),any_V2_complete_seeds=len({r['seed'] for r in rr if r['V2_complete']}),dominated_joint_complete_candidates=sum(r['joint_complete'] and not r['is_front'] for r in rr)))
    dump(OUT/'FULL_POPULATION_SUMMARY.json',summary);dump(OUT/'FULL_POPULATION_VALIDATION.json',dict(passed=True,rows=len(rows),unstable=0,seconds=time.perf_counter()-start,scope='all 200 retained candidates per run at generation 300, not historical eliminated candidates'))
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

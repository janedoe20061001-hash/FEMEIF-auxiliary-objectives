"""Whole-band score context for the frozen pairs; no new optimization."""
from score_pairs import initialize_score
import score_pairs
from local_geometry import *
from concurrent.futures import ProcessPoolExecutor,as_completed

def one(item):
    uid,lp,digest=item;model=score_pairs.MODEL;st=time.time();all_scores={}
    for n,ref in ((16,0),(32,1),(64,2)):
        values=[model.fixed(lp,w['block']['label'],n,ref) for w in model.windows]
        all_scores[n]=dict(mean={k:float(np.mean([v[k] for v in values])) for k in values[0]},windows=values)
    r=dict(uid=uid,bank_sha256=digest,log_parameters=lp,gauss16=all_scores[16]['mean'],gauss32=all_scores[32]['mean'],scores=all_scores[64]['mean'],window_scores=all_scores[64]['windows'],labels=[w['block']['label'] for w in model.windows],
        fine_converged=all(abs(all_scores[64]['mean'][k]-all_scores[32]['mean'][k])<=1e-4+1e-3*abs(all_scores[64]['mean'][k]) for k in all_scores[64]['mean']),
        seconds_all_eight_metrics_three_levels=time.time()-st,
        timing_scope='Diagnostic joint evaluation of eight outputs at three levels; not optimized individual metric runtime')
    dump(OUT/'global_scores'/f'{uid}.json',r);return r

def main(workers):
    bank=OUT/'SCORING_BANK.json';b=json.loads(bank.read_text());digest=sha(bank);items={}
    for p in b['pairs']:
        for side in ('better','worse'):
            v=p[side];items[v['uid']]=(v['uid'],v['log_parameters'],digest)
    result={}
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize_score) as pool:
        fs=[pool.submit(one,v) for v in items.values()]
        for f in as_completed(fs):
            r=f.result();result[r['uid']]=r;print('Whole-band context',len(result),'/',len(items),'converged',r['fine_converged'],flush=True)
    rows=[]
    for p in b['pairs']:
        a=result[p['better_uid']];c=result[p['worse_uid']]
        for k in a['scores']:
            rows.append(dict(pair_id=p['pair_id'],family=p['family'],region=p['region'],seed=p['seed'],metric=k,local_label_better_global_score=a['scores'][k],local_label_worse_global_score=c['scores'][k],
                global_order_favors_local_label=a['scores'][k]<c['scores'][k],both_fine_converged=a['fine_converged'] and c['fine_converged']))
    from analyze import csvout
    csvout('GLOBAL_PAIR_SCORE_CONTEXT.csv',rows)
    dump(OUT/'GLOBAL_SCORE_CHECKS.json',dict(candidate_evaluations=len(result),all_fine_converged=all(r['fine_converged'] for r in result.values()),bank_sha256=digest,
        scope='Whole-band context only; local structural labels do not define global superiority; not used to change the metric gate',optimization_started=False))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);main(parser.parse_args().workers)

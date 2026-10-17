"""Score a frozen candidate bank; retain all numerical and directional failures."""
from local_geometry import *
from analyze import csvout
from concurrent.futures import ProcessPoolExecutor,as_completed

MODEL=None
def initialize_score():
    global MODEL
    checks();MODEL=LocalGeometry()

def job(item):
    uid,label,lp,bank_sha=item;out=OUT/'metric_scores'/f'{uid}_{label}.json'
    if out.exists():
        value=json.loads(out.read_text());assert value['bank_sha256']==bank_sha and value['implementation_sha256']==sha(HERE/'local_geometry.py');return value
    st=time.time();r16=MODEL.fixed(lp,label,16,0);r32=MODEL.fixed(lp,label,32,1);r64=MODEL.fixed(lp,label,64,2)
    w=next(w for w in MODEL.windows if w['block']['label']==label);p=10**np.array(lp)
    legacy=MODEL.g.block(p,w,MODEL.g.modes(p),32,1,('J0','ARX','H'))
    for new,old in [('J0','J0'),('ARX_L','ARX'),('H','H')]:
        assert abs(r32[new]-legacy[old])<=1e-8+1e-8*abs(legacy[old]),('old_definition_mismatch',uid,label,new,r32[new],legacy[old])
    poly128=MODEL.polygon(lp,label,128);poly256=MODEL.polygon(lp,label,256)
    fine_ok=all(abs(r64[k]-r32[k])<=1e-4+1e-3*abs(r64[k]) for k in r64)
    poly_ok=abs(poly128-poly256)<=.01+.01*abs(poly256) and abs(r64['Local_ARX']-poly256)<=.01+.01*abs(poly256)
    truth=MODEL.fixed(np.log10(MODEL.c.p),label,32,1);truth_ok=max(truth.values())<1e-12
    result=dict(uid=uid,region=label,bank_sha256=bank_sha,implementation_sha256=sha(HERE/'local_geometry.py'),log_parameters=lp,
        scores=r64,gauss16=r16,gauss32=r32,polygon128_local=poly128,polygon256_local=poly256,
        fine_converged=fine_ok,independent_local_agrees=poly_ok,truth_near_zero=truth_ok,old_definitions_agree=True,
        numerical_passed=bool(fine_ok and poly_ok and truth_ok),truth=truth,
        target_floor_by_h=MODEL.floor_by_h,seconds=time.time()-st)
    dump(out,result);return result

def main(workers):
    bank_path=OUT/'SCORING_BANK.json';bank=json.loads(bank_path.read_text());digest=sha(bank_path)
    tasks={}
    for p in bank['pairs']:
        for side in ('better','worse'):
            row=p[side];tasks[int(row['uid']),p['region']]=(int(row['uid']),p['region'],row['log_parameters'],digest)
    scores={}
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize_score) as pool:
        fs=[pool.submit(job,t) for t in tasks.values()]
        for f in as_completed(fs):
            r=f.result();scores[r['uid'],r['region']]=r
            print(json.dumps(dict(scored=len(scores),total=len(tasks),uid=r['uid'],region=r['region'],passed=r['numerical_passed'])),flush=True)
    rows=[]
    for p in bank['pairs']:
        a=scores[p['better_uid'],p['region']];b=scores[p['worse_uid'],p['region']]
        for metric in a['scores']:
            aa=[a['scores'][metric],a['gauss32'][metric],a['gauss16'][metric]];bb=[b['scores'][metric],b['gauss32'][metric],b['gauss16'][metric]]
            if metric in ('Local_ARX','Combined'):
                for k in ('polygon128_local','polygon256_local'):
                    aa.append(a[k] if metric=='Local_ARX' else .5*a[k]+.5*a['scores']['ARX_S_L'])
                    bb.append(b[k] if metric=='Local_ARX' else .5*b[k]+.5*b['scores']['ARX_S_L'])
            eps=1e-8*max(max(abs(x) for x in aa),max(abs(x) for x in bb),1e-300)
            raw='correct' if a['scores'][metric]<b['scores'][metric]-eps else 'wrong' if a['scores'][metric]>b['scores'][metric]+eps else 'tie'
            robust='correct' if max(aa)<min(bb)-eps else 'wrong' if min(aa)>max(bb)+eps else 'uncertain'
            rows.append(dict(pair_id=p['pair_id'],family=p['family'],method=p['method'],seed=p['seed'],generation=p['generation'],region=p['region'],metric=metric,
                better_score=a['scores'][metric],worse_score=b['scores'][metric],raw_direction=raw,robust_direction=robust,numerical_passed=a['numerical_passed'] and b['numerical_passed'],
                better_uid=p['better_uid'],worse_uid=p['worse_uid']))
    csvout('PAIR_METRIC_DIRECTIONS.csv',rows)
    summaries=[]
    for metric in sorted({r['metric'] for r in rows}):
        rr=[r for r in rows if r['metric']==metric];clusters={}
        for r in rr:clusters.setdefault((r['better_uid'],r['worse_uid']),[]).append(r)
        successes=sum(all(v['robust_direction']=='correct' and v['numerical_passed'] for v in vs) for vs in clusters.values())
        extra=[r for r in rr if r['family']=='extra_structure']
        summaries.append(dict(metric=metric,entries=len(rr),robust_correct=sum(r['robust_direction']=='correct' for r in rr),robust_wrong=sum(r['robust_direction']=='wrong' for r in rr),uncertain=sum(r['robust_direction']=='uncertain' for r in rr),
            unique_parameter_pairs=len(clusters),correct_parameter_pairs=successes,extra_correct=sum(r['robust_direction']=='correct' for r in extra),extra_total=len(extra)))
    csvout('METRIC_DIRECTION_SUMMARY.csv',summaries)
    count_gate=bank['unique_parameter_pairs']>=12 and len(bank['seeds'])>=6 and len(bank['families'])>=3
    all_num=all(r['numerical_passed'] for r in scores.values())
    for r in summaries:
        r['development_gate_passed']=bool(count_gate and all_num and r['correct_parameter_pairs']>=.75*r['unique_parameter_pairs'] and r['extra_total']>0 and r['extra_correct']>=.8*r['extra_total'])
    dump(OUT/'METRIC_CHECKS.json',dict(bank_sha256=digest,unique_candidate_region_evaluations=len(scores),all_numerical_passed=all_num,count_gate_passed=count_gate,summary=summaries,
        scope='Development structural labels and old-objective matching; no optimization benefit or generalization claim',optimization_started=False))
    print(json.dumps(summaries),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);main(parser.parse_args().workers)

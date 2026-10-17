"""Freeze fresh shift controls and old regression/diagnostic cohorts."""
from correspondence import *
def main():
    OUT.mkdir(parents=True,exist_ok=True);prior=ROOT/'output/valley_perturbation_20261017/PLAN.json';p=json.loads(prior.read_text(encoding='utf-8'));tasks=[];expected={}
    for t in p['targets']:
        width=t['core'][1]-t['core'][0]
        controls=[dict(kind='fresh_shift',delta=k*width,shift_widths=k) for k in [-1.75,-1.25,-.75,-.375,0.,.375,.75,1.25,1.75]]
        for side,bound in [('left',t['core'][0]),('right',t['core'][1])]:
            controls += [dict(kind='boundary',delta=bound-t['center']+eps,boundary=side,epsilon=eps) for eps in [-5e-9,0.,5e-9]]
        for q in controls:
            ident=f"new_{len(tasks):03}";m=copy.deepcopy(t['model']);m['id']=ident;s=10**q['delta'];m['values']={key:v/s if key[0] in 'LC' else v for key,v in m['values'].items()};Physical(m)
            tasks.append(dict(id=ident,cohort='new_control',target=t['id'],model=m))
            cx=t['center']+q['delta'];expected[ident]=dict(**q,center_x=cx,location=location(cx,t))
    assert len(tasks)==75
    for item in p['tasks']:tasks.append(dict(id='reg_'+item['id'],cohort='physical_regression',target=item['target'],model=item['model'],source_id=item['id'],source_family=item['family']))
    oldpath=ROOT/'output/valley_process_20261017/COHORT.json';old=json.loads(oldpath.read_text(encoding='utf-8'));targets={t['id']:t for t in p['targets']}
    for ident,tid,parameters,source in old['tasks']:
        m=copy.deepcopy(targets[tid]['model']);m['id']='old_'+ident;m['values']=dict(zip(Physical(m).names,parameters));Physical(m)
        tasks.append(dict(id='old_'+ident,cohort='old_candidate',target=tid,model=m,source_id=ident,source_pair_id=source['pair_id'],source_side=source['side']))
    assert len(tasks)==373
    dump(OUT/'PLAN.json',dict(targets=p['targets'],tasks=tasks,expected_new_controls=expected,old_physical_plan_sha256=sha(prior),old_candidate_cohort_sha256=sha(oldpath),scope='Conservative occupancy and numerical translation consistency only; not a general structural matcher'))
    files=list(HERE.glob('*.py'))+[OUT/'PLAN.json',ROOT/'docs/谷过程频率归属_实现与核验方案_20261017.md',ROOT/'experiments/valley_perturbation_20261017/physical.py',prior,oldpath]
    dump(OUT/'SOURCE_MANIFEST.json',dict(files={q.relative_to(ROOT).as_posix():sha(q) for q in files},frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print('Frozen 75 controls + 266 physical regressions + 32 old candidates',flush=True)
if __name__=='__main__':main()

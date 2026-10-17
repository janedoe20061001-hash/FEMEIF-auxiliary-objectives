"""All prescribed tasks; no adaptive formula selection or optimization."""
from metric import *
from concurrent.futures import ProcessPoolExecutor,as_completed
MODELS={}
def get(tid):
    if tid not in MODELS:MODELS[tid]=Variation(tid)
    return MODELS[tid]
def job(task):
    ident,tid,p,labels,meta=task;p=np.array(p);start=time.perf_counter();m=get(tid);result={}
    try:
        for label in labels:
            den=m.scales(label);levels=[]
            for n in [64,128,256,512,1024]:
                v=m.polygon(p,label,n);levels.append(dict(n=n,scores=v))
                if n>=256 and agree(levels[-2]['scores'],v):break
            checks=[]
            for n,ref in [(16,0),(32,0),(64,1),(64,2)]:
                g=m.gaussian(p,label,n,ref);checks.append(dict(n=n,refinement=ref,scores=g))
                if len(checks)>=2 and agree(checks[-2]['scores'],g) and agree(levels[-1]['scores'],g):break
            ok=len(levels)>1 and agree(levels[-2]['scores'],levels[-1]['scores']) and len(checks)>1 and agree(checks[-2]['scores'],checks[-1]['scores']) and agree(levels[-1]['scores'],checks[-1]['scores']) and m.scale_audit[label]['passed']
            result[label]=dict(scores=levels[-1]['scores'],polygon_levels=levels,derivative_levels=checks,numerical_passed=bool(ok),target_scale=m.scale_audit[label])
        out=dict(id=ident,target=tid,parameters=p,meta=meta,windows=result,numerical_passed=all(r['numerical_passed'] for r in result.values()),seconds=time.perf_counter()-start,source_sha256=sha(HERE/'metric.py'))
    except Exception as exc:
        import traceback
        out=dict(id=ident,target=tid,meta=meta,numerical_passed=False,error=repr(exc),traceback=traceback.format_exc())
    dump(OUT/'scores'/f'{ident}.json',out);return out

def summarize():
    tasks=json.loads((OUT/'TASKS.json').read_text())['tasks'];records={t[0]:json.loads((OUT/'scores'/f'{t[0]}.json').read_text()) for t in tasks};rows=[]
    for task in tasks:
        ident,tid,p,labels,meta=task
        if meta.get('side')!='better':continue
        bad=ident.removesuffix('_better')+'_worse';a=records[ident];b=records[bad];label=labels[0]
        for metric in NAMES:
            ok=a['numerical_passed'] and b['numerical_passed']
            av=a.get('windows',{}).get(label);bv=b.get('windows',{}).get(label)
            if av and bv:
                aa=[x['scores'][metric] for x in av['polygon_levels'][-2:]+av['derivative_levels'][-2:]]
                bb=[x['scores'][metric] for x in bv['polygon_levels'][-2:]+bv['derivative_levels'][-2:]]
                eps=1e-8*max(aa+bb+[1e-300]);order='correct' if ok and max(aa)<min(bb)-eps else 'wrong' if ok and min(aa)>max(bb)+eps else 'uncertain'
                va,vb=av['scores'][metric],bv['scores'][metric]
            else:order='failure';va=vb=None
            rows.append(dict(scope='old_development' if ident.startswith('dev_') else 'old_regression',pair_id=meta['pair_id'],family=meta['family'],target=tid,region=label,metric=metric,better=va,worse=vb,direction=order,numerical_passed=ok))
    csvout(OUT/'DIRECTIONS.csv',rows);summary=[]
    for metric in NAMES:
        d=[r for r in rows if r['metric']==metric and r['scope']=='old_development'];r=[r for r in rows if r['metric']==metric and r['scope']=='old_regression']
        dc=sum(x['direction']=='correct' for x in d);rc=sum(x['direction']=='correct' for x in r)
        de=sum(x['direction']=='correct' and x['family']=='extra_structure' for x in d)
        families={f:sum(x['direction']=='correct' and x['family']==f for x in r) for f in ['missing_structure','existing_distortion','extra_structure']}
        known=all(any(x['pair_id']==pid and x['direction']=='correct' for x in d) for pid in ['pair_001','pair_002'])
        gate=dc>=9 and de>=5 and known and rc>=18 and all(x>=6 for x in families.values()) and families['extra_structure']>=7
        summary.append(dict(metric=metric,development_correct=dc,development_extra_correct=de,regression_correct=rc,regression_family_correct=families,two_known_correct=known,regression_direction_gate_passed=gate))
    control=[]
    for task in tasks:
        if task[1]!='controlled':continue
        r=records[task[0]]
        for label,w in r.get('windows',{}).items():control.append(dict(id=task[0],label=label,**task[-1],**w['scores'],numerical_passed=w['numerical_passed']))
    csvout(OUT/'CONTROLLED.csv',control)
    physical=[r for r in control if r['label']=='P_control' and r['family']=='controlled_grid' and r['amplitude']==1 and r['q_ratio']==1 and r['offset']>=0]
    physical=sorted(physical,key=lambda x:x['offset']);values=[r['JV'] for r in physical]
    monotone=all(y>=x-(1e-4+1e-3*abs(y)) for x,y in zip(values,values[1:]))
    out=dict(summary=summary,tasks=len(tasks),numerical_passed=sum(r['numerical_passed'] for r in records.values()),windows=sum(len(r.get('windows',{})) for r in records.values()),physical_translation=physical,physical_translation_monotone=monotone,optimization_started=False)
    dump(OUT/'SUMMARY.json',out);print(json.dumps(out,default=lambda v:v.tolist()),flush=True)

def main(workers):
    start=time.time();dump(OUT/'ANALYTIC_CHECKS.json',analytic_checks());tasks=json.loads((OUT/'TASKS.json').read_text())['tasks'];good=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(job,t) for t in tasks]
        for n,f in enumerate(as_completed(futures),1):
            r=f.result();good+=int(r['numerical_passed']);dump(OUT/'STATUS.json',dict(completed=n,total=len(tasks),numerical_passed=good,seconds=time.time()-start))
            if n%16==0 or n==len(tasks):print('COMPLETED',n,'/',len(tasks),'PASS',good,flush=True)
    summarize();dump(OUT/'COMPLETE.json',dict(tasks=len(tasks),numerical_passed=good,seconds=time.time()-start,optimization_started=False))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=32);p.add_argument('--summarize',action='store_true');a=p.parse_args()
    summarize() if a.summarize else main(a.workers)

"""All controlled and frozen-pair scores; retain numerical failures and directions."""
from common import *
from concurrent.futures import ProcessPoolExecutor,as_completed
MODELS={}
METRICS=['J0','J0_W','ARX_L','ARX_S_L','Local_ARX','D','H','Root','B','J0_local_weighted','Root_delta_0p001','Root_delta_0p1']

def point_objectives(m,p,label,n,ref):
    w=next(w for w in m.windows if w['block']['label']==label)
    t,mass=nodes(m.g.panel_edges(w,m.g.modes(p),ref),n)
    z=m.response(w,t,p)[0];r=m.response(w,t,m.c.p)[0]
    return dict(MAG=float(mass@(20*np.log10(abs(z)/abs(r)))**2),PHA=float(mass@(np.angle(z)-np.angle(r))**2))

def getmodel(tid):
    if tid not in MODELS:MODELS[tid]=Geometry(control_target() if tid=='controlled' else target(tid))
    return MODELS[tid]

def preflight():
    assert CONFIG['delta']==.01
    r=np.array([-100.,-1.,-.01,0.,.01,1.,100.]);np.testing.assert_allclose(root_transform(r),np.sign(r)*(np.sqrt(abs(r)+.01)-.1),atol=1e-14)
    for h in CONFIG['half_steps']:
        t=np.linspace(h,1-h,101);z=lambda u:u+1j*u*u
        np.testing.assert_allclose(z(t)-(z(t-h)+z(t+h))/2,-1j*h*h,atol=1e-14)
    d=control_target();c=d['circuit'];np.testing.assert_allclose(c.response(np.array([c.f0]))[0],[c.R],rtol=1e-8,atol=1e-8)
    freq=np.unique(np.r_[np.geomspace(100,1e6,1001),c.f0]);err=[]
    for a,off,q in [(1,0,1),(.1,-4,.125),(2,4,8)]:
        p=c.params(a,off,q);rr,ll,cc=p;f0=1/(2*np.pi*np.sqrt(ll*cc));qq=rr*np.sqrt(cc/ll)
        independent=rr/(1+1j*qq*(freq/f0-f0/freq));z,dz=c.response(freq,p)
        np.testing.assert_allclose(z,independent,rtol=1e-8,atol=1e-8);err.append(float(np.max(abs(z-independent)/np.maximum(abs(z),1e-30))))
    # Only truth is evaluated here; no development or holdout candidate ranking.
    m=getmodel('controlled');truth={}
    for w in m.windows:
        name=w['block']['label'];v=m.block(c.p,name,32,1);assert max(v.values())<1e-12;truth[name]=v
    dump(OUT/'PREFLIGHT.json',dict(polynomial_B_passed=True,root_transform_passed=True,independent_RLC_max_relative=max(err),truth=truth,
        candidate_bank_already_frozen=(OUT/'HOLDOUT_BANK.json').exists(),common_source_sha256=sha(HERE/'common.py')))

def job(item):
    ident,tid,parameters,labels,meta=item;p=np.asarray(parameters);path=OUT/'scores'/f'{ident}.json'
    if path.exists():
        r=json.loads(path.read_text());assert r['implementation_sha256']==sha(HERE/'common.py');return r
    start=time.time();m=getmodel(tid);windows={}
    try:
        for label in labels:
            levels=[]
            for n,ref in [(16,0),(32,1),(64,2)]:levels.append(dict(order=n,refinement=ref,scores={**m.block(p,label,n,ref),**point_objectives(m,p,label,n,ref)}))
            if not converged(levels[-2]['scores'],levels[-1]['scores']):
                for ref in [3,4]:
                    levels.append(dict(order=64,refinement=ref,scores={**m.block(p,label,64,ref),**point_objectives(m,p,label,64,ref)}))
                    if converged(levels[-2]['scores'],levels[-1]['scores']):break
            independent={f'polygon{s}':m.polygon(p,label,s) for s in [128,256]}
            fine=levels[-1]['scores'];polyok=all(abs(fine[k]-independent['polygon256'][k])<=.01+.01*abs(independent['polygon256'][k]) and abs(independent['polygon128'][k]-independent['polygon256'][k])<=.01+.01*abs(independent['polygon256'][k]) for k in ['Root','B'])
            cv=converged(levels[-2]['scores'],fine)
            windows[label]=dict(scores=fine,levels=levels,**independent,numerical_passed=bool(cv and polyok),fine_converged=cv,independent_agrees=polyok)
        result=dict(id=ident,target=tid,parameters=parameters,meta=meta,windows=windows,numerical_passed=all(v['numerical_passed'] for v in windows.values()),
            seconds=time.time()-start,implementation_sha256=sha(HERE/'common.py'),config_sha256=sha(HERE/'config.json'))
    except Exception as exc:
        import traceback
        result=dict(id=ident,target=tid,parameters=parameters,meta=meta,numerical_passed=False,error=repr(exc),traceback=traceback.format_exc(),implementation_sha256=sha(HERE/'common.py'))
    dump(path,result);return result

def make_tasks():
    m=getmodel('controlled');c=m.c;tasks=[];labels=[w['block']['label'] for w in m.windows];ident=0
    for a in CONFIG['amplitudes']:
        for off in CONFIG['offsets']:
            for q in CONFIG['q_ratios']:
                p=c.params(a,off,q);tasks.append((f'control_{ident:04}','controlled',p.tolist(),labels,dict(family='controlled_grid',amplitude=a,offset=off,q_ratio=q)));ident+=1
    for off in [-8,-4,4,8]:
        for a in [.1,.3,1.]:
            for q in [.5,1.,2.]:
                p=np.r_[c.p,c.params(a,off,q)];tasks.append((f'control_{ident:04}','controlled',p.tolist(),labels,dict(family='extra_branch',amplitude=a,offset=off,q_ratio=q)));ident+=1
    assert ident==684
    old=json.loads((ROOT/'output/arx_structure_diagnosis_20261016/SCORING_BANK.json').read_text())
    for pair in old['pairs']:
        for side in ['better','worse']:
            tasks.append((f"dev_{pair['pair_id']}_{side}",'case1',(10**np.array(pair[side]['log_parameters'])).tolist(),[pair['region']],dict(pair_id=pair['pair_id'],side=side,family=pair['family'],scope='development')))
    bank=json.loads((OUT/'HOLDOUT_BANK.json').read_text())
    for pair in bank['selected']:
        for side in ['better','worse']:
            tasks.append((f"{pair['pair_id']}_{side}",pair['target'],(10**np.array(pair[side]['log_parameters'])).tolist(),[pair['region']],dict(pair_id=pair['pair_id'],side=side,family=pair['family'],scope='candidate_holdout')))
    dump(OUT/'SCORING_TASKS.json',dict(tasks=tasks,holdout_sha256=sha(OUT/'HOLDOUT_BANK.json'),old_bank_sha256=sha(ROOT/'output/arx_structure_diagnosis_20261016/SCORING_BANK.json'),
        common_sha256=sha(HERE/'common.py'),config_sha256=sha(HERE/'config.json'),protocol_sha256=sha(HERE/'PROTOCOL.md')))
    return tasks

def directions():
    old=json.loads((ROOT/'output/arx_structure_diagnosis_20261016/SCORING_BANK.json').read_text());held=json.loads((OUT/'HOLDOUT_BANK.json').read_text());rows=[]
    for scope,pairs in [('dev',old['pairs']),('held',held['selected'])]:
        for p in pairs:
            stem=f"dev_{p['pair_id']}" if scope=='dev' else p['pair_id']
            a=json.loads((OUT/'scores'/f'{stem}_better.json').read_text());b=json.loads((OUT/'scores'/f'{stem}_worse.json').read_text())
            for metric in METRICS:
                aa=a.get('windows',{}).get(p['region']);bb=b.get('windows',{}).get(p['region']);num=bool(a['numerical_passed'] and b['numerical_passed'])
                if aa and bb:
                    va=[v['scores'][metric] for v in aa['levels']];vb=[v['scores'][metric] for v in bb['levels']]
                    # Robust order uses the two finest converged levels; all coarser levels retained.
                    va=va[-2:];vb=vb[-2:]
                    if metric in ['Root','B']:
                        va += [aa[f'polygon{s}'][metric] for s in [128,256]];vb += [bb[f'polygon{s}'][metric] for s in [128,256]]
                    eps=1e-8*max(max(map(abs,va)),max(map(abs,vb)),1e-300)
                    order='correct' if num and max(va)<min(vb)-eps else 'wrong' if num and min(va)>max(vb)+eps else 'uncertain'
                    aval,bval=aa['scores'][metric],bb['scores'][metric]
                else:order='numerical_failure';aval=bval=None
                rows.append(dict(scope=scope,pair_id=p['pair_id'],family=p['family'],target='case1' if scope=='dev' else p['target'],seed=p['seed'],region=p['region'],metric=metric,
                    better_score=aval,worse_score=bval,direction=order,numerical_passed=num))
    csvout(OUT/'PAIR_DIRECTIONS.csv',rows);summaries=[]
    for metric in METRICS:
        item=dict(metric=metric)
        for scope in ['dev','held']:
            rr=[r for r in rows if r['metric']==metric and r['scope']==scope]
            item[scope+'_total']=len(rr);item[scope+'_correct']=sum(r['direction']=='correct' for r in rr);item[scope+'_wrong']=sum(r['direction']=='wrong' for r in rr);item[scope+'_uncertain']=sum(r['direction'] not in ['correct','wrong'] for r in rr)
            extra=[r for r in rr if r['family'] in ['extra_structure']];item[scope+'_extra_correct']=sum(r['direction']=='correct' for r in extra);item[scope+'_extra_total']=len(extra)
        item['two_P3_counterexamples_correct']=all(any(r['scope']=='dev' and r['pair_id']==pid and r['metric']==metric and r['direction']=='correct' for r in rows) for pid in ['pair_001','pair_002'])
        families={f:sum(r['scope']=='held' and r['metric']==metric and r['family']==f and r['direction']=='correct' for r in rows) for f in ['missing_structure','existing_distortion','extra_structure']}
        item['held_family_correct']=families
        item['direction_gates_passed']=bool(item['dev_correct']>=9 and item['dev_extra_correct']>=5 and item['two_P3_counterexamples_correct'] and held['complete'] and item['held_correct']>=18 and all(v>=6 for v in families.values()) and item['held_extra_correct']>=7)
        summaries.append(item)
    dump(OUT/'DIRECTION_SUMMARY.json',dict(summary=summaries,optimization_started=False,holdout_complete=held['complete']))
    for v in summaries:print(json.dumps(v),flush=True)

def main(workers):
    assert (OUT/'HOLDOUT_BANK.json').exists(),'Freeze candidate bank first'
    preflight();tasks=make_tasks();st=time.time();good=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(job,item) for item in tasks]
        for n,f in enumerate(as_completed(fs),1):
            r=f.result();good+=int(r['numerical_passed']);dump(OUT/'SCORING_STATUS.json',dict(completed=n,total=len(tasks),numerical_passed=good,seconds=time.time()-st))
            if n%16==0 or n==len(tasks):print('Scored',n,'/',len(tasks),'numerical passed',good,flush=True)
    directions();dump(OUT/'SCORING_COMPLETE.json',dict(tasks=len(tasks),numerical_passed=good,seconds=time.time()-st,implementation_sha256=sha(HERE/'common.py')))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=32);main(parser.parse_args().workers)

"""Freeze fresh pairs using old objectives and event labels, never new scores."""
from common import *
from core import case1_modes
from concurrent.futures import ProcessPoolExecutor,as_completed
from types import SimpleNamespace
CACHE={}
FAMILIES=['missing_structure','existing_distortion','extra_structure']

def init():
    global I,R,EXCLUDED
    I={};R={}
    p=ROOT/'output/arx_structure_diagnosis_20261016'
    with np.load(p/'INVENTORY.npz') as a:I={k:a[k] for k in a.files}
    with np.load(p/'STRUCTURE_RESULTS.npz') as a:R={k:a[k] for k in a.files}
    b=json.loads((p/'SCORING_BANK.json').read_text());EXCLUDED={np.asarray(v[s]['log_parameters'],dtype='<f8').tobytes() for v in b['pairs'] for s in ('better','worse')}

def model(tid):
    if tid not in CACHE:
        d=target(tid);c=d['circuit']
        from features import Features
        fast=Features(d) if os.name!='nt' else None
        q=d['quadrature'];grids={l:q.reference_grid(None,l) for l in (1,2,4,8)}
        CACHE[tid]=d,fast,grids
    return CACHE[tid]

def full(tid,lp):
    d,fast,grids=model(tid);c=d['circuit'];p=10**lp;prev=None;now=None
    modes=case1_modes(p) if tid=='case1' else c.modes(p)
    for level in (1,2,4,8):
        xx=np.unique(np.r_[grids[level],anchors(SimpleNamespace(modes=lambda _:modes),p,2.,6.,24*level)])
        zz=(fast.response(10**xx,p) if fast else c.response(10**xx,p))[0]
        fit=diag.events(xx,zz,BASECFG);rows,extra=diag.match(d['record']['events'],fit,BASECFG)
        complete=np.array([r['complete'] for r in rows]);matched=np.array([r['matched'] for r in rows])
        vals=np.array([[np.nan if r[k] is None else r[k] for k in ['shift_decades','height_delta_db','width_ratio']] for r in rows])
        now=dict(complete=complete,matched=matched,values=vals,extra=len(extra),nfit=len(fit))
        if prev is not None:
            stable=(complete==prev['complete'])&(matched==prev['matched'])
            stable &= np.isclose(vals[:,0],prev['values'][:,0],atol=1e-5,rtol=0,equal_nan=True)
            stable &= np.isclose(vals[:,1],prev['values'][:,1],atol=.02,rtol=0,equal_nan=True)
            stable &= np.isclose(vals[:,2],prev['values'][:,2],atol=1e-8,rtol=.01,equal_nan=True)
            if stable.all() and len(extra)==prev['extra'] and len(fit)==prev['nfit']:break
        prev=now
    return dict(log_parameters=lp.tolist(),events=rows,extra_ids=extra,fit=fit,stable=stable.tolist(),level=level)

def scan_run(item):
    tid,mi,si,requested_generation=item;m=CFG['methods'][mi];seed=CFG['seeds'][si];out=OUT/'bank_candidates'/tid/m/f'{seed}_{requested_generation}.json'
    if out.exists():return json.loads(out.read_text())
    d,_,_=model(tid);event=d['record']['events'];folder=DATA/'jobs'/tid/m/f'seed_{seed}';summary=json.loads((folder/'summary.json').read_text());best={}
    # Final populations first. Historical saved populations are only inspected if needed.
    for gen in [requested_generation]:
        fp=folder/f'generation_{gen:04}.npz';assert sha(fp)==summary['files_sha256'][fp.name]
        with np.load(fp) as a:X=a['X'];F=a['F'][:,:2]
        records={};labels={}
        for i,lp in enumerate(X):
            if np.asarray(lp,dtype='<f8').tobytes() in EXCLUDED:continue
            if tid=='case1':
                uid=int(I['lookup'][mi,si,gen//20,i]);valid=R['evaluated_full'][uid] and R['stable'][uid].all()
                if valid:records[i]=dict(complete=R['complete'][uid],matched=R['matched'][uid],values=R['values'][uid],extra=int(R['extra'][uid]),nfit=int(R['nfit'][uid]))
            if i not in records:
                # Candidate geometry is cached by exact parameter digest across stages.
                key=hashlib.sha256(np.asarray(lp,dtype='<f8').tobytes()).hexdigest();cp=OUT/'bank_geometry'/tid/f'{key}.json'
                if cp.exists():v=json.loads(cp.read_text())
                else:v=full(tid,lp);dump(cp,v)
                if not all(v['stable']):continue
                labels[i]=v;ev=v['events']
                records[i]=dict(complete=np.array([r['complete'] for r in ev]),matched=np.array([r['matched'] for r in ev]),values=np.array([[np.nan if r[k] is None else r[k] for k in ['shift_decades','height_delta_db','width_ratio']] for r in ev]),extra=len(v['extra_ids']),nfit=len(v['fit']))
        indices=np.array(sorted(records));proposals={f:[] for f in FAMILIES if f not in best}
        if not len(indices):continue
        floors=np.array(CFG['audit']['base_atol']);rel=abs(F[indices,None,:]-F[None,indices,:])/np.maximum(np.maximum(F[indices,None,:],F[None,indices,:]),floors)
        ai,bi=np.where(np.all(rel<=.05,axis=2))
        for aa,bb in zip(ai,bi):
            i,j=int(indices[aa]),int(indices[bb])
            if i==j or np.array_equal(X[i],X[j]):continue
            a,b=records[i],records[j];order=(float(max(rel[aa,bb])),float(sum(rel[aa,bb])),i,j)
            for ei,e in enumerate(event):
                if not a['complete'][ei]:continue
                width=e['width_3db_decades']
                if width is None or abs(a['values'][ei,0])>width:continue
                vals=b['values'][ei]
                if not b['matched'][ei] and 'missing_structure' in proposals:proposals['missing_structure'].append((order,ei))
                elif b['matched'][ei] and not b['complete'][ei] and 'existing_distortion' in proposals:
                    q=max(abs(vals[0])/.02,abs(vals[1])/3,abs(np.log(vals[2]))/np.log(2)) if np.isfinite(vals).all() and vals[2]>0 else 0
                    if q>=2:proposals['existing_distortion'].append((order,ei))
            if 'extra_structure' in proposals and a['extra']<b['extra'] and sum(a['complete'])==sum(b['complete']):proposals['extra_structure'].append((order,None))
        for family,opts in proposals.items():
            for order,ei in sorted(opts):
                _,_,i,j=order
                if i not in labels:labels[i]=full(tid,X[i])
                if j not in labels:labels[j]=full(tid,X[j])
                a,b=labels[i],labels[j]
                if not(all(a['stable']) and all(b['stable'])):continue
                if family=='extra_structure':
                    if sum(r['complete'] for r in a['events'])!=sum(r['complete'] for r in b['events']):continue
                    extras=[f for f in b['fit'] if f['event_id'] in b['extra_ids']];block=None
                    for ev in extras:
                        v=next((bl for bl in d['record']['blocks'] if not bl['event'] and bl['lo']<=ev['x']<=bl['hi']),None)
                        if v is not None and not any(v['lo']<=f['x']<=v['hi'] for f in a['fit']):block=v;break
                    if block is None:continue
                    region=block['label']
                else:region=event[ei]['event_id']
                best[family]=dict(target=tid,method=m,seed=seed,generation=gen,region=region,family=family,better_index=i,worse_index=j,better_primary=F[i].tolist(),worse_primary=F[j].tolist(),better=a,worse=b,
                    source_stage=str(fp.relative_to(DATA)),source_stage_sha256=sha(fp),max_primary_relative=order[0]);break
        if len(best)==3:break
        # Exhaustive all-stage mixed scans would needlessly repeat thousands of curves
        # once all three families are present; otherwise follow the frozen latest-first order.
    result=dict(target=tid,method=m,seed=seed,proposals=list(best.values()))
    dump(out,result);return result

def main(workers,generation=300):
    OUT.mkdir(parents=True,exist_ok=True);items=[(tid,mi,si,generation) for tid in target_ids() for mi in range(6) for si in range(12)];allrows=[];st=time.time()
    with ProcessPoolExecutor(max_workers=workers,initializer=init) as pool:
        fs=[pool.submit(scan_run,item) for item in items]
        for n,f in enumerate(as_completed(fs),1):
            r=f.result();dump(OUT/'BANK_STATUS.json',dict(generation=generation,completed=n,total=len(items),seconds=time.time()-st))
            if n%8==0 or n==len(items):print('Candidate source runs',generation,n,'/',len(items),flush=True)
    for file in sorted((OUT/'bank_candidates').rglob('*.json')):
        value=json.loads(file.read_text());allrows.extend(value['proposals'])
    # Keep the latest eligible generation for each run/family before balanced draw.
    latest={}
    for r in allrows:
        k=(r['target'],r['method'],r['seed'],r['family'])
        if k not in latest or r['generation']>latest[k]['generation']:latest[k]=r
    allrows=list(latest.values())
    # Choose a balanced 3x3 allocation before scoring; no score fields exist here.
    tids=target_ids();allocation=[[3,3,2],[2,3,3],[3,2,3]];rng=np.random.default_rng(CONFIG['draw_seed'])
    ordered=sorted(allrows,key=lambda r:(r['target'],r['family'],r['seed'],r['method'],r['generation']))
    ranks=rng.permutation(len(ordered));ordered=[ordered[i] for i in ranks]
    selected=[];used=set();used_seed_condition=set();counts=np.zeros((3,3),int)
    # Deterministic backtracking enforces distinct seed within each condition and
    # at least twelve seed numbers overall; no metric-dependent substitutions.
    cells=[(i,j) for i in range(3) for j in range(3) for _ in range(allocation[i][j])]
    pools={(i,j):[r for r in ordered if r['target']==tids[i] and r['family']==FAMILIES[j]] for i in range(3) for j in range(3)}
    tries=[0]
    def choose(k):
        tries[0]+=1
        if tries[0]>200000:return False
        if k==24:return len({r['seed'] for r in selected})>=12
        i,j=cells[k]
        for r in pools[i,j]:
            pairkeys=[np.asarray(r[s]['log_parameters'],dtype='<f8').tobytes() for s in ['better','worse']]
            sk=(r['target'],r['seed'])
            if sk in used_seed_condition or any(v in used for v in pairkeys):continue
            selected.append(r);used.update(pairkeys);used_seed_condition.add(sk)
            if choose(k+1):return True
            selected.pop();used.difference_update(pairkeys);used_seed_condition.remove(sk)
        return False
    passed=choose(0)
    exact_status=None
    if not passed:
        # The finite backtracking cap is an engineering guard, not evidence that
        # a balanced bank is mathematically unavailable. Check exact feasibility.
        from scipy.optimize import milp,Bounds,LinearConstraint
        from scipy.sparse import lil_matrix
        candidates=[r for (i,j) in pools for r in pools[i,j]]
        if candidates:
            constraints=[];lower=[];upper=[]
            def add(ids,lo,hi):constraints.append(ids);lower.append(lo);upper.append(hi)
            for i in range(3):
                for j in range(3):add([k for k,r in enumerate(candidates) if r['target']==tids[i] and r['family']==FAMILIES[j]],allocation[i][j],allocation[i][j])
            for tid in tids:
                for seed in CFG['seeds']:add([k for k,r in enumerate(candidates) if r['target']==tid and r['seed']==seed],0,1)
            for seed in CFG['seeds']:add([k for k,r in enumerate(candidates) if r['seed']==seed],1,np.inf)
            parameter_usage={}
            for k,r in enumerate(candidates):
                for side in ['better','worse']:parameter_usage.setdefault(np.asarray(r[side]['log_parameters'],dtype='<f8').tobytes(),set()).add(k)
            for ids in parameter_usage.values():
                if len(ids)>1:add(list(ids),0,1)
            matrix=lil_matrix((len(constraints),len(candidates)))
            for i,ids in enumerate(constraints):matrix[i,ids]=1
            answer=milp(np.arange(1,len(candidates)+1)/len(candidates),integrality=np.ones(len(candidates)),bounds=Bounds(0,1),constraints=LinearConstraint(matrix.tocsr(),lower,upper),options={'time_limit':30})
            exact_status=dict(status=int(answer.status),message=answer.message)
            if answer.x is not None:
                selection=np.flatnonzero(answer.x>.5);selected=[candidates[k] for k in selection];passed=len(selected)==24
                if passed:
                    for row,lo,hi in zip(constraints,lower,upper):assert lo-1e-7<=sum(answer.x[k]>.5 for k in row)<=hi+1e-7
    if not passed:selected=[]
    for i,r in enumerate(selected):r['pair_id']=f'held_{i:03}'
    result=dict(scope='Pairs frozen before any new metric scoring; existing target development data, not independent target validation',selected=selected,required=24,
        complete=passed,seeds=sorted({r['seed'] for r in selected}),pool_counts={f'{tids[i]}|{FAMILIES[j]}':len(pools[i,j]) for i in range(3) for j in range(3)},
        source_run_stage_files=len(list((OUT/'bank_candidates').rglob('*.json'))),all_proposals=len(allrows),last_generation_checked=generation,backtracking_attempts=tries[0],exact_feasibility=exact_status,selection_code_sha256=sha(HERE/'bank.py'),config_sha256=sha(HERE/'config.json'))
    dump(OUT/'HOLDOUT_BANK.json',result);print('BANK FROZEN',passed,len(selected),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=32);p.add_argument('--generation',type=int,default=300);args=p.parse_args();main(args.workers,args.generation)

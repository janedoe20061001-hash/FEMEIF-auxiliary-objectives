"""Freeze existing-parameter pairs by old objectives and structural labels only."""
from scan import *
import scan
from analyze import csvout

def relative(a,b):
    floor=np.array(CFG['audit']['base_atol'])
    return abs(a[:,None,:]-b[None,:,:])/np.maximum(np.maximum(a[:,None,:],b[None,:,:]),floor)

def main():
    inv=np.load(OUT/'INVENTORY.npz');res=np.load(OUT/'STRUCTURE_RESULTS.npz');X=inv['X'];data=target('case1');events=data['record']['events'];ti=2
    initialize();w=scan._worker;good=res['complete'][:,ti]&res['stable'][:,ti]
    centered=good&(abs(res['values'][:,ti,0])<=events[ti]['width_3db_decades'])
    q=np.max(np.stack([abs(res['values'][:,ti,0])/.02,abs(res['values'][:,ti,1])/3,abs(np.log(res['values'][:,ti,2]))/np.log(2)]),axis=0)
    q[(res['evaluated_full'])&(~res['matched'][:,ti])]=np.inf
    lower=(events[ti]['height_db']-20*np.log10(np.maximum(res['amplitude_upper_bound'],1e-300)))/3
    q[~res['evaluated_full']]=lower[~res['evaluated_full']]
    robust_bad=res['stable'][:,ti]&(~res['complete'][:,ti])&(q>=2)
    full_cache={};pairs=[]
    def full(uid):
        uid=int(uid)
        if uid not in full_cache:
            w['force'].add(uid);r=evaluate(uid);p=10**X[uid]
            xx=w['q'].reference_grid(p,int(r['level']));z,_=w['c'].response(10**xx,p)
            fit=diag.events(xx,z,w['cfg']);rows,extra=diag.match(events,fit,w['cfg'])
            full_cache[uid]=dict(uid=uid,log_parameters=X[uid].tolist(),events=rows,extra_ids=extra,fit=fit,stable=r['stable'].tolist())
        return full_cache[uid]
    def attach(mi,si,gi,a,b,region,family):
        u=inv['lookup'][mi,si,gi];ua,ub=int(u[a]),int(u[b]);fa,fb=full(ua),full(ub)
        if ua==ub:return
        pairs.append(dict(pair_id=f'pair_{len(pairs):03}',family=family,region=region,method=METHODS[mi],seed=SEEDS[si],generation=STAGES[gi],
            better_index=int(a),worse_index=int(b),better_uid=ua,worse_uid=ub,
            better_primary=inv['primary'][mi,si,gi,a].tolist(),worse_primary=inv['primary'][mi,si,gi,b].tolist(),
            better=fa,worse=fb,scope='Local structural label only; other structures may deteriorate; not a replacement final selection'))
    def ordered_pairs(F,aa,bb):
        if not len(aa) or not len(bb):return []
        v=relative(F[aa],F[bb]);rr,cc=np.nonzero(np.all(v<=.05,axis=2));items=[]
        for r,c in zip(rr,cc):items.append((float(max(v[r,c])),float(sum(v[r,c])),int(aa[r]),int(bb[c])))
        return sorted(items)
    # Latest same-population near-MSE pair; no new metric used.
    for mi in range(6):
        for si in range(12):
            for gi in range(15,0,-1):
                u=inv['lookup'][mi,si,gi];F=inv['primary'][mi,si,gi]
                options=ordered_pairs(F,np.flatnonzero(centered[u]),np.flatnonzero(robust_bad[u]))
                if options:
                    _,_,a,b=options[0];attach(mi,si,gi,a,b,'P3','P3_missing_or_distorted');break
    # Previously published diagnostic pairs, independent of new auxiliary scores.
    old=json.loads((OLD/'SELECTOR_STRUCTURE_DIAGNOSIS.json').read_text())['records']
    for r in old:
        si=SEEDS.index(r['seed']);a=r['diagnostic_candidate_index'];b=r['original_selected_index'];F=inv['primary'][0,si,-1]
        if not np.all(relative(F[[a]],F[[b]])[0,0]<=.05):continue
        fa=full(inv['lookup'][0,si,-1,a]);fb=full(inv['lookup'][0,si,-1,b])
        for region in ('P1','P2'):
            j=next(j for j,e in enumerate(events) if e['event_id']==region);ea,eb=fa['events'][j],fb['events'][j]
            if not (ea['matched'] and eb['matched'] and fa['stable'][j] and fb['stable'][j]):continue
            if ea['width_ratio'] is None or eb['width_ratio'] is None:continue
            if abs(eb['height_delta_db'])-abs(ea['height_delta_db'])>=3 and abs(np.log(eb['width_ratio']))-abs(np.log(ea['width_ratio']))>=np.log(1.5) and abs(ea['shift_decades'])<=events[j]['width_3db_decades']:
                attach(0,si,15,a,b,region,'existing_peak_height_width')
    # Extra-event candidate pairs: evaluated structures only; geometry label checked before freezing.
    for mi in range(6):
        for si in range(12):
            found=False
            for gi in range(15,0,-1):
                u=inv['lookup'][mi,si,gi];F=inv['primary'][mi,si,gi]
                eligible=res['evaluated_full'][u]&res['stable'][u].all(axis=1)
                index=np.flatnonzero(eligible)
                options=ordered_pairs(F,index,index)
                for _,_,a,b in options:
                    if res['extra'][u[b]]<=res['extra'][u[a]]:continue
                    if sum(res['complete'][u[a]])!=sum(res['complete'][u[b]]):continue
                    fa,fb=full(u[a]),full(u[b]);extra=[f for f in fb['fit'] if f['event_id'] in fb['extra_ids']]
                    for e in extra:
                        block=next((v for v in data['record']['blocks'] if not v['event'] and v['lo']<=e['x']<=v['hi']),None)
                        if block is None:continue
                        if any(block['lo']<=v['x']<=block['hi'] for v in fa['fit']):continue
                        attach(mi,si,gi,a,b,block['label'],'extra_structure');found=True;break
                    if found:break
                if found:break
    # Deduplicate same directional parameter pair / region, keep all selected source provenance.
    unique=[];seen=set()
    for p in pairs:
        key=(p['better_uid'],p['worse_uid'],p['region'])
        if key in seen:continue
        seen.add(key);p['pair_id']=f'pair_{len(unique):03}';unique.append(p)
    result=dict(scope='Development candidate pairs frozen before new metric scores',pair_protocol_sha256=sha(HERE/'PAIR_PROTOCOL.md'),
        source_inventory_sha256=sha(OUT/'INVENTORY.npz'),selection_code_sha256=sha(HERE/'pairs.py'),pairs=unique,
        entries=len(unique),unique_parameter_pairs=len({(p['better_uid'],p['worse_uid']) for p in unique}),seeds=sorted({p['seed'] for p in unique}),families=sorted({p['family'] for p in unique}),
        optimization_count_gate_minimum_pairs=12,optimization_count_gate_minimum_seeds=6,optimization_count_gate_minimum_families=3)
    dump(OUT/'FROZEN_PAIR_BANK.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='pairs'}),flush=True)
    for p in unique:print(p['pair_id'],p['family'],p['method'],p['seed'],p['generation'],p['region'],flush=True)

if __name__=='__main__':main()

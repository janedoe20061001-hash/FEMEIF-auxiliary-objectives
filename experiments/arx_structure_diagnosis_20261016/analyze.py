"""Aggregate read-only structure diagnostics, keeping unknowns and selector provenance."""
from scan import *
import csv

def csvout(name,rows):
    if not rows:return
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    recovered=OUT/'retrieved/output/arx_structure_diagnosis_20261016'
    rec=json.loads((OUT/'RECOVERY.json').read_text());assert rec['verified']
    inv=np.load(OUT/'INVENTORY.npz');X=inv['X'];n=len(X)
    source=json.loads((OUT/'INVENTORY.json').read_text());assert sha(OUT/'INVENTORY.npz')==source['inventory_sha256']
    arrays=dict(matched=np.zeros((n,11),bool),complete=np.zeros((n,11),bool),values=np.full((n,11,3),np.nan),stable=np.zeros((n,11),bool),extra=np.full(n,-1,np.int16),nfit=np.full(n,-1,np.int16),level=np.zeros(n,np.int8),evaluated_full=np.zeros(n,bool),amplitude_upper_bound=np.full(n,np.nan),certified_height_incomplete=np.zeros(n,bool))
    seen=np.zeros(n,bool);batch_hashes={}
    for path in sorted((recovered/'batches').glob('*.npz')):
        batch_hashes[path.name]=sha(path)
        with np.load(path) as batch:
            ids=batch['uid'];repeat=seen[ids]
            for key,value in arrays.items():
                if repeat.any():np.testing.assert_array_equal(value[ids[repeat]],batch[key][repeat])
                value[ids]=batch[key]
            seen[ids]=True
    assert seen.all(),int(sum(~seen))
    np.savez_compressed(OUT/'STRUCTURE_RESULTS.npz',**arrays)
    ti=2;good=arrays['stable'][:,ti]&arrays['complete'][:,ti];known=arrays['stable'][:,ti]
    target_event=next(e for e in target('case1')['record']['events'] if e['event_id']=='P3')
    centered=good&(abs(arrays['values'][:,ti,0])<=target_event['width_3db_decades'])
    records=[];stages=[];audit=[]
    for mi,m in enumerate(METHODS):
        for si,s in enumerate(SEEDS):
            row=dict(method=m,seed=s)
            presence=[];centered_presence=[]
            for gi,gen in enumerate(STAGES):
                u=inv['lookup'][mi,si,gi];a=good[u];b=centered[u];k=known[u]
                stages.append(dict(method=m,seed=s,generation=gen,records=len(u),unique_parameters=len(set(u.tolist())),P3_complete_records=int(a.sum()),P3_centered_records=int(b.sum()),P3_uncertain_records=int((~k).sum()),selected_P3_complete=None if gen==0 else bool(a[inv['selected'][mi,si,gi]])))
                presence.append(bool(a.any()));centered_presence.append(bool(b.any()))
            final_u=inv['lookup'][mi,si,-1];F=inv['primary'][mi,si,-1];idx=int(inv['selected'][mi,si,-1]);_,sel=knee(F)
            front=np.zeros(200,bool);front[sel['front_indices']]=True;near=np.all(F<=1.1*F[idx]+1e-14,axis=1)
            gg=good[final_u];cc=centered[final_u]
            old=json.loads((DATA/'jobs/case1'/m/f'seed_{s}/audit.json').read_text());expected=next(e for e in old['amplitude_events'] if e['event_id']=='P3')['complete']
            assert known[final_u[idx]] and bool(gg[idx])==expected,(m,s,'original_audit_mismatch')
            audit.append(dict(method=m,seed=s,original=expected,replayed=bool(gg[idx]),passed=True))
            row.update(final_P3_complete_records=int(gg.sum()),final_P3_centered_records=int(cc.sum()),final_P3_uncertain_records=int((~known[final_u]).sum()),
                final_P3_on_front=int((gg&front).sum()),final_P3_within_10pct=int((gg&near).sum()),selected_P3_complete=bool(gg[idx]),selected_P3_centered=bool(cc[idx]),
                initial_any_P3=presence[0],saved_any_P3=any(presence[1:]),saved_any_centered=any(centered_presence[1:]),initial_any_centered=centered_presence[0],
                any_P3_before_final=any(presence[:-1]),P3_seen_then_absent_final=any(presence[:-1]) and not presence[-1],
                first_saved_P3_generation=next((g for g,p in zip(STAGES,presence) if p),None),last_saved_P3_generation=next((g for g,p in zip(STAGES[::-1],presence[::-1]) if p),None))
            records.append(row)
    csvout('RUN_SUMMARY.csv',records);csvout('SAVED_STAGE_COUNTS.csv',stages);csvout('ORIGINAL_FINAL_AUDIT_REPLAY.csv',audit)
    groups=[]
    for m in METHODS:
        rr=[r for r in records if r['method']==m]
        groups.append(dict(method=m,runs=12,final_complete_records=sum(r['final_P3_complete_records'] for r in rr),final_centered_records=sum(r['final_P3_centered_records'] for r in rr),
            runs_final_P3=sum(r['final_P3_complete_records']>0 for r in rr),runs_front_P3=sum(r['final_P3_on_front']>0 for r in rr),runs_near_P3=sum(r['final_P3_within_10pct']>0 for r in rr),
            selected_complete=sum(r['selected_P3_complete'] for r in rr),selected_centered=sum(r['selected_P3_centered'] for r in rr),initial_any=sum(r['initial_any_P3'] for r in rr),
            saved_any=sum(r['saved_any_P3'] for r in rr),saved_any_centered=sum(r['saved_any_centered'] for r in rr),seen_then_absent=sum(r['P3_seen_then_absent_final'] for r in rr),
            uncertain_final=sum(r['final_P3_uncertain_records'] for r in rr)))
    csvout('METHOD_SUMMARY.csv',groups)
    coverage=dict(unique_candidates=n,all_records=int(inv['lookup'].size),final_records=14400,
        full_band_evaluations=int(arrays['evaluated_full'].sum()),analytic_negative_certificates=int((~arrays['evaluated_full']).sum()),
        full_evaluated_and_certificate_negative=int((arrays['evaluated_full']&arrays['certified_height_incomplete']).sum()),
        P3_known=int(known.sum()),P3_uncertain=int((~known).sum()),P3_complete_unique=int(good.sum()),P3_centered_unique=int(centered.sum()),
        original_final_audits_replayed=len(audit),batch_hashes=batch_hashes,all_uid_covered=True)
    dump(OUT/'SCAN_CHECKS.json',coverage)
    print(json.dumps({k:v for k,v in coverage.items() if k!='batch_hashes'}),flush=True)
    for g in groups:print(json.dumps(g),flush=True)

if __name__=='__main__':main()

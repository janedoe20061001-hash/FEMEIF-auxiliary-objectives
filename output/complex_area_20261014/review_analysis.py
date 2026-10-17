"""Read-only analysis of verified final populations and saved knee trajectories."""
from pathlib import Path
import os,sys,json,csv,time,hashlib
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
WORK=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
SNAP=OUT/'review_inputs';CODE=SNAP/'experiments/complex_area_20261014'
sys.path.insert(0,str(CODE))
import numpy as np
from core import target,CFG,target_ids,sha,knee
from features import Features,mode_points
from concurrent.futures import ProcessPoolExecutor,as_completed
DATA=SNAP/'output/complex_area_20261014';AN=OUT/'analysis';AN.mkdir(exist_ok=True)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda a:a.tolist()),encoding='utf-8')
def csvout(name,rows):
    if not rows:return
    with (AN/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def band_metric(data):
    g=Features(data);ident=data['id']
    hz=(45000.,58000.) if ident.endswith('/base') else (38500.,45000.) if ident!='case1' else (10**g.lo,10**g.hi)
    x=np.log10(hz);xx=np.unique(np.r_[np.linspace(*x,8193),mode_points(g.target_modes,*x),[np.log10(f) for f,_ in g.target_modes if hz[0]<=f<=hz[1]]])
    ref,_=g.c.response(10**xx);scale=float(max(abs(ref)));w=dict(x=x,scale=scale)
    def metric(lp,n=16,refinement=0):
        p=10**lp
        return g.block(p,w,g.modes(p),n,refinement,('J0',))['J0']
    return g,metric,dict(band_hz=hz,scale_ohm=scale,measure='uniform log10(f), fixed target-only scale')

def one_job(job):
    ident,method,seed=job;d=DATA/'jobs'/ident/method/f'seed_{seed}';a=read(d/'audit.json');s=read(d/'summary.json')
    data=target(ident);g,metric,scope=band_metric(data)
    with np.load(d/'generation_0300.npz') as p:
        X=p['X'];F=p['F'];index,sel=knee(F[:,:2]);assert index==int(p['selected_index'])
    local=np.array([metric(x) for x in X]);fine=metric(X[index],32,1)
    assert abs(local[index]-fine)<=1e-9+1e-4*abs(fine),(ident,method,seed,local[index],fine)
    # Spot-check deterministic population indices independently of the final selector.
    errors=[]
    for i in (0,49,99,149,199):
        v=metric(X[i],32,1);errors.append(abs(local[i]-v)/max(abs(v),1e-12))
        assert abs(local[i]-v)<=1e-9+1e-4*abs(v),(ident,method,seed,i,local[i],v)
    threshold=F[index,:2]*1.10+1e-14;near=np.all(F[:,:2]<=threshold,axis=1)
    minimum=float(min(local[near]));allmin=float(min(local))
    focus=[e for e in a['amplitude_events'] if ident=='case1' or e['event_id'] in ('P2','V3')]
    row=dict(target=ident,method=method,seed=seed,accepted=a['accepted'],mag=a['objective'][0],pha=a['objective'][1],
        local_complex_error=fine,focused_complete=sum(e['complete'] for e in focus),focused_total=len(focus),
        all_complete=sum(e['complete'] for e in a['amplitude_events']),all_total=len(a['amplitude_events']),
        extra_amplitude=len(a['extra_amplitude_events']),phase_complete=sum(e['complete'] for e in a['phase_events']),phase_total=len(a['phase_events']),
        extra_phase=len(a['extra_phase_events']),cpu_seconds=s['cpu_seconds'],seconds=s['seconds'],
        base_queries=s['base_frequency_queries'],auxiliary_queries=s['auxiliary_frequency_queries'],
        quadrature_refined=s['quadrature_refined'],quadrature_max_order=s['quadrature_max_order'],selection_reason=sel['reason'])
    population=dict(target=ident,method=method,seed=seed,selected_local=fine,min_any=allmin,min_within_10pct=minimum,
        any_half_better=allmin<.5*fine,within10pct_half_better=minimum<.5*fine,near_candidates=int(sum(near)),
        max_spotcheck_relative_error=max(errors))
    tag=ident.replace('/','_')+'_'+method+'_'+str(seed)
    np.savez_compressed(AN/'populations'/f'{tag}.npz',X=X,F=F,local_complex_error=local,selected_index=index)
    stages=[]
    with np.load(d/'review_stages.npz') as p:
        assert p['generations'].tolist()==list(range(20,301,20))
        for gen,x,f,ff,idx in zip(p['generations'],p['X_selected'],p['F_selected'],p['F_population'],p['selected_indices']):
            k,_=knee(ff[:,:2]);assert k==idx and np.array_equal(ff[k],f)
            if seed in (46000,46004):
                stages.append(dict(target=ident,method=method,seed=seed,generation=int(gen),mag=f[0],pha=f[1],local_complex_error=metric(x)))
    events=[dict(target=ident,method=method,seed=seed,accepted=a['accepted'],**e) for e in a['amplitude_events']]
    contribution=[dict(target=ident,method=method,seed=seed,**v) for v in a['local']]
    return row,population,stages,events,contribution,scope

def main():
    assert read(OUT/'REVIEW_INPUTS_RECOVERY.json')['verified']
    manifest=read(DATA/'MANIFEST.json')
    for name,h in manifest['hashes'].items():assert sha(SNAP/name)==h,('source',name)
    jobs=[(t,m,s) for t in target_ids() for m in CFG['methods'] for s in CFG['seeds']]
    initial={};shapes=0
    for t,m,s in jobs:
        d=DATA/'jobs'/t/m/f'seed_{s}';summary=read(d/'summary.json');data=target(t);x=np.load(d/'initial.npy')
        assert summary['evaluations']==60000 and summary['generations_saved']==list(range(20,301,20))
        np.testing.assert_array_equal(x,np.random.default_rng(s).uniform(data['lower'],data['upper'],size=x.shape))
        key=(t,s);assert initial.setdefault(key,summary['initial_sha256'])==summary['initial_sha256']
        for name in ('initial.npy','generation_0300.npz','history.json','selection_0300.json'):
            assert sha(d/name)==summary['files_sha256'][name],(t,m,s,name)
        assert [v['evaluations'] for v in read(d/'history.json')]==list(range(200,60001,200))
        with np.load(d/'generation_0300.npz') as p:
            assert p['X'].shape==x.shape and p['F'].shape==(200,2 if m.startswith('MAG_PHA') else 3);shapes+=1
    (AN/'populations').mkdir(exist_ok=True);rows=[];pop=[];stages=[];events=[];contrib=[];scopes={}
    with ProcessPoolExecutor(max_workers=min(4,os.cpu_count() or 1)) as pool:
        fs=[pool.submit(one_job,j) for j in jobs]
        for f in as_completed(fs):
            r,p,ss,ee,cc,scope=f.result();rows.append(r);pop.append(p);stages.extend(ss);events.extend(ee);contrib.extend(cc);scopes[r['target']]=scope
            if len(rows)%8==0:print('reviewed jobs',len(rows),'/',len(jobs),flush=True)
    rows.sort(key=lambda r:(r['target'],r['method'],r['seed']))
    csvout('REVIEWED_SEEDS.csv',rows);csvout('POPULATION_OPPORTUNITY.csv',pop);csvout('SELECTED_STAGE_TRAJECTORY.csv',stages);csvout('REVIEWED_EVENTS.csv',events);csvout('REVIEWED_CONTRIBUTIONS.csv',contrib)
    dump(AN/'METRIC_SCOPES.json',scopes)
    lookup={(r['target'],r['method'],r['seed']):r for r in rows};pairs=[];groups=[];rng=np.random.default_rng(20261014)
    for ident in target_ids():
        for method in CFG['methods']:
            rr=[lookup[ident,method,s] for s in CFG['seeds']];bb=[lookup[ident,'MAG_PHA',s] for s in CFG['seeds']]
            dif=np.array([r['local_complex_error']-b['local_complex_error'] for r,b in zip(rr,bb)])
            boot=np.mean(dif[rng.integers(0,len(dif),(32768,len(dif)))],axis=1)
            joint={q:0 for q in (1.,1.10,1.25)};valid=0
            for r,b in zip(rr,bb):
                eligible=r['accepted'] and b['accepted'];valid+=eligible
                for q in joint:joint[q]+=int(eligible and r['local_complex_error']<b['local_complex_error'] and r['mag']<=q*b['mag'] and r['pha']<=q*b['pha'] and r['extra_amplitude']<=b['extra_amplitude'])
                pairs.append(dict(target=ident,method=method,seed=r['seed'],both_accepted=eligible,local_delta=r['local_complex_error']-b['local_complex_error'],
                    local_ratio=r['local_complex_error']/max(b['local_complex_error'],1e-30),mag_ratio=r['mag']/max(b['mag'],1e-30),pha_ratio=r['pha']/max(b['pha'],1e-30),
                    complete_delta=r['focused_complete']-b['focused_complete'],extra_delta=r['extra_amplitude']-b['extra_amplitude']))
            groups.append(dict(target=ident,method=method,accepted=sum(r['accepted'] for r in rr),median_local=float(np.median([r['local_complex_error'] for r in rr])),
                mean_local_delta=float(dif.mean()),paired_mean_ci_lo=float(np.quantile(boot,.025)),paired_mean_ci_hi=float(np.quantile(boot,.975)),
                median_mag=float(np.median([r['mag'] for r in rr])),median_pha=float(np.median([r['pha'] for r in rr])),median_complete=float(np.median([r['focused_complete'] for r in rr])),
                focused_complete_total=sum(r['focused_complete'] for r in rr),focused_possible_total=sum(r['focused_total'] for r in rr),
                local_better_all8=int(sum(dif<0)),valid_pairs=valid,joint_100=joint[1.],joint_110=joint[1.10],joint_125=joint[1.25],
                total_cpu_hours=sum(r['cpu_seconds'] for r in rr)/3600,cpu_ratio_to_base=sum(r['cpu_seconds'] for r in rr)/sum(b['cpu_seconds'] for b in bb),
                extras_total=sum(r['extra_amplitude'] for r in rr)))
    csvout('REVIEWED_PAIRS.csv',pairs);csvout('REVIEWED_GROUPS.csv',groups)
    dump(OUT/'LOCAL_ANALYSIS_CHECKS.json',dict(compact_source_hashes_verified=True,initial_and_final_file_hashes_verified=True,
        paired_initializations_verified=True,budget_history_verified=True,saved_stage_selections_verified=True,final_populations_checked=shapes,
        final_candidates_local_loss_checked=len(rows)*200,selected_stage_local_losses_checked=len(stages),accepted=sum(r['accepted'] for r in rows),
        complete_archive_verification_required=True,report_manual_review_complete=False,updated_unix=time.time()))
    print('COMPACT ANALYSIS COMPLETE',flush=True)

if __name__=='__main__':main()

"""Audit 100 seeds/Case; summarize structure metrics without averaging curves."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,csv,time
import numpy as np
from analyze import Model,ind,diagnose,improved_dense_grid,HERE,ROOT,OUT,COLS,dump,csvout,table,n,link,write

def job(j):
    c,s=j;m=Model(c);r=m.record;runs=HERE/'server_run/retrieved/runs';d=runs/c/f'seed_{s}';old=ROOT/'experiments/optimized_curve_structure_20261007/server_run/retrieved/runs'/c/f'seed_{s}'
    np.testing.assert_array_equal(np.load(d/'initial_log_parameters.npy'),np.load(old/'initial_log_parameters.npy'));rows=[];ds=[];bs=[];checked=0;stageparams=[]
    history=json.loads((d/'history.json').read_text());assert len(history)==300
    for g in range(20,301,20):
        with np.load(d/f'generation_{g:03}.npz') as pop:
            x=pop['log_parameters'];assert np.isfinite(x).all() and np.all(x>=m.lower) and np.all(x<=m.upper)
            v=ind.independent_metrics(ind.independent_spectra(c,m.f,10**x),m.target,m.alpha);np.testing.assert_allclose(v,pop['metrics'],rtol=1e-8,atol=1e-8);np.testing.assert_allclose(v[:,:2],pop['optimization_objective'],rtol=1e-8,atol=1e-8)
            F=pop['optimization_objective'];dom=np.all(F[:,None,:]<=F[None,:,:],axis=2)&np.any(F[:,None,:]<F[None,:,:],axis=2);front=np.flatnonzero(~dom.any(axis=0));np.testing.assert_array_equal(front,np.sort(pop['front_indices']))
            ix=int(pop['selected_index']);bx=int(pop['balanced_index']);exact=pop['metrics'];assert ix==int(front[np.argmin(exact[front,2])]);assert bx==int(front[np.argmin(np.maximum(exact[front,0],m.alpha*exact[front,1]))])
            np.testing.assert_allclose(v[ix],history[g-1]['metrics'],rtol=1e-8,atol=1e-8);assert history[g-1]['evaluations']==g*r['population']
            if g==300:
                with np.load(old/'final_population.npz') as op:np.testing.assert_array_equal(x,op['log_parameters'])
            checked+=len(x);parameters=x[[ix,bx]];stageparams.append(parameters)
            for k,rule in enumerate(['min_wsmse','balanced_minmax']):rows.append(dict(circuit=c,seed=s,generation=g,rule=rule,**dict(zip(COLS,v[[ix,bx][k]].tolist()))))
        prev=None;stable=np.ones(2,dtype=bool)
        for ref in ([1,2,4,8] if c=='case1' else [0]):
            f=improved_dense_grid(r,10**parameters,ref) if c=='case1' else m.f;target=ind.independent_spectra(c,f,r['target_parameters'])[0] if c=='case1' else m.target;z=ind.independent_spectra(c,f,10**parameters);cur=[diagnose(f,target,zi) for zi in z]
            if prev:
                stable=np.array([all(a[k]==b[k] for k in ['amp_missed','amp_extra','phase_missed','phase_extra']) and all(abs(a[k]-b[k])/max(abs(b[k]),floor)<=.01 for k,floor in [('amplitude_rmse_db',.1),('phase_rmse_deg',.5),('mean_distance_logweighted_ohm',1e-8)]) for (a,_,_),(b,_,_) in zip(prev,cur)])
                if stable.all():break
            prev=cur
        for k,(diag,bands,events) in enumerate(cur):
            base=dict(circuit=c,seed=s,generation=g,rule=['min_wsmse','balanced_minmax'][k],grid_stable=bool(stable[k]),grid_refinement=ref)
            ds.append(dict(**base,**diag));bs.extend(dict(**base,**b) for b in bands)
    return dict(circuit=c,seed=s,checked=checked,metrics=rows,diagnostics=ds,bands=bs,parameters=np.asarray(stageparams).tolist())

def main():
    OUT.mkdir(exist_ok=True,parents=True);cache=OUT/'background_chunks';cache.mkdir(exist_ok=True);start=time.time();done=[]
    jobs=[(c,s) for c in ['case1','case2'] for s in range(2300,2400)];pending=[]
    for c,s in jobs:
        p=cache/f'{c}_{s}.json'
        if p.exists():done.append(json.loads(p.read_text()))
        else:pending.append((c,s))
    with ProcessPoolExecutor(max_workers=int(os.environ.get('STRUCTURE_ANALYSIS_WORKERS','6'))) as pool:
        for future in as_completed([pool.submit(job,j) for j in pending]):
            r=future.result();dump(cache/f'{r["circuit"]}_{r["seed"]}.json',r);done.append(r);dump(OUT/'BACKGROUND_STATUS.json',dict(stage='diagnostics',completed=len(done),total=200,seconds=time.time()-start));print(f'Background verified {len(done)}/200 seeds',flush=True)
    metrics=[row for r in done for row in r['metrics']];diagnostics=[row for r in done for row in r['diagnostics']];bands=[row for r in done for row in r['bands']]
    for rows in [metrics,diagnostics,bands]:rows.sort(key=lambda r:(r['circuit'],r['generation'],r['seed'],r['rule']))
    csvout(OUT/'all_seed_metrics.csv',metrics);csvout(OUT/'all_seed_diagnostics.csv',diagnostics);csvout(OUT/'all_seed_bands.csv',bands)
    summaries=[];keys=['amplitude_rmse_db','phase_rmse_deg','amp_missed','amp_extra','phase_missed','phase_extra','amp_mean_abs_shift_decades','amp_mean_abs_width_log2_ratio','worst_band_phase_rmse_deg']
    for c in ['case1','case2']:
        for g in range(20,301,20):
            for rule in ['min_wsmse','balanced_minmax']:
                rr=[r for r in diagnostics if (r['circuit'],r['generation'],r['rule'])==(c,g,rule)];assert len(rr)==100
                row=dict(circuit=c,generation=g,rule=rule,seeds=100,stable=sum(r['grid_stable'] for r in rr),amp_miss_seeds=sum(r['amp_missed']>0 for r in rr),phase_miss_seeds=sum(r['phase_missed']>0 for r in rr))
                for k in keys:
                    a=[r[k] for r in rr if r[k] is not None];row[k]=float(np.mean(a)) if a else None
                summaries.append(row)
    csvout(OUT/'all_seed_stage_summary.csv',summaries)
    # Paired late-stage changes; no curve average is computed.
    rng=np.random.default_rng(2026100801);draws=rng.integers(0,100,(10000,100));pairs=[]
    for c in ['case1','case2']:
        for rule in ['min_wsmse','balanced_minmax']:
            lookup={(r['seed'],r['generation']):r for r in diagnostics if r['circuit']==c and r['rule']==rule}
            for k in ['amplitude_rmse_db','phase_rmse_deg','amp_missed','phase_missed','worst_band_phase_rmse_deg']:
                a=np.array([lookup[(s,300)][k]-lookup[(s,200)][k] for s in range(2300,2400)]);ci=np.quantile(a[draws].mean(axis=1),[.025,.975]);pairs.append(dict(circuit=c,rule=rule,metric=k,mean_300_minus_200=float(a.mean()),ci95=ci.tolist(),numerically_stable=all(lookup[(s,g)]['grid_stable'] for s in range(2300,2400) for g in [200,300])))
    unstable=sum(not r['grid_stable'] for r in diagnostics);checked=sum(r['checked'] for r in done);assert checked==525000
    dump(OUT/'BACKGROUND_VALIDATION.json',dict(passed=True,runs=200,seeds_per_case=100,stages=3000,independent_individuals_recomputed=checked,final_populations_exactly_equal=True,diagnostic_rows=6000,unstable_rows=unstable,paired_changes=pairs,stage_summary=summaries))
    parts=['# 全部100seeds：结构随代数的变化','后台每个seed每20代选一条代表解，然后汇总诊断数值；没有计算平均阻抗曲线。两种选解都保留。',f'3000个seed阶段、525,000个阶段个体独立核验；6000条主要/均衡诊断，其中{unstable}条未通过网格稳定检查。Case2仅测点诊断；Case1解析连续诊断。']
    for c in ['case1','case2']:
        rr=[r for r in summaries if r['circuit']==c and r['rule']=='min_wsmse'];parts.extend([f'## {c}',table(['代','幅值RMSE/dB','相位RMSE/°','幅值漏失/新增均数','相位漏失/新增均数','幅值漏失seed数','稳定seed数'],[[r['generation'],n(r['amplitude_rmse_db']),n(r['phase_rmse_deg']),f"{n(r['amp_missed'])}/{n(r['amp_extra'])}",f"{n(r['phase_missed'])}/{n(r['phase_extra'])}",r['amp_miss_seeds'],r['stable']] for r in rr])])
    parts.extend(['## 200→300代配对变化',table(['Case','规则','诊断','300−200','95%区间','数值稳定'],[[r['circuit'],r['rule'],r['metric'],n(r['mean_300_minus_200']),f"[{n(r['ci95'][0])}, {n(r['ci95'][1])}]",r['numerically_stable']] for r in pairs]),'区间以配对seed重采样10000次，RNG=2026100801，未作多重比较校正。负值表示改善；不稳定项不能支持精确的连续诊断结论。末代残差不能证明双目标不存在零误差解。',f'[逐seed数据]({link(OUT/"all_seed_diagnostics.csv")})；[所有频带]({link(OUT/"all_seed_bands.csv")})；[均衡规则阶段摘要]({link(OUT/"all_seed_stage_summary.csv")})；[完整核验]({link(OUT/"BACKGROUND_VALIDATION.json")})',f'[返回图文报告]({link(OUT/"REPORT.md")})']);write(OUT/'ALL_SEEDS.md',parts)
    dump(OUT/'BACKGROUND_STATUS.json',dict(stage='complete',completed=200,total=200,seconds=time.time()-start));print(json.dumps(dict(complete=True,checked=checked,unstable=unstable,seconds=time.time()-start)))
if __name__=='__main__':main()

"""Independent full-final-population checks and paired seed analysis."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
GROUPS=['baseline','old_pair','signed_phase_pair']
LABELS=['Baseline','I-03 + II-02','I-03 + D_phi']
COLORS=['#64748b','#0284c7','#dc2626']
OLD=('reactance_crossing_direction_consistency','transition_speed_relative_error')
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m; spec.loader.exec_module(m);return m
def csvout(p,rows):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def evaluator(case):
    legacy=module('analysis_legacy_'+case,HERE/'frozen'/case/'legacy_objectives.py')
    cfg=json.loads((HERE/'frozen'/case/'config.json').read_text())
    if case=='case1':
        f=np.geomspace(100,1e6,180);network=legacy.impedance;target=network(f,legacy.TARGET)[0]
        lower,upper=legacy.LOWER,legacy.UPPER
    else:
        n=module('analysis_network_case2',HERE/'frozen'/case/'case2.py')
        f,target=n.load_target(cfg['target_data']);network=n.impedance;lower,upper=n.LOWER,n.UPPER
    old=legacy.Objectives(f,aux=OLD,target_z=target)
    phase=np.angle(target);h=np.diff(np.log10(f));delta=np.diff(phase)
    energy=np.sum(delta**2/h)
    u=(np.log10(f)-np.log10(f[0]))/np.log10(f[-1]/f[0]);bins=np.minimum((u*8).astype(int),7)
    def score(x):
        z=network(f,10**x);v=old.evaluate_spectra(z)
        # Algebraically independent expression from the online slope implementation.
        dphi=np.sqrt(np.sum((np.diff(np.angle(z),axis=1)-delta)**2/h,axis=1)/energy)
        pe=(np.angle(z)-phase)**2
        worst=np.sqrt(np.max(np.stack([pe[:,bins==j].mean(axis=1) for j in range(8)]),axis=0))
        np.testing.assert_allclose(v[:,0],np.mean((20*np.log10(abs(z))-20*np.log10(abs(target)))**2,axis=1),atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(v[:,1],pe.mean(axis=1),atol=1e-12,rtol=1e-10)
        return z,np.column_stack([v,dphi,worst])
    return cfg,f,target,lower,upper,score,delta**2/h/energy
def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    manifest=json.loads((a.runs/'manifest.json').read_text()); assert manifest['protocol_sha256']==sha(HERE/'PROTOCOL.md')
    for path,digest in manifest['frozen_sha256'].items():assert sha(HERE/path)==digest
    assert manifest['runner_sha256']==sha(HERE/'run.py')
    assert json.loads((a.runs/'status.json').read_text())['completed']==120
    rows=[];comparisons=[];means=[];cases={};checked=0;evaluations=0
    rng=np.random.default_rng(20261007)
    for case in ('case1','case2'):
        cfg,f,target,lo,hi,score,shares=evaluator(case); alpha=cfg['final_selection']['phase_weight']
        records=[]; curves=[]; parameters=[];hist=[]
        for seed in range(2200,2220):
            initials=[];seedcurves=[];seedpars=[];seedhist=[]
            for group in GROUPS:
                d=a.runs/case/group/f'seed_{seed}';r=json.loads((d/'summary.json').read_text());assert r['status']=='complete'
                assert (r['case'],r['group'],r['seed'])==(case,group,seed)
                assert r['evaluations']==cfg['population']*300 and r['generations']==300
                evaluations+=r['evaluations'];initial=np.load(d/'initial_log_parameters.npy');initials.append(initial)
                expected=np.random.default_rng(seed).uniform(lo,hi,(cfg['population'],len(lo)))
                np.testing.assert_array_equal(initial,expected)
                assert hashlib.sha256(initial.tobytes()).hexdigest()==r['initial_population_sha256']
                with np.load(d/'final_population.npz') as saved:
                    x=saved['log_parameters'];F=saved['objectives'];assert x.shape==(cfg['population'],len(lo))
                    assert np.isfinite(x).all() and np.all(x>=lo) and np.all(x<=hi)
                    z,v=score(x);checked+=len(x)
                    expectedF=v[:,:2] if group=='baseline' else v[:,[0,1,2,3 if group=='old_pair' else 4]]
                    np.testing.assert_allclose(F,expectedF,atol=1e-9,rtol=1e-9)
                    ix=int(np.argmin(F[:,0]+alpha*F[:,1]));assert ix==int(saved['selected_index'])==r['wsmse_selected_index']
                    np.testing.assert_allclose(saved['selected_primary_mse'],v[ix,:2],atol=1e-10,rtol=1e-9)
                    np.testing.assert_array_equal(saved['parameters'],10**x)
                    np.testing.assert_array_equal(saved['selected_log_parameters'],x[ix])
                    np.testing.assert_allclose([r['wsmse_selected_magnitude_mse'],r['wsmse_selected_phase_mse']],v[ix,:2],rtol=1e-9,atol=1e-10)
                    row=dict(case=case,seed=seed,group=group,magnitude_mse=float(v[ix,0]),phase_mse=float(v[ix,1]),
                        I03=float(v[ix,2]),II02=float(v[ix,3]),D_phi=float(v[ix,4]),worst_band_phase_rmse=float(v[ix,5]),
                        wsmse=float(v[ix,0]+alpha*v[ix,1]),elapsed_seconds=r['elapsed_seconds'])
                    rows.append(row);records.append(row);seedcurves.append(z[ix]);seedpars.append(x[ix])
                history=list(csv.DictReader((d/'history.csv').open(encoding='utf-8-sig')))
                assert len(history)==300
                assert [int(r['generation']) for r in history]==list(range(1,301))
                assert [int(r['evaluations']) for r in history]==[cfg['population']*g for g in range(1,301)]
                seedhist.append([[float(r['wsmse_selected_value']),float(r['wsmse_selected_magnitude_mse']),float(r['wsmse_selected_phase_mse'])] for r in history])
            for initial in initials[1:]: np.testing.assert_array_equal(initial,initials[0])
            curves.append(seedcurves);parameters.append(seedpars);hist.append(seedhist)
        vals={k:np.array([r[k] for r in records]).reshape(20,3) for k in ['magnitude_mse','phase_mse','wsmse','worst_band_phase_rmse','II02','D_phi']}
        cases[case]=dict(target_top5_phase_derivative_energy_share=float(np.sort(shares)[-5:].sum()),
            max_target_phase_step_deg=float(np.rad2deg(np.max(abs(np.diff(np.angle(target)))))),
            old_II02_lower_phase_worse_vs_baseline=int(np.sum((vals['II02'][:,1]<vals['II02'][:,0])&(vals['phase_mse'][:,1]>vals['phase_mse'][:,0]))),
            candidate_Dphi_lower_phase_worse_vs_baseline=int(np.sum((vals['D_phi'][:,2]<vals['D_phi'][:,0])&(vals['phase_mse'][:,2]>vals['phase_mse'][:,0]))))
        boot=rng.integers(0,20,(10000,20))
        for j,group in enumerate(GROUPS):
            means.append(dict(case=case,group=group,**{k:float(v[:,j].mean()) for k,v in vals.items()}))
        for treatment,reference in [(2,1),(2,0),(1,0)]:
            for metric in ['wsmse','magnitude_mse','phase_mse','worst_band_phase_rmse']:
                delta=vals[metric][:,treatment]-vals[metric][:,reference]
                comparisons.append(dict(case=case,treatment=GROUPS[treatment],reference=GROUPS[reference],metric=metric,
                    mean_difference=float(delta.mean()),bootstrap95=np.quantile(delta[boot].mean(axis=1),[.025,.975]).tolist(),
                    lower_error_seeds=int(np.sum(delta<0)),higher_error_seeds=int(np.sum(delta>0)),ties=int(np.sum(delta==0)),
                    relative_change_of_means_pct=float(100*delta.mean()/vals[metric][:,reference].mean())))
        np.savez_compressed(out/f'{case}_selected_curves.npz',frequency=f,target=target,curves=np.asarray(curves),
            log_parameters=np.asarray(parameters),groups=np.asarray(GROUPS),seeds=np.arange(2200,2220),history=np.asarray(hist))
        fig,ax=plt.subplots(1,3,figsize=(13,3.8))
        for axis,metric,label in zip(ax,['wsmse','phase_mse','worst_band_phase_rmse'],['WSMSE','Phase MSE (rad²)','Worst-band phase RMSE (rad)']):
            for line in vals[metric]:axis.plot(range(3),line,color='#cbd5e1',lw=.6,zorder=0)
            b=axis.boxplot([vals[metric][:,j] for j in range(3)],positions=range(3),patch_artist=True,tick_labels=['Base','Old pair','Candidate'])
            for patch,color in zip(b['boxes'],COLORS):patch.set_facecolor(color);patch.set_alpha(.35)
            axis.set_ylabel(label);axis.grid(axis='y',alpha=.2)
        fig.suptitle(f'{case}: 20 paired seeds, fixed final selection');fig.tight_layout();fig.savefig(out/'figures'/f'{case}_paired_metrics.png',dpi=170);plt.close(fig)
        fig,ax=plt.subplots(1,2,figsize=(11,3.7));history=np.asarray(hist)
        for j,(label,color) in enumerate(zip(LABELS,COLORS)):
            for axis,col in zip(ax,[0,2]):
                q=np.quantile(history[:,j,:,col],[.25,.5,.75],axis=0)
                axis.plot(np.arange(1,301),q[1],label=label,color=color);axis.fill_between(np.arange(1,301),q[0],q[2],color=color,alpha=.12)
        ax[0].set_ylabel('Selected WSMSE (median / IQR)');ax[1].set_ylabel('Selected phase MSE (rad²)')
        for axis in ax:axis.set_xlabel('Generation');axis.grid(alpha=.2);axis.legend(fontsize=8)
        fig.suptitle(case);fig.tight_layout();fig.savefig(out/'figures'/f'{case}_convergence.png',dpi=170);plt.close(fig)
        difference=vals['wsmse'][:,2]-vals['wsmse'][:,1]
        medianix=int(np.argsort(difference,kind='stable')[9]);regressix=int(np.argmax(difference))
        cases[case]['representative_seeds']=dict(lower_median_paired_difference=2200+medianix,largest_candidate_regression=2200+regressix)
        for kind,idx in [('median',medianix),('largest_regression',regressix)]:
            fig,ax=plt.subplots(2,1,figsize=(10,6),sharex=True)
            for axis,values in [(ax[0],20*np.log10(abs(target))),(ax[1],np.rad2deg(np.angle(target)))]:axis.semilogx(f,values,'k--',lw=1.5,label='Target')
            for j,(label,color) in enumerate(zip(LABELS,COLORS)):
                ax[0].semilogx(f,20*np.log10(abs(curves[idx][j])),label=label,color=color)
                ax[1].semilogx(f,np.rad2deg(np.angle(curves[idx][j])),label=label,color=color)
            ax[0].set_ylabel('Magnitude (dBΩ)');ax[1].set_ylabel('Phase (degree)');ax[1].set_xlabel('Frequency (Hz)')
            for axis in ax:axis.grid(alpha=.2);axis.legend(fontsize=8)
            fig.suptitle(f'{case}, seed {2200+idx}: {kind.replace("_"," ")}');fig.tight_layout();fig.savefig(out/'figures'/f'{case}_{kind}_curves.png',dpi=170);plt.close(fig)
    assert len(rows)==120 and checked==21000 and evaluations==6300000
    csvout(out/'selected_metrics.csv',rows);csvout(out/'group_means.csv',means)
    dump(out/'SUMMARY.json',dict(means=means,comparisons=comparisons,cases=cases,seed_count_per_group=20,bootstrap_replicates=10000))
    dump(out/'VALIDATION.json',dict(passed=True,complete_runs=120,final_individuals_recomputed=checked,total_evaluations=evaluations,
        paired_initial_populations=True,initial_seed_reproduction=True,parameter_bounds=True,fixed_selection=True,
        all_final_objectives_recomputed=True,history_generation_and_evaluation_counts=True,frozen_source_hashes=True,
        analysis_sha256=sha(__file__)))
    report=['# 相位斜率指标：服务器优化验证结果（2026-10-07）','',
        '两个 Case、三组对照、每组 20 个配对新种子，120 次运行全部完成。每次 300 代，总计 630 万次评价。最终按固定 WSMSE 选解，辅助值不参与最终选择。', '',
        '本次是在预筛未通过之后，按用户明确要求继续的独立优化验证。不能用优化结果追改预筛门槛。负的配对差表示候选改善；统计单位是种子。','']
    for case in ('case1','case2'):
        report+=['## '+case,'','| 组别 | 幅值 MSE（dB²） | 相位 MSE（rad²） | WSMSE | 最差频带相位 RMSE（rad） |','|---|---:|---:|---:|---:|']
        for j,group in enumerate(GROUPS):
            m=next(m for m in means if m['case']==case and m['group']==group)
            report.append(f"| {LABELS[j]} | {m['magnitude_mse']:.6g} | {m['phase_mse']:.6g} | {m['wsmse']:.6g} | {m['worst_band_phase_rmse']:.6g} |")
        report+=['','### 候选相对原 I-03＋II-02（同为四目标）','','| 量 | 配对均值差 | 95% bootstrap 区间 | 均值相对变化 | 改善种子数 |','|---|---:|---|---:|---:|']
        for metric,label in [('wsmse','WSMSE'),('magnitude_mse','幅值 MSE'),('phase_mse','相位 MSE'),('worst_band_phase_rmse','最差频带相位 RMSE')]:
            c=next(c for c in comparisons if c['case']==case and c['metric']==metric and c['treatment']=='signed_phase_pair' and c['reference']=='old_pair')
            low,high=c['bootstrap95'];report.append(f"| {label} | {c['mean_difference']:+.6g} | [{low:+.6g}, {high:+.6g}] | {c['relative_change_of_means_pct']:+.1f}% | {c['lower_error_seeds']}/20 |")
        report+=['',f"原组 II-02 下降但相位 MSE 比基线上升：{cases[case]['old_II02_lower_phase_worse_vs_baseline']}/20；候选组 D_phi 下降但相位 MSE 比基线上升：{cases[case]['candidate_Dphi_lower_phase_worse_vs_baseline']}/20。",'',
            f'![配对结果](figures/{case}_paired_metrics.png)',f'![进化历史](figures/{case}_convergence.png)',
            f'![按配对差中位数选择的曲线](figures/{case}_median_curves.png)',
            f'![候选 WSMSE 相对原组退步最大的曲线](figures/{case}_largest_regression_curves.png)','']
    report+=['## 如何解读','',
        '主要证据是候选与原组合的配对主误差，指标自身降低并不等于拟合改善。区间跨零时，20 个种子不足以分清方向；区间不跨零也只支持本电路、本预算下的探索结果。多个诊断量未作多重比较校正，不据此作普适胜出声明。', '',
        f"Case1 目标相位导数能量最强的 5 个区间占 {cases['case1']['target_top5_phase_derivative_energy_share']:.1%}；Case2 占 {cases['case2']['target_top5_phase_derivative_energy_share']:.1%}。Case2 最大相邻相位差 {cases['case2']['max_target_phase_step_deg']:.2f} 度，所有原始测量点均保留。候选对该局部跳变的重视程度是既知限制。",'',
        '这轮直接检验指标用于进化时的收益，尚未验证对 16 种自设计拓扑的推广，也未新增实测电路。下一步选择应依据两项主误差及局部频带误差的联合结果，不凭 D_phi 单项变小做替换。','',
        '核验：120 个完整记录，21,000 个末代个体全部独立重算；初始种群配对、种子复现、参数边界、选解、每代评价次数及冻结代码哈希均通过。', '',
        '[逐种子数据](selected_metrics.csv) · [组均值](group_means.csv) · [完整统计](SUMMARY.json) · [核验](VALIDATION.json)']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,runs=120,final_individuals=checked,evaluations=evaluations)))
if __name__=='__main__': main()

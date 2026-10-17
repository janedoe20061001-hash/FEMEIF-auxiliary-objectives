"""Independently audit snapshots; plot separate seeds, never averaged curves."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,csv,hashlib,shutil,importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'output/dual_trajectory_20261008'
sys.path.insert(0,str(HERE/'frozen_dual'))
from model import Model
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
ind=module('trajectory_independent',ROOT/'experiments/seed_replication_20261007/analyze.py')
sys.path.insert(0,str(ROOT/'experiments/optimized_curve_structure_20261007'))
from diagnostics import diagnose,weights
from refined_grid import improved_dense_grid
COLS=['magnitude_mse_db2','phase_mse_rad2','wsmse','mean_distance_ohm']
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def csvout(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def table(h,rows):return '\n'.join(['| '+' | '.join(h)+' |','| '+' | '.join(['---']*len(h))+' |',*['| '+' | '.join(str(v) for v in r)+' |' for r in rows]])
def n(v):return '—' if v is None else f'{v:.5g}'
def link(p):return p.resolve().as_posix()
def write(p,parts):p.write_text('\n\n'.join(parts)+'\n',encoding='utf8')

def main():
    cfg=json.loads((HERE/'CONFIG.json').read_text());runs=HERE/'server_run/retrieved/runs';status=json.loads((runs/'status.json').read_text());assert status['status']=='complete' and status['runs']==200
    for name,h in json.loads((HERE/'SOURCES.json').read_text()).items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==h,name
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'figures').mkdir(exist_ok=True);(OUT/'selected').mkdir(exist_ok=True);(OUT/'schematics').mkdir(exist_ok=True)
    seeds=cfg['seeds'];stages=cfg['stages'];rows=[];diags=[];bands=[];parameters=[];checked=0;unstable=0;case_summaries=[]
    colors=['#2674ad','#cf741c','#309661']
    for c in cfg['cases']:
        m=Model(c);r=m.record;pp=np.zeros((15,3,2,len(m.lower)));vv=np.zeros((15,3,2,4))
        for si,s in enumerate(seeds):
            d=runs/c/f'seed_{s}';old=ROOT/'experiments/optimized_curve_structure_20261007/server_run/retrieved/runs'/c/f'seed_{s}'
            np.testing.assert_array_equal(np.load(d/'initial_log_parameters.npy'),np.load(old/'initial_log_parameters.npy'))
            history=json.loads((d/'history.json').read_text());assert len(history)==300
            for ti,g in enumerate(stages):
                with np.load(d/f'generation_{g:03}.npz') as p:
                    x=p['log_parameters'];assert x.shape==(r['population'],len(m.lower));assert np.all(x>=m.lower) and np.all(x<=m.upper)
                    v=ind.independent_metrics(ind.independent_spectra(c,m.f,10**x),m.target,m.alpha)
                    np.testing.assert_allclose(v,p['metrics'],rtol=1e-8,atol=1e-8);np.testing.assert_allclose(v[:,:2],p['optimization_objective'],rtol=1e-8,atol=1e-8)
                    F=p['optimization_objective'];dom=np.all(F[:,None,:]<=F[None,:,:],axis=2)&np.any(F[:,None,:]<F[None,:,:],axis=2);front=np.flatnonzero(~dom.any(axis=0));np.testing.assert_array_equal(front,np.sort(p['front_indices']))
                    ix=int(p['selected_index']);bx=int(p['balanced_index']);stored=p['metrics'];assert ix==int(front[np.argmin(stored[front,2])]);assert bx==int(front[np.argmin(np.maximum(stored[front,0],m.alpha*stored[front,1]))])
                    for ki,k in enumerate([ix,bx]):
                        pp[ti,si,ki]=x[k];vv[ti,si,ki]=v[k];base=dict(circuit=c,seed=s,generation=g,rule=['min_wsmse','balanced_minmax'][ki]);rows.append(dict(**base,front_size=len(front),**dict(zip(COLS,v[k].tolist()))))
                        parameters.extend(dict(**base,parameter=name,unit={'R':'ohm','L':'H','C':'F'}[name[0]],value=float(10**x[k,j])) for j,name in enumerate(r['parameter_names']))
                    np.testing.assert_allclose(v[ix],history[g-1]['metrics'],rtol=1e-8,atol=1e-8);assert history[g-1]['evaluations']==g*r['population'];checked+=len(x)
                    if g==300:
                        with np.load(old/'final_population.npz') as op:np.testing.assert_array_equal(x,op['log_parameters'])
        # Only Case1 has analytic between-point truth. Case2 remains measured-point diagnostics.
        stageplots=[]
        for ti,g in enumerate(stages):
            flat=pp[ti].reshape(6,-1);previous=None;stable=np.ones(6,dtype=bool)
            for refinement in ([1,2,4,8] if c=='case1' else [0]):
                f=improved_dense_grid(r,10**flat,refinement) if c=='case1' else m.f
                target=ind.independent_spectra(c,f,r['target_parameters'])[0] if c=='case1' else m.target
                z=ind.independent_spectra(c,f,10**flat);current=[diagnose(f,target,curve) for curve in z]
                if previous:
                    stable=np.array([all(a[k]==b[k] for k in ['amp_missed','amp_extra','phase_missed','phase_extra']) and all(abs(a[k]-b[k])/max(abs(b[k]),floor)<=.01 for k,floor in [('amplitude_rmse_db',.1),('phase_rmse_deg',.5),('mean_distance_logweighted_ohm',1e-8)]) for (a,_,_),(b,_,_) in zip(previous,current)])
                    if stable.all():break
                previous=current
            unstable+=int((~stable).sum());stageplots.append((f,target,z.reshape(3,2,-1)))
            # Store original-point spectra plus exact parameters; analytic plot grids are reproducible.
            np.savez_compressed(OUT/'selected'/f'{c}_generation_{g:03}.npz',frequency=m.f,target=m.target,seeds=np.asarray(seeds),rules=np.asarray(['min_wsmse','balanced_minmax']),curves=ind.independent_spectra(c,m.f,10**flat).reshape(3,2,-1),log_parameters=pp[ti],training_metrics=vv[ti],grid_stable=stable.reshape(3,2),analytic_plot_refinement=refinement)
            for i,(diag,bb,events) in enumerate(current):
                si,ki=divmod(i,2);base=dict(circuit=c,seed=seeds[si],generation=g,rule=['min_wsmse','balanced_minmax'][ki],grid='analytic' if c=='case1' else 'measured',grid_stable=bool(stable[i]),grid_refinement=refinement,points=len(f))
                diags.append(dict(**base,**diag));bands.extend(dict(**base,**b) for b in bb)
            print(f'Diagnosed {c} generation {g}',flush=True)
        fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True,layout='constrained')
        for si,s in enumerate(seeds):
            for ki,style in enumerate(['-','--']):
                axes[0].plot(stages,vv[:,si,ki,0],style,color=colors[si],label=f'seed {s}, '+['min WSMSE','balanced'][ki]);axes[1].plot(stages,vv[:,si,ki,1],style,color=colors[si])
        for ax in axes:ax.set_yscale('log');ax.grid(True,alpha=.2)
        axes[0].set_ylabel('MAG MSE (dB^2)');axes[1].set_ylabel('PHA MSE (rad^2)');axes[1].set_xlabel('Generation');axes[0].legend(fontsize=8,ncol=2);fig.suptitle(c+': separate seed trajectories; solid/display vs dashed/balanced')
        fig.savefig(OUT/'figures'/f'{c}_metrics.png',dpi=150);plt.close(fig)
        for page in range(5):
            fig,axes=plt.subplots(3,2,figsize=(13,9.8),layout='constrained')
            for rr,ti in enumerate(range(page*3,page*3+3)):
                f,t,z=stageplots[ti]
                for si,s in enumerate(seeds):
                    axes[rr,0].semilogx(f,20*np.log10(abs(z[si,0])),color=colors[si],lw=1.0,label=f'seed {s}');axes[rr,1].semilogx(f,np.rad2deg(np.angle(z[si,0])),color=colors[si],lw=1.0)
                axes[rr,0].semilogx(f,20*np.log10(abs(t)),'k--',lw=1.3,label='Target');axes[rr,1].semilogx(f,np.rad2deg(np.angle(t)),'k--',lw=1.3)
                axes[rr,0].set_ylabel('Magnitude (dB re 1 ohm)');axes[rr,1].set_ylabel('Phase (degree)')
                for ax in axes[rr]:ax.grid(True,alpha=.2);ax.set_xlim(f[0],f[-1]);ax.set_title(f'Generation {stages[ti]}');ax.set_xlabel('Frequency (Hz)')
            axes[0,0].legend(fontsize=9);fig.suptitle(c+': MAG/PHA dual optimization; each color is one fixed seed',fontsize=13);fig.savefig(OUT/'figures'/f'{c}_stages_{page+1}.png',dpi=160);plt.close(fig)
        # Pareto fronts, to expose the non-uniqueness of the representative solution.
        fig,axes=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
        for ax,g in zip(axes,[20,100,300]):
            for si,s in enumerate(seeds):
                with np.load(runs/c/f'seed_{s}'/f'generation_{g:03}.npz') as p:
                    v=p['metrics'];front=p['front_indices'];ax.scatter(v[front,0],v[front,1],s=9,alpha=.5,color=colors[si],label=f'seed {s}');ix=int(p['selected_index']);bx=int(p['balanced_index']);ax.scatter(*v[ix,:2],s=65,marker='*',color=colors[si]);ax.scatter(*v[bx,:2],s=35,marker='s',facecolors='none',edgecolors=colors[si])
            ax.set(xscale='log',yscale='log',xlabel='MAG MSE (dB^2)',ylabel='PHA MSE (rad^2)',title=f'Generation {g}');ax.grid(True,alpha=.2)
        axes[0].legend(fontsize=8);fig.suptitle(c+': fronts; star=min WSMSE, hollow square=balanced');fig.savefig(OUT/'figures'/f'{c}_fronts.png',dpi=150);plt.close(fig)
        shutil.copyfile(ROOT/'output/primary_loss_comparison_20261007/schematics'/f'{c}.svg',OUT/'schematics'/f'{c}.svg')
        chosen=[d for d in diags if d['circuit']==c and d['rule']=='min_wsmse' and d['generation'] in [20,100,200,300]]
        diagtable=table(['seed','代','幅值RMSE/dB','相位RMSE/°','幅值漏失/新增','相位事件漏失/新增','网格稳定'],[[d['seed'],d['generation'],n(d['amplitude_rmse_db']),n(d['phase_rmse_deg']),f"{d['amp_missed']}/{d['amp_extra']}",f"{d['phase_missed']}/{d['phase_extra']}",d['grid_stable']] for d in chosen])
        last=[d for d in diags if d['circuit']==c and d['rule']=='min_wsmse' and d['generation']==300]
        summary=dict(circuit=c,seeds=seeds,final_amp_missed=[d['amp_missed'] for d in last],final_phase_missed=[d['phase_missed'] for d in last],final_amp_rmse=[d['amplitude_rmse_db'] for d in last],final_phase_rmse=[d['phase_rmse_deg'] for d in last]);case_summaries.append(summary)
        targettable=table(['元件','目标值/SI','log10下界','log10上界'],[[name,n(r['target_parameters'][j]) if r['target_parameters'] is not None else '实测目标，无真值',n(m.lower[j]),n(m.upper[j])] for j,name in enumerate(r['parameter_names'])])
        parts=[f'# {c}：每20代查看3个固定seed',f'随机seed：{seeds}；颜色全程固定；每个seed各取本阶段前沿的最小WSMSE代表解。没有平均任何曲线。',f'![电路]({link(OUT/"schematics"/f"{c}.svg")})',targettable,'## 每个seed的阶段诊断',diagtable,'Case1在解析加密网格上诊断；Case2仅在原实测点上诊断，点间峰谷不能确认。峰谷阈值1dB、相位事件阈值5°，同类型在0.25 decade内匹配；这些是诊断阈值，不是新优化目标。RMSE按log f积分，和优化使用的频点等权MSE有所不同。',f'![逐seed指标]({link(OUT/"figures"/f"{c}_metrics.png")})',f'![前沿选解]({link(OUT/"figures"/f"{c}_fronts.png")})','## 所有20代阶段的曲线']
        parts.extend(f'### {stages[page*3]}–{stages[page*3+2]}代\n\n![独立seed曲线]({link(OUT/"figures"/f"{c}_stages_{page+1}.png")})' for page in range(5))
        parts.extend([f'[所有阶段元件参数]({link(OUT/"parameters.csv")})；[逐seed局部结构诊断]({link(OUT/"diagnostics.csv")})；[逐频带数据]({link(OUT/"bands.csv")})',f'[返回主报告]({link(OUT/"REPORT.md")})']);write(OUT/f'{c}.md',parts)
    csvout(OUT/'metrics.csv',rows);csvout(OUT/'diagnostics.csv',diags);csvout(OUT/'bands.csv',bands);csvout(OUT/'parameters.csv',parameters)
    dump(OUT/'VALIDATION.json',dict(passed=True,replayed_runs=6,stages=90,individuals_recomputed=checked,final_populations_identical_to_old=True,initial_populations_identical_to_old=True,independent_metrics_and_fronts=True,unstable_diagnostic_curves=unstable,summary=case_summaries))
    write(OUT/'REPORT.md',['# MAG/PHA双目标：每20代的优化曲线',
        '**已按老师的建议完成：随机固定3个seed，每20代把各seed的代表曲线与目标画在同图；不平均曲线。** 后台重放Case1和Case2各100seeds、各15阶段，共3000个seed阶段；图文部分展示随机90个seed阶段。保留全部阶段的种群、前沿与参数。',
        '## 怎么看这些图',
        '先看各Case的逐seed指标，再看20→100→200→300代局部结构诊断，最后查看所有阶段曲线。同色始终对应同一个seed；每阶段选出的个体可以不同。后期仍存在的缺陷是本预算、采样和选解规则下的残差，不能断言MAG/PHA理论上无法优化该结构。',
        *[f'- [{c}：电路参数、阶段诊断和全部曲线]({link(OUT/f"{c}.md")})' for c in cfg['cases']],
        '## 双目标的“最好”怎么定义',
        'MAG=mean[(20log10|Z|−20log10|Z*|)²]，PHA=mean[(arg Z−arg Z*)²]，相位单位rad。NSGA-II只最小化这两个目标。前沿通常有多个相互取舍的解，为延续上一轮，画图取前沿中MAG+αPHA最小的代表解：Case1 α=25.12，Case2 α=17.38。α只用于画图选解，并没有把双目标改成WSMSE单目标。',
        '同时保存均衡规则min max(MAG,αPHA)。指标图用虚线展示它，前沿图用空心方块标出；若同一结构结论随规则变化，应先查前沿取舍。暂不把三个seed或两个Case的数值比值解释成规律。',
        '## 随机选择、复算与边界',
        f'在重跑前用RNG=20261008从2300–2399无放回抽取{seeds}用于画图，两Case共用。沿用原实验频点、参数边界、算法、算子、200/150种群和300代；全部重放新增10,500,000次评价。随机展示的6次末代完整种群与旧实验逐项完全一致，后台另核验全部200次运行。',
        f'独立公式重算{checked:,}个阶段种群个体，核对非支配前沿、两种选解、历史、初始化和参数边界。Case1解析网格逐阶段加密核验，{unstable}条代表曲线未通过最终网格稳定检查；Case2只在测量点上评价。三个随机seed适合看过程，不足以确认100seed的总体缺陷频率。',
        f'[全部100seeds的阶段诊断与后期配对比较]({link(OUT/"ALL_SEEDS.md")})。后台汇总每条曲线的诊断数值，不平均曲线。',
        f'[运行前冻结方案]({link(HERE/"PROTOCOL.md")})；[可视化核验记录]({link(OUT/"VALIDATION.json")})；[每阶段独立指标]({link(OUT/"metrics.csv")})；[完整阶段曲线NPZ]({link(OUT/"selected")})',
        '## 上一轮加密采样',
        f'[1680次新增实验的独立报告]({link(ROOT/"output/sampling_refinement_20261007/REPORT.md")})。它检查采样缺口；本轮检查原双目标运行中结构随代数的变化。后续新增指标应在采样表达可靠的条件下验证，并以MAG/PHA为基础目标。',
        '全部结果仅保存本地，没有上传网站。'])
    dump(OUT/'SUMMARY.json',case_summaries);print(json.dumps(dict(passed=True,checked=checked,unstable=unstable,summary=case_summaries),ensure_ascii=False))
if __name__=='__main__':main()

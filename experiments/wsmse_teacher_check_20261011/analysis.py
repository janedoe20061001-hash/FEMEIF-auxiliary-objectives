"""Verify all final populations; select knees; draw target/scalar/dual comparisons."""
from check import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from html import escape

RETRIEVED=HERE/'server_run/retrieved/wsmse_teacher_check_20261011'
RUNS=RETRIEVED/'runs'
LABELS={'wsmse':'WSMSE 单目标','dual_knee':'双目标膝点','dual_min_wsmse':'双目标最小WSMSE（核查）'}
COLORS={'wsmse':'#d97706','dual_knee':'#2563eb','dual_min_wsmse':'#7b8794'}

def nondominated(v):
    dominates=np.all(v[:,None,:]<=v[None,:,:],axis=2)&np.any(v[:,None,:]<v[None,:,:],axis=2)
    return np.flatnonzero(~dominates.any(axis=0))

def knee(v):
    """Discrete normalized maximum bulge; mark fronts lacking a convex knee."""
    front=nondominated(v)
    _,first=np.unique(v[front],axis=0,return_index=True)
    indices=front[first];values=v[indices]
    low=values.min(axis=0);high=values.max(axis=0);span=high-low
    safe=np.maximum(span,1e-14*np.maximum(1,np.maximum(abs(low),abs(high))))
    normalized=(values-low)/safe
    ideal=np.sum(normalized**2,axis=1)
    distance=(1-normalized.sum(axis=1))/np.sqrt(2)
    if len(values)==1:reason='unique_front_point'
    elif np.any(span<=1e-14*np.maximum(1,np.maximum(abs(low),abs(high)))):reason='degenerate_span'
    elif len(values)<3:reason='insufficient_front_points'
    elif np.max(distance)<=1e-12:reason='no_convex_knee'
    else:reason='convex_knee'
    if reason=='convex_knee':
        eligible=np.flatnonzero(distance>=np.max(distance)-1e-12)
        local=eligible[np.lexsort((indices[eligible],ideal[eligible]))[0]]
    else:local=np.lexsort((indices,ideal))[0]
    return int(indices[local]),dict(reason=reason,front_indices=front.tolist(),unique_indices=indices.tolist(),
                                   minimum=low.tolist(),maximum=high.tolist(),normalized=normalized.tolist(),
                                   signed_distances=distance.tolist(),selected_index=int(indices[local]),
                                   distance=float(distance[local]))

def selection_checks():
    v=np.array([[0.,1.],[.2,.2],[1.,0.],[.6,.6]])
    assert knee(v)[0]==1
    assert knee(v*np.array([100,1e-3]))[0]==1
    order=np.array([2,0,3,1]);assert order[knee(v[order])[0]]==1
    assert knee(np.array([[.1,.1],[.1,.1],[.2,.3]]))[1]['reason']=='unique_front_point'
    assert knee(np.array([[0,1],[.5,.5],[1,0]]))[1]['reason']=='no_convex_knee'
    assert knee(np.array([[0,1],[1,0]]))[1]['reason']=='insufficient_front_points'

def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def verify():
    status=json.loads((RETRIEVED/'STATUS.json').read_text());assert status['status']=='complete' and status['completed']==206 and not status['failed']
    selection_checks();rows=[];selections={};replays=[];total=0
    for case in ['case1','case2']:
        m=Model(case);r=m.record
        for seed in range(2300,2400):
            d=RUNS/case/'wsmse'/f'seed_{seed}'
            initial=np.load(d/'initial_log_parameters.npy')
            np.testing.assert_array_equal(initial,np.load(OLD_DUAL/case/f'seed_{seed}/initial_log_parameters.npy'))
            np.testing.assert_array_equal(initial,np.load(OLD_SCALAR/case/'wsmse'/f'seed_{seed}/initial_log_parameters.npy'))
            with np.load(d/'final_population.npz') as a:
                x=a['log_parameters'];v=independent_metrics(m.spectrum(x),m.target,m.alpha)
                np.testing.assert_allclose(a['metrics'],v,rtol=1e-11,atol=1e-9)
                np.testing.assert_allclose(a['optimization_objective'],v[:,2:3],rtol=1e-11,atol=1e-9)
                ix=int(a['selected_index']);assert ix==int(np.argmin(v[:,2]))
                assert x.shape==(r['population'],len(m.lower)) and np.all(x>=m.lower) and np.all(x<=m.upper)
                record=dict(case=case,seed=seed,group='wsmse',selected_index=ix,**dict(zip(COLS,v[ix].tolist())))
                rows.append(record);selections[case,seed,'wsmse']=x[ix].copy()
                with np.load(OLD_SCALAR/case/'wsmse'/f'seed_{seed}/final_population.npz') as old:
                    oldx=old['log_parameters'];oldv=old['selected_metrics']
                replays.append(dict(case=case,seed=seed,group='wsmse',population_max_log_parameter_difference=float(abs(x-oldx).max()),
                                    selected_wsmse_difference=float(v[ix,2]-oldv[2])))
            summary=json.loads((d/'summary.json').read_text());assert summary['evaluations']==300*r['population'] and summary['n_obj']==1
            total+=summary['evaluations']
            history=np.genfromtxt(d/'history.csv',delimiter=',',names=True,encoding='utf-8-sig')
            assert len(history)==300 and int(history['generation'][-1])==300
            assert np.all(np.diff(history['wsmse'])<=1e-9)
            np.testing.assert_allclose(history['wsmse'][-1],record['wsmse'],rtol=1e-11,atol=1e-9)
            with np.load(OLD_DUAL/case/f'seed_{seed}/generation_300.npz') as a:
                dx=a['log_parameters'];dv=independent_metrics(m.spectrum(dx),m.target,m.alpha)
                np.testing.assert_allclose(a['metrics'],dv,rtol=1e-11,atol=1e-9)
                np.testing.assert_allclose(a['optimization_objective'],dv[:,:2],rtol=1e-11,atol=1e-9)
                assert dx.shape==(r['population'],len(m.lower)) and np.all(dx>=m.lower) and np.all(dx<=m.upper)
                front=nondominated(dv[:,:2]);np.testing.assert_array_equal(np.sort(front),np.sort(a['front_indices']))
                ki,info=knee(dv[:,:2]);wi=int(np.argmin(dv[:,2]));assert ki in front and wi in front
                dump(OUT/'knees'/case/f'seed_{seed}.json',info)
                for group,index in [('dual_knee',ki),('dual_min_wsmse',wi)]:
                    rows.append(dict(case=case,seed=seed,group=group,selected_index=index,**dict(zip(COLS,dv[index].tolist()))))
                    selections[case,seed,group]=dx[index].copy()
            if seed in SHOW:
                d=RUNS/case/'dual'/f'seed_{seed}'
                np.testing.assert_array_equal(np.load(d/'initial_log_parameters.npy'),initial)
                with np.load(d/'final_population.npz') as a:
                    replay=a['log_parameters'];rv=independent_metrics(m.spectrum(replay),m.target,m.alpha)
                    np.testing.assert_allclose(a['optimization_objective'],rv[:,:2],rtol=1e-11,atol=1e-9)
                    np.testing.assert_allclose(replay,dx,rtol=1e-12,atol=1e-12)
                    replays.append(dict(case=case,seed=seed,group='dual',population_max_log_parameter_difference=float(abs(replay-dx).max()),selected_wsmse_difference=float(rv[:,2].min()-dv[:,2].min())))
                s=json.loads((d/'summary.json').read_text());assert s['n_obj']==2 and s['evaluations']==300*r['population'];total+=s['evaluations']
    assert total==10815000
    write_csv(OUT/'all_seed_metrics.csv',rows);write_csv(OUT/'reproduction.csv',replays)
    means=[];best={};rng=np.random.default_rng(20261011);pairs=[]
    for case in ['case1','case2']:
        for group in LABELS:
            records=[x for x in rows if x['case']==case and x['group']==group]
            means.append(dict(case=case,group=group,seeds=len(records),**{k:float(np.mean([x[k] for x in records])) for k in COLS}))
            best[case,group]=min(records,key=lambda x:(x['wsmse'],x['seed']))
        for group in ['dual_knee','dual_min_wsmse']:
            for metric in COLS:
                w=np.array([next(x for x in rows if x['case']==case and x['seed']==s and x['group']=='wsmse')[metric] for s in range(2300,2400)])
                dual=np.array([next(x for x in rows if x['case']==case and x['seed']==s and x['group']==group)[metric] for s in range(2300,2400)])
                delta=w-dual;boot=delta[rng.integers(0,100,(10000,100))].mean(axis=1)
                pairs.append(dict(case=case,comparison='wsmse minus '+group,metric=metric,mean_difference=float(delta.mean()),
                                  lower_95=float(np.quantile(boot,.025)),upper_95=float(np.quantile(boot,.975)),wsmse_better_seeds=int(np.sum(delta<0))))
    fallback={case:sum(json.loads((OUT/'knees'/case/f'seed_{s}.json').read_text())['reason']!='convex_knee' for s in range(2300,2400)) for case in ['case1','case2']}
    summary=dict(means=means,best=list(best.values()),paired=pairs,knee_fallback_counts=fallback)
    dump(OUT/'SUMMARY.json',summary)
    dump(OUT/'VALIDATION.json',dict(passed=True,server_runs=206,scalar_runs=200,dual_replays=6,paired_seeds_per_case=100,
                                  total_evaluations=total,final_individuals_recomputed=71050,initialization_identical=True,
                                  knee_selection_checks=True,dual_replay_identical=True,scalar_replay_max_parameter_difference=max(x['population_max_log_parameter_difference'] for x in replays if x['group']=='wsmse'),
                                  scalar_replay_max_wsmse_difference=max(abs(x['selected_wsmse_difference']) for x in replays if x['group']=='wsmse')))
    return rows,selections,best,summary

def dense_grid(m,parameters,refine=1):
    if m.record['id']=='case2':return m.f
    logs=[np.linspace(np.log10(m.f[0]),np.log10(m.f[-1]),8192*refine+1),np.log10(m.f)]
    for p in [np.asarray(m.record['target_parameters']),*parameters]:
        for r,l,c in p.reshape(6,3):
            fc=1/(2*np.pi*np.sqrt(l*c));q=r*np.sqrt(c/l);dx=min(.5,max(6/q,1e-12))/np.log(10);center=np.log10(fc)
            if not m.f[0]<=fc<=m.f[-1]:continue
            logs.append(np.linspace(center-dx,center+dx,384*refine+1))
            offsets=np.geomspace(dx,max(dx,.5),192*refine+1)
            logs.extend([center-offsets,center+offsets])
    x=np.unique(np.concatenate(logs));x=x[(x>=np.log10(m.f[0]))&(x<=np.log10(m.f[-1]))]
    return 10**x

def axes_style(ax):
    ax.grid(True,which='major',alpha=.2);ax.spines[['top','right']].set_visible(False)

def plot_curves(case,selections,best):
    m=Model(case);groups=['wsmse','dual_knee'];params=[10**selections[case,best[case,g]['seed'],g] for g in groups]
    f=dense_grid(m,params);target=m.target if case=='case2' else m.network(f,np.asarray(m.record['target_parameters']))[0]
    curves=[m.network(f,p)[0] for p in params]
    diagnostics=[]
    for g,z in zip(groups,curves):
        da=20*np.log10(abs(z)/abs(target));dp=np.rad2deg(np.angle(z)-np.angle(target))
        record=dict(case=case,group=g,seed=best[case,g]['seed'],grid_points=len(f),
                    maximum_amplitude_error_db=float(abs(da).max()),maximum_amplitude_error_frequency_hz=float(f[np.argmax(abs(da))]),
                    maximum_phase_error_deg=float(abs(dp).max()),maximum_phase_error_frequency_hz=float(f[np.argmax(abs(dp))]))
        if case=='case1':
            fine=dense_grid(m,params,2);fine_target=m.network(fine,np.asarray(m.record['target_parameters']))[0]
            fine_fit=m.network(fine,10**selections[case,best[case,g]['seed'],g])[0]
            record['refined_maximum_amplitude_error_db']=float(abs(20*np.log10(abs(fine_fit)/abs(fine_target))).max())
            record['refined_maximum_phase_error_deg']=float(abs(np.rad2deg(np.angle(fine_fit)-np.angle(fine_target))).max())
        diagnostics.append(record)
    dump(OUT/f'{case}_curve_diagnostics.json',diagnostics)
    (OUT/'figures').mkdir(exist_ok=True)
    fig,axs=plt.subplots(2,2,figsize=(14,8),layout='constrained')
    for i,ax in enumerate(axs.flat):
        y=lambda z:20*np.log10(abs(z)) if i%2==0 else np.rad2deg(np.angle(z))
        ax.semilogx(f,y(target),color='#111827',lw=1.8,label='目标曲线',zorder=4)
        for g,z in zip(groups,curves):
            r=best[case,g];ax.semilogx(f,y(z),color=COLORS[g],lw=1.2,label=f"{LABELS[g]} · seed {r['seed']}")
        ax.set(xlabel='频率 / Hz',ylabel='幅值 / dBΩ' if i%2==0 else '相位 / °');axes_style(ax)
        if i>=2:
            limits=(2e4,3e4) if case=='case1' else (2e6,1.1e8)
            ax.set_xlim(limits);mask=(f>=limits[0])&(f<=limits[1])
            local=np.concatenate([y(z)[mask] for z in [target,*curves]])
            low,high=local.min(),local.max();margin=max(1.,.08*(high-low));ax.set_ylim(low-margin,high+margin)
            if case=='case2':ax.semilogx(f[mask],y(target)[mask],'.',color='#111827',ms=4,zorder=5)
        ax.set_title('全频段' if i<2 else ('局部：20–30 kHz' if case=='case1' else '局部：2–110 MHz'))
        if i==0:ax.legend(fontsize=9)
    fig.suptitle(f'{case.upper()}：目标与两组各自100次运行的最佳代表（300代）',fontsize=15)
    fig.savefig(OUT/'figures'/f'{case}_best.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax in axs:
        ax.plot(target.real,target.imag,color='#111827',lw=1.8,label='目标曲线')
        for g,z in zip(groups,curves):ax.plot(z.real,z.imag,color=COLORS[g],lw=1.3,label=LABELS[g])
        ax.set(xlabel='Re Z / Ω',ylabel='Im Z / Ω');axes_style(ax)
    axs[0].set_title('复阻抗全尺度');axs[0].legend()
    t=m.target
    re=np.quantile(t.real,[.02,.98]);im=np.quantile(t.imag,[.02,.98])
    axs[1].set_xlim(re[0]-.1*np.ptp(re),re[1]+.1*np.ptp(re));axs[1].set_ylim(im[0]-.1*np.ptp(im),im[1]+.1*np.ptp(im));axs[1].set_title('目标2%–98%范围放大（截断处见全尺度）')
    fig.suptitle(f'{case.upper()}：同一套最佳参数的复阻抗曲线（纵轴为 Im Z）')
    fig.savefig(OUT/'figures'/f'{case}_nyquist.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(3,2,figsize=(14,11),layout='constrained')
    for row,seed in enumerate(SHOW):
        ps=[10**selections[case,seed,g] for g in groups];ff=dense_grid(m,ps)
        tt=m.target if case=='case2' else m.network(ff,np.asarray(m.record['target_parameters']))[0]
        zz=[m.network(ff,p)[0] for p in ps]
        for col in range(2):
            ax=axs[row,col];transform=lambda z:20*np.log10(abs(z)) if col==0 else np.rad2deg(np.angle(z))
            ax.semilogx(ff,transform(tt),color='#111827',lw=1.7,label='目标曲线')
            for g,z in zip(groups,zz):ax.semilogx(ff,transform(z),color=COLORS[g],lw=1.1,label=LABELS[g])
            ax.set(title=f'seed {seed}',xlabel='频率 / Hz',ylabel='幅值 / dBΩ' if col==0 else '相位 / °');axes_style(ax)
            if row==0 and col==0:ax.legend()
    fig.suptitle(f'{case.upper()}：原网站固定3个seed的配对结果（未挑选冠军）')
    fig.savefig(OUT/'figures'/f'{case}_paired.png',dpi=150);plt.close(fig)
    # The displayed best dual front makes the choice reviewable.
    seed=best[case,'dual_knee']['seed']
    with np.load(OLD_DUAL/case/f'seed_{seed}/generation_300.npz') as a:v=a['metrics'][:,:2]
    index,info=knee(v);nf=np.asarray(info['normalized']);wi=int(np.argmin(v[:,0]+m.alpha*v[:,1]))
    lo=np.asarray(info['minimum']);span=np.maximum(np.asarray(info['maximum'])-lo,1e-14)
    fig,ax=plt.subplots(figsize=(7,6),layout='constrained');ax.scatter(nf[:,0],nf[:,1],s=16,alpha=.6,label='末代非支配前沿')
    ax.plot([0,1],[1,0],'--',color='#6b7280',label='两极值连线')
    kn=(v[index]-lo)/span;wn=(v[wi]-lo)/span
    ax.scatter(*kn,s=180,marker='*',color=COLORS['dual_knee'],label='膝点 / 折中代表',zorder=5)
    ax.scatter(*wn,s=70,marker='x',color='#d97706',label='前沿最小WSMSE（核查）',zorder=5)
    ax.set(xlabel='归一化幅值MSE',ylabel='归一化相位MSE',title=f'{case.upper()} · seed {seed} · {info["reason"]}');axes_style(ax);ax.legend()
    fig.savefig(OUT/'figures'/f'{case}_front.png',dpi=160);plt.close(fig)
    curve_rows=[]
    for i,frequency in enumerate(f):
        row=dict(frequency_hz=float(frequency),target_real=float(target[i].real),target_imag=float(target[i].imag))
        for g,z in zip(groups,curves):row.update({g+'_real':float(z[i].real),g+'_imag':float(z[i].imag)})
        curve_rows.append(row)
    write_csv(OUT/f'{case}_best_curves.csv',curve_rows)
    params_rows=[]
    for i,name in enumerate(m.record['parameter_names']):
        params_rows.append(dict(parameter=name,unit='ohm' if name.startswith('R') else ('H' if name.startswith('L') else 'F'),
                                target=None if case=='case2' else m.record['target_parameters'][i],wsmse=float(params[0][i]),dual_knee=float(params[1][i])))
    write_csv(OUT/f'{case}_best_parameters.csv',params_rows)

def report(rows,best,summary):
    lines=['# WSMSE单主目标与双主目标膝点：基础排查结果','',
           '2026年10月11日。服务器完成两Case各100次WSMSE单主目标优化，另重放6次双目标作复现核查。双目标100次结果使用同条件历史末代种群重新选膝点。目标数据、频点、边界、初始种群、算子和300代预算一致。','',
           '**结果：单目标的平均WSMSE较低，但换回单目标没有让两Case的局部误差全部消失。** Case1两组最佳解仍存在尖峰高度与位置偏差；Case2两组在高频仍有明显相位误差。Case1双目标膝点的最佳训练分数低于WSMSE最佳分数，Case2则是WSMSE略低，因此不能笼统说“双目标一定比单目标差”。平均相位MSE反而是双目标膝点较低，两项误差仍有取舍。','',
           '**主图同时放目标、WSMSE拟合、双目标膝点拟合。** 单目标每seed取末代WSMSE最小值；双目标每seed取末代非支配前沿膝点。每组再按同一WSMSE尺度选100个代表中最好的一个，用于回答“最好能拟合到什么程度”；它不是典型seed，不代表平均水平。','',
           '膝点采用归一化前沿距极值连线向理想点方向的最大垂距，定义及退化处理已在运行前固定。膝点是一种折中选择，不保证WSMSE最小。[方法出处：Deb与Gupta，定义5](https://www.egr.msu.edu/~kdeb/papers/k2010005.pdf)。','']
    for case in ['case1','case2']:
        m=Model(case);w=best[case,'wsmse'];d=best[case,'dual_knee'];old=best[case,'dual_min_wsmse']
        means={x['group']:x for x in summary['means'] if x['case']==case}
        lines += [f'## {case.upper()}：先看最好的曲线','',
                  f"WSMSE最好为seed **{w['seed']}**，WSMSE={w['wsmse']:.6g}；双目标膝点代表中最好为seed **{d['seed']}**，WSMSE={d['wsmse']:.6g}。两条来自各组独立排名，不是同seed配对。",'',
                  '| 最佳代表 | seed | 幅值RMSE / dB | 相位RMSE / ° | WSMSE | 平均复阻抗距离 / Ω |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for x in [w,d]:lines.append(f"| {LABELS[x['group']]} | {x['seed']} | {np.sqrt(x[COLS[0]]):.4f} | {np.rad2deg(np.sqrt(x[COLS[1]])):.4f} | {x['wsmse']:.6g} | {x[COLS[3]]:.6g} |")
        curve_diagnostics=json.loads((OUT/f'{case}_curve_diagnostics.json').read_text())
        lines += ['',('上述RMSE只在原训练频点计算；加密检查仍显示尖峰高度/位置严重失真。' if case=='case1' else '上述RMSE在201原测点计算；相位误差仍集中在高频。'),'',
                  '| 最佳代表 | 最大幅值偏差 / dB | 所在频率 / Hz | 最大相位偏差 / ° | 所在频率 / Hz |',
                  '| --- | ---: | ---: | ---: | ---: |']
        for x in curve_diagnostics:
            lines.append(f"| {LABELS[x['group']]} | {x['maximum_amplitude_error_db']:.3f} | {x['maximum_amplitude_error_frequency_hz']:.7g} | {x['maximum_phase_error_deg']:.3f} | {x['maximum_phase_error_frequency_hz']:.7g} |")
        if case=='case1':lines+=['','最大偏差来自目标和两条拟合曲线的分支中心/宽度加密网格；网格加倍后的最大偏差一致。它们是展示与诊断值，不是优化训练分数。']
        lines += ['',f'![{case}最佳曲线对照](figures/{case}_best.png)','',f'![{case}复阻抗](figures/{case}_nyquist.png)','',
                  f'![{case}膝点选择](figures/{case}_front.png)','',
                  f"本Case有 {summary['knee_fallback_counts'][case]}/100 次前沿没有可辨识凸膝点，采用预定折中解并逐seed标注。",'',
                  f"作为选解核查，原前沿最小WSMSE规则跨100seed最好是 {old['wsmse']:.6g}（seed {old['seed']}）。这一规则与膝点结果不同，不能把膝点改变带来的分数变化都归因于优化目标。",'',
                  '### 全部100次的平均结果','',
                  '| 组别 | 幅值MSE / dB² | 相位MSE / rad² | WSMSE | 平均复阻抗距离 / Ω |',
                  '| --- | ---: | ---: | ---: | ---: |']
        for group in LABELS:
            x=means[group];lines.append(f"| {LABELS[group]} | {x[COLS[0]]:.6g} | {x[COLS[1]]:.6g} | {x['wsmse']:.6g} | {x[COLS[3]]:.6g} |")
        change=means['wsmse']['wsmse']/means['dual_min_wsmse']['wsmse']-1
        kneechange=means['wsmse']['wsmse']/means['dual_knee']['wsmse']-1
        lines += ['',f'沿用相同最小WSMSE选解时，单目标平均WSMSE相对双目标变化 **{change:+.2%}**；按用户要求改用膝点后，对应变化 **{kneechange:+.2%}**。负数表示WSMSE单目标较低。此处比较的是当前实现与预算，不能推出所有双目标算法都更差。','',
                  f'![{case}固定seed配对](figures/{case}_paired.png)','',
                  f'[最佳曲线逐点数据]({case}_best_curves.csv) · [两组最佳参数]({case}_best_parameters.csv)','']
    validation=json.loads((OUT/'VALIDATION.json').read_text())
    lines += ['## 简单检查发现了什么','',
              '- 目标数据和代码哈希一致；Case2全部201行均保留，幅值Ω转dB、相位degree转rad检查通过。两Case独立串并联计算与现有电路计算一致，独立幅相损失重算一致。',
              '- 单/双目标同seed初始化逐项一致，单目标维数为1；200次WSMSE及6次双目标重放的预算、历史和末代种群检查通过。全部100次历史双目标末代个体重新核算并选膝点。',
              f"- 历史WSMSE复现的最大末代log10参数差为 {validation['scalar_replay_max_parameter_difference']:.3g}，最大选解WSMSE差为 {validation['scalar_replay_max_wsmse_difference']:.3g}；6次双目标重放与历史种群一致。",
              '- 以上基础检查没有发现单位、目标接线或末代最小值选解错误；这不能排除所有程序问题，也不能将尚未修复的局部误差解释为“只是放大”。','',
              'Case1图用目标和拟合支路模态加密展示，180点训练仍可能漏掉窄峰；图上局部偏差应按实际曲线判断。Case2只比较201测点，包含一个相位107°、实部为负的点；当前正值无源RLC模型不能精确表示该点，但它不能解释全部残差。Case2没有已知真参数，本轮不判断测量噪声来源。','',
              '首次服务器预检因打包时漏带根目录目标CSV而停止，尚未开始优化；补传同哈希原文件后预检及全部206次运行通过。数据回收核对1006个文件哈希。所有正式优化结果完整保留，未删除不良seed或增加预算。','',
              '## 结论怎么使用','',
              '先看主图能否重合，再看平均与固定seed对照。WSMSE分数较低只说明该统一尺度下更好，仍需承认图中漏峰、尖峰或局部相位误差。若WSMSE也未恢复全部结构，当前结果不支持“只把双目标换回单目标就能完全恢复”。膝点只负责在已有前沿选折中解，不会创造尚未搜索到的正确参数。','',
              '[全部seed误差表](all_seed_metrics.csv) · [统计与最佳seed记录](SUMMARY.json) · [核验结果](VALIDATION.json) · [逐seed复现记录](reproduction.csv)','']
    text='\n'.join(lines)
    import re
    absolute_text=re.sub(r'\]\((?!https?://)([^)]+)\)',lambda match:']('+str(OUT/match.group(1)).replace('\\','/')+')',text)
    (OUT/'REPORT.md').write_text(absolute_text,encoding='utf8')
    sys.path.append(str(ROOT/'tmp/curve_refinement_deps'))
    import markdown
    body=markdown.markdown(text,extensions=['tables'])
    html='<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>WSMSE与双目标膝点对照</title><style>body{max-width:1200px;margin:40px auto;padding:0 24px;font:17px/1.7 system-ui;color:#17233b}h1,h2,h3{line-height:1.35}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:15px;margin:20px 0}th,td{padding:8px;border:1px solid #d7dfeb;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#eaf0f8}a{color:#2563eb}</style>'+body+'</html>'
    (OUT/'index.html').write_text(html,encoding='utf8')

def main():
    fonts={f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    rows,selections,best,summary=verify()
    for case in ['case1','case2']:plot_curves(case,selections,best)
    report(rows,best,summary)
    print(json.dumps(dict(validation='passed',best=list(best.values()),means=summary['means'],knee_fallback_counts=summary['knee_fallback_counts']),ensure_ascii=False))

if __name__=='__main__':main()

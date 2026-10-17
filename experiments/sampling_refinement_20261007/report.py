"""Local Markdown delivery with three representative circuit figures."""
from pathlib import Path
import sys,json,csv,hashlib,shutil
import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'output/sampling_refinement_20261007';sys.path.insert(0,str(HERE))
import analyze as ana
ARM={'original':'原采样','dense_interpolated':'加密·旧点插值','dense_true':'加密·解析真值'};GROUP={'wsmse':'WSMSE','mean_distance':'原Ω距离','dual_mse':'双MSE·最小WSMSE'}
def n(v):return '不适用' if v is None else f'{float(v):.6g}'
def link(p):return p.resolve().as_posix()
def table(headers,rows):
    esc=lambda s:str(s).replace('|',r'\|')
    return '\n'.join(['| '+' | '.join(map(esc,headers))+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(esc,r))+' |' for r in rows]])
def write(p,txt):p.write_text('\n\n'.join(txt)+'\n',encoding='utf-8')
def main():
    s=json.loads((OUT/'SUMMARY.json').read_text());validation=json.loads((OUT/'VALIDATION.json').read_text());assert validation['passed'];prep=json.loads((HERE/'PREPARATION.json').read_text());config=json.loads((HERE/'CONFIG.json').read_text());status=json.loads((HERE/'server_run/retrieved/runs/status.json').read_text())
    records={r['id']:r for r in json.loads((HERE/'frozen_scalar/circuits.json').read_text(encoding='utf-8'))};means=s['structure_means'];pairs=s['paired_comparisons'];aggregate=s['aggregate_comparisons']
    lookup={(r['circuit'],r['arm'],r['group'],r['rule']):r for r in means};plook={(r['circuit'],r['baseline'],r['group'],r['rule'],r['metric']):r for r in pairs}
    (OUT/'figures').mkdir(exist_ok=True);(OUT/'schematics').mkdir(exist_ok=True);examples=[];rep=[]
    for c in ['case1','two_parallel_units_c3','rlc_bridge_c2']:
        d=np.load(OUT/'selected'/f'{c}.npz');v=d['original_metrics'];ranks=np.column_stack([rankdata(v[:,ai,gi,k],method='average') for ai in range(3) for gi in range(3) for k in [0,1,3]]);score=ranks.mean(axis=1);median=float(np.median(score));ix=int(np.argmin(abs(score-median)));seed=int(d['seeds'][ix]);r=records[c];p=10**d['log_parameters'][ix].reshape(9,-1);f=ana.improved_dense_grid(r,p,2);target=ana.spectra(r,f,r['target_parameters'])[0];z=ana.spectra(r,f,p).reshape(3,3,len(f))
        fig,axes=plt.subplots(2,3,figsize=(15.5,7.2),sharex=True,layout='constrained');colors=['#2674ad','#9b9b9b','#29936f'];labels=['Original samples','Dense: old-data interpolation','Dense: analytic target'];titles=['Optimize WSMSE','Optimize raw mean distance','Dual MSE: min WSMSE selection']
        for gi in range(3):
            for ai in range(3):
                axes[0,gi].semilogx(f,20*np.log10(abs(z[ai,gi])),lw=1.1,color=colors[ai],label=labels[ai]);axes[1,gi].semilogx(f,np.rad2deg(np.angle(z[ai,gi])),lw=1.1,color=colors[ai])
            axes[0,gi].semilogx(f,20*np.log10(abs(target)),'k--',lw=1.1,label='Analytic target');axes[1,gi].semilogx(f,np.rad2deg(np.angle(target)),'k--',lw=1.1);axes[0,gi].set_title(titles[gi],fontsize=11);axes[1,gi].set_xlabel('Frequency (Hz)')
            for a in axes[:,gi]:a.grid(True,alpha=.18);a.set_xlim(f[0],f[-1])
        axes[0,0].set_ylabel('Magnitude (dB re 1 ohm)');axes[1,0].set_ylabel('Phase (degree)');axes[0,0].legend(fontsize=8,loc='best');fig.suptitle(f'{c}: representative seed {seed}; common analytic grid',fontsize=13);fig.savefig(OUT/'figures'/f'{c}.png',dpi=170);plt.close(fig)
        shutil.copyfile(ROOT/'output/primary_loss_comparison_20261007/schematics'/f'{c}.svg',OUT/'schematics'/f'{c}.svg')
        texts=[f'# {c}：典型seed {seed}',f'27项原频点幅相MSE/Ω距离排名的平均秩={n(score[ix])}，seed中位平均秩={n(median)}。本例不是按加密获益最大选择。',f'![电路结构]({link(OUT/"schematics"/f"{c}.svg")})',f'![三目标×三采样对照]({link(OUT/"figures"/f"{c}.png")})','三列分别对应WSMSE、原Ω距离、双MSE；每列颜色区分采样条件。所有线由相同解析检查网格计算，未把折线插值当成真实拟合曲线。主报告统计全部seeds，本图仅供解释。']
        for gi,g in enumerate(ana.GROUPS):
            texts.extend([f'## {GROUP[g]}：元件参数',table(['参数/SI单位','目标值','原采样','加密插值','加密真值'],[[name+'/'+{'R':'Ω','L':'H','C':'F'}[name[0]],n(r['target_parameters'][j]),*[n(p[ai*3+gi,j]) for ai in range(3)]] for j,name in enumerate(r['parameter_names'])])])
        texts.extend(['## 同一个seed的原频点误差',table(['组','采样','幅值MSE/dB²','相位MSE/rad²','WSMSE','原距离/Ω'],[[GROUP[g],ARM[arm],*[n(x) for x in v[ix,ai,gi]]] for gi,g in enumerate(ana.GROUPS) for ai,arm in enumerate(ana.ARMS)]),'这里是原频点评分；图中的解析加密检查与原点训练损失不是同一统计口径。',f'[返回主报告]({link(OUT/"REPORT.md")})']);write(OUT/f'{c}_example.md',texts);examples.append(f'[{c}：seed {seed}，电路、参数和三组曲线]({link(OUT/f"{c}_example.md")})');rep.append(dict(circuit=c,seed=seed,mean_rank=float(score[ix]),median_mean_rank=median,rule='closest median mean rank among 27 original-point metrics; exact tie smaller seed'))
    (OUT/'REPRESENTATIVES.json').write_text(json.dumps(rep,indent=2),encoding='utf-8')
    case_table=table(['组','采样','解析幅值RMSE/dB','解析相位RMSE/°','幅值漏失/新增','相位漏失/新增','最坏频带相位RMSE/°','解析平均Ω距离'],[[GROUP[g],ARM[arm],n(r['amplitude_rmse_db']),n(r['phase_rmse_deg']),f"{n(r['amp_missed'])}/{n(r['amp_extra'])}",f"{n(r['phase_missed'])}/{n(r['phase_extra'])}",n(r['worst_band_phase_rmse_deg']),n(r['mean_distance_logweighted_ohm'])] for g in ana.GROUPS for arm in ana.ARMS for r in [lookup[('case1',arm,g,'primary')]]])
    compact=[];decision=[]
    for cfg in config['records']:
        c=cfg['circuit']
        for g in ana.GROUPS:
            row=lookup[(c,'dense_true',g,'primary')];base=lookup[(c,'original',g,'primary')];a=plook[(c,'original',g,'primary','amplitude_rmse_db')];p=plook[(c,'original',g,'primary','phase_rmse_deg')];events=plook[(c,'original',g,'primary','amp_missed')]
            both=a['ci95'][1]<0 and p['ci95'][1]<0 and a['numerically_stable'] and p['numerically_stable'];decision.append(dict(circuit=c,group=g,both_amplitude_phase_improved_ci95=both,amplitude_mean_improved=a['new_minus_baseline']<0,phase_mean_improved=p['new_minus_baseline']<0,amp_missed_improved_ci95=events['ci95'][1]<0 and events['numerically_stable']))
            compact.append([c,GROUP[g],n(base['amplitude_rmse_db'])+'→'+n(row['amplitude_rmse_db']),n(base['phase_rmse_deg'])+'→'+n(row['phase_rmse_deg']),n(base['amp_missed'])+'→'+n(row['amp_missed']),'两项区间均支持降低' if both else '未同时支持降低'])
    case_pairs=table(['组','对照采样','指标','真值加密−对照','95%配对区间'],[[GROUP[r['group']],ARM[r['baseline']],{'amplitude_rmse_db':'幅值RMSE/dB','phase_rmse_deg':'相位RMSE/°','amp_missed':'幅值漏失数','worst_band_phase_rmse_deg':'最坏频带相位RMSE/°'}[r['metric']],n(r['new_minus_baseline']),f"[{n(r['ci95'][0])}, {n(r['ci95'][1])}]"] for r in pairs if r['circuit']=='case1' and r['rule']=='primary' and r['metric'] in ['amplitude_rmse_db','phase_rmse_deg','amp_missed','worst_band_phase_rmse_deg']])
    agg_table=table(['电路集合','组','对照','指标','电路等权平均差','95%区间','均值改善电路数'],[[r['cohort'],GROUP[r['group']],ARM[r['baseline']],{'amplitude_rmse_db':'幅值RMSE/dB','phase_rmse_deg':'相位RMSE/°','amp_missed':'幅值漏失数','worst_band_phase_rmse_deg':'最坏频带相位RMSE/°'}[r['metric']],n(r['equal_circuit_mean_difference']),f"[{n(r['ci95'][0])}, {n(r['ci95'][1])}]",f"{r['improved_circuit_means']}/{r['circuits']}"] for r in aggregate if r['metric'] in ['amplitude_rmse_db','phase_rmse_deg','amp_missed','worst_band_phase_rmse_deg']])
    improvement=[]
    for g in ana.GROUPS:
        rr=[r for r in decision if r['group']==g and r['circuit'] not in ['rc_series_c1','rlc_bridge_c2']];improvement.append(f"{GROUP[g]}：8个缺口目标中，{sum(r['both_amplitude_phase_improved_ci95'] for r in rr)}个的幅相RMSE配对区间都支持降低，{sum(r['amp_missed_improved_ci95'] for r in rr)}个的幅值漏失区间支持降低。")
    grid_table=table(['电路','原点→加密点','原目标最大缺口/dB','新目标独立检查误差/dB','新相位检查误差/°'],[[r['circuit'],f"{r['original_points']}→{r['dense_points']}",n(r['original_gap_db']),n(r['amplitude_max_holdout_gap_db']),n(r['phase_max_holdout_gap_deg'])] for r in prep['grid_checks']])
    case_summary=[]
    for g in ana.GROUPS:
        a=plook[('case1','original',g,'primary','amplitude_rmse_db')];p=plook[('case1','original',g,'primary','phase_rmse_deg')]
        case_summary.append(f"{GROUP[g]}：幅值RMSE均值差{n(a['new_minus_baseline'])}dB，相位RMSE均值差{n(p['new_minus_baseline'])}°。")
    # Retain all full diagnostic columns in a compact appendix, avoiding wide main tables.
    detailrows=list(csv.DictReader((OUT/'per_seed_diagnostics.csv').open(encoding='utf-8-sig')))
    gaprows=[]
    for c in ['case1','two_parallel_units_c3','rlc_bridge_c2']:
        for arm in ana.ARMS:
            for g in ana.GROUPS:
                rr=[r for r in detailrows if r['circuit']==c and r['arm']==arm and r['group']==g and r['rule']=='primary'];vv=[float(r['fit_sampling_gap_max_db']) for r in rr];gaprows.append([c,GROUP[g],ARM[arm],n(np.median(vv)),n(np.quantile(vv,.9))])
    sensitivity=table(['Case1双MSE规则','对照','指标','真值加密−对照','95%区间'],[[r['rule'],ARM[r['baseline']],r['metric'],n(r['new_minus_baseline']),f"[{n(r['ci95'][0])}, {n(r['ci95'][1])}]"] for r in pairs if r['circuit']=='case1' and r['group']=='dual_mse' and r['metric'] in ['amplitude_rmse_db','phase_rmse_deg','amp_missed']])
    write(OUT/'DETAILS.md',['# 加密采样：完整比较附表','## 每电路真值加密相对原采样',table(['电路','组','幅值RMSE 原→新/dB','相位RMSE 原→新/°','幅值漏失 原→新','幅相均改善的区间判断'],compact),'## 两个电路集合的等权比较',agg_table,'gap8为预先选定8个缺口目标；controls2为2个采样充分的对照。这些区间只重采样电路内seeds，固定这些电路，不代表从所有电路总体抽样的不确定性。','## 候选曲线在训练点之间的隐藏形变',table(['电路','组','采样','候选折线缺口中位/dB','90%分位/dB'],gaprows),'这里对每个拟合电路重新计算连续解析曲线，比较训练频点折线与共同诊断网格的最大幅值差。目标采样足够并不保证所有候选曲线的窄峰都被训练频点捕捉。','## 双目标选解敏感性',sensitivity,'primary为前沿最小本条件WSMSE，balanced为min max(M_A,αM_θ)。两种规则来自相同运行，不是两组独立seed。',f'[返回主报告]({link(OUT/"REPORT.md")})'])
    summary=' '.join(improvement)
    ag=lambda base,key:next(x for x in aggregate if x['cohort']=='gap8' and x['group']=='dual_mse' and x['baseline']==base and x['metric']==key)
    ma,mp=ag('original','amplitude_rmse_db'),ag('original','phase_rmse_deg');ip=ag('dense_interpolated','phase_rmse_deg')
    dual_interpretation=f"本轮以MAG/PHA双目标为重点看：8个缺口电路等权汇总，真值加密相比原采样，幅值RMSE变化{n(ma['equal_circuit_mean_difference'])}dB（95%区间[{n(ma['ci95'][0])}, {n(ma['ci95'][1])}]），相位RMSE变化{n(mp['equal_circuit_mean_difference'])}°（[{n(mp['ci95'][0])}, {n(mp['ci95'][1])}]），体现幅相取舍。相比频点/权重相同的插值加密组，相位RMSE变化{n(ip['equal_circuit_mean_difference'])}°（[{n(ip['ci95'][0])}, {n(ip['ci95'][1])}]）；补入真实目标细节有相位收益，但不等于相对原实验整体占优。"
    txt=['# 加密采样能否改善阻抗曲线拟合',
      '**目标的采样缺口已经修复；优化收益需要按目标和电路分别判断。** '+summary,
      '10个解析目标、280个配对seed块。新增1680次运行全部完成；复用840次原粗采样运行。后台比较2520条主要选解及840条双目标均衡选解，仅绘制3个典型seed。Case2没有测点之间的目标真值，本轮不对它生成伪加密数据。未更新网站。',
      '## 1. 本轮只改变采样条件',
      '三种采样：①原点等权，复用既有选解；②加密频点但目标仅由旧点dB幅值/相位插值；③同一加密频点上的解析目标真值。后两组的频点、权重、算法、初始化和预算完全一致，区别只有目标细节。每种条件运行WSMSE、原Ω距离、幅相双MSE三组。',
      '新增频点按原log f单元作梯形积分，每个原单元总权重保持1/N；不会因窄峰附近点多就提高该单元的总权重。因此原采样→真值加密包含了候选检查与离散积分的变化，而插值加密→真值加密更直接检查“补入真实目标结构”的作用。训练损失跨采样条件不能直接比大小，所有选解须在共同解析网格上复评。',
      '## 2. 采样本身是否修复',grid_table,
      '全部新网格通过另一档目标加密网格检查，幅值最大折线误差<0.2dB、相位<1°。Case1在约25.23kHz原来漏掉的约3kΩ窄峰已被表达。网格由目标信息准备，未用任何拟合结果；目标真参数用于合成目标查询和网格准备，没有用于初始化、收缩边界或优化选解。这是理想可取得目标细节条件下的实验，不验证测量仪器如何自动找到窄峰。',
      '## 3. Case1：所有100个seed的选解',
      '下面每个seed都在包含目标和全部拟合模态的共同解析网格上计算，再对seed平均。RMSE、Ω距离按log f积分；峰谷阈值1dB、相位事件5°，同类型在0.25 decade内配对。数值为每条曲线的平均事件数，不是物理谐振数量的证明。',case_table,
      ' '.join(case_summary),case_pairs,
      '表中差值为真值加密组减对照，负值表示误差降低；区间跨零时，当前seeds不能确认稳定改善方向。幅值、相位、漏失和局部误差可能出现取舍，不能只取一个改善的诊断宣布所有结构被修复。',
      '## 4. 跨电路验证与仍需解决的结构',dual_interpretation,summary,
      f'[全部10个电路的比较、候选隐藏窄峰及双目标选解敏感性]({link(OUT/"DETAILS.md")})',
      '8个缺口目标是按上一轮采样诊断提前选定；2个对照原采样缺口都约0.002dB，用来观察即便没有缺失目标结构，增加候选检查点与改变离散积分是否也影响搜索。跨电路摘要先在每电路平均seed，再等权平均电路；共享的20个seed跨电路联动抽样，Case1其余80个seed另作一层。其区间条件于这10个电路，不证明所有电路的普遍规律。',
      '目标采样充分与候选检查充分是两个问题：目标固定网格仍可能漏过优化器生成的极窄拟合峰，需查看附表的候选折线缺口和新增事件。如果改善集中在某些峰位/宽度或频带，应在这些持续缺陷上设计下一轮候选指标；不能只因为目标折线更准确就认定300代优化一定更好。',
      '频点表达可靠与特征在平均误差中的权重足够，也是两个问题。Case1约25.23kHz窄峰对应单元的半功率宽度约13.33Hz；当前冻结网格在该核心区域有10个点，总积分权重仅约0.00575%。这个描述来自目标网络，不依赖优化结果。补到峰不等于把峰作为单独特征等权惩罚；若仍漏峰，应检查局部结构权重与搜索困难，而不是继续无区别地堆频点。',
      '## 5. 三个典型例子',
      '按预登记规则，Case1、最大缺口自设计two_parallel_units_c3及采样充分对照rlc_bridge_c2各取一例。每电路按三种采样×三种目标的原幅相MSE/Ω距离共27项seed内平均秩，选最接近中位平均秩的seed；精确并列取较小seed。',
      '\n'.join('- '+x for x in examples),
      '## 6. 核验、预算与结论边界',
      f"1680次新运行、{config['planned_evaluations']:,}次评价，服务器正式耗时{status['elapsed_seconds']/60:.2f}分钟。Case1人口200、100seeds；其余人口120、各20seeds；均300代。所有三组保持log10编码、NSGA-II、SBX(prob=0.9, eta=1)、PM(prob=1, eta=1)、变异概率、α和参数边界不变。相同seed的初始种群与三组旧运行逐项一致。候选评价次数相同，但新增频点使每次评价计算量更高，不是相同墙钟时间或浮点运算预算。",
      f"独立电路/误差公式重算{validation['final_individuals_recomputed']:,}个末代个体，核对训练指标、原点评分、非支配前沿、选解、预算、每代历史、哈希和参数边界。旧选解保留，不按新的检查指标重新选择旧组。全部{validation['grid_rows_checked']}条解析诊断作网格敏感性检查，其中{validation['grid_unstable_rows']}条不稳定。",
      '网格检查先比较1/2档；若任何主要或均衡选解不稳定，整块seed继续比较2/4、4/8档，不改优化结果。积分相对变化限1%（低误差采用幅值0.1dB、相位0.5°、Ω距离1e−8Ω数值底值），幅相漏失/新增事件数需一致。不稳定项保留标记，不能用它们支持精确的解析结构优势。',
      '所有目标仍有结构正确、真值在边界内的零误差解。此处比较固定300代的搜索结果，不证明各目标全局最优不同。Bootstrap共10,000次，单位为电路内配对seed，RNG=2026100711；区间未作多重比较校正。峰位和宽度只对匹配事件统计，缺失事件未当成零偏差。',
      '\n'.join(f'- [{label}]({link(path)})' for label,path in [('运行前方案',HERE/'PROTOCOL.md'),('分析实现说明',HERE/'ANALYSIS_NOTES.md'),('目标细节和窄峰权重',HERE/'TARGET_INFORMATION.json'),('网格准备与输入哈希',HERE/'PREPARATION.json'),('所有统计与区间JSON',OUT/'SUMMARY.json'),('逐seed原点/训练指标',OUT/'selected_metrics.csv'),('逐seed解析诊断',OUT/'per_seed_diagnostics.csv'),('所有频带数据',OUT/'band_diagnostics.csv'),('网格核验',OUT/'grid_refinement.csv'),('完整复算说明',HERE/'README.md'),('独立核验记录',OUT/'VALIDATION.json'),('服务器归档哈希',HERE/'server_run/ARCHIVES.json')])]
    write(OUT/'REPORT.md',txt);dump=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');dump(OUT/'DECISIONS.json',decision);files=[HERE/'analyze.py',HERE/'report.py',HERE/'ANALYSIS_NOTES.md',HERE/'TARGET_INFORMATION.json',ROOT/'experiments/seed_replication_20261007/analyze.py',ROOT/'experiments/optimized_curve_structure_20261007/diagnostics.py',ROOT/'experiments/optimized_curve_structure_20261007/refined_grid.py'];dump(OUT/'ANALYSIS_SOURCES.json',{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files});print(json.dumps(dict(report=link(OUT/'REPORT.md'),summary=summary,case1=case_summary,unstable=s['grid_unstable_rows']),ensure_ascii=False,indent=2))
if __name__=='__main__':main()

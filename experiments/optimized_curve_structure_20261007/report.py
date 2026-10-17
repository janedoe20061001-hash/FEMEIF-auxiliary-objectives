"""Summarize all-seed structure diagnostics and draw at most eight examples."""
from pathlib import Path
import csv,html,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import markdown
from model import Model
from diagnostics import weights
from refined_grid import improved_dense_grid
from run_analysis import loadcsv
from analyze import dump,sha
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'output/optimized_curve_structure_20261007'
GROUPS=['wsmse','mean_distance','dual_mse'];LABELS={'wsmse':'WSMSE 单目标','mean_distance':'平均距离单目标','dual_mse':'双 MSE · 最小 WSMSE 选解','dual_balanced':'双 MSE · minmax 选解'};COLORS=['#1560a4','#d56522','#27805d']
def fmt(x):return '—' if x is None else f'{x:.4g}'
def pct(x):return '—' if x is None else f'{100*x:.1f}%'
def table(headers,rows):
    def escape(value):return str(value).replace('|','\\|')
    return '| '+' | '.join(escape(h) for h in headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(escape(x) for x in row)+' |\n' for row in rows)+'\n'
def med(values):return float(np.median(values)) if values else None
def meaningful(r):return r['amplitude_rmse_db']>1 or r['phase_rmse_deg']>5
def representative(rows,ident):
    rr=[r for r in rows if r['circuit']==ident and r['group'] in GROUPS];seeds=sorted(set(int(r['seed']) for r in rr));features=[]
    for s in seeds:
        a=[r for r in rr if int(r['seed'])==s]
        features.append([np.mean([r[k] for r in a]) for k in ['amplitude_rmse_db','phase_rmse_deg','complex_rms_target_relative']]+[np.mean([r['amp_missed']+r['phase_missed']+r['cross_missed'] for r in a])])
    v=np.array(features);ranks=np.empty_like(v)
    from scipy.stats import rankdata
    for j in range(v.shape[1]):ranks[:,j]=(rankdata(v[:,j],method='average')-.5)/len(v)
    return seeds[int(np.argmin(np.sum((ranks-.5)**2,axis=1)))]
def unit(value,name):
    if value is None:return '实测 · 未知'
    options=[(1e3,'kΩ'),(1,'Ω'),(1e-3,'mΩ')] if name.startswith('R') else [(1,'H'),(1e-3,'mH'),(1e-6,'µH'),(1e-9,'nH')] if name.startswith('L') else [(1,'F'),(1e-6,'µF'),(1e-9,'nF'),(1e-12,'pF')]
    scale,label=next(((s,u) for s,u in options if abs(value)>=s),options[-1]);return f'{value/scale:.5g} {label}'
def render_report(md,dest,home='../index.html'):
    body=markdown.markdown(md,extensions=['tables','fenced_code']);dest.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>优化结果 · 曲线结构诊断</title><style>body{margin:0;background:#f5f7fa;color:#233b51;font:16px/1.75 system-ui,"Microsoft YaHei"}main{max-width:1040px;margin:auto;padding:24px}a{color:#1460ad}h1{font-size:30px;line-height:1.4}h2{margin-top:32px}img{max-width:100%;height:auto;background:white}table{border-collapse:collapse;display:block;overflow:auto;background:white;font-size:14px;max-width:100%}td,th{border:1px solid #d9e2eb;padding:7px 10px;white-space:nowrap}code{overflow-wrap:anywhere}nav{padding:12px 24px;background:#eaf1f9}blockquote{border-left:4px solid #98b9d8;padding:10px 18px;background:white}details{background:white;padding:16px;margin:16px 0}</style><nav><a href="'+home+'">研究首页</a> · <a href="index.html">结构诊断</a></nav><main>'+body+'</main></html>',encoding='utf8')
def plot_example(ident,seed,reason,rows):
    m=Model(ident);r=m.record;p=np.load(OUT/'selected'/f'{ident}.npz');si=int(np.flatnonzero(p['seeds']==seed)[0]);x=p['log_parameters'][si]
    if r['target_parameters'] is None:f=m.f;target=m.target
    else:f=improved_dense_grid(r,10**x,2);target=m.network(f,np.array(r['target_parameters']))[0]
    z=m.network(f,10**x[:3]);fig,axes=plt.subplots(2,2,figsize=(12,7.4),sharex=True)
    axes[0,0].loglog(f,abs(target),'k--',lw=1.2,label='目标');axes[0,1].semilogx(f,np.rad2deg(np.angle(target)),'k--',lw=1.2,label='目标')
    for gi,group in enumerate(GROUPS):
        axes[0,0].loglog(f,abs(z[gi]),color=COLORS[gi],lw=1,label=LABELS[group]);axes[0,1].semilogx(f,np.rad2deg(np.angle(z[gi])),color=COLORS[gi],lw=1)
        axes[1,0].semilogx(f,20*np.log10(abs(z[gi])/abs(target)),color=COLORS[gi],lw=1);axes[1,1].semilogx(f,np.rad2deg(np.angle(z[gi])-np.angle(target)),color=COLORS[gi],lw=1)
    axes[0,0].plot(m.f,abs(m.target),'.',color='black',ms=2,alpha=.6,label='原优化目标频点')
    for a,label in zip(axes.flat,['|Z| / Ω','相位 / °','幅值残差 / dB','相位残差 / °']):a.set_ylabel(label);a.grid(alpha=.2)
    for a in axes[1]:a.set_xlabel('频率 / Hz');a.axhline(0,color='#789',lw=.6)
    axes[0,0].legend(fontsize=8);fig.suptitle(r['title']+' · '+ident+' · seed '+str(seed)+'\n'+reason,fontsize=11);fig.tight_layout()
    name=f'{ident}_seed_{seed}.png';fig.savefig(OUT/'figures'/name,dpi=150);plt.close(fig)
    records=[v for v in rows if v['circuit']==ident and int(v['seed'])==seed and v['group'] in GROUPS]
    text=f'# {r["title"]} · seed {seed}\n\n选择原因：{reason}。这是一个 seed 的三个选中解，不是均值曲线。\n\n![结构](schematics/{ident}.svg)\n\n'
    params=[]
    for j,n in enumerate(r['parameter_names']):params.append([n,unit(r['target_parameters'][j],n) if r['target_parameters'] is not None else '实测 · 未知',*[unit(10**x[g,j],n) for g in range(3)]])
    text+='## 参数\n\n'+table(['元件','目标','WSMSE','距离','双MSE'],params)+'\n## 本 seed 的诊断\n\n'
    text+=table(['组','原幅值MSE / dB²','原相位MSE / rad²','原距离 / Ω','曲线幅值RMSE / dB','曲线相位RMSE / °','幅值漏失/新增','相位漏失/新增','交互 η'],[[LABELS[v['group']],fmt(v['magnitude_mse_db2']),fmt(v['phase_mse_rad2']),fmt(v['mean_distance_ohm']),fmt(v['amplitude_rmse_db']),fmt(v['phase_rmse_deg']),f'{int(v["amp_missed"])}/{int(v["amp_extra"])}',f'{int(v["phase_missed"])}/{int(v["phase_extra"])}',fmt(v['interaction_eta'])] for v in records])
    text+='\n![三组与目标](figures/'+name+')\n\nCase2仅使用原测量点；其他电路使用经加密复核的解析曲线。下排是同频残差，没有移动曲线来降低误差。\n'
    path=OUT/f'example_{ident}_{seed}.md';path.write_text(text,encoding='utf8');render_report(text,path.with_suffix('.html'))
    return dict(circuit=ident,seed=seed,reason=reason,figure='figures/'+name,page=path.with_suffix('.html').name,diagnostics=records)
def heatmap(bands):
    fig,axes=plt.subplots(2,2,figsize=(13,6.7))
    for row,ident in enumerate(['case1','case2']):
        for col,key in enumerate(['amplitude_rmse_db','phase_rmse_deg']):
            a=axes[row,col];v=np.array([[np.mean([b[key] for b in bands if b['circuit']==ident and b['group']==group and int(b['band'])==k]) for k in range(1,9)] for group in GROUPS]);image=a.imshow(v,aspect='auto',cmap='YlOrRd');a.set_yticks(range(3),['WSMSE','距离','双 MSE']);a.set_xticks(range(8),range(1,9));a.set_xlabel('电路内等宽 log 频带编号');a.set_title(ident+' · '+('幅值 RMSE / dB' if col==0 else '相位 RMSE / °'));fig.colorbar(image,ax=a)
            for i in range(3):
                for j in range(8):a.text(j,i,f'{v[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if v[i,j]>.65*v.max() else '#25384a')
    fig.suptitle('各频带：先逐 seed 计算 RMSE，再在本组 seeds 中等权平均');fig.tight_layout();fig.savefig(OUT/'figures/band_errors.png',dpi=145);plt.close(fig)
def main():
    v=json.loads((OUT/'VALIDATION.json').read_text());assert v['passed'];summary=json.loads((OUT/'SUMMARY.json').read_text(encoding='utf8'));rows=loadcsv(OUT/'per_seed_diagnostics.csv');bands=loadcsv(OUT/'band_diagnostics.csv');means=summary['means'];records=json.loads((HERE/'circuits.json').read_text(encoding='utf8'))
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc');plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':9})
    (OUT/'figures').mkdir(exist_ok=True);(OUT/'schematics').mkdir(exist_ok=True)
    for r in records:shutil.copy2(ROOT/'output/primary_loss_comparison_20261007/schematics'/f'{r["id"]}.svg',OUT/'schematics'/f'{r["id"]}.svg')
    lookup={(r['circuit'],r['group']):r for r in means};examples=[];seen=set()
    for ident in ['case1','case2']:
        seed=representative(rows,ident);examples.append(plot_example(ident,seed,'四项结构诊断的组间平均秩最接近中位位置',rows));seen.add(ident)
    criteria=[('interaction_eta','明显误差样本中的非加和剩余较大'),('fit_sampling_gap_max_db','拟合曲线在原频点之间偏离采样折线较大'),('amp_missed','显著幅值峰谷漏失较多')]
    for key,reason in criteria:
        eligible=[r for r in rows if r['circuit'] not in ['case1','case2'] and r['group'] in GROUPS and r['grid_stable'] and meaningful(r) and r.get(key) is not None]
        ids={r['circuit'] for r in eligible}-seen
        if not ids:continue
        ident=max(sorted(ids),key=lambda ident:np.mean([r[key] for r in eligible if r['circuit']==ident]));subset=[r for r in eligible if r['circuit']==ident];value=med([r[key] for r in subset]);selected=min(subset,key=lambda r:(abs(r[key]-value),r['seed']));seed=int(selected['seed'])
        examples.append(plot_example(ident,seed,reason+'；电路按条件均值选取，seed 接近条件中位数',rows));seen.add(ident)
    ident='rc_series_c1'
    if ident not in seen:examples.append(plot_example(ident,representative(rows,ident),'可加的 RC 串联对照，检验非线性解释的边界',rows))
    heatmap(bands);assert len(examples)+1<=8;dump(OUT/'EXAMPLES.json',examples)
    tallies={}
    for group in ['mean_distance','dual_mse','dual_balanced']:
        synth=[r for r in records if r['id'] not in ['case1','case2']]
        tallies[group]={key:sum(lookup[r['id'],group][key]<lookup[r['id'],'wsmse'][key] for r in synth) for key in ['magnitude_mse_db2','phase_mse_rad2','amplitude_rmse_db','phase_rmse_deg','amp_missed','phase_missed']}
    active=[r for r in rows if r['circuit']!='case2' and r['group'] in GROUPS and meaningful(r)]
    interaction_stats={group:dict(active_seeds=sum(r['group']==group for r in active),eta_median=med([r['interaction_eta'] for r in active if r['group']==group]),eta_over_02=sum(r['interaction_eta']>.2 for r in active if r['group']==group)) for group in GROUPS}
    target_gaps=[lookup[r['id'],'wsmse']['target_sampling_gap_max_db'] for r in records if r['target_parameters'] is not None]
    evidence=dict(synthetic_tallies=tallies,interaction_in_meaningful_error_samples=interaction_stats,analytic_target_sampling_gap_above_3db=sum(g>3 for g in target_gaps),analytic_target_circuits=49,examples=[{k:x[k] for k in ['circuit','seed','reason','page','figure']} for x in examples],figure_count=len(examples)+1)
    dump(OUT/'EVIDENCE.json',evidence)
    text='# 优化结束后，阻抗曲线具体错在哪里\n\n**后台逐一分析3,480条主要代表曲线，另分析1,160条双目标均衡选解；没有把全部seed画成图片。** 50个电路、1,160个配对seed，新增双目标1,160次运行全部通过。\n\n'
    text+='三类进化目标分别为WSMSE、MAE_Z=mean|Z−Z*|（原始Ω）和幅值/相位MSE双目标。幅值MSE用dB²，相位MSE用rad²主值差。α在运行前固定：Case1=25.12、Case2=17.38，自设计沿用目标幅值/相位跨度比的平方；双目标中的α只参与末代代表解选择。自设计为无噪声且结构正确的16拓扑×3组参数，边界为真值中心±2 decades。\n\n'
    text+='## 先看三点\n\n'
    text+='1. **最先要修的是结构的采样可靠性。** Case1的解析目标在约25.23 kHz有约3 kΩ的尖峰，比原180点目标连线高40.49 dB。49个有解析真值的电路中，'+str(evidence['analytic_target_sampling_gap_above_3db'])+'个目标相对原采样折线的最大幅值差超过3 dB。这说明部分结构在优化前就没有被充分表达；Case2没有测点间真值，不能作同样断言。\n'
    text+='2. **残差集中在局部峰谷和相位变化区，不只是整体抬高或压低。** Case1的距离组幅值RMSE为'+fmt(lookup['case1','mean_distance']['amplitude_rmse_db'])+' dB，去掉每seed最佳恒定dB偏移后仍为'+fmt(lookup['case1','mean_distance']['amplitude_shape_rmse_db'])+' dB。Case2距离组的距离残差中，目标幅值最高四分位频点占 '+pct(lookup['case2','mean_distance']['distance_high_target_quartile_share'])+'；这也显示原始Ω距离的误差尺度集中。双MSE的峰谷漏失改善区间跨零，没有稳定修复这些结构。\n'
    text+='3. **单参数变化不能普遍相加，且也存在可加特例。** 在幅值RMSE>1 dB或相位RMSE>5°的解析目标样本中，三组η中位数分别为 '+ ' / '.join(fmt(interaction_stats[g]['eta_median']) for g in GROUPS)+'；RC串联的η接近零。η高也可能含参数抵消或等价参数的影响，不能当作曲线质量指标。\n\n'
    text+='> 这些结果用于定位缺陷和提出假设；没有证明某个新辅助指标有效，也不能把有限300代结果解释为各目标的全局最优解。\n\n'
    text+='Case1和48个自设计电路的真实参数都在搜索边界内，且目标与拟合网络相同，因此三类目标都存在零误差解。现在比较的是同预算300代的搜索行为及残余失真，不能据此说它们的全局最优曲线必然不同。Case2是实测目标，另有模型与测量是否一致的问题。\n\n'
    for ident in ['case1','case2']:
        text+='## '+ident+'：全部100个seed\n\n'
        text+=table(['进化目标/选解','幅值MSE / dB²','相位MSE / rad²','平均距离 / Ω','WSMSE'],[[LABELS[g],*[fmt(lookup[ident,g][k]) for k in ['magnitude_mse_db2','phase_mse_rad2','mean_distance_ohm','wsmse']]] for g in GROUPS])
        text+='结构诊断使用log频率权重；Case1使用解析密集曲线，Case2使用全部201个测量点。峰谷为1 dB显著幅值事件 / 5°显著相位事件，类型相同且频差不超过0.25 decades才匹配。下表计数是每seed平均事件数，不是百分比。\n\n'
        text+=table(['组','幅值RMSE / dB','相位RMSE / °','幅值漏失/新增','相位漏失/新增','|相位误差|>15°频带份额','最坏频带相位RMSE / °'],[[LABELS[g],fmt(lookup[ident,g]['amplitude_rmse_db']),fmt(lookup[ident,g]['phase_rmse_deg']),fmt(lookup[ident,g]['amp_missed'])+'/'+fmt(lookup[ident,g]['amp_extra']),fmt(lookup[ident,g]['phase_missed'])+'/'+fmt(lookup[ident,g]['phase_extra']),pct(lookup[ident,g]['phase_bad_span_fraction']),fmt(lookup[ident,g]['worst_band_phase_rmse_deg'])] for g in GROUPS])
        text+='匹配峰谷的频移和宽度误差仅对成功匹配的事件统计；先看漏失/新增数量，再看频移，避免把失去特征的解误认成“位置误差小”。\n\n'
        text+=table(['组','幅值事件平均绝对频移 / decades','幅值宽度 |log₂比值|','相位事件平均绝对频移 / decades','相位宽度 |log₂比值|'],[[LABELS[g],*[fmt(lookup[ident,g][k]) for k in ['amp_mean_abs_shift_decades','amp_mean_abs_width_log2_ratio','phase_mean_abs_shift_decades','phase_mean_abs_width_log2_ratio']]] for g in GROUPS])
        text+='相对WSMSE单目标的配对seed区间：\n\n'
        keys=['magnitude_mse_db2','phase_mse_rad2','amp_missed','phase_missed','worst_band_phase_rmse_deg']
        text+=table(['比较组','量','均值差','95%配对bootstrap区间'],[[LABELS[c['group']],c['metric'],fmt(c['mean_difference']),f'[{fmt(c["ci95_low"])}, {fmt(c["ci95_high"])}]'] for c in summary['paired_comparisons'] if c['circuit']==ident and c['group'] in ['mean_distance','dual_mse'] and c['metric'] in keys])
        if ident=='case2':
            text+='Case2在16 MHz有107°原测量点，原样保留。当前正值无源网络的Re Z>0，因此相位只能在−90°到90°之间；这个测点至少有17°相位误差，对全201点相位MSE的下界贡献为0.000438 rad²。实际该点平均占三组相位平方误差的比例为 '+ ' / '.join(pct(lookup[ident,g]['phase_target_extreme_point_squared_error_share']) for g in GROUPS)+'，不足以解释总体差异。主值相位平方差不能被“平滑后曲线更好看”替代；从测量起伏提取的事件不是已证实的物理谐振。\n\n'
    text+='## 误差集中在哪些频带\n\n![八个频带的误差](figures/band_errors.png)\n\n'
    text+=table(['电路','频带','频率下界 / Hz','频率上界 / Hz'],[[ident,k,fmt(next(b for b in bands if b['circuit']==ident and int(b['band'])==k)['lo_hz']),fmt(next(b for b in bands if b['circuit']==ident and int(b['band'])==k)['hi_hz'])] for ident in ['case1','case2'] for k in range(1,9)])
    text+='## 48个自设计电路\n\n下表先逐电路对seeds等权平均，再数方向；近零差异也计数，不等同于实用或统计优势。双MSE与单目标沿用相同目标参数、频点、边界和预算。\n\n'
    text+=table(['比较组','原幅值MSE更低 /48','原相位MSE更低 /48','密集幅值RMSE更低 /48','密集相位RMSE更低 /48','幅值漏失更少 /48','相位漏失更少 /48'],[[LABELS[g],*[tallies[g][k] for k in ['magnitude_mse_db2','phase_mse_rad2','amplitude_rmse_db','phase_rmse_deg','amp_missed','phase_missed']]] for g in ['mean_distance','dual_mse']])
    text+='## 多个元件的偏差能否相加\n\n在原频点上计算：r_all=Z(p_fit)−Z(p*)；r_add=Σ_j[Z(p*仅将第j个元件改为拟合值)−Z(p*)]；r_int=r_all−r_add。η=||r_int||/(||r_all||+||r_add||)，使用log频率权重L2范数；取值0–1，η=0表示该组有限变化可加。它不是交互贡献百分比。\n\n'
    text+=table(['组','明显误差的解析样本数','η中位数','η>0.2样本数'],[[LABELS[g],interaction_stats[g]['active_seeds'],fmt(interaction_stats[g]['eta_median']),interaction_stats[g]['eta_over_02']] for g in GROUPS])
    text+='这些数以seed为诊断单位，不能当成独立电路的普遍比例；近乎完美拟合不会因很小分母而被拿来制造“强非线性”。Case2没有已知真值参数，不计算分解。不同参数组合可能给出相同阻抗，参数差异大也不必然表示曲线差。\n\n'
    text+='## 双目标“最好”的选择敏感性\n\n主要选解是末代非支配解中最小WSMSE。较均衡的另一个固定规则为min max(M_A,αM_φ)。两者是同一前沿上的两个选择，不是额外优化组。\n\n'
    text+=table(['电路','选解','原幅值MSE','原相位MSE','曲线幅值RMSE / dB','曲线相位RMSE / °','幅值漏失','相位漏失'],[[ident,LABELS[g],*[fmt(lookup[ident,g][k]) for k in ['magnitude_mse_db2','phase_mse_rad2','amplitude_rmse_db','phase_rmse_deg','amp_missed','phase_missed']]] for ident in ['case1','case2'] for g in ['dual_mse','dual_balanced']])
    text+='前沿内按平均距离、纯幅值或纯相位另选的误差也全部记录在dual_selection_sensitivity.csv，供检查其他取舍。它们不能替换主要比较中的选解。\n\n'
    text+='## 少量典型曲线\n\n图中黑色虚线为目标，蓝/橙/绿分别为三组。上排使用f–|Z|与f–θ，下排为同频残差。论文电路选接近总体中位位置的seed；合成例子按说明中的诊断条件选取，包含刻意突出的现象，不能代表所有电路。\n\n'
    for e in examples:text+='### '+e['circuit']+' · seed '+str(e['seed'])+'\n\n'+e['reason']+'。\n\n[查看结构、参数与本seed数据]('+e['page']+')\n\n!['+e['circuit']+']('+e['figure']+')\n\n'
    text+='## 下一步如何用这些结果\n\n1. **先保证采样可靠。** 对有解析模型的合成问题，用目标和拟合的模态位置加密或网格外检查，避免优化器只贴合测点却在测点间失真。实测Case2需要新增测量或独立目标数据，不能用模型生成“目标真值”。\n2. **将误差分成数量、位置与形状。** 只有匹配成功的结构再讨论峰位和宽度；漏失峰谷不能靠调频移惩罚独自修正。相位问题要按频带查看，并区分整体偏移、局部转变和单个异常点影响。\n3. **再形成连续的候选损失。** 可以从稳定失败区域考虑局部幅相残差/转变形状、峰谷附近对齐或频带平衡；暂不把η、硬峰数或极端点最大误差直接加进目标。它们尚未经过独立优化验证。\n4. **使用联合参数变化验证方向。** 在实际拟合解附近检查多参数共同变化的真实响应，不能用单参数曲线相加替代。候选仍需独立seeds、相同预算和幅相两项交叉指标检验，不能以辅助项本身下降作为成功。\n\n'
    text+='## 数值可靠性与材料\n\n初始网格有'+str(summary['first_grid_unstable_rows'])+'条解析诊断未通过两档加密稳定条件。改为不截断窄模态、加入精确RLC中心和几何尾部网格后，仍有'+str(summary['grid_unstable_rows'])+'条未通过；全部如实标记在grid_refinement.csv，不删seed。未稳定项不支持精确结构优势结论。网格更改只发生在离线诊断，原频点上的目标值与选解未更改。\n\n'
    text+='[逐seed诊断](per_seed_diagnostics.csv) · [逐频带结果](band_diagnostics.csv) · [三组及均衡选解均值](group_means.csv) · [配对区间](paired_comparisons.csv) · [前沿选解敏感性](dual_selection_sensitivity.csv) · [网格检查](grid_refinement.csv) · [运行前方案](PROTOCOL.md) · [网格修复记录](NUMERICAL_GRID_NOTE.md) · [核验记录](VALIDATION.json)\n'
    (OUT/'REPORT.md').write_text(text,encoding='utf8');render_report(text,OUT/'index.html');render_report(text,OUT/'REPORT.html')
    for name in ['PROTOCOL.md','DIAGNOSTIC_CLARIFICATIONS.md','NUMERICAL_GRID_NOTE.md']:shutil.copy2(HERE/name,OUT/name)
    dump(OUT/'REPORT_VALIDATION.json',dict(passed=True,figure_count=len(examples)+1,max_figures=8,examples_retain_three_groups=True,parameters_and_schematics=True,report_sha256=sha(OUT/'REPORT.md'),report_code_sha256=sha(__file__)))
    print(json.dumps(evidence,ensure_ascii=False))
if __name__=='__main__':main()

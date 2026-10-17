"""Keep baseline, parent and combination comparisons distinct after full audit."""
from pathlib import Path
import json,sys
import numpy as np
root=Path(__file__).resolve().parents[1];run=root/'experiments/curve3d_directional_20261003'
audit=json.loads((run/'server_run/download_audit.json').read_text());assert audit['status']=='passed'
rows=json.loads((run/'RESULTS.json').read_text(encoding='utf-8'))
parents=json.loads((run/'PARENT_COMPARISONS.json').read_text(encoding='utf-8'))
checks=json.loads((run/'software_checks.json').read_text())
old=json.loads((root/'experiments/curve3d_refinement_20261003/RESULTS.json').read_text(encoding='utf-8'))
base={(r['case'],r['id']):r for r in rows}
direct={(r['case'],r['id'],r['parent']):r for r in parents}
codes=['Q-05-B','Q-05-A','Q-05-D','Q-COMB']
def supported(r):
    return all(r['reduction_pct'][k]>0 and r['reduction_pct_bootstrap95'][k][0]>0
               and r['holm_adjusted_p_44_primary_comparisons'][k]<.05 for k in range(2))
def mean_positive(r):return all(r['reduction_pct'][k]>0 for k in range(2))
baseline_supported=[code for code in codes if all(supported(base[(case,code)]) for case in ['case1','case2'])]
increment_supported=[];increment_mean=[]
for code in codes:
    relevant=[r for r in parents if r['id']==code]
    if all(mean_positive(r) for r in relevant):increment_mean.append(code)
    if all(supported(r) for r in relevant) and code in baseline_supported:increment_supported.append(code)
text=['# 第四轮：继续优化有没有收益，以及是否应收窄指标','',
    '## 结论的判断口径','',
    '本轮完成 1,400 次正式运行，73,500,000 次评估；每 Case 7 组，各 100 个新配对种子（200–299）。三个单辅助修改、一个双辅助组合，以及两个原累计指标和基线均使用原完整预算。',
    '不能只看“比基线下降”；新单辅助还要对累计弧长原版比较，双辅助组合要对两个单辅助分别比较。所有 44 项主误差检验统一修正，不跨轮挑最好降幅。','',
    '- 两个 Case 的两项主误差相对基线均有改善支持的新组：'+('、'.join(baseline_supported) or '无')+'。',
    '- 两个 Case 的两项主误差相对对应父组均值均下降的新组：'+('、'.join(increment_mean) or '无')+'；这是均值方向，不是明确改善证明。',
    '- 同时对基线和所有对应父组，跨 Case 双主误差均有改善支持的新组：'+('、'.join(increment_supported) or '无')+'。','']
if increment_supported:
    text.append('本轮存在有证据的进一步改善，建议优先保留上述方向；仍不能把新随机种子验证当作独立测量数据验证。')
else:
    text.append('**本轮尚未证明某一改法或组合在两个 Case 的两项误差上都进一步优于对应原方案。** 有个别改善时应按 Case、主误差和比较对象分别说，不宣布所有新指标都有效。')
text+=['','## 当前取舍建议','',
    '- 优先复核 Q-05-B 两端对称累计弧长：改法简单、理由明确，两个 Case 对原累计弧长的两项均值方向都改善；但本轮直接对原版的 Holm 检验均未给出明确支持，暂不宣布替代原版。此优先级是本轮探索结果后的判断，不是预先独立确认。',
    '- Q-05-D 方向分解作为次级候选：Case II 相对基线两项有支持，但对原累计弧长的增量仍不明确；不因为数学上更能区分曲线，就称更优。',
    '- Q-05-A 单纯分轴不继续优先：对原版没有显示明确收益，且相位／Case II 均值不如原版；这也说明弧长的结构盲点不一定就是优化效果欠佳的原因。',
    '- 本轮不采用 Q-COMB：Case II 对原累计弧长，幅值／相位 MSE 均值分别恶化 12.61%／10.91%；配对区间虽为负，44 项修正检验仍未达到两项都明确，不能宣称所有条件下必然更差。',
    '- 不继续无限增添公式或凭某一次均值挑最终目标。更合适的后续是收窄至原累计弧长与对称变体，做独立重复及新数据／已知噪声验证；同时检查原电路与测量的适配，而不是直接删点或改主目标。','']
text+=['','## 为什么不直接继续堆公式','',
    '下列是公式可直接指出的限制或待检验假设，不是已经证明的全部负结果原因。','',
    '|限制|实际含义|本轮对应检验|','|---|---|---|',
    '|长度是一个标量|幅值、相位变化可能互相补偿；相同长度不代表相同方向|Q-05-A 分轴；Q-05-D 分方向|',
    '|低频端单侧累计|靠近高频末端的偏差影响较短积分区间|Q-05-B 两端对称|',
    '|辅助目标未必帮助主误差|区分更多形状不等于优化更好，且多目标排序会改变|全部比较主误差；Q-COMB 直接检验两个辅助是否互补|',
    '|测量目标与电路可行域可能不一致|任何指标都不能让原正参数被动网络精确产生不可达的测量点|保留数据原样，单独报告限制，不擅自删点|',
    '|变化量对噪声敏感|细碎噪声也增加累计长度或总变化量|不宣称此次指标能去噪；没有任意平滑窗口|','',
    '方向分解并非完全新信息：它等价于总变化量差与相对起点位移差的组合（系数来自恒等式）；这部分与主误差有重合，公式说明中明确列出。','',
    '两种分轴指标先除目标范围、再除目标轴总变化量，范围因子在比值中消去，等价于直接在 dB／rad 上比较相对目标总变化量的差；不用新增轴比例参数，但两轴等相对重要性仍是约定。',
    '对称弧长也不是全新独立信息：当候选与目标总长完全相同时，反向误差是前向误差的相反数，因此 Q-05-B 与原 R-05-C 完全相同；它只改变存在总长偏差时两端的惩罚分配。','',
    '## 相对本轮基线的完整结果','',
    '|组|Case I 幅值／相位 MSE 下降|Case II 幅值／相位 MSE 下降|','|---|---:|---:|']
for code in ['R-03-C','R-05-C']+codes:
    a,b=base[('case1',code)],base[('case2',code)]
    text.append(f"|{code} {a['label']}|{a['reduction_pct'][0]:.2f}%／{a['reduction_pct'][1]:.2f}%|{b['reduction_pct'][0]:.2f}%／{b['reduction_pct'][1]:.2f}%|")
text+=['','百分比是 100 次运行的 MSE 均值比例下降；正数为改善，负数为变差，不是幅值／相位本身下降。两个主误差来自同一最终个体。','',
    '## 新组是否优于对应父组','',
    '|新组|比较对象|Case I 幅值／相位 MSE 下降|Case II 幅值／相位 MSE 下降|','|---|---|---:|---:|']
for code,parent in [('Q-05-B','R-05-C'),('Q-05-A','R-05-C'),('Q-05-D','R-05-C'),('Q-COMB','R-05-C'),('Q-COMB','R-03-C')]:
    a,b=direct[('case1',code,parent)],direct[('case2',code,parent)]
    text.append(f"|{code}|{parent}|{a['reduction_pct'][0]:.2f}%／{a['reduction_pct'][1]:.2f}%|{b['reduction_pct'][0]:.2f}%／{b['reduction_pct'][1]:.2f}%|")
text+=['','## 原累计指标在不同种子区块的表现','',
    '|指标|Case|上一轮 100–199 幅值／相位 MSE 下降|本轮 200–299 幅值／相位 MSE 下降|',
    '|---|---|---:|---:|']
for code in ['R-03-C','R-05-C']:
    for case in ['case1','case2']:
        previous=next(r for r in old if r['case']==case and r['id']==code);now=base[(case,code)]
        text.append(f"|{code}|{case}|{previous['reduction_pct'][0]:.2f}%／{previous['reduction_pct'][1]:.2f}%|{now['reduction_pct'][0]:.2f}%／{now['reduction_pct'][1]:.2f}%|")
text+=['','两个区块分别与自身基线比较，不把不对应的种子配对，不选择更好区块当总结果。','',
    '## 运行分布：均值之外同时报告中位数','',
    '|Case|组|幅值 MSE 均值／中位数（dB²）|相位 MSE 均值／中位数（rad²）|',
    '|---|---|---:|---:|']
for case in ['case1','case2']:
    selected=[r for r in rows if r['case']==case]
    representative=selected[0]
    b,m=representative['baseline_mean'],representative['baseline_median']
    text.append(f'|{case}|基线|{b[0]:.5g}／{m[0]:.5g}|{b[1]:.5g}／{m[1]:.5g}|')
    for r in selected:
        b,m=r['auxiliary_mean'],r['auxiliary_median']
        text.append(f"|{case}|{r['id']}|{b[0]:.5g}／{m[0]:.5g}|{b[1]:.5g}／{m[1]:.5g}|")
text+=['','中位数为补充描述，不用它替换预先规定的均值主表，也不按它重新挑最有利的统计口径。种子间的优化差异和两区块差异，均不等于独立测量数据上的泛化能力。','',
    '## 不确定性与修正检验','',
    '|比较对象|Case|组|幅值 MSE 下降的 95% 区间|相位 MSE 下降的 95% 区间|幅值／相位 Holm p|',
    '|---|---|---|---|---|---|']
for r in rows+parents:
    m,p=r['reduction_pct_bootstrap95'][:2];pm,pp=r['holm_adjusted_p_44_primary_comparisons']
    text.append(f"|{r.get('parent','基线')}|{r['case']}|{r['id']}|[{m[0]:.2f}%, {m[1]:.2f}%]|[{p[0]:.2f}%, {p[1]:.2f}%]|{pm:.4g}／{pp:.4g}|")
text+=['','10,000 次配对重采样均值比例区间未作同时覆盖校正；Wilcoxon 针对配对差异，未显著不等于等效。统计反映优化随机性，不包括新测量数据及噪声分布的变化。','',
    '## 目标曲线的几何诊断（未用来删点或调参）','']
for d in checks['target_only_diagnostics']:
    f1,f2=d['largest_segment_frequency_hz']
    text.append(f"- {d['case']}：目标三维总长度 {d['target_total_length']:.5g}；最大单段占全长 {100*d['largest_segment_length_share']:.2f}%，位于 {f1:.6g}–{f2:.6g} Hz。两投影总变化量（目标范围坐标）为 {d['target_axis_total_variations']}。")
text+=['','Case II 最大单段位于 16–17.2 MHz，附近包含已知主值相位跳变；这提示变化量型指标对它敏感的风险，但不证明它导致所有效果不佳。所有 201 原点保留，不改相位、不裁剪、不使用候选专属尺度。','',
    '### 可行域问题的定量检查：不是把负结果都归因于一个点','',
    r'当前每个并联块的导纳实部为 $a_k=1/R_{2k+2}+R_{2k+1}/[R_{2k+1}^2+(\omega L_{k+1})^2]>0$（$k=0,1,2$；$\omega=2\pi f$ 为角频率）；令 $b_k$ 为该块导纳虚部，其阻抗实部为 $a_k/(a_k^2+b_k^2)>0$，再加串联 $R_7>0$，所以网络实部严格为正，主值相位在 $(-\pi/2,\pi/2)$ 内。',
    r'因此原相位 MSE 有下界 $E_\phi\ge N^{-1}\sum_i[\max(|\phi_i^*|-\pi/2,0)]^2$。这只是下界，不保证电路可以同时达到。','']
folder=run/'server_run/raw_results/case2';manifest=json.loads((folder/'runs/manifest.json').read_text())
frequency=np.array(manifest['frequency_hz']);target=np.array(manifest['target_real'])+1j*np.array(manifest['target_imag'])
phase=np.angle(target);mask=abs(phase)>np.pi/2;bound=float(np.mean(np.maximum(abs(phase)-np.pi/2,0)**2))
reference=next(r for r in rows if r['case']=='case2')['baseline_mean'][1]
text.append(f'该下界为 {bound:.8g} rad²，是本轮基线相位 MSE 均值的 {100*bound/reference:.3f}%；目标超出模型相位域的频率为 {frequency[mask].tolist()} Hz。不能据这个下界推断该点的真实误差贡献，下面直接复算其贡献。')
text+=['','|Case II 组|超出模型相位域的原点，对相位 MSE 均值的贡献（rad²）|占该组相位 MSE 均值|','|---|---:|---:|']
sys.path.insert(0,str(folder));from case2 import impedance
for group in ['baseline']+[r['group'] for r in rows if r['case']=='case2']:
    components=[];totals=[]
    for seed in range(200,300):
        with np.load(folder/f'runs/{group}/seed_{seed:03d}/final_population.npz') as saved:
            z=impedance(frequency,saved['selected_parameters'])[0]
        residual=(np.angle(z)-phase)**2;components.append(float(residual[mask].sum()/len(phase)));totals.append(float(residual.mean()))
    component=float(np.mean(components));total=float(np.mean(totals))
    expected=reference if group=='baseline' else next(r for r in rows if r['case']=='case2' and r['group']==group)['auxiliary_mean'][1]
    np.testing.assert_allclose(total,expected,rtol=1e-9,atol=1e-11)
    label=next((r['id'] for r in rows if r['group']==group),'基线')
    text.append(f'|{label}|{component:.8g}|{100*component/total:.2f}%|')
text+=['','本轮该点的直接贡献约 3%–5%，不是总相位误差的大多数；但它可能改变其他频段的优化折衷，不能据此断言影响很小。比例由完整最终同一个体复算，是误差分解而非删点后实验或因果检验；没有修改任何训练数据、目标、统计主表或选解方法。较大贡献不能证明测量错误，较小数学下界也不能证明该点没有重要影响。','',
    '## 增加辅助目标后，优化中的区别（描述性诊断）','',
    '|Case|组|最后一代全目标第一层非支配解占比均值|原双主目标第一层非支配解占比均值|',
    '|---|---|---:|---:|']
for case in ['case1','case2']:
    groups=['baseline']+[r['group'] for r in rows if r['case']==case]
    population=200 if case=='case1' else 150
    for group in groups:
        records=[json.loads((run/'server_run/raw_results'/case/f'runs/{group}/seed_{s:03d}/summary.json').read_text()) for s in range(200,300)]
        full=sum(r['full_front_size'] for r in records)/100/population
        primary=sum(r['primary_front_size'] for r in records)/100/population
        label=next((r['id'] for r in rows if r['group']==group),'基线')
        text.append(f'|{case}|{label}|{100*full:.2f}%|{100*primary:.2f}%|')
text+=['','第一层占比高表示更多个体在全目标 Pareto 排序下互不支配；它与采用更多辅助目标后的排序变化有关，但这张末代描述表不能单独证明选择压力是效果变化的原因，也不以该占比代替拟合误差。','',
    '## 文件与审计','',
    f"源文件、配置与原实验非组别／非种子设置已核对；全部 {audit['independent_final_individuals']:,} 个最终个体独立复算通过，主误差最大绝对差 {audit['independent_primary_max_abs_difference']:.3g}，辅助误差最大绝对差 {audit['independent_aux_max_abs_difference']:.3g}。",
    '100 个同 seed 初始种群、300 代历史、实际评估数量、最终固定 WSMSE 选择、全部 22 个比较的均值／比例／区间／检验均已核对。',
    f"归档 SHA-256：`{audit['sha256']}`。",'',
    '- [详细公式与冻结方案](../experiments/curve3d_directional_20261003/README.md)',
    '- [完整结果](../experiments/curve3d_directional_20261003/RESULTS.md)',
    '- [独立复算审计](../experiments/curve3d_directional_20261003/server_run/download_audit.json)',
    '- [对基线结果图](../experiments/curve3d_directional_20261003/figures/baseline_intervals.png)',
    '- [对原方案结果图](../experiments/curve3d_directional_20261003/figures/parent_intervals.png)',
    '- [结构盲点数学示例](../experiments/curve3d_directional_20261003/figures/structural_examples.png)','',
    '未修改既有 PDF、讲稿与 HTML PPT；本轮先完成指标筛选，不把未证实的新项直接写成有效指标。']
output=root/'output/三维曲线指标_方向与组合优化验证_20261003.md'
output.write_text('\n'.join(text)+'\n',encoding='utf-8')
print(json.dumps(dict(report=str(output),baseline_supported=baseline_supported,increment_supported=increment_supported,
                     increment_mean_positive=increment_mean),ensure_ascii=True))

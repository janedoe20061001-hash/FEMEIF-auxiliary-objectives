"""Concise rational decisions after audited formal runs, retaining negative results."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[1];run=root/'experiments/curve3d_refinement_20261003'
audit=json.loads((run/'server_run/download_audit.json').read_text());assert audit['status']=='passed'
formula=(run/'README.md').read_text(encoding='utf-8')
formula=formula.replace('各自把总弧长归一化为 1，丢失总长度；某段偏差会改变后面的对应',
    '没有直接比较总长度及长度积累；某段偏差会改变后面的对应')
formula=formula.replace('## R-05-C：不用各自归一化隐藏总长度',
    '## R-05-C：显式比较实际累计长度')
formula=formula.replace('同时比较全长和长度沿频率的积累顺序，没有候选自己的长度归一化，没有频率扭曲。',
    '直接比较长度沿频率的积累过程，也会对全长不同作出反应，但不是另外加一个终点长度惩罚。没有候选自己的长度归一化，没有频率扭曲。原表示仍含位置信息，可从完整曲线求回长度；此处只是改变直接比较的特征，并非恢复完全丢失的信息。')
formula='> 本文为讲解用修订稿。仅修正原方案中“丢失总长度”的解释，不改实验公式。运行前冻结原稿 README.md 保留不变；详见 CLARIFICATION.md。\n\n'+formula
(run/'FORMULAS.md').write_text(formula,encoding='utf-8')
rows=json.loads((run/'RESULTS.json').read_text(encoding='utf-8'))
parents=json.loads((run/'PARENT_COMPARISONS.json').read_text(encoding='utf-8'))
old=json.loads((root/'experiments/curve3d_logpolar_native_20261003/RESULTS.json').read_text(encoding='utf-8'))
new={(r['case'],r['id']):r for r in rows};direct={(r['case'],r['id']):r for r in parents}
codes=['R-03-C','R-05-C','R-03-S','R-05-S']
labels={r['id']:r['label'] for r in rows}
def all_positive(code,data):return all(data[(c,code)]['reduction_pct'][k]>0 for c in ['case1','case2'] for k in range(2))
def all_supported(code,data):
    return all(data[(c,code)]['reduction_pct'][k]>0 and data[(c,code)]['reduction_pct_bootstrap95'][k][0]>0
               and data[(c,code)]['holm_adjusted_p_40_primary_comparisons'][k]<.05 for c in ['case1','case2'] for k in range(2))
positive=[c for c in codes if all_positive(c,new)]
supported=[c for c in codes if all_supported(c,new) and all_supported(c,direct)]
baseline_supported=[c for c in codes if all_supported(c,new)]
mean_better_both=[c for c in codes if all_positive(c,new) and all_positive(c,direct)]
text=['# 指标是否够用，以及修改后有没有改善：第三轮正式实验结论','',
    '## 先说结论','',
    '仅凭前两轮结果不够定稿。本轮已完成四个针对性变体、两项原指标对照与基线的正式复核，采用新的 100 个配对随机种子（100–199），两个 Case 共 1,400 次完整运行、73,500,000 次评估。',
    '新种子复核不是新测量数据上的独立验证；不能消除此前根据结果筛选候选的偏差。','']
if supported:
    text.append('在本轮统计口径下，同时优于基线及对应原版、且两个 Case 的两项主误差均有支持的变体：'+ '、'.join(supported)+'。仍需新数据验证，不能据此断言普遍有效。')
else:
    text.append('**没有变体同时证明：在两个 Case 的两项主误差上，既优于基线，又优于对应原版。** 因此不能因为公式改得更完整，就把它定为最终目标。')
if baseline_supported:
    text+=['','但本轮有实质进展：'+ '、'.join(baseline_supported)+' 在两个 Case 的两项主误差上，相对基线的改善均有 95% 区间和 40 项修正检验支持。建议优先保留此方向进入后续验证，不必继续无边界增加候选公式。']
if 'R-05-C' in baseline_supported:
    text+=['','具体看累计弧长 R-05-C：Case II 对原版两项均有改善支持；Case I 虽然对原版均值更好，但直接比较的区间仍跨零，不能宣布两 Case 都已明确胜过原版。']
text+=['','## 修改了什么，为什么值得做','',
    '|变体|改动|要检验的问题|','|---|---|---|',
    '|R-03-C|终点面积改为全过程累计面积|原面积是否把中途正负变化抵消掉了？|',
    '|R-05-C|改为同频率处的累计实际弧长|直接约束长度及其积累顺序，是否比按归一化弧长对应位置更有帮助？|',
    '|R-03-S|面积原公式不变，仅换目标标准差尺度|原最大最小值尺度是否影响面积指标效果？|',
    '|R-05-S|弧长原公式不变，仅换目标标准差尺度|原最大最小值尺度是否影响弧长指标效果？|','',
    '没有增加转折点数量、事件阈值、窗口、拟合权重、配准规则或固定重采样数量；不同时叠加多个改动。',
    '**表述修正：** 原弧长表示仍含位置信息，理论上可从完整曲线求回长度；准确说法是原损失没有显式比较总长度与同频率处的累计长度，而非“完全丢失长度”。详见 [表述修正](../experiments/curve3d_refinement_20261003/CLARIFICATION.md)。','',
    '## 原指标能否在新种子上重复','',
    '|指标|Case|上一轮 0–99：幅值／相位 MSE 下降|本轮 100–199：幅值／相位 MSE 下降|',
    '|---|---|---:|---:|']
for code in ['D-03-LP','D-05-LP']:
    for case in ['case1','case2']:
        previous=next(r for r in old if r['id']==code and r['case']==case);now=new[(case,code)]
        text.append(f"|{code}|{case}|{previous['reduction_pct'][0]:.2f}%／{previous['reduction_pct'][1]:.2f}%|{now['reduction_pct'][0]:.2f}%／{now['reduction_pct'][1]:.2f}%|")
text+=['','两个 seed 区块各自使用同区块的基线，不将两区块直接配对，也不选较好的区块代表最终结论。所有百分比均为 MSE 均值比例下降，负数为恶化。','',
    '## 变体相对本轮基线','',
    '|变体|Case I 幅值／相位 MSE 下降|Case II 幅值／相位 MSE 下降|','|---|---:|---:|']
for code in codes:
    a,b=new[('case1',code)],new[('case2',code)]
    text.append(f"|{code} {labels[code]}|{a['reduction_pct'][0]:.2f}%／{a['reduction_pct'][1]:.2f}%|{b['reduction_pct'][0]:.2f}%／{b['reduction_pct'][1]:.2f}%|")
text+=['','## 变体相对本轮对应原版','',
    '|变体|对应原版|Case I 幅值／相位 MSE 下降|Case II 幅值／相位 MSE 下降|','|---|---|---:|---:|']
for code in codes:
    a,b=direct[('case1',code)],direct[('case2',code)]
    text.append(f"|{code}|{a['parent']}|{a['reduction_pct'][0]:.2f}%／{a['reduction_pct'][1]:.2f}%|{b['reduction_pct'][0]:.2f}%／{b['reduction_pct'][1]:.2f}%|")
text+=['','“比基线好”与“改版优于原版”是两个不同问题，不可混用百分比。','',
    '## 95% 区间与统一修正检验','',
    '|比较|Case|指标|幅值下降区间|相位下降区间|幅值／相位 Holm p|','|---|---|---|---|---|']
for comparator,data in [('基线',rows),('对应原版',parents)]:
    for r in data:
        m,p=r['reduction_pct_bootstrap95'][:2];pm,pp=r['holm_adjusted_p_40_primary_comparisons']
        text.append(f"|{comparator}|{r['case']}|{r['id']}|[{m[0]:.2f}%, {m[1]:.2f}%]|[{p[0]:.2f}%, {p[1]:.2f}%]|{pm:.4g}／{pp:.4g}|")
text+=['','共 40 项主目标检验统一 Holm 修正；10,000 次配对 seed 重采样区间未作同时覆盖校正。Wilcoxon 针对配对运行差异，不是专门检验均值比例；区间与检验共同报告，不把未显著解释成等效。','',
    '## 当前筛选判断','',
    '- 优先保留进入后续验证：'+('、'.join(baseline_supported) or '尚无跨 Case 双主目标改善均有支持的变体')+'；这里的支持仅针对相对基线。',
    '- 对基线跨 Case 四项均值都下降的变体：'+('、'.join(positive) or '无')+'；这是方向描述，不是有效性证明。',
    '- 对基线及对应原版跨 Case 四项均值都下降的变体：'+('、'.join(mean_better_both) or '无')+'。',
    '- 对基线及对应原版跨 Case 四项改善均有区间与修正检验支持的变体：'+('、'.join(supported) or '无')+'。',
    '- 不据单 Case、单主误差或 WSMSE 的优势宣布找到通用目标；所有负结果保留。',
    '- 累计面积 R-03-C 可作次级候选：Case I 对基线和原版均有明显支持，Case II 仍不明确。标准差尺度两个变体未显示明确改进；标准差弧长在两个 Case 的均值都不如原弧长，不建议以此替换原版。',
    '- 若后续继续研究，应优先处理验证问题而非无限增加公式：独立数据、已知噪声条件、Case II 数据与电路可行域的核对。该核对不等于授权删点或改原主目标。','',
    '## 审计与文件','',
    f"完整运行 1,400 次；独立重算全部最终种群 {audit['independent_final_individuals']:,} 个个体。主目标复算最大绝对差 {audit['independent_primary_max_abs_difference']:.3g}，辅助目标最大绝对差 {audit['independent_aux_max_abs_difference']:.3g}。",
    '源文件与配置 hash、同 seed 初始种群、300 代历史、评估数量、最终同一个体选择均已核对。原版控制在两个 Case 全初始种群上的目标值与旧代码逐元素完全一致。',
    f"结果归档 SHA-256：`{audit['sha256']}`。",'',
    '- [详细公式与解释（表述修订，不改实验公式）](../experiments/curve3d_refinement_20261003/FORMULAS.md)',
    '- [运行前冻结原方案](../experiments/curve3d_refinement_20261003/README.md)',
    '- [原弧长信息表述修正](../experiments/curve3d_refinement_20261003/CLARIFICATION.md)',
    '- [完整结果及区间](../experiments/curve3d_refinement_20261003/RESULTS.md)',
    '- [下载与独立复算审计](../experiments/curve3d_refinement_20261003/server_run/download_audit.json)',
    '- [数学示意图](../experiments/curve3d_refinement_20261003/figures/mechanism_examples.png)',
    '- [相对基线的结果图](../experiments/curve3d_refinement_20261003/figures/baseline_intervals.png)',
    '- [相对原版的结果图](../experiments/curve3d_refinement_20261003/figures/parent_intervals.png)','',
    '本轮没有改动既有指标 PDF、讲稿和 HTML 结果 PPT；先完成筛选，再决定哪些值得正式纳入。']
output=root/'output/三维曲线指标_改进与新种子复核_20261003.md'
output.write_text('\n'.join(text)+'\n',encoding='utf-8')
print(json.dumps(dict(report=str(output),positive_against_baseline=positive,positive_against_baseline_and_parent=mean_better_both,
                     supported_against_baseline=baseline_supported,supported_against_baseline_and_parent=supported),ensure_ascii=False))

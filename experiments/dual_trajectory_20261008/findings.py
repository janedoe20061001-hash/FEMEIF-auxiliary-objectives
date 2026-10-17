"""Read completed diagnostics and explain persistent late-stage residuals."""
from pathlib import Path
import json,csv
import numpy as np
from analyze import OUT,ROOT,table,n,link,write,dump
def main():
    audit=json.loads((OUT/'BACKGROUND_VALIDATION.json').read_text());assert audit['passed'];summaries=audit['stage_summary'];pairs=audit['paired_changes']
    bands=list(csv.DictReader((OUT/'all_seed_bands.csv').open(encoding='utf-8-sig')));diags=list(csv.DictReader((OUT/'all_seed_diagnostics.csv').open(encoding='utf-8-sig')))
    parts=['# 后期残差在哪里：以MAG/PHA双目标为基础',
        '**主要结论：整体误差仍在改善，局部结构没有同步全部修复；目前不能称作“MAG/PHA优化不动”。** 200→300代，Case1的幅值RMSE均值下降0.2768dB、相位下降1.8372°；Case2分别下降0.1070dB、0.5440°，配对95%区间均低于零。两个Case的相位事件漏失均只下降0.05个/seed，区间均跨零。',
        '末代Case1有73/100seeds至少漏失一个幅值事件，平均漏失1.56个；Case2为100/100、平均5.77个，但后者只表示测量点曲线上的阈值事件，不能当作真实连续物理模态数。匹配峰谷的位置在改善，但Case1的平均宽度偏差仍约1.06倍频程，提示后续值得检查局部宽度和相位转变。',
        '这是同一批运行的描述性分析。训练目标没有新增指标；图中不平均曲线，后台只汇总误差和事件诊断。','## 全部100seeds的阶段变化']
    summaryrows=[];final=[];bandrows=[]
    for c in ['case1','case2']:
        for g in [20,100,200,300]:
            r=next(r for r in summaries if (r['circuit'],r['generation'],r['rule'])==(c,g,'min_wsmse'));summaryrows.append([c,g,n(r['amplitude_rmse_db']),n(r['phase_rmse_deg']),n(r['amp_missed']),n(r['phase_missed']),r['amp_miss_seeds'],r['stable']])
        for rule in ['min_wsmse','balanced_minmax']:
            r=next(r for r in summaries if (r['circuit'],r['generation'],r['rule'])==(c,300,rule));final.append([c,rule,n(r['amp_missed']),n(r['amp_mean_abs_shift_decades']),n(r['amp_mean_abs_width_log2_ratio']),n(r['worst_band_phase_rmse_deg']),r['stable']])
        # Rank each stage's frequency bands by mean squared phase residual across seeds.
        for g in [20,100,200,300]:
            rr=[r for r in bands if (r['circuit'],int(r['generation']),r['rule'])==(c,g,'min_wsmse')];scores=[]
            for b in range(1,9):
                bb=[r for r in rr if int(r['band'])==b];assert len(bb)==100
                scores.append((b,float(np.mean([float(r['phase_rmse_deg'])**2 for r in bb])),bb[0]))
            total=sum(x[1] for x in scores)
            for b,score,example in sorted(scores,key=lambda x:x[1],reverse=True)[:2]:
                bandrows.append([c,g,b,n(float(example['lo_hz'])),n(float(example['hi_hz'])),n(np.sqrt(score)),f'{100*score/total:.1f}%'])
    parts.extend([table(['Case','代','幅值RMSE/dB均值','相位RMSE/°均值','幅值漏失均数','相位漏失均数','幅值漏失seed数','网格稳定seed数'],summaryrows),
        '这些RMSE是每条曲线的诊断数值再取均值，和优化的等权训练点MSE不是同一个口径。Case1用解析加密网格；Case2只在测量点之间连线诊断。',
        '## 末代仍存在什么结构误差',table(['Case','选解规则','幅值漏失均数','匹配峰谷位移/decade','匹配峰谷宽度偏差/倍频程','最坏频带相位RMSE/°均值','稳定seed数'],final),
        '峰位和宽度只统计已经匹配到的事件；漏峰不能被当作零位置/宽度误差。宽度偏差是abs(log2(拟合宽度/目标宽度))，数值1对应宽度差一倍。这里的峰谷事件基于曲线阈值，不等于电路模态数。',
        '## 相位误差集中的频段',table(['Case','代','频带','起始/Hz','结束/Hz','频带合并RMSE/°','相位平方误差份额'],bandrows),
        '每个Case按log f等宽划8段；份额先在各seed/频带计算平方误差，再跨seed汇总。这里的合并RMSE用于定位能量分布，不能与上表“每seed RMSE再平均”互换。相同频段连续成为主要残差才值得作为后续局部指标的候选。',
        '## 200到300代是否还在改善',table(['Case','规则','诊断','配对变化300−200','95%区间','数值稳定'],[[r['circuit'],r['rule'],r['metric'],n(r['mean_300_minus_200']),f"[{n(r['ci95'][0])}, {n(r['ci95'][1])}]",r['numerically_stable']] for r in pairs]),
        '误差下降但峰谷漏失仍高，意味着平均目标的改善没有同步消除全部局部缺陷；后期不再稳定改善，才提供停滞的证据。两种规则结论不一致时，先考虑前沿取舍。即使两者一致，也只是相同300代预算下的现象，不能证明双目标在理论上优化不到。',
        '## 后续指标的设计方向',
        '保持MAG/PHA两个基础目标。根据这里持续存在的缺陷选择少量候选：峰谷漏失、位置和宽度偏差可用特征匹配约束；集中在少数频带的相位残差可用最坏频带或局部相位转变约束。没有漏失和局部偏差证据的结构，不应先增加惩罚。',
        f'指标实验之前应结合[加密采样报告]({link(ROOT/"output/sampling_refinement_20261007/REPORT.md")})。目标窄峰没被表达、或候选窄峰落在固定点之间时，应先处理采样/评价可靠性。若窄峰表达充分但在平均误差中权重很小，再检验结构指标是否改善它，避免把所有问题混为“平均MSE不好”。',
        f'[查看随机seed曲线]({link(OUT/"REPORT.md")})；[所有100seed完整阶段表]({link(OUT/"ALL_SEEDS.md")})'])
    write(OUT/'FINDINGS.md',parts)
    report=OUT/'REPORT.md';txt=report.read_text(encoding='utf8');txt=txt.replace('## 怎么看这些图',f'## 先看后台结构结论\n\n[100seeds：阶段变化、持续峰谷偏差和相位残差频段]({link(OUT/"FINDINGS.md")})。\n\n## 怎么看这些图');report.write_text(txt,encoding='utf8')
    dump(OUT/'ANALYSIS_SOURCES.json',{str(p.relative_to(ROOT)):__import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    print(json.dumps(dict(findings=str(OUT/'FINDINGS.md'),stage_rows=summaryrows),ensure_ascii=False))
if __name__=='__main__':main()

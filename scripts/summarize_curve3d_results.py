"""Plain-language complete summary, generated only after verified completion."""
from pathlib import Path
import json

root=Path(__file__).resolve().parents[1]
run=root/'experiments/curve3d_objectives_20261003'
audit=json.loads((run/'server_run/download_audit.json').read_text())
assert audit['status']=='passed' and audit['independent_all_final_population_formula_checks']
rows=json.loads((run/'RESULTS.json').read_text(encoding='utf-8'))
by_id={r['id']:{v['case']:v for v in rows if v['id']==r['id']} for r in rows}
supported=[];mean_improved=[]
for code,cases in by_id.items():
    if all(all(v>0 for v in r['reduction_pct'][:2]) for r in cases.values()):mean_improved.append(code)
    if all(all(v>0 for v in r['reduction_pct'][:2]) and
           all(ci[0]>0 for ci in r['reduction_pct_bootstrap95'][:2]) and
           all(p<.05 for p in r['holm_adjusted_p_40_primary_comparisons']) for r in cases.values()):supported.append(code)
lines=['# 三维曲线指标：正式实验总结','',
       '已完成 5 个几何方向 × 2 种坐标的检验。两个 Case 各 11 组，每组 100 个配对种子；共 2,200 次运行、115,500,000 次评估。没有缩减为预实验。',
       'Case I 种群 200 × 300 代；Case II 种群 150 × 300 代。原双主目标、原电路、原数据、算子与最后选择方法均保持不变。','',
       '## 最主要的判断','']
if supported:
    lines.append('以下候选在两个 Case 的两项误差上均有改善方向、正的配对区间下界和本轮 Holm 修正后的检验支持：'+', '.join(supported)+'。这仍属于探索筛选，不是独立数据验证。')
else:
    lines.append('**本轮尚未找到一项在两个 Case 都有明确双主目标改善证据的三维辅助指标。** 能解释几何意义，不等于能帮助这套优化算法找到更好的拟合参数。')
lines += ['', '两个 Case 的幅值、相位四个均值都下降的候选：'+(', '.join(mean_improved) if mean_improved else '没有')+'。仅均值方向一致不代表已证明有效。','',
          '## 全部结果，不只列改善组','',
          '正数表示误差下降，负数表示变差。百分比 = 100 ×（基线均值 − 辅助组均值）/ 基线均值。幅值是 dB² 的 MSE，相位是 rad² 的 MSE，不是阻抗幅值或相位本身减少了多少。',
          'LIN：目标跨度固定的线性坐标；ASH：先做保留正负号的 asinh 压缩，再用目标跨度归一化。两种坐标均包含归一化的对数频率。','',
          '|候选|方向|Case I 幅值下降|Case I 相位下降|Case II 幅值下降|Case II 相位下降|',
          '|---|---|---:|---:|---:|---:|']
for code,cases in by_id.items():
    a=cases['case1'];b=cases['case2']
    label=a['label'].split('（')[0]
    lines.append(f"|{code}|{label}|{a['reduction_pct'][0]:.2f}%|{a['reduction_pct'][1]:.2f}%|{b['reduction_pct'][0]:.2f}%|{b['reduction_pct'][1]:.2f}%|")
lines += ['','### 两个 Case 分别说明了什么','']
for r in rows:
    if r['case']=='case1' and all(v>0 for v in r['reduction_pct'][:2]):
        a,b=r['reduction_pct'][:2]
        ci=r['reduction_pct_bootstrap95']
        lines.append(f"- Case I 的 {r['id']}：幅值均值下降 {a:.2f}%，相位均值下降 {b:.2f}%；两项 95% 区间分别为 [{ci[0][0]:.2f}%, {ci[0][1]:.2f}%]、[{ci[1][0]:.2f}%, {ci[1][1]:.2f}%]，不能当成明确的双项改善。")
case2=[r for r in rows if r['case']=='case2']
if all(all(v<0 for v in r['reduction_pct'][:2]) and
       all(ci[1]<0 for ci in r['reduction_pct_bootstrap95'][:2]) and
       all(p<.05 for p in r['holm_adjusted_p_40_primary_comparisons']) for r in case2):
    lines.append('- Case II 的十项候选：幅值、相位均值全部变差，两项下降率的 95% 区间全部低于零，并且本轮 40 项检验的 Holm 修正后仍支持差异。这批结果不支持将任何一项加入当前辅助目标集合。')
lines.append('- 这不否定把阻抗画成三维曲线用于解释和诊断；本轮不支持的是这些特定公式在当前算法、数据和预算下作为第三目标的效果。')
lines += ['','![均值下降和配对区间](../experiments/curve3d_objectives_20261003/figures/primary_reduction_intervals.png)','',
          '点是 100 个配对种子的均值比例下降，线是配对自助 95% 区间。区间跨零说明均值改善方向还不确定；这些区间未作同时覆盖修正。完整报告同时列出 40 项比较的 Holm 修正 p 值。','',
          '## 如何据此筛选','',
          '- 不能只看辅助指标自己下降，也不能分别挑两个不同个体各自的最低误差。本表仍按原固定 WSMSE 从最后种群选同一个体。',
          '- 不把单 Case 的改善推广为跨 Case 有效；不把十项中事后最好的坐标当成独立验证。',
          '- 不据此直接上双、三辅助组合：单项尚无支持时，多加目标可能增加搜索负担。组合效果应单独冻结方案、正式检验。',
          '- 指标的数学意义、坐标选择的依据和解释不了的弱点都已写明；没有靠特定转折数量、窗口长度或经验阈值筛点。','',
          '## 结果可信度检查','',
          '服务器审计：每运行完整 300 代、评估次数、100 个种子的初始化、最后完整种群的目标复算、最终个体选择与源代码指纹均通过。',
          '下载后：完整原始包 SHA256 与服务器一致；代码、配置与启动前冻结版本一致；每个最终种群的指标又以另一套公式实现复算。协方差和弧长使用独立 Gauss 积分，面积用逐平面鞋带公式，距离用完整距离矩阵，而不是重复调用同一个损失函数。',
          '两个 Case 的全部历史 baseline seed 逐项对照结果：'+ '; '.join(f"{case} 最大绝对差 {v['max_abs_difference']:.3g}" for case,v in audit['baseline_reproduction'].items())+'。','',
          f"跨 Linux/Windows 复算的主误差最大绝对差为 {audit['independent_primary_max_abs_difference']:.3g}，最大相对差为 {audit['independent_primary_max_relative_difference']:.3g}；辅助公式最大绝对差为 {audit['independent_max_abs_difference']:.3g}。容许精度与实测差异均保留在审计 JSON 中，不更改服务器的实验记录。",'',
          '统计只反映这批随机优化运行，不涵盖测量噪声、新电路或新数据。前面多轮指标探索也会带来选择偏差，不能靠本轮多重检验完全消除。','',
          '## 文件','',
          '- [候选表及三维坐标图](三维阻抗曲线指标候选_20261003.md)',
          '- [全部公式、计算步骤和原理出处](../experiments/curve3d_objectives_20261003/README.md)',
          '- [详细结果、区间和检验](../experiments/curve3d_objectives_20261003/RESULTS.md)',
          '- [下载与独立公式复算审计](../experiments/curve3d_objectives_20261003/server_run/download_audit.json)',
          '- 原始 2,200 次运行的初始种群、每代历史、最终完整种群和汇总均在 `experiments/curve3d_objectives_20261003/server_run/raw_results/`；完整包为同目录上一级的 `formal_results.tar.gz`。']
target=root/'output/三维曲线指标正式实验总结_20261003.md'
target.write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(target)
print(json.dumps(dict(cross_case_supported=supported,cross_case_all_means_improved=mean_improved),ensure_ascii=False))

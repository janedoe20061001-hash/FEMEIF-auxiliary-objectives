"""Keep all coordinate variants and both negative/positive results in one table."""
from pathlib import Path
import json,csv

root=Path(__file__).resolve().parents[1]
first=root/'experiments/curve3d_objectives_20261003'
second=root/'experiments/curve3d_logpolar_native_20261003'
rows=[];audits=[]
for folder in [first,second]:
    audit=json.loads((folder/'server_run/download_audit.json').read_text());assert audit['status']=='passed'
    audits.append(audit)
    data=json.loads((folder/'RESULTS.json').read_text(encoding='utf-8'))
    for r in data:
        r['round']=1 if folder==first else 2
        key='holm_adjusted_p_40_primary_comparisons' if folder==first else 'holm_adjusted_p_24_primary_comparisons'
        r['within_round_holm_p']=r[key];rows.append(r)
by_id={r['id']:{v['case']:v for v in rows if v['id']==r['id']} for r in rows}
mean_improved=[];supported=[]
for code,cases in by_id.items():
    if all(all(v>0 for v in r['reduction_pct'][:2]) for r in cases.values()):mean_improved.append(code)
    if all(all(v>0 for v in r['reduction_pct'][:2]) and all(ci[0]>0 for ci in r['reduction_pct_bootstrap95'][:2])
           and all(p<.05 for p in r['within_round_holm_p']) for r in cases.values()):supported.append(code)
text=['# 三维阻抗曲线指标：两轮正式实验汇总','',
      '已检验六种几何方向：全频率对距离、整体坐标变化关系、三平面有向面积、走向与变化快慢、按弧长对应的形状，以及新增的全频率对空间方向差。',
      '本次比较三种坐标：LIN（实部/虚部线性归一化）、ASH（实部/虚部有符号对数式压缩）、LP（对数幅值/主值相位）。全部坐标都包含归一化的对数频率，目标和候选共用目标尺度。',
      '六种方向不等于十六个独立新原理：五类方向各比较三种坐标，方向差在第二轮新增，共十六种具体候选。','',
      '两轮共有 3,600 次完整正式运行、189,000,000 次评估。第一轮各 Case 11 组 × 100 seeds；第二轮各 Case 7 组 × 100 seeds。两轮各自重跑基线，不是将重跑当成更多独立种子。',
      'Case I 种群 200 × 300 代；Case II 种群 150 × 300 代；每轮均是原双主目标加单辅助。慢实现曾中止的 302 次完成运行不纳入这些结果。','',
      '## 主要结论','']
if supported:text.append('以下候选在两 Case 的两项误差上均有改善方向、正的配对区间下界和各轮 Holm 检验支持：'+', '.join(supported)+'。仍需在新数据/新种子方案上独立验证，不能把探索中的筛选结果包装成定论。')
else:text.append('**尚未发现一项在两个 Case 都有明确双主目标改善证据的三维辅助指标。** 当前结果不支持只因几何意义容易解释就直接纳入目标集合。')
text += ['', '两个 Case 四个误差均值都下降的候选：'+(', '.join(mean_improved) if mean_improved else '没有')+'。均值方向一致与有统计支持不是同一件事。','',
         '## 全部下降比例','',
         '正数表示误差下降，负数表示变差；均值下降 = 100 ×（基线均值 − 辅助组均值）/ 基线均值。幅值为 dB² 的 MSE，相位为 rad² 的 MSE，不是幅值或相位本身的下降。','',
         '|方向与坐标|Case I 幅值下降|Case I 相位下降|Case II 幅值下降|Case II 相位下降|',
         '|---|---:|---:|---:|---:|']
table=[]
for code,cases in sorted(by_id.items()):
    a=cases['case1'];b=cases['case2'];label=a['label'].split('（')[0]
    values=a['reduction_pct'][:2]+b['reduction_pct'][:2]
    text.append(f"|{code} {label}|"+'|'.join(f'{v:.2f}%' for v in values)+'|')
    table.append(dict(metric=code,label=label,case1_magnitude_reduction_pct=values[0],case1_phase_reduction_pct=values[1],
                      case2_magnitude_reduction_pct=values[2],case2_phase_reduction_pct=values[3]))
text += ['','### 当前建议：保留复核候选，不宣称已经筛选成功','',
         '- 优先复核 D-05-LP（按弧长对应的形状）：两 Case 的两项均值方向都改善；Case II 幅值下降 24.60%，配对区间 [16.32%, 31.76%]，本轮 Holm p≈0.000140，幅值证据较明确。但是相位下降 1.79% 的区间为 [-6.31%, 9.45%]，尚不确定，Case I 也未获得明确双项支持。',
         '- D-03-LP（三平面有向面积）可作次级复核候选：两个 Case 的两项均值都下降，但四项区间均跨零，证据弱于 D-05-LP 的 Case II 幅值结果。',
         '- 新增 D-06-LP（全频率对空间方向差）目前不推荐：没有在两个 Case 实现双项均值改善。',
         '- 不能把“相位未检出显著变差”说成“证明相位不变或改善”；这需要另行设计非劣性或独立验证，当前没有做。']
text += ['','## 第二轮：哪些均值变化值得说明','']
for r in rows:
    if r['round']!=2:continue
    a,b=r['reduction_pct'][:2];c1,c2=r['reduction_pct_bootstrap95'][:2]
    text.append(f"- {r['case']} / {r['id']}：幅值下降 {a:.2f}%，区间 [{c1[0]:.2f}%, {c1[1]:.2f}%]；相位下降 {b:.2f}%，区间 [{c2[0]:.2f}%, {c2[1]:.2f}%]；本轮 Holm p = {r['within_round_holm_p'][0]:.4g} / {r['within_round_holm_p'][1]:.4g}。")
text += ['','![第二轮误差下降和配对区间](../experiments/curve3d_logpolar_native_20261003/figures/primary_reduction_intervals.png)','',
         '各区间由 10,000 次配对种子重采样得到，反映优化随机性；未作同时覆盖修正。各轮检验分别作 Holm 修正（40 项、24 项），不是对历轮探索的全部选择偏差作了校正。第二轮是在第一轮负结果后提出，不是独立数据确认。','',
         '## 不能忽略的问题','',
         '- 大多数几何量不能唯一决定曲线。距离、协方差、面积、方向差等也可能忽略整体平移；原幅值、相位目标不可去掉。',
         '- 直接对 LP 坐标作相同频率逐点距离平均会退化成原两误差的固定加权和，本轮没有将它当成新指标。',
         '- Case II 在 16 MHz 有一个相位 107°、负实部的测量点，正参数被动 RLC 不能精确复现；最大相邻主值相位跳变为 184.9°。这是一项可行域限制，不足以证明实验变差的原因，也不能据此认定测量错误。原始全部 201 点保留，没有删点、截相位或展开相位来改善成绩。',
         '- 单辅助不确定或变差时，不直接以“多加几个”作为改进理由；双、三辅助组合仍需要另行冻结与正式对照。','',
         '对上述可行域限制，当前 Case II 每个并联块的导纳实部为 $a_k=1/R_{2k}+R_{2k-1}/[R_{2k-1}^2+(\\omega L_k)^2]>0$；若导纳为 $a_k+jb_k$，阻抗实部为 $a_k/(a_k^2+b_k^2)>0$。三个块串联再加 $R_7>0$，总实部严格为正。这个论证只说明无法精确复现负实部点，不证明需要删掉它或它就是全部恶化的原因。','',
         '## 完整预算与数值核验','',
         '第二轮初版的全点对方向差因临时数组开销慢，停止时已完成 302 次，所有文件保留但不纳入结果。更高效的 native_v2 使用同一公式、全部点对，软件检查在两个 Case 的完整初始种群上与原 NumPy 公式逐位相同；随后从头跑完全部 1,400 次正式运行。没有用原中途结果判断候选效果或改变数学定义。',
         '两轮都通过服务器完整代数/评估数、配对初始化、源代码与最终完整种群审计；原始包下载 hash 一致，全部最终个体用独立公式复算。两轮分别检查 385,000 和 245,000 个最终个体；重跑基线与历史全部 100 seeds 完全一致。','']
for folder,audit in zip([first,second],audits):
    text.append(f"- {folder.name}：主误差跨平台最大绝对差 {audit['independent_primary_max_abs_difference']:.3g}，辅助公式最大绝对差 {audit['independent_max_abs_difference']:.3g}。")
text += ['','## 文件','',
         '- [第一轮候选表与坐标图](三维阻抗曲线指标候选_20261003.md)',
         '- [第二轮全部公式、原理与限制](../experiments/curve3d_logpolar_native_20261003/README.md)',
         '- [第一轮完整结果](../experiments/curve3d_objectives_20261003/RESULTS.md)',
         '- [第二轮完整结果](../experiments/curve3d_logpolar_native_20261003/RESULTS.md)',
         '- [第二轮数值版本说明](../experiments/curve3d_logpolar_native_20261003/NUMERICAL_VERSION.md)',
         '- 原始包、每代历史、初始种群和最终完整种群分别保存在各轮 `server_run/`。']
output=root/'output/三维曲线指标_两轮正式实验汇总_20261003.md'
output.write_text('\n'.join(text)+'\n',encoding='utf-8')
with output.with_suffix('.csv').open('w',encoding='utf-8-sig',newline='') as fh:
    writer=csv.DictWriter(fh,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
print(output);print(json.dumps(dict(cross_case_supported=supported,cross_case_all_means_improved=mean_improved),ensure_ascii=False))

"""Local Markdown only; no site build or publication."""
from pathlib import Path
import json,csv,re,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'output/seed_replication_20261007'
summary=json.loads((OUT/'SUMMARY.json').read_text(encoding='utf-8'));validation=json.loads((OUT/'VALIDATION.json').read_text(encoding='utf-8'));assert validation['passed']
env=json.loads((HERE/'server_run/retrieved/server_environment.json').read_text(encoding='utf-8'));status=json.loads((HERE/'server_run/retrieved/runs/status.json').read_text(encoding='utf-8'))
primary=summary['primary_Q'];decision=summary['decision'];q=primary['new'];lo,hi=primary['new_ci95']
if decision=='supports_stable_closeness_within_5pct':
    lead='新种子支持“两个跨Case比例在本实验条件下稳定接近”：新Q的95%区间完全落在预设±5%范围内。它不是仅原100个种子才出现的结果，但本实验仍不能证明产生这种比例的机制。'
elif decision=='does_not_replicate_5pct_closeness':
    lead='新种子未复现预设±5%的比例接近：新Q的95%区间与[0.95,1.05]完全分离。原来的接近不能当作稳定规律。'
else:
    lead='新种子对“±5%内稳定接近”的证据仍不充分：新Q的95%区间没有完全落在[0.95,1.05]内，也没有与该范围完全分离。要把接近解释为稳定规律，目前仍需保留不确定性。'
names={'wsmse':'WSMSE单目标','mean_distance':'平均距离单目标'}
def fmt(v):return f'{v:.6g}'
def ci(v):return f'[{v[0]:.6g}, {v[1]:.6g}]'
rows=['# 两个Case：新种子单目标复现实验','',f'**{lead}**','',
      '400次新运行全部完成，零失败；只改变优化随机种子，保持目标、采样、α、边界、算法与预算一致。未修改原结果，未更新网站。','',
      '## 本轮要检验什么','',
      '原距离组中，Case1/Case2的幅值MSE比为1.96514，相位MSE比为1.99033。新批要检验这种接近是否会在另一批种子中重现。定义：','',
      '- R_A＝Case1幅值MSE种子均值／Case2幅值MSE种子均值。',
      '- R_φ＝Case1相位MSE种子均值／Case2相位MSE种子均值。',
      '- Q＝R_A／R_φ；Q＝1表示两个跨Case比值相等。Q不受幅值或相位的共同单位换算影响。','',
      '读取新结果前，将Q∈[0.95,1.05]登记为“接近”的操作性范围。这个±5%是复现判据，不是物理定律。主要判断对象为平均距离组，WSMSE组为对照；不按结果换组或换种子。','',
      '## 核心结果：比例是否重现','',
      '| 优化组 | 比值 | 旧批估计 | 旧批95%区间 | 新批估计 | 新批95%区间 |','|---|---|---:|---|---:|---|']
for g in ['mean_distance','wsmse']:
    for name in ['R_A','R_phi','Q']:
        v=next(r for r in summary['ratios'] if r['group']==g and r['ratio']==name)
        rows.append(f'| {names[g]} | {name} | {fmt(v["old"])} | {ci(v["old_ci95"])} | {fmt(v["new"])} | {ci(v["new_ci95"])} |')
rows+=['',f'平均距离组新Q={q:.6g}，95%区间={ci(primary["new_ci95"])}。新Q/旧Q={primary["new_over_old"]:.6g}，区间={ci(primary["new_over_old_ci95"])}。此处直接对两批独立抽样的比值比较，不以两个置信区间是否重叠替代差异分析。','',
       'Bootstrap为10,000次，单位是seed；同一批中共享seed的两个Case、两组选解一起重采样，新旧批次分别抽样。区间描述随机优化种子的变化，不是测量误差区间，也不能把频点当成独立样本。','',
       '## 新旧四项误差均值','',
       '每次在该seed末代种群中按本组自己的目标选最小个体，再对100个seeds等权平均。表中四项误差来自同一个选中个体；不是对均值曲线算误差。','',
       '| Case | 优化组 | 批次 | 幅值MSE / dB² | 相位MSE / rad² | WSMSE | 同频平均距离 / Ω |','|---|---|---|---:|---:|---:|---:|']
for c in ['case1','case2']:
    for g in ['wsmse','mean_distance']:
        for batch in ['old','new']:
            v=next(v for v in summary['means'] if (v['circuit'],v['group'],v['batch'])==(c,g,batch))
            rows.append('| '+c+' | '+names[g]+' | '+('旧' if batch=='old' else '新')+' | '+' | '.join(fmt(v[k]) for k in ['magnitude_mse_db2','phase_mse_rad2','wsmse','mean_distance_ohm'])+' |')
rows+=['','新批平均距离组减WSMSE组的配对比较：','',
       '| Case | 分量 | 均值差 | 95%配对区间 | 距离组更低的seeds |','|---|---|---:|---|---:|']
labels={'magnitude_mse_db2':'幅值MSE / dB²','phase_mse_rad2':'相位MSE / rad²','wsmse':'WSMSE','mean_distance_ohm':'平均距离 / Ω'}
for c in ['case1','case2']:
    for k in labels:
        v=next(v for v in summary['component_comparisons'] if v['circuit']==c and v['metric']==k)
        rows.append(f'| {c} | {labels[k]} | {v["distance_minus_wsmse"]:+.6g} | {ci(v["ci95"])} | {v["distance_lower_seeds"]}/100 |')
components=summary['component_comparisons'];direction=all(next(v for v in components if v['circuit']==c and v['metric']==k)['ci95'][0]>0 for c in ['case1','case2'] for k in ['magnitude_mse_db2','phase_mse_rad2']) and all(next(v for v in components if v['circuit']==c and v['metric']=='mean_distance_ohm')['ci95'][1]<0 for c in ['case1','case2'])
rows+=['',('本轮重复出现原先的方向：WSMSE组幅值与相位MSE均更低，平均距离组Ω距离更低；这六个配对均值差的95%区间均支持相应方向。' if direction else '部分交叉误差方向或区间需要区别判断，不能笼统说原优势全部复现。逐项以上表为准。'),'','## 每个seed是否也存在固定幅相比','',
       '下面M_A/M_φ是使用dB²和rad²时的数值比，随单位选择变化；只用于描述这些解的分布，不是物理贡献比例。','',
       '| Case | 批次（距离组） | 均值幅相数值比 | 每seed比值10%–90%范围 | 幅相误差Pearson相关 |','|---|---|---:|---|---:|']
for c in ['case1','case2']:
    for batch in ['old','new']:
        v=next(v for v in summary['amp_phase_statistics'] if (v['circuit'],v['group'],v['batch'])==(c,'mean_distance',batch))
        rows.append(f'| {c} | {batch} | {v["amp_over_phase_numeric_ratio"]:.6g} | {v["seed_ratio_p10"]:.3f}–{v["seed_ratio_p90"]:.3f} | {v["amp_phase_pearson"]:.4f} |')
rows+=['','平均比例的复现不要求每个seed比例完全相同；相关性表示共同变化，不能证明为什么两个Case的均值比接近。','',
       '## 能得到与不能得到的结论','',
       lead,'',
       '这次只换seed，检验的是这两个固定目标和固定300代搜索的种子稳健性。若要解释机制，还需检查误差来自哪些频带/结构，并改变目标电路或采样作受控验证。新种子重复结果也不能证明概率上的“绝不可能是巧合”，不能推出其他电路均有相同比例。','',
       'Case1的采样尖峰缺口本轮原样保留，保证新旧条件相同；这不是采样修复实验。Case2保持原测量点，包括107°点。Case1有零误差全局解，本轮比较的是300代后的搜索结果。','',
       '## 实验条件与核验','',
       '- 原seed：2300–2399；新seed：2026100700–2026100799，两批不重叠。',
       '- 两种单目标：WSMSE=M_A+αM_φ；D=mean abs(Z−Z目标)，原实部/虚部Ω、同频对应。幅值dB，相位rad主值。',
       '- Case1：180频点、α=25.12、种群200；Case2：全部201测点、α=17.38、种群150。',
       '- 算法沿用NSGA-II单目标模式，log10参数编码，SBX/PM与边界完全不变；同seed两组初始种群一致。',
       f'- 400次新运行、21,000,000次评价；28 workers，正式优化耗时{status["elapsed_seconds"]:.2f}秒。',
       '- 70,000个末代个体全部独立重算，检查原始来源哈希、参数边界、末代与历史最小选解、配对初始化；全部通过。',
       '- 跨平台路径修复发生在正式运行之前；第一次目录没有启动优化，新正式结果单独归档。','',
       '## 本地数据','',
       '[逐seed四指标](selected_metrics.csv) · [新旧组均值](group_means.csv) · [比例与区间](ratio_comparison.csv) · [每seed幅相比](seed_ratios.csv) · [逐频带分量贡献](band_contributions.csv) · [完整统计结果](SUMMARY.json) · [核验记录](VALIDATION.json) · [运行前方案](PROTOCOL.md)','']
shutil.copy2(HERE/'PROTOCOL.md',OUT/'PROTOCOL.md')
(OUT/'REPORT.md').write_text('\n'.join(rows),encoding='utf-8')
# Validate all report data links and Markdown table widths, without generating a website.
links=0
for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',(OUT/'REPORT.md').read_text(encoding='utf-8')):
    assert (OUT/target).exists();links+=1
text=(OUT/'REPORT.md').read_text(encoding='utf-8');tables=0;expected=None
for line in text.splitlines():
    if line.startswith('|'):
        n=len(line.split('|'))-2
        if expected is None:expected=n;tables+=1
        assert n==expected
    else:expected=None
dump=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
dump(OUT/'REPORT_VALIDATION.json',{'passed':True,'local_links':links,'tables':tables,'website_updated':False,'report_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
dump(OUT/'INTEGRITY.json',{'files_sha256':{p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='INTEGRITY.json'}})
print(json.dumps({'report_complete':True,'report':str(OUT/'REPORT.md'),'decision':decision,'primary_Q_new':q,'primary_Q_ci95':[lo,hi],'component_direction_reproduced':direction,'local_links':links,'tables':tables},ensure_ascii=False,indent=2))

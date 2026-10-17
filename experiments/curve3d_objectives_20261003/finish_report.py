from pathlib import Path
import csv
import hashlib
import json
import tarfile
import numpy as np

root=Path(__file__).resolve().parent
rows=[]
for case in ['case1','case2']:
    assert json.loads((root/case/'audit.json').read_text())['status']=='passed'
    rows.extend(json.loads((root/case/'analysis.json').read_text()))
p=np.array([v for r in rows for v in r['wilcoxon_two_sided_p']])
order=np.argsort(p);adj=np.empty_like(p)
adj[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
for i,r in enumerate(rows):
    r['holm_adjusted_p_40_primary_comparisons']=adj[2*i:2*i+2].tolist()
(root/'RESULTS.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
table=[]
for r in rows:
    a,b,w=r['reduction_pct']
    table.append(dict(case=r['case'],metric=r['id'],label=r['label'],
        magnitude_reduction_pct=a,phase_reduction_pct=b,wsmse_reduction_pct=w,
        both_lower_seeds=r['both_lower_seeds'],wsmse_lower_seeds=r['wsmse_lower_seeds'],
        aux_dominates_baseline_front_mean=r['aux_dominates_baseline_front_mean'],
        baseline_dominates_aux_front_mean=r['baseline_dominates_aux_front_mean']))
with (root/'RESULTS.csv').open('w',encoding='utf-8-sig',newline='') as fh:
    writer=csv.DictWriter(fh,fieldnames=table[0].keys());writer.writeheader();writer.writerows(table)
text=['# 三维曲线指标：Case I / Case II 正式实验结果','',
      '两个 Case 各 11 组、每组 100 个配对随机种子；共 2,200 次运行、115,500,000 次评估，全部完成并通过审计。',
      'Case I：种群 200、300 代；Case II：种群 150、300 代。两个 Case 均在本次同一环境中重跑基线。',
      '主表为最后一代按既有固定 WSMSE 选出的同一个体；先取各 seed 的两项误差，再跨 100 seeds 等权平均。',
      '**正数表示下降，负数表示变差。** 百分比=100×(基线均值−辅助组均值)/基线均值。','',
      '|Case|辅助指标|幅值误差下降|相位误差下降|WSMSE下降|同seed两项同时下降|',
      '|---|---|---:|---:|---:|---:|']
for r in rows:
    a,b,w=r['reduction_pct'];text.append(f"|{r['case']}|{r['id']} {r['label']}|{a:.3f}%|{b:.3f}%|{w:.3f}%|{r['both_lower_seeds']}/100|")
text+=['','## 均值、配对区间与显著性','',
       '以下 95% 区间由 10,000 次配对 seed 重采样获得，统计对象为均值比例下降，不是逐 seed 百分比的平均。',
       '同时给出配对双侧 Wilcoxon 检验，并对 2 Cases × 10 辅助指标 × 2 主目标共 40 项比较做 Holm 修正；区间本身未做同时覆盖校正。',
       '这些统计反映随机优化运行的差异，不包括测量噪声与换数据集带来的不确定性。','']
for r in rows:
    text += [f"### {r['case']} / {r['id']} {r['label']}",'']
    for i,label in enumerate(['幅值 MSE (dB²)','相位 MSE (rad²)']):
        lo,hi=r['reduction_pct_bootstrap95'][i]
        text.append(f"- {label}：基线均值 {r['baseline_mean'][i]:.8g}，辅助组均值 {r['auxiliary_mean'][i]:.8g}；下降 {r['reduction_pct'][i]:.3f}%，配对 95% 区间 [{lo:.3f}%, {hi:.3f}%]；Holm 修正 p={r['holm_adjusted_p_40_primary_comparisons'][i]:.6g}。")
    text += [f"- 原双主目标前沿：辅助组严格支配的基线前沿点比例均值为 {100*r['aux_dominates_baseline_front_mean']:.2f}%；反方向为 {100*r['baseline_dominates_aux_front_mean']:.2f}%。此比例是补充描述，受前沿采样密度影响，不等于统一超体积增益。",'']
text += ['## 解释范围','',
         '五个几何方向各比较两种坐标。所有组保留双主目标并仅添加一个辅助目标；单辅助结果不能推断组合效果。',
         '仅在一个 Case 降低两项误差不代表跨 Case 稳健；不能只依据 WSMSE 或辅助指标自身变小宣称成功。',
         '未进行事后阈值、积分窗口或权重调优；两种坐标在实验前冻结，不能把事后选出的最好坐标视为独立验证。候选定义、取样、算子和预算在启动前冻结。','',
         '完整定义见 README.md；代码/配置来源见各 Case 的 SOURCE.json；审计见 audit.json；每代历史和最终完整种群见 runs/。']
(root/'RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
archive=root/'formal_results.tar.gz'
with tarfile.open(archive,'w:gz') as tf:
    for pth in sorted(root.rglob('*')):
        if not pth.is_file() or '.venv' in pth.parts or '__pycache__' in pth.parts or pth==archive:
            continue
        if pth.suffix in ['.py','.json','.md','.csv','.npy','.npz','.txt','.log']:
            tf.add(pth,arcname=str(pth.relative_to(root)))
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
(root/'formal_results.sha256').write_text(digest+'  formal_results.tar.gz\n')
print(json.dumps({'archive':str(archive),'sha256':digest,'bytes':archive.stat().st_size}))

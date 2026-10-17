"""Report both baseline and parent comparisons, correcting one 20-test family."""
from pathlib import Path
import csv,hashlib,json,tarfile
import numpy as np
from scipy.stats import wilcoxon
from global_metrics import CATALOG,COMPARISONS

root=Path(__file__).resolve().parent;rows=[];parents=[]
ids={name:(code,label) for code,name,label in CATALOG}
bootstrap=np.random.default_rng(20261003).integers(0,100,(10000,100))
fields=['wsmse_selected_magnitude_mse','wsmse_selected_phase_mse','wsmse_selected_value']
for case in ['case1','case2']:
    audit=json.loads((root/case/'audit.json').read_text());assert audit['status']=='passed' and audit['runs']==400
    rows.extend(json.loads((root/case/'analysis.json').read_text()))
    for name,parent in COMPARISONS:
        def values(group):
            return np.array([[json.loads((root/case/f'runs/{group}/seed_{s:03d}/summary.json').read_text())[k]
                              for k in fields] for s in range(1100,1200)])
        a,b=values(name),values(parent)
        pct=100*(1-a.mean(axis=0)/b.mean(axis=0))
        ci=np.quantile(100*(1-a[bootstrap].mean(axis=1)/b[bootstrap].mean(axis=1)),[.025,.975],axis=0).T
        p=[float(wilcoxon(a[:,i]-b[:,i],method='approx').pvalue) if np.any(a[:,i]!=b[:,i]) else 1. for i in range(2)]
        parents.append(dict(case=case,id=ids[name][0],group=name,label=ids[name][1],parent=ids[parent][0],
            comparator='unchanged_parent_same_seeds',paired_seeds=100,baseline_mean=b.mean(axis=0).tolist(),
            auxiliary_mean=a.mean(axis=0).tolist(),reduction_pct=pct.tolist(),reduction_pct_bootstrap95=ci.tolist(),
            wilcoxon_two_sided_p=p,both_lower_seeds=int(np.all(a[:,:2]<b[:,:2],axis=1).sum())))
all_rows=rows+parents;p=np.array([v for r in all_rows for v in r['wilcoxon_two_sided_p']]);assert len(p)==20
order=np.argsort(p);adj=np.empty_like(p);adj[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
for i,row in enumerate(all_rows):row['holm_adjusted_p_20_primary_comparisons']=adj[2*i:2*i+2].tolist()
for name,data in [('RESULTS.json',rows),('PARENT_COMPARISONS.json',parents)]:
    (root/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
table=[dict(case=r['case'],id=r['id'],label=r['label'],comparator=r.get('parent','baseline'),
    magnitude_reduction_pct=r['reduction_pct'][0],phase_reduction_pct=r['reduction_pct'][1],wsmse_reduction_pct=r['reduction_pct'][2]) for r in all_rows]
with (root/'RESULTS.csv').open('w',encoding='utf-8-sig',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=table[0].keys());w.writeheader();w.writerows(table)
text=['# 全区间与逐段弧长：正式实验','',
    '两个 Case 各 4 组，每组 100 个新配对种子（1100–1199）；800 次完整运行，42,000,000 次评估。',
    'Case I：种群 200 × 300 代；Case II：种群 150 × 300 代。原数据、电路、边界、双主目标和最终同一个体的固定 WSMSE 选择不变。',
    '百分比是主目标 MSE 的均值比例下降，不是幅值／相位本身变化，也不是逐 seed 下降比例的均值。正为改善、负为变差。','']
for heading,data in [('相对本轮基线：原版能否重复，变体是否有用',rows),('相对对应原版：修改是否真正更好',parents)]:
    text += ['## '+heading,'','|Case|指标|比较对象|幅值 MSE 下降|相位 MSE 下降|WSMSE 下降|','|---|---|---|---:|---:|---:|']
    for r in data:
        text.append(f"|{r['case']}|{r['id']} {r['label']}|{r.get('parent','基线')}|{r['reduction_pct'][0]:.3f}%|{r['reduction_pct'][1]:.3f}%|{r['reduction_pct'][2]:.3f}%|")
    text+=['','### 置信区间与检验','']
    for r in data:
        text.append(f"- {r['case']} / {r['id']} 对 {r.get('parent','基线')}：")
        for k,label in enumerate(['幅值','相位']):
            lo,hi=r['reduction_pct_bootstrap95'][k]
            text.append(f"  - {label} MSE：比较组均值 {r['baseline_mean'][k]:.9g}，本组均值 {r['auxiliary_mean'][k]:.9g}；下降 {r['reduction_pct'][k]:.3f}%，95% 区间 [{lo:.3f}%, {hi:.3f}%]，Holm p={r['holm_adjusted_p_20_primary_comparisons'][k]:.6g}。")
        text.append(f"  - 同 seed 两项同时下降：{r['both_lower_seeds']}/100。")
    text.append('')
text+=['## 解释边界','',
    '所有 12 项对基线和 8 项对累计弧长的主目标比较统一 Holm 修正；区间由 10,000 次配对 seed 重采样得到，未作同时覆盖修正。新随机种子不是新数据集，不消除跨轮选择偏差。',
    '改进有明确数学理由，但不等于效果必然更好。必须同时看对基线、对原版、跨 Case 及两个主目标；不能用辅助指标自己变小替代主误差改善。',
    '本轮检验全区间长度和逐段长度平方根两个改法；二者不消除平移、噪声与不可达目标的问题。保留原 16 MHz／107° 点，不修改相位与主目标。',
    '全部公式和预先冻结方案见 README.md，软件检查见 software_checks.json，来源指纹见 FROZEN.json 与两 Case 的 manifest。']
(root/'RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
archive=root/'formal_results.tar.gz'
with tarfile.open(archive,'w:gz') as tf:
    for path in sorted(root.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path!=archive and path.suffix in ['.py','.json','.md','.csv','.npy','.npz','.txt','.log']:
            tf.add(path,arcname=str(path.relative_to(root)))
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
(root/'formal_results.sha256').write_text(digest+'  formal_results.tar.gz\n')
print(json.dumps(dict(archive=str(archive),sha256=digest,bytes=archive.stat().st_size)))

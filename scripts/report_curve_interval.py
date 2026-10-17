"""Audited result summary, exact mechanism examples, and uncertainty figures."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

root=Path(__file__).resolve().parents[1]
run=root/'experiments/curve3d_interval_20261003'
sys.path.insert(0,str(run))
from global_metrics import interval_loss,integral_squared,hellinger_loss
ap=argparse.ArgumentParser();ap.add_argument('--mechanism-only',action='store_true');args=ap.parse_args()
fonts={f.name for f in font_manager.fontManager.ttflist}
plt.rcParams['font.sans-serif']=[f for f in ['Microsoft YaHei','SimHei','Noto Sans CJK SC'] if f in fonts]+['DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False
out=run/'figures';out.mkdir(exist_ok=True)

u=np.array([0.,.2,.8,1.]);h=np.diff(u)
early=np.array([0.,.2,.2,.2]);late=np.array([0.,0.,0.,.2])
lb=[float(np.sqrt(integral_squared(e[:,None],h))) for e in [early,late]]
li=[float(interval_loss(e,h)) for e in [early,late]]
fig,axes=plt.subplots(1,2,figsize=(12,4.9),layout='constrained')
axes[0].plot(u,early,'o-',color='#137c89',label='低频附近新增长度')
axes[0].plot(u,late,'s--',color='#c85226',label='高频附近新增长度')
axes[0].set(title='同样的总长偏差，发生位置不同',xlabel='归一化对数频率 u',ylabel='累计长度相对误差 e')
axes[0].text(.30,.43,f'原累计弧长：{lb[0]:.3f} / {lb[1]:.3f}\n全区间弧长：{li[0]:.3f} / {li[1]:.3f}',transform=axes[0].transAxes,
             bbox=dict(facecolor='white',alpha=.9,edgecolor='none'),fontsize=10)
axes[0].legend(loc='center',bbox_to_anchor=(.5,.72),fontsize=9)
t=np.array([1.,2.,1.,2.]);p=np.array([2.,1.,2.,1.]);x=np.arange(4)
axes[1].bar(x-.18,t,width=.36,label='目标段长',color='#137c89')
axes[1].bar(x+.18,p,width=.36,label='候选段长',color='#c85226')
axes[1].set(title='总长度相同，沿频率分配仍不同',xlabel='示意段编号（实际实验用全部原始段）',ylabel='段长',xticks=x,xticklabels=['1','2','3','4'],ylim=(0,2.9))
axes[1].text(.04,.91,f'目标与候选总长均为 6\n逐段弧长平方根误差 = {hellinger_loss(p,t):.3f}',transform=axes[1].transAxes,va='top',fontsize=10)
axes[1].legend(loc='upper right',fontsize=9)
for ax in axes:ax.grid(axis='y',alpha=.15)
fig.suptitle('数学示例：检验两种长度信息；不代表拟合收益',fontsize=14)
fig.savefig(out/'mechanism_examples.png',dpi=160,bbox_inches='tight',pad_inches=.2);plt.close(fig)
if args.mechanism_only:
    print(out/'mechanism_examples.png');raise SystemExit

audit=json.loads((run/'server_run/download_audit.json').read_text())
assert audit['status']=='passed' and audit['completed_runs']==800
rows=json.loads((run/'RESULTS.json').read_text(encoding='utf-8'))
parents=json.loads((run/'PARENT_COMPARISONS.json').read_text(encoding='utf-8'))
for data,filename,heading in [(rows,'baseline_intervals.png','相对双主目标基线'),
                              (parents,'parent_intervals.png','相对原累计弧长')]:
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained',sharey=True)
    for ax,case in zip(axes,['case1','case2']):
        subset=[r for r in data if r['case']==case];y=np.arange(len(subset))
        for k,(label,color,offset) in enumerate([('幅值 MSE','#137c89',-.12),('相位 MSE','#c85226',.12)]):
            ci=np.array([r['reduction_pct_bootstrap95'][k] for r in subset]);mean=[r['reduction_pct'][k] for r in subset]
            ax.hlines(y+offset,ci[:,0],ci[:,1],colors=color,lw=1.8)
            ax.scatter(mean,y+offset,c=color,s=28,label=label,zorder=3)
        ax.axvline(0,c='#777777',ls='--',lw=1)
        ax.set_yticks(y,[r['id'] for r in subset]);ax.tick_params(axis='y',labelleft=True)
        ax.set(title=case,xlabel='MSE 均值下降（%）；正数为改善')
        ax.grid(axis='x',alpha=.18)
    axes[0].invert_yaxis();handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2)
    fig.suptitle(heading+'\n100 个配对种子；95% 区间未作同时覆盖校正，修正检验见表',fontsize=13)
    fig.savefig(out/filename,dpi=160,bbox_inches='tight',pad_inches=.2);plt.close(fig)

def supported(data):
    return all(all(r['reduction_pct'][i]>0 and r['holm_adjusted_p_20_primary_comparisons'][i]<.05 for i in range(2)) for r in data)
text=['# 全区间弧长和逐段弧长：公式改进与正式实验','',
    '本轮完成 800 次完整正式运行、42,000,000 次评估。两个 Case 各 4 组，每组 100 个新配对种子（1100—1199）；原累计弧长作直接对照。所有 140,000 个最终个体、配对初始化、代数、评估数、固定选解以及 20 项修正检验通过独立复算。','',
    '## 做了什么','',
    '- G-05-I 全区间弧长误差：比较任意两个频率之间沿曲线走过的长度，检验固定从低频端累计的影响。',
    '- G-05-H 逐段弧长平方根误差：用 Hellinger 型差异比较长度在频率上的分配，保留总长信息。它是旧 SRVF 损失中长度部分的改写，未宣称新发明的距离。',
    '- 原 R-05-C 累计弧长、原双主目标、坐标尺度、频点、电路、预算及最终 WSMSE 选择保持一致。','',
    '## 结果判断','']
for code in ['R-05-C','G-05-I','G-05-H']:
    own=[r for r in rows if r['id']==code]
    assert len(own)==2
    text.append(f"- {code} 对基线："+('本轮两个 Case、两项主误差均有改善及修正检验支持。' if supported(own) else '尚未在本轮同时得到两个 Case、两项主误差的明确改善支持。'))
    direct=[r for r in parents if r['id']==code]
    if direct:
        text.append('  '+('对原累计弧长，两个 Case 的两项改善均有修正检验支持。' if supported(direct) else '对原累计弧长，尚未获得两个 Case、两项主误差均进一步改善的明确支持。'))
text += ['', '本轮筛选决定：', '',
    '- 原累计弧长 R-05-C 仍作优先候选。Case2 两项误差再次下降，且均有修正检验支持；Case1 均值下降，但本轮两项均未通过预定修正检验，不能写成两个 Case 都已稳定有效。',
    '- 全区间弧长 G-05-I 留作备选，不替换原版。它在 Case2 相对基线有效，但相对原版只多下降约 2.07%／0.28%；Case1 幅值反而变差约 1.37%，相位改善约 7.42%。四项直接比较的区间均跨 0，Holm p 均为 1；这不是等效性证明，也不是升级成功。',
    '- 逐段弧长平方根 G-05-H 退出当前优先候选。在 Case1 相对基线两项明显变差；相对原累计弧长，两个 Case 的两项也均变差且有修正检验支持。公式能解释不代表适合作为辅助优化目标。',
    '- 不据此继续堆叠辅助目标，也不根据本轮结果调整权重、频点或主值相位定义。本轮没有找到明确超过原累计弧长的通用改进。', '',
    '后续更值得检验的是原累计弧长在新测量、重复测量或有依据的噪声条件下能否保持收益，以及 Case1 为什么对随机初始化更敏感。没有测量噪声依据时，不把随意指定噪声幅度的模拟当成真实性验证。']
for heading,data in [('相对本轮基线',rows),('相对原累计弧长：新公式是否进一步改善',parents)]:
    text += ['', '## '+heading,'','|指标|Case1 幅值／相位 MSE 下降|Case2 幅值／相位 MSE 下降|','|---|---:|---:|']
    for code in dict.fromkeys(r['id'] for r in data):
        pair=[next(r for r in data if r['id']==code and r['case']==case) for case in ['case1','case2']]
        text.append('|'+code+' '+pair[0]['label'].replace('对照：','')+'|'+ '|'.join(f"{r['reduction_pct'][0]:.2f}%／{r['reduction_pct'][1]:.2f}%" for r in pair)+'|')
text += ['', '百分比为 100 次运行的 MSE 均值比例下降，正数改善、负数变差。两个主误差来自同一个最终选中解；区间和检验分别报告，未显著不表示等效。','',
    '下降比例为 100 ×（比较组 MSE 均值 − 本组 MSE 均值）／比较组 MSE 均值。配对 bootstrap 区间针对均值比例，Wilcoxon 针对配对差值的秩，且后者进行了 20 项 Holm 修正；因此未修正区间不跨 0 与修正检验未通过可以同时出现，不能择其一宣布成功。','',
    '## 区间和修正检验','', '|Case|指标|比较对象|幅值下降 95% 区间|相位下降 95% 区间|幅值／相位 Holm p|',
    '|---|---|---|---|---|---|']
for r in rows+parents:
    ci=r['reduction_pct_bootstrap95'];p=r['holm_adjusted_p_20_primary_comparisons']
    text.append(f"|{r['case']}|{r['id']}|{r.get('parent','基线')}|[{ci[0][0]:.2f}%, {ci[0][1]:.2f}%]|[{ci[1][0]:.2f}%, {ci[1][1]:.2f}%]|{p[0]:.6g}／{p[1]:.6g}|")
text += ['', '## 如何解释','',
    '两种改法均有可直接推导的数学意义。全区间误差相当于把累计长度误差减去其频率平均值后再衡量波动，并非额外独立的物理约束；这也可能减弱对端点附近长度偏差的关注。逐段平方根误差没有累计后的正负抵消，但更直接依赖相邻点变化，不能未经实验就称抗噪。',
    '逐段平方根误差使用 Hellinger 的平方根密度表示，见 [Liero、Mielke、Savaré 原文第 2 节](https://arxiv.org/pdf/1509.00068)。与 [SRV 原始表示](https://pubmed.ncbi.nlm.nih.gov/20921581/) 的长度／方向分解在冻结公式中逐项推导；外层平方根也会影响 NSGA-II 拥挤距离，不能把新旧效果差异完全归因于去掉方向项。',
    '当前结果仍基于原两条目标曲线。频率采样、噪声、Case2 主值相位跳变和模型可行域限制仍在；同一数据上新种子不等于新测量验证，跨轮探索选择偏差也没有被本轮 Holm 校正全部消除。','',
    '## 公式和文件','',
    f'- [详细公式、推导、局限与运行前冻结方案]({(run/"README.md").as_posix()})',
    f'- [完整结果与检验]({(run/"RESULTS.md").as_posix()})',
    f'- [独立复算审计]({(run/"server_run/download_audit.json").as_posix()})',
    f'![相对基线]({(out/"baseline_intervals.png").as_posix()})',
    f'![相对原累计弧长]({(out/"parent_intervals.png").as_posix()})',
    f'![公式解释的数学示例]({(out/"mechanism_examples.png").as_posix()})']
path=root/'output/全区间与逐段弧长_改进实验总结_20261003.md'
path.write_text('\n'.join(text)+'\n',encoding='utf-8')
print(path)
for r in rows+parents:
    print(json.dumps({k:r[k] for k in ['case','id','reduction_pct','holm_adjusted_p_20_primary_comparisons']},ensure_ascii=False))

"""Scientific figures, not illustration-dependent scientific parameters."""
from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
root=Path(__file__).resolve().parents[1];run=root/'experiments/curve3d_directional_20261003'
sys.path.insert(0,str(run));from global_metrics import cumulative_length,variations
fonts={f.name for f in font_manager.fontManager.ttflist}
plt.rcParams['font.sans-serif']=[f for f in ['Microsoft YaHei','SimHei','Noto Sans CJK SC'] if f in fonts]+['DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False
out=run/'figures';out.mkdir(exist_ok=True)
u=np.linspace(0,1,401);p=np.column_stack([u,u*u,u]);q=np.column_stack([u,u,u*u]);opposite=np.column_stack([u,-u*u,-u])
fig,axes=plt.subplots(2,2,figsize=(11.8,7.3),layout='constrained')
axes[0,0].plot(u,cumulative_length(p),c='#137c89',lw=3,label='参考曲线');axes[0,0].plot(u,cumulative_length(q),c='#c85226',ls='--',label='幅值与相位走法交换')
axes[0,0].set(title='两轴交换，三维累计长度仍完全相同',xlabel='频率坐标 u',ylabel='累计长度')
vp,vn=variations(p);qp,qn=variations(q);np_,nn=variations(opposite)
axes[1,0].plot(u,(vp+vn)[:,0],c='#137c89');axes[1,0].plot(u,(qp+qn)[:,0],c='#c85226',ls='--')
axes[1,0].set(title='分开比较幅值变化量，就能看到区别',xlabel='频率坐标 u',ylabel='第一投影累计变化量')
axes[0,1].plot(u,(vp+vn)[:,0],c='#137c89',lw=3);axes[0,1].plot(u,(np_+nn)[:,0],c='#c85226',ls='--')
axes[0,1].set(title='走向反转，总变化量仍完全相同',xlabel='频率坐标 u',ylabel='第一投影累计变化量')
axes[1,1].plot(u,vp[:,0],c='#137c89');axes[1,1].plot(u,np_[:,0],c='#c85226',ls='--')
axes[1,1].set(title='区分上升／下降，就能看到区别',xlabel='频率坐标 u',ylabel='第一投影累计上升量')
for ax in axes.flat:ax.grid(alpha=.18)
fig.suptitle('公式的结构盲点：数学示例，不是拟合结果\n左列为两轴交换，右列为走向反转；实际指标使用全部原频点',fontsize=14)
handles=[plt.Line2D([],[],color='#137c89',lw=3),plt.Line2D([],[],color='#c85226',ls='--')]
fig.legend(handles,['参考曲线','结构不同的曲线'],loc='outside lower center',ncol=2)
fig.savefig(out/'structural_examples.png',dpi=150,bbox_inches='tight',pad_inches=.25);plt.close(fig)
for filename,name,title in [('RESULTS.json','baseline_intervals.png','相对本轮基线'),('PARENT_COMPARISONS.json','parent_intervals.png','相对对应原方案')]:
    if not (run/filename).exists():continue
    rows=json.loads((run/filename).read_text(encoding='utf-8'));fig,axes=plt.subplots(1,2,figsize=(12.4,6.6),layout='constrained',sharey=True)
    for ax,case in zip(axes,['case1','case2']):
        data=[r for r in rows if r['case']==case];y=np.arange(len(data))
        for k,(label,color,shift) in enumerate([('幅值 MSE','#137c89',-.13),('相位 MSE','#c85226',.13)]):
            means=[r['reduction_pct'][k] for r in data];ci=np.array([r['reduction_pct_bootstrap95'][k] for r in data])
            ax.hlines(y+shift,ci[:,0],ci[:,1],colors=color,lw=1.6);ax.scatter(means,y+shift,c=color,s=25,label=label)
        labels=[r['id']+(' / '+r['parent'] if 'parent' in r else '') for r in data]
        ax.axvline(0,c='grey',ls='--',lw=1);ax.grid(axis='x',alpha=.18)
        ax.set(title=case,xlabel='MSE 均值下降（%）；右侧为改善');ax.set_yticks(y,labels);ax.tick_params(axis='y',labelleft=True)
    axes[0].invert_yaxis();handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2)
    fig.suptitle('第四轮：'+title+'\n100 个新配对种子，95% 区间未作同时覆盖校正；组合对两个单辅助分别比较',fontsize=13)
    fig.savefig(out/name,dpi=150,bbox_inches='tight',pad_inches=.25);plt.close(fig)
print(out)

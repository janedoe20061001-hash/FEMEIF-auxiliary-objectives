"""Scientific mechanism illustration and complete reduction intervals."""
from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
root=Path(__file__).resolve().parents[1];run=root/'experiments/curve3d_refinement_20261003'
sys.path.insert(0,str(run))
from global_metrics import cumulative_area,cumulative_length
available={f.name for f in font_manager.fontManager.ttflist}
plt.rcParams['font.sans-serif']=[f for f in ['Microsoft YaHei','SimHei','Noto Sans CJK SC'] if f in available]+['DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False
out=run/'figures';out.mkdir(exist_ok=True)
u=np.linspace(0,1,601);straight=np.column_stack([u,0*u,0*u])
detour=np.column_stack([u,np.sin(2*np.pi*u),0*u])
wiggle=np.column_stack([u,.025*np.sin(20*np.pi*u),0*u])
fig,axes=plt.subplots(2,2,figsize=(11.8,7.8),layout='constrained')
axes[0,0].plot(u,straight[:,1],c='#137c89',label='直线');axes[0,0].plot(u,detour[:,1],c='#c85226',label='先上绕、再下绕')
axes[0,0].set(title='终点面积都为零，中途走法不同',xlabel='频率坐标 u',ylabel='第二坐标');axes[0,0].legend()
axes[1,0].plot(u,cumulative_area(straight)[:,2],c='#137c89');axes[1,0].plot(u,cumulative_area(detour)[:,2],c='#c85226')
axes[1,0].set(title='累计面积保留了中途的区别',xlabel='频率坐标 u',ylabel='累计有向面积（第三分量）')
axes[0,1].plot(u,straight[:,1],c='#137c89',label='直线');axes[0,1].plot(u,wiggle[:,1],c='#c85226',label='位置偏差很小的起伏')
axes[0,1].set(title='位置很近，走过的长度仍可不同',xlabel='频率坐标 u',ylabel='第二坐标');axes[0,1].legend()
axes[1,1].plot(u,cumulative_length(straight),c='#137c89');axes[1,1].plot(u,cumulative_length(wiggle),c='#c85226')
axes[1,1].set(title='累计弧长直接比较走过多少距离',xlabel='频率坐标 u',ylabel='未各自归一化的累计长度')
for ax in axes.flat:ax.grid(alpha=.18)
fig.suptitle('为什么值得修改：数学示例，不是拟合结果\n示例中的曲线取值不是指标参数；实际指标用全部原始频点',fontsize=14)
fig.savefig(out/'mechanism_examples.png',dpi=150,bbox_inches='tight',pad_inches=.25);plt.close(fig)
for filename,name,title in [('RESULTS.json','baseline_intervals.png','相对本轮基线'),
                            ('PARENT_COMPARISONS.json','parent_intervals.png','相对对应原版')]:
    if not (run/filename).exists():continue
    rows=json.loads((run/filename).read_text(encoding='utf-8'))
    fig,axes=plt.subplots(1,2,figsize=(12,6),layout='constrained',sharey=True)
    for ax,case in zip(axes,['case1','case2']):
        data=[r for r in rows if r['case']==case];y=np.arange(len(data))
        for k,(label,color,shift) in enumerate([('幅值 MSE','#137c89',-.13),('相位 MSE','#c85226',.13)]):
            mean=[r['reduction_pct'][k] for r in data];ci=np.array([r['reduction_pct_bootstrap95'][k] for r in data])
            ax.hlines(y+shift,ci[:,0],ci[:,1],colors=color,lw=1.6);ax.scatter(mean,y+shift,c=color,s=25,label=label)
        ax.axvline(0,c='grey',ls='--',lw=1);ax.grid(axis='x',alpha=.18)
        ax.set(title=case,xlabel='MSE 均值下降（%）；右侧为改善')
        ax.set_yticks(y,[r['id'] for r in data]);ax.tick_params(axis='y',labelleft=True)
    axes[0].invert_yaxis()
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,fontsize=10)
    fig.suptitle('第三轮：'+title+'\n100 个新配对种子，均值降幅及 95% 区间（区间未作同时覆盖校正）',fontsize=13)
    fig.savefig(out/name,dpi=150,bbox_inches='tight',pad_inches=.25);plt.close(fig)
print(out)

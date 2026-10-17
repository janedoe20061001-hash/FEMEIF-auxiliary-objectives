"""Deterministic scientific figures, no representative seed cherry-picking."""
from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

root=Path(__file__).resolve().parents[1]
run=root/'experiments/curve3d_objectives_20261003'
sys.path.insert(0,str(run));sys.path.insert(0,str(run/'case1'))
from legacy_objectives import impedance,TARGET
sys.path.insert(0,str(run/'case2'))
from case2 import load_target
from global_metrics import GlobalObjectives
fonts=['Microsoft YaHei','SimHei','Noto Sans CJK SC']
installed={font.name for font in font_manager.fontManager.ttflist}
plt.rcParams['font.sans-serif']=[f for f in fonts if f in installed]+['DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False
output=run/'figures';output.mkdir(exist_ok=True)
fig=plt.figure(figsize=(12,9),layout='constrained')
for index,case in enumerate(['case1','case2']):
    cfg=json.loads((run/case/'config.json').read_text(encoding='utf-8'))
    if case=='case1':f=np.geomspace(100,1e6,180);z=impedance(f,TARGET)[0]
    else:f,z=load_target({**cfg['target_data'],'file':str(run/case/cfg['target_data']['file'])})
    obj=GlobalObjectives(f,target_z=z)
    for j,(coord,name) in enumerate([('linear','线性归一化'),('asinh','有符号对数式压缩后归一化')]):
        ax=fig.add_subplot(2,2,2*index+j+1,projection='3d')
        p=obj.reference[coord]['p']
        ax.plot(*p.T,color='#137c89',lw=1.8)
        ax.scatter(*p[0],c='#2356b4',s=32,label='最低频率')
        ax.scatter(*p[-1],c='#c85226',s=32,label='最高频率')
        ax.set(xlabel='对数频率 u',ylabel='实部坐标',zlabel='虚部坐标',title=f'{case}：{name}')
        ax.set_box_aspect((1,1,1));ax.view_init(elev=24,azim=-58)
        ax.legend(loc='upper left',fontsize=8)
fig.suptitle('同一条目标阻抗曲线，坐标变换会改变空间形状',fontsize=16)
fig.savefig(output/'target_coordinates.png',dpi=160,bbox_inches='tight',pad_inches=.25);plt.close(fig)

results=run/'RESULTS.json'
if results.exists():
    rows=json.loads(results.read_text(encoding='utf-8'))
    fig,axes=plt.subplots(1,2,figsize=(13,7),layout='constrained',sharey=True)
    for ax,case in zip(axes,['case1','case2']):
        r=[row for row in rows if row['case']==case];y=np.arange(len(r))
        for i,(label,color,offset) in enumerate([('幅值 MSE','#137c89',-.12),('相位 MSE','#c85226',.12)]):
            mean=np.array([v['reduction_pct'][i] for v in r]);ci=np.array([v['reduction_pct_bootstrap95'][i] for v in r])
            ax.hlines(y+offset,ci[:,0],ci[:,1],color=color,linewidth=1.7)
            ax.scatter(mean,y+offset,color=color,s=25,label=label,zorder=3)
        ax.axvline(0,c='grey',ls='--',lw=1)
        ax.set(title=case,xlabel='相对基线误差下降（%）；右侧为改善')
        ax.set_yticks(y,[v['id'] for v in r])
        ax.grid(axis='x',alpha=.18);ax.legend(loc='upper right')
    axes[0].invert_yaxis()
    fig.suptitle('100 个配对种子的均值下降及 95% 区间\nLIN：线性坐标；ASH：有符号对数式坐标；区间未作同时覆盖修正',fontsize=13)
    fig.savefig(output/'primary_reduction_intervals.png',dpi=160,bbox_inches='tight',pad_inches=.25);plt.close(fig)
print(output)

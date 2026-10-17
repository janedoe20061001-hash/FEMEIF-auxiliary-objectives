"""New target coordinates and all paired-seed reductions, without seed selection."""
from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

root=Path(__file__).resolve().parents[1];run=root/'experiments/curve3d_logpolar_native_20261003'
sys.path.insert(0,str(run/'case1'))
from legacy_objectives import impedance,TARGET
sys.path.insert(0,str(run/'case2'))
from case2 import load_target
installed={f.name for f in font_manager.fontManager.ttflist}
plt.rcParams['font.sans-serif']=[f for f in ['Microsoft YaHei','SimHei','Noto Sans CJK SC'] if f in installed]+['DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False
output=run/'figures';output.mkdir(exist_ok=True)
fig=plt.figure(figsize=(12,5.7),layout='constrained')
for j,case in enumerate(['case1','case2']):
    cfg=json.loads((run/case/'config.json').read_text(encoding='utf-8'))
    if case=='case1':f=np.geomspace(100,1e6,180);z=impedance(f,TARGET)[0]
    else:f,z=load_target({**cfg['target_data'],'file':str(run/case/cfg['target_data']['file'])})
    logf=np.log10(f);u=(logf-logf[0])/(logf[-1]-logf[0]);m=20*np.log10(abs(z));phase=np.angle(z)
    p=np.column_stack([u,(m-m.min())/np.ptp(m),(phase-phase.min())/np.ptp(phase)])
    ax=fig.add_subplot(1,2,j+1,projection='3d');ax.plot(*p.T,c='#137c89',lw=1.6)
    ax.scatter(*p[0],c='#2356b4',s=30,label='最低频率');ax.scatter(*p[-1],c='#c85226',s=30,label='最高频率')
    neg=z.real<0
    if np.any(neg):ax.scatter(*p[neg].T,c='#b12c51',marker='x',s=50,label='16 MHz，相位 107°')
    ax.set(xlabel='对数频率 u',ylabel='对数幅值坐标',zlabel='主值相位坐标',title=case)
    ax.set_box_aspect((1,1,1));ax.view_init(elev=24,azim=-58);ax.legend(loc='upper left',fontsize=8)
fig.suptitle('第二轮：把频率、对数幅值、相位放进同一条三维曲线',fontsize=15)
fig.savefig(output/'target_logpolar.png',dpi=160,bbox_inches='tight',pad_inches=.25);plt.close(fig)

if (run/'RESULTS.json').exists():
    rows=json.loads((run/'RESULTS.json').read_text(encoding='utf-8'))
    fig,axes=plt.subplots(1,2,figsize=(12,5.3),layout='constrained',sharey=True)
    for ax,case in zip(axes,['case1','case2']):
        selected=[r for r in rows if r['case']==case];y=np.arange(len(selected))
        for k,(label,color,shift) in enumerate([('幅值 MSE','#137c89',-.12),('相位 MSE','#c85226',.12)]):
            mean=[r['reduction_pct'][k] for r in selected];ci=np.array([r['reduction_pct_bootstrap95'][k] for r in selected])
            ax.hlines(y+shift,ci[:,0],ci[:,1],colors=color,lw=1.7);ax.scatter(mean,y+shift,c=color,s=26,label=label)
        ax.axvline(0,c='grey',ls='--',lw=1);ax.grid(axis='x',alpha=.18)
        ax.set(title=case,xlabel='相对基线误差下降（%）；右侧为改善');ax.set_yticks(y,[r['id'] for r in selected]);ax.legend(loc='lower left' if case=='case1' else 'upper right')
    axes[0].invert_yaxis()
    fig.suptitle('对数幅值—相位坐标：100 个配对种子的均值下降及 95% 区间\n区间未作同时覆盖修正',fontsize=13)
    fig.savefig(output/'primary_reduction_intervals.png',dpi=160,bbox_inches='tight',pad_inches=.25);plt.close(fig)
print(output)

"""Figures from frozen structural labels and metric scores; no re-selection."""
from scan import *
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)

def save(fig,name):
    fig.savefig(FIG/name,dpi=170,bbox_inches='tight');plt.close(fig)

def main():
    inv=np.load(OUT/'INVENTORY.npz');res=np.load(OUT/'STRUCTURE_RESULTS.npz')
    bank=json.loads((OUT/'SCORING_BANK.json').read_text());data=target('case1');c=data['circuit']
    event=next(e for e in data['record']['events'] if e['event_id']=='P3')
    good=res['stable'][:,2]&res['complete'][:,2]
    centered=good&(abs(res['values'][:,2,0])<=event['width_3db_decades'])
    fig,axs=plt.subplots(3,2,figsize=(13,12),layout='constrained')
    for mi,(m,ax) in enumerate(zip(METHODS,axs.flat)):
        counts=centered[inv['lookup'][mi]].sum(axis=-1)
        im=ax.imshow(counts,aspect='auto',vmin=0,vmax=max(1,int(counts.max())),cmap='Blues')
        for si,gi in zip(*np.where(counts>0)):ax.text(gi,si,str(counts[si,gi]),ha='center',va='center',fontsize=8,color='white' if counts[si,gi]>.5*counts.max() else 'black')
        ax.set(xticks=range(16),xticklabels=STAGES,yticks=range(12),yticklabels=SEEDS,title=m,xlabel='保存代数',ylabel='seed')
        ax.tick_params(axis='x',rotation=60,labelsize=8);fig.colorbar(im,ax=ax,shrink=.8,label='该保存种群的记录数')
    fig.suptitle('P3完整且峰中心偏移≤一个目标3dB宽度\n数字为记录数；重复参数与同seed阶段并非独立样本',fontsize=14)
    save(fig,'saved_centered_P3.png')

    rows=list(csv.DictReader((OUT/'PAIR_METRIC_DIRECTIONS.csv').open(encoding='utf-8-sig')))
    metrics=['ARX_L','ARX_S_L','Local_ARX','Combined','H','J0','J0_W','J0_local_weighted']
    matrix=np.array([[{'correct':1,'wrong':0,'uncertain':.5}[next(r['robust_direction'] for r in rows if r['pair_id']==p['pair_id'] and r['metric']==m)] for m in metrics] for p in bank['pairs']])
    fig,ax=plt.subplots(figsize=(11,7),layout='constrained');ax.imshow(matrix,cmap=ListedColormap(['#d57c72','#d9d9d9','#80b79d']),vmin=0,vmax=1,aspect='auto')
    ax.set(xticks=range(8),xticklabels=metrics,yticks=range(12),yticklabels=[f"{p['pair_id']} · {p['region']} · {p['seed']}" for p in bank['pairs']],title='固定候选对的局部判别：1方向符合局部标签，0方向相反')
    ax.tick_params(axis='x',rotation=30)
    for i in range(12):
        for j in range(8):ax.text(j,i,'1' if matrix[i,j]==1 else '0',ha='center',va='center',fontsize=14)
    save(fig,'pair_directions.png')

    p=bank['pairs'][1];f0=event['frequency_hz'];f=np.unique(np.r_[np.linspace(f0-220,f0+220,8000),np.linspace(f0-30,f0+30,4000)])
    responses=[c.response(f,c.p)[0]]+[c.response(f,10**np.array(p[s]['log_parameters']))[0] for s in ['better','worse']]
    labels=['目标','P3较好候选','P3退化候选'];colors=['black','#2878a4','#c76542']
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for z,label,color in zip(responses,labels,colors):
        axs[0,0].plot(f-f0,20*np.log10(abs(z)),label=label,color=color,lw=1.5)
        axs[0,1].plot(f-f0,np.angle(z,deg=True),color=color,lw=1.5)
        axs[1,0].plot(z.real,z.imag,color=color,lw=1.5)
    axs[0,0].set(xlabel='相对目标峰中心 / Hz',ylabel='20 log10|Z| / dBΩ',title='同一保存种群：较好候选仍有其他频段缺陷');axs[0,0].legend()
    axs[0,1].set(xlabel='相对目标峰中心 / Hz',ylabel='相位 / °',title='未移动或配准候选频率')
    axs[1,0].set(xlabel='Re Z / Ω',ylabel='Im Z / Ω',title='上述同一频段的复平面轨迹');axs[1,0].set_aspect('equal',adjustable='datalim')
    mm=['ARX_S_L','J0','Local_ARX','H','Combined'];ratios=[]
    for m in mm:
        r=next(r for r in rows if r['pair_id']==p['pair_id'] and r['metric']==m);ratios.append(float(r['better_score'])/float(r['worse_score']))
    axs[1,1].barh(mm,ratios,color=['#80b79d' if r<1 else '#d57c72' for r in ratios]);axs[1,1].set_xscale('log');axs[1,1].axvline(1,color='black',ls='--');axs[1,1].set(xlabel='较好候选分数 / 退化候选分数（越小越好）',title='局部面积将较好峰罚得更重')
    from matplotlib.ticker import FixedLocator,FixedFormatter,NullLocator
    axs[1,1].xaxis.set_major_locator(FixedLocator([.2,.5,1,2,5]));axs[1,1].xaxis.set_major_formatter(FixedFormatter(['0.2','0.5','1','2','5']));axs[1,1].xaxis.set_minor_locator(NullLocator())
    fig.suptitle('反例 pair_001：G0，seed 47007，第40代',fontsize=14)
    save(fig,'P3_counterexample.png')

    fig=plt.figure(figsize=(10,7),layout='constrained');ax=fig.add_subplot(projection='3d')
    for z,label,color in zip(responses,labels,colors):ax.plot(f-f0,z.real,z.imag,label=label,color=color,lw=1.5)
    ax.set(xlabel='f − 目标峰中心 / Hz',ylabel='Re Z / Ω',zlabel='Im Z / Ω',title='pair_001：保留频率标记的三维复阻抗轨迹');ax.legend();save(fig,'P3_3D.png')

    from local_geometry import LocalGeometry
    model=LocalGeometry();pp=bank['pairs'][6];ww=next(w for w in model.windows if w['block']['label']==pp['region'])
    ff=np.logspace(*ww['x'][[0,-1]],6000);zz=[c.response(ff,c.p)[0]]+[c.response(ff,10**np.array(pp[s]['log_parameters']))[0] for s in ['better','worse']]
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for z,label,color in zip(zz,['目标','窗口内无额外事件','窗口内有额外事件'],colors):
        axs[0].plot(ff,20*np.log10(abs(z)),label=label,color=color);axs[1].plot(z.real,z.imag,color=color)
    axs[0].set(xlabel='f / Hz',ylabel='20 log10|Z| / dBΩ',title='pair_006：额外结构标签与点误差存在取舍');axs[0].set_xscale('log');axs[0].legend(fontsize=9)
    axs[1].set(xlabel='Re Z / Ω',ylabel='Im Z / Ω',title='同一间隙窗口的复平面轨迹');save(fig,'extra_structure_tradeoff.png')

    probe=json.loads((OUT/'CURVATURE_COUNTEREXAMPLE.json').read_text())
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained');x=np.arange(3)
    ax.bar(x-.2,[v['better_area_over_target'] for v in probe['ratios']],.4,label='P3较好候选 / 目标',color='#2878a4')
    ax.bar(x+.2,[v['worse_area_over_target'] for v in probe['ratios']],.4,label='P3退化候选 / 目标',color='#c76542')
    ax.axhline(1,color='black',ls='--');ax.axhline(2,color='gray',ls=':');ax.set(xticks=x,xticklabels=['1/32','1/16','1/8'],xlabel='固定半窗口 h',ylabel='同号局部有向面积比',title='目标峰中心处：较好候选面积超过目标2倍，退化候选接近0');ax.legend();save(fig,'local_area_overshoot.png')

    mi=METHODS.index('ARX_S_L');si=SEEDS.index(47000);u=inv['lookup'][mi,si,-1];F=inv['primary'][mi,si,-1];idx=int(inv['selected'][mi,si,-1]);_,sel=knee(F)
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained');ax.scatter(F[:,0],F[:,1],s=18,color='#b9bec3',label='最终候选')
    front=np.array(sel['front_indices']);ax.scatter(F[front,0],F[front,1],s=35,facecolors='none',edgecolors='black',label='MAG/PHA二维前沿')
    cg=centered[u];ax.scatter(F[cg,0],F[cg,1],s=65,color='#2878a4',marker='D',label='P3完整且居中')
    ax.scatter(*F[idx],s=180,marker='*',color='#c76542',label='原正式膝点');ax.set(xlabel='MAG MSE',ylabel='PHA MSE',title='ARX_S_L，seed 47000：保留的居中P3均不在二维前沿');ax.legend();save(fig,'P3_front_example.png')

if __name__=='__main__':main()

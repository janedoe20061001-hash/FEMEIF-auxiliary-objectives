"""Tables and plots from immutable mechanism scores; no parameter tuning."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False})

def main():
    records=[json.loads(p.read_text()) for p in sorted((OUT/'scores').glob('*.json'))];control=[];fail=[];pairrows=[]
    model=Geometry(control_target());win=model.windows;global_scale=max(w['scale'] for w in win)
    for r in records:
        if not r['numerical_passed']:fail.append(dict(id=r['id'],target=r['target'],error=r.get('error'),window_failures=[k for k,v in r.get('windows',{}).items() if not v['numerical_passed']]))
        if r['target']!='controlled':continue
        if 'windows' not in r:continue
        mainrow=r['windows']['P_control'];row=dict(id=r['id'],**r['meta'],numerical_passed=r['numerical_passed'])
        row.update(mainrow['scores'])
        row['G0_global']=sum(r['windows'][w['block']['label']]['scores']['J0']*(w['scale']/global_scale)**2*(w['x'][-1]-w['x'][0])/4 for w in win)
        for m in ['MAG','PHA']:row[m+'_global']=sum(r['windows'][w['block']['label']]['scores'][m]*(w['x'][-1]-w['x'][0])/4 for w in win)
        for m in ['Root','B','J0','ARX_S_L','Local_ARX','D','H']:row[m+'_all_windows']=float(np.mean([v['scores'][m] for v in r['windows'].values()]))
        control.append(row)
    csvout(OUT/'CONTROLLED_SCORES.csv',control);dump(OUT/'NUMERICAL_FAILURES.json',fail)
    single=[]
    for axis,baseline in [('amplitude',1.),('offset',0.),('q_ratio',1.)]:
        other=[v for v in ['amplitude','offset','q_ratio'] if v!=axis]
        rows=[r for r in control if r['family']=='controlled_grid' and all(r[v]==(0 if v=='offset' else 1) for v in other)]
        for r in sorted(rows,key=lambda r:r[axis]):single.append(dict(axis=axis,value=r[axis],**{m:r[m] for m in ['J0','Local_ARX','Root','B','D','H','ARX_S_L','MAG','PHA']}))
    csvout(OUT/'SINGLE_FACTOR_RESPONSES.csv',single)
    # A physically controlled analogue of the known area-overshoot failure.
    good=next(r for r in control if r['family']=='controlled_grid' and r['amplitude']==1.25 and r['offset']==0 and r['q_ratio']==1.5)
    bad=next(r for r in control if r['family']=='controlled_grid' and r['amplitude']==.1 and r['offset']==0 and r['q_ratio']==1)
    probe=dict(scope='Predeclared physical analogue, not all combinations or a new gate',better_parameters=dict(amplitude=1.25,offset=0,q_ratio=1.5),worse_parameters=dict(amplitude=.1,offset=0,q_ratio=1),
        scores={m:dict(better=good[m],worse=bad[m],ratio=good[m]/bad[m]) for m in ['J0','Local_ARX','Root','B','D','H','ARX_S_L']})
    dump(OUT/'CONTROLLED_COUNTEREXAMPLE.json',probe)
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    fig,axs=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,axis,xlabel in zip(axs,['amplitude','offset','q_ratio'],['R / R*','中心偏移 / 目标对数半功率带宽','Q / Q*']):
        rows=[r for r in single if r['axis']==axis]
        for m in ['J0','Local_ARX','Root','B']:
            scale=max(r[m] for r in rows)
            ax.plot([r['value'] for r in rows],[r[m]/scale if scale else 0 for r in rows],marker='.',label=m)
        ax.set(xlabel=xlabel,ylabel='损失 / 该指标在本图的最大值',ylim=(0,1.05));ax.legend(fontsize=8)
        if axis=='q_ratio':ax.set_xscale('log',base=2)
    fig.suptitle('单因素响应：仅为显示而分别归一，不改变实际评分或排名')
    fig.savefig(figdir/'single_factors.png',dpi=160);plt.close(fig)
    metrics=['J0','Local_ARX','Root','B'];fig,axs=plt.subplots(1,4,figsize=(16,4),layout='constrained')
    for ax,m in zip(axs,metrics):
        matrix=np.array([[next(r[m] for r in control if r['family']=='controlled_grid' and r['amplitude']==a and r['offset']==0 and r['q_ratio']==q) for q in CONFIG['q_ratios']] for a in CONFIG['amplitudes']])
        image=ax.imshow(np.log10(np.maximum(matrix,1e-10)),origin='lower',aspect='auto');ax.set(title=m,xticks=range(9),xticklabels=CONFIG['q_ratios'],yticks=range(8),yticklabels=CONFIG['amplitudes'],xlabel='Q / Q*',ylabel='R / R*');ax.tick_params(axis='x',rotation=60,labelsize=7);fig.colorbar(image,ax=ax,label='log10损失')
    fig.suptitle('固定中心：高度与Q的交互，各图颜色单独定标')
    fig.savefig(figdir/'height_Q_interaction.png',dpi=160);plt.close(fig)
    c=model.c;w=next(w for w in win if w['block']['label']=='P_control');f=np.logspace(*w['x'][[0,-1]],8000)
    curves=[c.response(f,c.p)[0],c.response(f,c.params(1.25,0,1.5))[0],c.response(f,c.params(.1,0,1))[0]]
    fig=plt.figure(figsize=(12,5),layout='constrained');ax=fig.add_subplot(121,projection='3d');ax2=fig.add_subplot(122)
    for z,label,color in zip(curves,['目标','R×1.25，Q×1.5','R×0.1，Q不变'],['black','#2878a4','#c76542']):
        ax.plot(f-c.f0,z.real,z.imag,label=label,color=color);ax2.plot(f-c.f0,20*np.log10(abs(z)),label=label,color=color)
    ax.set(xlabel='f − f* / Hz',ylabel='Re Z / Ω',zlabel='Im Z / Ω',title='独立控制峰阻抗与Q的物理电路');ax.legend(fontsize=8)
    ax2.set(xlabel='f − f* / Hz',ylabel='20 log10|Z| / dBΩ',title='预设受控对照：近目标结构与严重衰减');ax2.legend(fontsize=8)
    fig.savefig(figdir/'controlled_3D.png',dpi=160);plt.close(fig)
    import csv
    directions=list(csv.DictReader((OUT/'PAIR_DIRECTIONS.csv').open(encoding='utf-8-sig')))
    for scope in ['dev','held']:
        rr=[r for r in directions if r['scope']==scope];pairs=sorted({r['pair_id'] for r in rr})
        if not pairs:continue
        names=['ARX_S_L','Local_ARX','Root','B','J0','J0_W','D','H']
        vals=np.array([[{'correct':1,'wrong':0,'uncertain':.5,'numerical_failure':.5}[next(r['direction'] for r in rr if r['pair_id']==p and r['metric']==m)] for m in names] for p in pairs])
        fig,ax=plt.subplots(figsize=(10,max(4,len(pairs)*.3)),layout='constrained');ax.imshow(vals,cmap=matplotlib.colors.ListedColormap(['#d57c72','#c0c0c0','#80b79d']),vmin=0,vmax=1,aspect='auto')
        ax.set(xticks=range(len(names)),xticklabels=names,yticks=range(len(pairs)),yticklabels=pairs,title=f'{scope}：1符合局部标签，0相反，?数值或方向不确定')
        for i in range(len(pairs)):
            for j in range(len(names)):ax.text(j,i,'1' if vals[i,j]==1 else '0' if vals[i,j]==0 else '?',ha='center',va='center')
        fig.savefig(figdir/f'{scope}_directions.png',dpi=160);plt.close(fig)
    rr=[r for r in directions if r['scope']=='dev' and r['pair_id']=='pair_001' and r['metric'] in ['J0','Local_ARX','Root','B','D','H']]
    if rr and all(r['better_score'] and r['worse_score'] for r in rr):
        ratios=[float(r['better_score'])/float(r['worse_score']) for r in rr]
        fig,ax=plt.subplots(figsize=(8,4),layout='constrained');ax.barh([r['metric'] for r in rr],ratios,color=['#80b79d' if v<1 else '#d57c72' for v in ratios]);ax.axvline(1,color='black',ls='--');ax.set(xlabel='较好候选 / 退化候选的局部损失',title='真实候选反例 pair_001：数值小于1才符合局部标签')
        fig.savefig(figdir/'actual_counterexample.png',dpi=160);plt.close(fig)
    bank=json.loads((OUT/'HOLDOUT_BANK.json').read_text());pair=next(p for p in bank['selected'] if p['pair_id']=='held_017')
    m=Geometry(target(pair['target']));w=next(w for w in m.windows if w['block']['label']==pair['region']);f=np.logspace(*w['x'][[0,-1]],8000)
    pars=[m.c.p]+[10**np.array(pair[s]['log_parameters']) for s in ['better','worse']]
    zz=[m.c.response(f,p)[0] for p in pars];labels=['目标','P5符合原结构门槛','P5未匹配'];colors=['black','#2878a4','#c76542']
    fig,axs=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for z,label,col in zip(zz,labels,colors):axs[0].plot(f/1000,20*np.log10(abs(z)),color=col,label=label);axs[1].plot(z.real,z.imag,color=col)
    axs[0].set(xlabel='f / kHz',ylabel='20 log10|Z| / dBΩ',title='新留出反例 held_017：P5');axs[0].legend(fontsize=8)
    axs[1].set(xlabel='Re Z / Ω',ylabel='Im Z / Ω',title='同一目标窗口的复平面轨迹')
    h=m.steps[0];t=np.linspace(h,1-h,3000);values=[]
    for p in pars:
        v=m.response(w,np.r_[t-h,t,t+h],p)[0];vm,vc,vp=np.split(v,3);values.append(vc-(vm+vp)/2)
    j=m.windows.index(w);ff=10**(w['x'][0]+(w['x'][-1]-w['x'][0])*t)/1000
    for v,label,col in zip(values[1:],labels[1:],colors[1:]):axs[2].plot(ff,abs(v-values[0])/m.bden[j,0],label=label,color=col)
    axs[2].set(xlabel='中心频率 / kHz',ylabel='|B − B*| / 固定目标尺度',title='h=1/32：保留了峰，局部差分仍可受重罚');axs[2].legend(fontsize=8)
    fig.savefig(figdir/'held_failure_P5.png',dpi=160);plt.close(fig)
    summary=json.loads((OUT/'DIRECTION_SUMMARY.json').read_text())
    dump(OUT/'ANALYSIS_STATUS.json',dict(score_records=len(records),controlled_records=len(control),numerical_failure_records=len(fail),
        source_common_sha256=sha(HERE/'common.py'),controlled_counterexample=probe,candidate_direction_summary=summary,optimization_started=False))
    print(json.dumps(dict(records=len(records),controlled=len(control),numerical_failures=len(fail))),flush=True)

if __name__=='__main__':main()

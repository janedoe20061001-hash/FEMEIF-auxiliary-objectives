from core import *
from diagnose import dense_grid,csvwrite
import csv,shutil,zipfile,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
DATA=HERE/'server_run/retrieved/teacher_doubts_validation_20261011'
LABEL={'scalar_nsga':'WSMSE / 当前框架','scalar_ga':'WSMSE / 传统GA','dual_nsga':'双目标 / 膝点'}
COLOR={'scalar_ga':'#d97706','scalar_nsga':'#059669','dual_nsga':'#2563eb'}
METHODS=['scalar_ga','scalar_nsga','dual_nsga']

def lookup(condition,case,method,g=1000,selection=None):
    selection=selection or ('knee' if method=='dual_nsga' else 'minimum')
    return next(s for s in SUMMARY if (s['condition'],s['case'],s['method'],s['generation'],s['selection'])==(condition,case,method,g,selection))

def style(ax):
    ax.grid(True,which='major',alpha=.22);ax.spines[['top','right']].set_visible(False)

def curves(condition,case,g=1000):
    m=model(case,condition);methods=METHODS if condition=='full' else ['scalar_ga','dual_nsga']
    parameters=[10**np.array(lookup(condition,case,k,g)['best']['log_parameters']) for k in methods]
    # Always display the full measured target. Band-trained extrapolation is labelled.
    display=model(case);f=dense_grid(display,parameters);target=display.network(f,np.array(display.record['target_parameters']))[0] if case=='case1' else display.target
    fitted=[display.network(f,p)[0] for p in parameters]
    fig,axes=plt.subplots(2,2,figsize=(13,7.5),layout='constrained')
    for row in range(2):
        values=20*np.log10(abs(target)) if row==0 else np.rad2deg(np.angle(target))
        for col in range(2):
            ax=axes[row,col];ax.semilogx(f,values,color='#172033',lw=1.8,label='目标曲线',zorder=6)
            for method,z in zip(methods,fitted):
                v=20*np.log10(abs(z)) if row==0 else np.rad2deg(np.angle(z))
                b=lookup(condition,case,method,g)['best'];ax.semilogx(f,v,lw=1.1,alpha=.88,color=COLOR[method],label=f'{LABEL[method]} · seed {b["seed"]}')
            ax.set(xlabel='频率 / Hz',ylabel='幅值 / dBΩ' if row==0 else '相位 / °');style(ax)
            if col==1:ax.set_xlim(*( [20000,30000] if case=='case1' else [4e7,1.1e8] if condition=='full' else [1e4,3e7]))
            if condition=='paper_band':ax.axvspan(1e4,3e7,color='#93c5fd',alpha=.09)
    axes[0,0].legend(fontsize=8,loc='best');axes[0,0].set_title('全频曲线');axes[0,1].set_title('局部放大' if condition=='full' else '训练频段（108原测点）')
    fig.suptitle(f'{case.upper()} · {g}代 · 每组100个最终代表中的最佳WSMSE'+(' · 频段外仅显示外推' if condition=='paper_band' else ''),fontsize=13)
    filename=f'{case}_{condition}_best_{g}.png';fig.savefig(OUT/'figures'/filename,dpi=160);plt.close(fig)
    rows=[]
    for i,frequency in enumerate(f):
        r=dict(frequency_hz=float(frequency),target_real=float(target[i].real),target_imag=float(target[i].imag))
        for method,z in zip(methods,fitted):r.update({method+'_real':float(z[i].real),method+'_imag':float(z[i].imag)})
        rows.append(r)
    csvwrite(OUT/f'{case}_{condition}_curves_{g}.csv',rows)
    csvwrite(OUT/f'{case}_{condition}_parameters_{g}.csv',[dict(parameter=name,unit='ohm' if name[0]=='R' else 'H' if name[0]=='L' else 'F',**{method:float(p[i]) for method,p in zip(methods,parameters)}) for i,name in enumerate(m.record['parameter_names'])])

def evolution(case):
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for method in METHODS:
        for sel in (['knee','minimum'] if method=='dual_nsga' else ['minimum']):
            group=[r for r in ROWS if r['condition']=='full' and r['case']==case and r['method']==method and r['selection']==sel]
            generations=sorted(set(r['generation'] for r in group));data=np.array([[r['wsmse'] for r in group if r['generation']==g] for g in generations])
            label=LABEL[method] if sel!='minimum' or method!='dual_nsga' else '双目标 / 前沿最小WSMSE（诊断）'
            for ax,kind in zip(axes,['mean','best']):
                value=data.mean(axis=1) if kind=='mean' else data.min(axis=1)
                ax.semilogy(generations,value,color=COLOR[method],ls='--' if method=='dual_nsga' and sel=='minimum' else '-',lw=1.5,label=label)
    for ax in axes:ax.axvline(300,color='#6b7280',ls=':',lw=1);ax.set(xlabel='代数',ylabel='WSMSE');style(ax)
    axes[0].set_title('100 seed平均（每代重新选代表）');axes[1].set_title('100 seed中的最好值');axes[0].legend(fontsize=8)
    fig.suptitle(f'{case.upper()} · 300代并非已证明收敛');fig.savefig(OUT/'figures'/f'{case}_evolution.png',dpi=160);plt.close(fig)
    # Predeclared fixed seeds, same seed at 300 and 1000; never switch seeds mid-path.
    m=model(case);fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
    for row,seed in enumerate(SHOW):
        allx=[];entries=[]
        for method in METHODS:
            with np.load(DATA/'runs/full'/case/method/f'seed_{seed}/trajectory.npz') as a:
                for g in [300,1000]:
                    x=a['log_parameters'][list(a['generations']).index(g),0];allx.append(10**x);entries.append((method,g,x))
        f=dense_grid(m,allx);target=m.network(f,np.array(m.record['target_parameters']))[0] if case=='case1' else m.target
        for col in range(2):
            ax=axes[row,col];ax.semilogx(f,20*np.log10(abs(target)) if col==0 else np.rad2deg(np.angle(target)),color='#172033',label='目标',lw=1.5,zorder=6)
            for method,g,x in entries:
                z=m.network(f,10**x)[0];ax.semilogx(f,20*np.log10(abs(z)) if col==0 else np.rad2deg(np.angle(z)),color=COLOR[method],ls=':' if g==300 else '-',lw=.9,label=f'{LABEL[method]} {g}代')
            ax.set(xlabel='频率 / Hz',ylabel='幅值 / dBΩ' if col==0 else '相位 / °',title=f'seed {seed}');style(ax)
    axes[0,0].legend(fontsize=7,ncols=2);fig.suptitle(f'{case.upper()} · 预先固定seed曲线，点线300代 / 实线1000代')
    fig.savefig(OUT/'figures'/f'{case}_fixed_seeds.png',dpi=150);plt.close(fig)

def polish_plot(case):
    entries=[r for r in POLISH if r['case']==case and r['condition']=='full' and r['generation']==1000]
    m=model(case);ps=[];data=[]
    for r in entries:
        name=f"full_{case}_{r['method']}_1000"
        with np.load(OUT/'polish'/f'{name}.npz') as a:x=a['initial_log_parameters'];y=a['log_parameters']
        ps.extend([10**x,10**y]);data.append((r['method'],x,y))
    f=dense_grid(m,ps);t=m.network(f,np.asarray(m.record['target_parameters']))[0] if case=='case1' else m.target
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for col,ax in enumerate(axes):
        ax.semilogx(f,20*np.log10(abs(t)) if col==0 else np.rad2deg(np.angle(t)),color='#172033',lw=1.8,label='目标',zorder=6)
        for method,x,y in data:
            for xx,ls,label in [(x,':','进化最终解'),(y,'-','有界最小二乘后')]:
                z=m.network(f,10**xx)[0];ax.semilogx(f,20*np.log10(abs(z)) if col==0 else np.rad2deg(np.angle(z)),color=COLOR[method],ls=ls,lw=1,label=LABEL[method]+' '+label)
        ax.set(xlabel='频率 / Hz',ylabel='幅值 / dBΩ' if col==0 else '相位 / °');style(ax)
    axes[0].legend(fontsize=7);fig.suptitle(f'{case.upper()} · 可达性诊断：额外预算，不能混入主实验')
    fig.savefig(OUT/'figures'/f'{case}_polish.png',dpi=160);plt.close(fig)

def paired():
    idx={(r['condition'],r['case'],r['method'],r['selection'],r['generation'],r['seed']):r for r in ROWS};rng=np.random.default_rng(20261011);results=[]
    def add(label,left,right,metric='wsmse'):
        delta=np.array([idx[(*left,s)][metric]-idx[(*right,s)][metric] for s in range(2300,2400)])
        boot=delta[rng.integers(0,100,(10000,100))].mean(axis=1)
        results.append(dict(comparison=label,metric=metric,mean_left_minus_right=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),left_better_seeds=int(sum(delta<0))))
    for case in ['case1','case2']:
        for g in [300,1000]:
            ga=('full',case,'scalar_ga','minimum',g);ws=('full',case,'scalar_nsga','minimum',g);du=('full',case,'dual_nsga','knee',g);dm=('full',case,'dual_nsga','minimum',g)
            for label,a,b in [('GA-current WSMSE',ga,ws),('GA-dual knee',ga,du),('current WSMSE-dual knee',ws,du),('knee-front minimum',du,dm)]:add(f'{case} {g} {label}',a,b)
        for method in METHODS:
            sel='knee' if method=='dual_nsga' else 'minimum';add(f'{case} {method} 1000-300',('full',case,method,sel,1000),('full',case,method,sel,300))
    for method in ['scalar_ga','dual_nsga']:
        sel='knee' if method=='dual_nsga' else 'minimum'
        for g in [300,1000]:add(f'case2 {method} {g} band-full on same 108 points',('paper_band','case2',method,sel,g),('full','case2',method,sel,g),'paper_band_wsmse')
    dump(OUT/'PAIRED.json',results)

def main():
    global SUMMARY,ROWS,POLISH
    assert json.loads((DATA/'VALIDATION.json').read_text())['passed']
    for p in DATA.iterdir():
        if p.is_file():shutil.copy2(p,OUT/p.name)
    shutil.copytree(DATA/'polish',OUT/'polish',dirs_exist_ok=True)
    SUMMARY=json.loads((OUT/'SUMMARY.json').read_text());POLISH=json.loads((OUT/'POLISH_SUMMARY.json').read_text())
    with (OUT/'all_seed_stage_metrics.csv').open(encoding='utf-8-sig') as f:ROWS=list(csv.DictReader(f))
    for r in ROWS:
        for k in ['seed','generation']:r[k]=int(r[k])
        for k in METRICS+['paper_band_wsmse']:r[k]=float(r[k]) if r[k] else None
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    (OUT/'figures').mkdir(exist_ok=True)
    for case in ['case1','case2']:
        for g in [300,1000]:curves('full',case,g)
        evolution(case);polish_plot(case)
    curves('paper_band','case2');paired()
    print(json.dumps({'figures':len(list((OUT/'figures').glob('*.png'))),'analysis':'complete'}))

if __name__=='__main__':main()

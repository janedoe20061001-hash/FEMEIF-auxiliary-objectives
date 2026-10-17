from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
def target_plot():
    fig,ax=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for s in [.25,1,8]:
        m=Target(s);f=improved_dense_grid(m.record,np.empty((0,18)),4);z=spectra(f,m.record['target_parameters'])[0]
        for a in ax[0]:a.plot(f/1000,20*np.log10(abs(z)),label=f's={s:g}')
        for a in ax[1]:a.plot(f/1000,np.angle(z,deg=True),label=f's={s:g}')
    for a in ax[:,0]:a.set_xlim(18,27);a.grid(alpha=.2)
    for a in ax[:,1]:a.set_xlim(25.15,25.30);a.grid(alpha=.2)
    ax[0,0].set_title('目标的峰谷组合：谷也跟着移动');ax[0,1].set_title('同一尖峰：位置、峰高近似不变，宽度不同')
    for a in ax[1]:a.set_xlabel('频率 / kHz')
    for a in ax[0]:a.set_ylabel('幅值 / dBΩ')
    for a in ax[1]:a.set_ylabel('相位 / °')
    ax[0,0].legend();fig.savefig(FIG/'target_conditions.png',dpi=160);plt.close(fig)
def old_pair_plot():
    m=Target();d=OLD/'server_run/retrieved/runs/adaptive/seed_2324/generation_1000.npz'
    with np.load(d) as data:x=data['log_parameters'][[15,78]]
    f=improved_dense_grid(m.record,10**x,4);t=spectra(f,m.record['target_parameters'])[0];zz=spectra(f,10**x)
    fig,ax=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for col in range(2):
        ax[0,col].plot(f,20*np.log10(abs(t)),'k--',label='目标');ax[1,col].plot(f,np.angle(t,deg=True),'k--',label='目标')
        for z,n in zip(zz,[15,78]):ax[0,col].plot(f,20*np.log10(abs(z)),label=f'候选{n}');ax[1,col].plot(f,np.angle(z,deg=True),label=f'候选{n}')
        for a in ax[:,col]:a.set_xlim((23000,26000) if col==0 else (25190,25255));a.grid(alpha=.2)
    ax[0,0].set_title('诊断例：seed 2324，1000代');ax[0,1].set_title('峰顶放大：整体成绩几乎相同，峰高差约12 dB')
    for a in ax[0]:a.set_ylabel('幅值 / dBΩ')
    for a in ax[1]:a.set_ylabel('相位 / °');a.set_xlabel('频率 / Hz')
    ax[0,0].legend();fig.savefig(FIG/'similar_scores_different_peak.png',dpi=160);plt.close(fig)
def pilot_plots():
    cfg=json.loads((HERE/'CONFIG.json').read_text());params=[]
    for seed in cfg['plot_seeds']:
        fig,ax=plt.subplots(2,3,figsize=(15,7),layout='constrained')
        for col,s in enumerate(cfg['run_scales']):
            m=Target(s)
            with np.load(HERE/'batch'/label(s)/f'seed_{seed}'/'generation_300.npz') as d:x=d['log_parameters'][int(d['selected_index'])];index=int(d['selected_index']);F=d['optimization_objective'][index]
            params.append(dict(seed=seed,scale=s,index=index,log_parameters=x.tolist(),train_mag=F[0],train_phase=F[1]))
            f=improved_dense_grid(m.record,10**x[None,:],4);t=spectra(f,m.record['target_parameters'])[0];z=spectra(f,10**x)[0]
            ax[0,col].plot(f/1000,20*np.log10(abs(t)),'k--',label='目标');ax[0,col].plot(f/1000,20*np.log10(abs(z)),label='300代代表解')
            ax[1,col].plot(f/1000,np.angle(t,deg=True),'k--');ax[1,col].plot(f/1000,np.angle(z,deg=True))
            for a in ax[:,col]:a.set_xlim(18,27);a.grid(alpha=.2)
            ax[0,col].set_title(f's={s:g}；MAG={F[0]:.2f}，PHA={F[1]:.3f}')
            ax[1,col].set_xlabel('频率 / kHz')
        for a in ax[0]:a.set_ylim(-25,85)
        for a in ax[1]:a.set_ylim(-95,95)
        ax[0,0].set_ylabel('幅值 / dBΩ');ax[1,0].set_ylabel('相位 / °');ax[0,0].legend();fig.suptitle(f'预先固定的seed {seed}：逐条件显示实际曲线')
        fig.savefig(FIG/f'pilot_seed_{seed}.png',dpi=160);plt.close(fig)
    dump(OUT/'PLOT_PARAMETERS.json',params)
    fig,ax=plt.subplots(2,3,figsize=(15,7),layout='constrained')
    for col,s in enumerate(cfg['run_scales']):
        m=Target(s);pp=[p for p in params if p['scale']==s];xx=np.array([p['log_parameters'] for p in pp]);f=improved_dense_grid(m.record,10**xx,4);t=spectra(f,m.record['target_parameters'])[0]
        ax[0,col].plot(f/1000,20*np.log10(abs(t)),'k--',label='目标',linewidth=1.6);ax[1,col].plot(f/1000,np.angle(t,deg=True),'k--',linewidth=1.6)
        for p,z in zip(pp,spectra(f,10**xx)):
            ax[0,col].plot(f/1000,20*np.log10(abs(z)),label=f'seed {p["seed"]}',alpha=.8);ax[1,col].plot(f/1000,np.angle(z,deg=True),alpha=.8)
        for a in ax[:,col]:a.set_xlim(18,27);a.grid(alpha=.2)
        ax[0,col].set_title(f's={s:g}：同条件、不同起点');ax[1,col].set_xlabel('频率 / kHz')
    for a in ax[0]:a.set_ylim(-25,85)
    for a in ax[1]:a.set_ylim(-95,95)
    ax[0,0].set_ylabel('幅值 / dBΩ');ax[1,0].set_ylabel('相位 / °');ax[0,0].legend(fontsize=9);fig.suptitle('300代实际曲线：预先固定3个seed，无曲线平均')
    fig.savefig(FIG/'pilot_three_seeds.png',dpi=160);plt.close(fig)
def scoring_probe_plot():
    fig,ax=plt.subplots(2,3,figsize=(15,7),layout='constrained')
    for col,s in enumerate([.25,1,8]):
        m=Target(s);p0=np.array(m.record['target_parameters']);p=p0.copy();p[3]/=2;f=improved_dense_grid(m.record,p[None,:],4);t=spectra(f,p0)[0];z=spectra(f,p)[0];cat=target_catalog(m);e=next(x for x in cat if x['event_id']=='P3');width=10**(e['x']+e['width_3db_decades']/2)-10**(e['x']-e['width_3db_decades']/2)
        for a in ax[:,col]:a.plot(f,20*np.log10(abs(t)),'k--',label='目标');a.plot(f,20*np.log10(abs(z)),label='只有R2减半');a.grid(alpha=.2);a.set_ylabel('幅值 / dBΩ')
        ax[0,col].set_xscale('log');ax[0,col].set_xlim(100,1e6);ax[0,col].set_ylim(-25,80);ax[0,col].set_title(f's={s:g}：全频段幅值')
        ax[1,col].set_xlim(e['frequency_hz']-4*width,e['frequency_hz']+4*width);ax[1,col].set_ylim(48,72);ax[1,col].set_title(f'峰顶放大；目标宽度约{width:.1f} Hz');ax[1,col].set_xlabel('频率 / Hz')
    ax[0,0].legend();fig.suptitle('直接评分诊断：其余17个参数正确，尖峰仍错约6 dB且宽度约翻倍')
    fig.savefig(FIG/'scoring_probe_peak.png',dpi=160);plt.close(fig)
if __name__=='__main__':
    target_plot();old_pair_plot();scoring_probe_plot()
    if (HERE/'batch/status.json').exists() and json.loads((HERE/'batch/status.json').read_text())['status']=='complete':pilot_plots()

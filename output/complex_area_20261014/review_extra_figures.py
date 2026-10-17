from review_figures import *

data=target('case1');g=Features(data);c=data['circuit'];event=max((e for e in data['record']['events'] if e['kind']=='peak'),key=lambda e:e['height_db'])
fc=event['frequency_hz'];width=event['width_3db_hz'];f=np.linspace(fc-8*width,fc+8*width,4097);truth,_=c.response(f)
fig,axs=plt.subplots(1,3,figsize=(15,4.5))
for m in ['MAG_PHA','G0','J0','A','ARX']:
    with np.load(DATA/'jobs/case1'/m/'seed_46004/generation_0300.npz') as a:p=10**a['X'][int(a['selected_index'])]
    z,_=c.response(f,p)
    axs[0].plot(f,20*np.log10(abs(z)),color=COLORS[m],lw=1.2,label=LABELS[m]);axs[1].plot(f,np.rad2deg(np.angle(z)),color=COLORS[m],lw=1.2);axs[2].plot(z.real,z.imag,color=COLORS[m],lw=1.2)
axs[0].plot(f,20*np.log10(abs(truth)),color='black',lw=2,label='目标');axs[1].plot(f,np.rad2deg(np.angle(truth)),color='black',lw=2);axs[2].plot(truth.real,truth.imag,color='black',lw=2)
axs[0].set_xlabel('频率 / Hz');axs[0].set_ylabel('幅值 / dBΩ');axs[1].set_xlabel('频率 / Hz');axs[1].set_ylabel('相位 / °');axs[2].set_xlabel('ReZ / Ω');axs[2].set_ylabel('ImZ / Ω')
axs[0].legend(fontsize=8)
for ax in axs:ax.grid(alpha=.2)
fig.suptitle('Case1 · seed46004：约25.23kHz处的目标窄峰仍未恢复；评分数值可靠不代表结构正确',fontsize=13)
fig.tight_layout();fig.savefig(FIG/'case1_dominant_peak_zoom.png',dpi=180);plt.close(fig)

sel=read(OUT/'SELECTOR_STRUCTURE_DIAGNOSIS.json');record=next(v for v in sel['records'] if v['seed']==46002)
with np.load(AN/'populations/case1_MAG_PHA_46002.npz') as a:
    ps=[10**a['X'][record['original_selected_index']],10**a['X'][record['diagnostic_candidate_index']]]
x=[np.linspace(g.lo,g.hi,8193)]
for p in [c.p,*ps]:
    modes=g.modes(p);x.append(mode_points(modes,g.lo,g.hi))
    for f0,w in modes:
        cc=np.log10(f0)
        if g.lo<=cc<=g.hi:x.append(np.clip(cc+np.linspace(-w,w,1025),g.lo,g.hi))
x=np.unique(np.concatenate(x));tt,_=c.response(10**x)
fig,axs=plt.subplots(1,2,figsize=(13,5))
for p,color,label in zip(ps,['#d94c48','#29945d'],['原最终膝点（3个峰谷达标）','种群中的诊断候选（5个峰谷达标，未替换最终结果）']):
    zz,_=c.response(10**x,p);axs[0].plot(10**x,20*np.log10(abs(zz)),color=color,lw=1.2,label=label);axs[1].plot(10**x,np.rad2deg(np.angle(zz)),color=color,lw=1.2)
axs[0].plot(10**x,20*np.log10(abs(tt)),color='black',lw=1.7,label='目标');axs[1].plot(10**x,np.rad2deg(np.angle(tt)),color='black',lw=1.7)
for ax in axs:ax.set_xscale('log');ax.set_xlabel('频率 / Hz');ax.grid(alpha=.2)
axs[0].set_ylabel('幅值 / dBΩ');axs[1].set_ylabel('相位 / °');axs[0].legend(fontsize=8)
fig.suptitle('Case1 · seed46002：诊断候选的幅值MSE高约0.99%，相位MSE高约1.37%，但多恢复2个峰谷',fontsize=12)
fig.tight_layout();fig.savefig(FIG/'case1_score_structure_mismatch.png',dpi=180);plt.close(fig)
print('extra figures complete')

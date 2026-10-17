from review_figures import *

def main():
    rows=loadrows('REVIEWED_GROUPS.csv');fig,axs=plt.subplots(1,3,figsize=(15,4.6))
    for ax,t in zip(axs,target_ids()):
        rr=[next(r for r in rows if r['target']==t and r['method']==m) for m in CFG['methods']]
        v=[float(r['cpu_ratio_to_base']) for r in rr]
        ax.bar(range(6),v,color=[COLORS[m] for m in CFG['methods']])
        ax.set_xticks(range(6),['基础','G0','J0','J0_W','ARX_L','ARX_S_L'])
        for j,x in enumerate(v):ax.text(j,x+.04,f'{x:.2f}',ha='center',fontsize=9)
        ax.set_ylim(0,max(v)*1.18);ax.set_title(NAMES[t]);ax.set_ylabel('优化CPU核时 / 同条件基础组CPU核时');ax.grid(axis='y',alpha=.15)
    fig.suptitle('相同60000候选预算；包含主目标、辅助目标和搜索开销',fontsize=13);fig.tight_layout();fig.savefig(FIG/'optimization_cost.png',dpi=180);plt.close(fig)
    data=target('case1');g=Features(data);c=data['circuit'];event=next(e for e in data['record']['events'] if e['event_id']=='P3')
    fc,w=event['frequency_hz'],event['width_3db_hz'];f=np.linspace(fc-8*w,fc+8*w,4097);truth,_=c.response(f)
    fig,axs=plt.subplots(2,3,figsize=(15,8))
    for aa,seed in zip(axs,(47000,47004)):
        for m in CFG['methods']:
            with np.load(DATA/'jobs/case1'/m/f'seed_{seed}/generation_0300.npz') as a:p=10**a['X'][int(a['selected_index'])]
            z,_=c.response(f,p)
            aa[0].plot(f,20*np.log10(abs(z)),color=COLORS[m],lw=1,label=LABELS[m]);aa[1].plot(f,np.rad2deg(np.angle(z)),color=COLORS[m],lw=1);aa[2].plot(z.real,z.imag,color=COLORS[m],lw=1)
        aa[0].plot(f,20*np.log10(abs(truth)),color='black',lw=2,label='目标');aa[1].plot(f,np.rad2deg(np.angle(truth)),color='black',lw=2);aa[2].plot(truth.real,truth.imag,color='black',lw=2)
        aa[0].set_ylabel(f'seed {seed}\n幅值 / dBΩ');aa[1].set_ylabel('相位 / °');aa[2].set_ylabel('ImZ / Ω')
        aa[0].set_xlabel('频率 / Hz');aa[1].set_xlabel('频率 / Hz');aa[2].set_xlabel('ReZ / Ω')
        for ax in aa:ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=8);fig.suptitle(f'Case1固定窄峰P3局部：中心约{fc:.1f}Hz，目标3dB宽度约{w:.2f}Hz',fontsize=14)
    fig.tight_layout();fig.savefig(FIG/'case1_P3_zoom.png',dpi=180);plt.close(fig)
    r=next(r for r in read(OUT/'SELECTOR_STRUCTURE_DIAGNOSIS.json')['records'] if r['seed']==47001)
    with np.load(AN/'populations/case1_MAG_PHA_47001.npz') as a:params=[10**a['X'][r['original_selected_index']],10**a['X'][r['diagnostic_candidate_index']]]
    fig,axs=plt.subplots(1,2,figsize=(13,4.8))
    for ax,event_id in zip(axs,('P1','P2')):
        e=next(x for x in data['record']['events'] if x['event_id']==event_id);center=e['frequency_hz'];width=e['width_3db_hz'];f=np.linspace(center-8*width,center+8*width,4097);z,_=c.response(f)
        ax.plot(f,20*np.log10(abs(z)),color='black',lw=2,label='目标')
        for p,color,label in zip(params,['#d94c48','#009e8e'],['正式膝点','种群诊断候选，未替换正式结果']):
            z,_=c.response(f,p);ax.plot(f,20*np.log10(abs(z)),color=color,label=label)
        ax.set_title(event_id);ax.set_xlabel('频率 / Hz');ax.set_ylabel('幅值 / dBΩ');ax.grid(alpha=.2)
    axs[0].legend(fontsize=9);fig.suptitle('Case1 seed47001：候选幅相MSE分别高约0.40%、0.33%，P1/P2高度和宽度却更接近目标',fontsize=12)
    fig.tight_layout();fig.savefig(FIG/'case1_score_structure_mismatch.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()

"""Scientific figures, individual seeds only; no averaged response curves."""
from review_analysis import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from reference_features import mode_points
font=Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams['axes.unicode_minus']=False
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
COLORS={'MAG_PHA':'#d94c48','G0':'#e69f00','J0':'#54a24b','J0_W':'#9467bd','ARX_L':'#2964bc','ARX_S_L':'#009e8e'}
LABELS={'MAG_PHA':'基础MAG/PHA','G0':'全带复误差G0','J0':'窗口复误差J0','J0_W':'匹配权重J0_W','ARX_L':'原公式轻量ARX_L','ARX_S_L':'新尺度ARX_S_L'}
NAMES={'mixed_resonators/base':'混合支路：原参数','mixed_resonators/lc_reciprocal_8p0':'混合支路：L3×8、C3÷8','case1':'Case1：极窄峰谷'}

def loadrows(name):
    with (AN/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def main():
    rows=loadrows('REVIEWED_SEEDS.csv');scope=read(AN/'METRIC_SCOPES.json');methods=CFG['methods']
    fig,axes=plt.subplots(1,3,figsize=(16,5.5))
    for ax,ident in zip(axes,target_ids()):
        for j,m in enumerate(methods):
            rr=sorted([r for r in rows if r['target']==ident and r['method']==m],key=lambda r:r['seed'])
            y=[float(r['local_complex_error']) for r in rr];jitter=np.linspace(-.18,.18,len(y))
            ax.scatter(j+jitter,y,color=COLORS[m],s=27,alpha=.85)
            ax.plot([j-.24,j+.24],[np.median(y)]*2,color=COLORS[m],lw=2)
            for xx,r in zip(j+jitter,rr):
                if r['accepted']=='False':ax.scatter(xx,float(r['local_complex_error']),marker='x',color='black',s=80,lw=2)
        ax.set_yscale('log');ax.set_xticks(range(6),['基础','G0','J0','J0_W','ARX_L','ARX_S_L']);ax.set_title(NAMES[ident]);ax.grid(alpha=.18,axis='y')
        ax.set_ylabel('固定频段、固定目标尺度的复误差（越低越好）')
    fig.suptitle('每个点是一条seed的最终膝点；短横线仅是误差中位数，未平均曲线',fontsize=13)
    fig.tight_layout();fig.savefig(FIG/'all_seeds_local_error.png',dpi=180);plt.close(fig)
    selected=methods
    for ident in target_ids():
        data=target(ident);g=Features(data);c=data['circuit'];lo,hi=np.log10(scope[ident]['band_hz']);scale=scope[ident]['scale_ohm']
        for seed in (47000,47004):
            parameters={};parts=[np.linspace(lo,hi,4097)]
            for m in selected:
                d=DATA/'jobs'/ident/m/f'seed_{seed}'
                with np.load(d/'generation_0300.npz') as p:parameters[m]=10**p['X'][int(p['selected_index'])]
            for p in [c.p,*parameters.values()]:
                modes=g.modes(p);parts.append(mode_points(modes,lo,hi))
                for fc,width in modes:
                    center=np.log10(fc)
                    if lo<=center<=hi:parts.append(np.clip(center+np.linspace(-width,width,1025),lo,hi))
            x=np.unique(np.concatenate(parts));x=x[(x>=lo)&(x<=hi)];truth,_=c.response(10**x)
            fig=plt.figure(figsize=(13,9));ax3=fig.add_subplot(221,projection='3d');ax2=fig.add_subplot(222);axm=fig.add_subplot(223);axp=fig.add_subplot(224)
            ax3.plot(x,truth.real/scale,truth.imag/scale,color='black',lw=2);ax2.plot(truth.real/scale,truth.imag/scale,color='black',lw=2,label='目标')
            axm.plot(10**x,20*np.log10(abs(truth)),color='black',lw=2);axp.plot(10**x,np.rad2deg(np.angle(truth)),color='black',lw=2)
            for m,p in parameters.items():
                z,_=c.response(10**x,p);color=COLORS[m]
                ax3.plot(x,z.real/scale,z.imag/scale,color=color,lw=1,alpha=.85)
                ax2.plot(z.real/scale,z.imag/scale,color=color,lw=1.1,alpha=.85,label=LABELS[m])
                axm.plot(10**x,20*np.log10(abs(z)),color=color,lw=1.1,alpha=.85);axp.plot(10**x,np.rad2deg(np.angle(z)),color=color,lw=1.1,alpha=.85)
            ax3.set_xlabel('log10(f/Hz)');ax3.set_ylabel('ReZ / S');ax3.set_zlabel('ImZ / S');ax3.set_title('带频率标记的三维轨迹')
            ax2.set_xlabel('ReZ / S');ax2.set_ylabel('ImZ / S');ax2.set_title('Re/Im平面投影');ax2.legend(fontsize=8,loc='best')
            axm.set_xscale('log');axp.set_xscale('log');axm.set_ylabel('幅值 / dBΩ');axp.set_ylabel('相位 / °');axm.set_xlabel('频率 / Hz');axp.set_xlabel('频率 / Hz')
            for ax in (ax2,axm,axp):ax.grid(alpha=.2)
            fig.suptitle(f'{NAMES[ident]} · seed {seed} · 每条都是独立结果，未平均或配准',fontsize=14)
            fig.tight_layout();name=ident.replace('/','_')+f'_seed{seed}.png';fig.savefig(FIG/name,dpi=170);plt.close(fig)
    stages=loadrows('SELECTED_STAGE_TRAJECTORY.csv');fig,axs=plt.subplots(1,3,figsize=(16,5))
    for ax,ident in zip(axs,target_ids()):
        for m in methods:
            for seed in (47000,47004):
                rr=sorted([r for r in stages if r['target']==ident and r['method']==m and int(r['seed'])==seed],key=lambda r:int(r['generation']))
                ax.plot([int(r['generation']) for r in rr],[float(r['local_complex_error']) for r in rr],color=COLORS[m],ls='-' if seed==47000 else '--',lw=1.3,label=LABELS[m] if seed==47000 else None)
        ax.set_yscale('log');ax.set_xlabel('代数');ax.set_ylabel('所选膝点的局部复误差');ax.set_title(NAMES[ident]);ax.grid(alpha=.2)
    axs[-1].legend(fontsize=8);fig.suptitle('固定两条seed的中间过程；实线47000，虚线47004；选中膝点会变，曲线无需单调下降',fontsize=12)
    fig.tight_layout();fig.savefig(FIG/'stage_trajectories.png',dpi=180);plt.close(fig)
    print('figures complete',flush=True)

if __name__=='__main__':main()

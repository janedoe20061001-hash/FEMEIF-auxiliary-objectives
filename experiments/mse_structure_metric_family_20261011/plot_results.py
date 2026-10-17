from core import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'MAG_PHA':'MAG/PHA','MAG_PHA_INTEGRAL':'C: original integral','MAG_PHA_RADIAL':'R: radial extreme','MAG_PHA_OCCUPANCY':'F: radial occupancy','MAG_PHA_ANGULAR':'T: angular progress','MAG_PHA_BRIDGE':'B: full bridge','MAG_PHA_EXCESS':'X: excess reversal'}

def main():
    seed=CFG['seeds'][0];out=OUT/'figures';out.mkdir(exist_ok=True)
    dump(OUT/'PLOT_SELECTION.json',dict(seed=seed,rule='first seed by frozen ID; same primary selector; no average curve or retrospective seed selection'))
    for ident in CFG['targets']:
        data=target(ident);c=data['circuit'];q=data['quadrature'];geo=Geometry(data)
        curves=[]
        for method in CFG['methods']:
            d=json.loads((OUT/f'analysis/details/{ident}_{method}_{seed}_primary.json').read_text(encoding='utf-8'))
            curves.append((method,np.array(d['log_parameters']),max(4,d['diagnostic']['refinement'])))
        grids=[q.reference_grid(c.p,4),*[q.reference_grid(10**p,level) for _,p,level in curves]]
        windows=list(geo.windows)
        if len(windows)>1:windows.append(dict(event='Full connecting span',lo=geo.span[0],hi=geo.span[1]))
        fig,axes=plt.subplots(2,len(windows),figsize=(6*len(windows),7),squeeze=False)
        for k,w in enumerate(windows):
            x=np.unique(np.r_[np.linspace(w['lo'],w['hi'],2049),*[g[(g>=w['lo'])&(g<=w['hi'])] for g in grids]])
            f=10**x;z,_=c.response(f)
            axes[0,k].plot(f/1000,20*np.log10(abs(z)),'k--',lw=2.4,label='Target',zorder=20)
            axes[1,k].plot(f/1000,np.rad2deg(np.angle(z)),'k--',lw=2.4,label='Target',zorder=20)
            for method,p,_ in curves:
                z,_=c.response(f,10**p)
                axes[0,k].plot(f/1000,20*np.log10(abs(z)),lw=1.2,label=LABELS[method])
                axes[1,k].plot(f/1000,np.rad2deg(np.angle(z)),lw=1.2,label=LABELS[method])
            axes[0,k].set_title(f"{ident} / {w['event']} / seed {seed}")
            axes[0,k].set_ylabel('Magnitude (dB ohm)');axes[1,k].set(xlabel='Frequency (kHz)',ylabel='Phase (degree)')
            for row in axes:
                ax=row[k];ax.grid(alpha=.2);ax.ticklabel_format(axis='x',style='plain',useOffset=False)
        axes[0,0].legend(fontsize=8,ncol=2);fig.suptitle('Same primary selection / generation 300',fontsize=12)
        fig.tight_layout();fig.savefig(out/f'{ident}_local.png',dpi=140);plt.close(fig)
        if ident in ('case1','ladder8'):
            lo,hi=geo.span;x=np.unique(np.r_[np.linspace(lo,hi,2049),*[g[(g>=lo)&(g<=hi)] for g in grids]])
            f=10**x;fig=plt.figure(figsize=(11,8));ax=fig.add_subplot(projection='3d')
            z,_=c.response(f);ax.plot(f/1000,z.real,z.imag,'k--',lw=2.4,label='Target')
            for method,p,_ in curves:
                z,_=c.response(f,10**p);ax.plot(f/1000,z.real,z.imag,lw=1.2,label=LABELS[method])
            ax.set(xlabel='Frequency (kHz)',ylabel='Re Z (ohm)',zlabel='Im Z (ohm)',title=f'{ident}: primary representative / seed {seed}')
            ax.view_init(25,-60);ax.set_box_aspect((1,1,1),zoom=.8);ax.tick_params(labelsize=9);ax.legend(fontsize=8)
            for a in (ax.xaxis,ax.yaxis,ax.zaxis):a.label.set_size(10);a.labelpad=6
            fig.subplots_adjust(left=.05,right=.87,bottom=.14,top=.9);fig.savefig(out/f'{ident}_3d.png',dpi=140);plt.close(fig)
    print('Seven fixed-seed figures generated',flush=True)

if __name__=='__main__':main()

"""Fixed seed curve and front plots; no ranking or averaged curves."""
from reselect import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'MAG_PHA':'MAG/PHA','MAG_PHA_GEOMETRY':'G: bottleneck','MAG_PHA_MAX':'G: bottleneck','MAG_PHA_MEAN':'A: mean bottleneck','MAG_PHA_CHORD':'D: chord','MAG_PHA_INTEGRAL':'C: path integral','MAG_PHA_RADIAL':'R: radial extreme','MAG_PHA_OCCUPANCY':'F: occupancy','MAG_PHA_ANGULAR':'T: angular progress','MAG_PHA_BRIDGE':'B: connecting path','MAG_PHA_EXCESS':'X: excess reversal'}

def main(batch):
    core,_=setup(batch);seed=core.CFG['seeds'][0];out=core.OUT/'figures';out.mkdir(exist_ok=True)
    dump(core.OUT/'PLOT_SELECTION.json',dict(seed=seed,rule='fixed first seed in frozen protocol; generation 300 MAG/PHA knee or marked fallback; no averaged curves'))
    for ident in core.CFG['targets']:
        data=core.target(ident);c=data['circuit'];q=data['quadrature'];geo=core.Geometry(data)
        curves=[]
        for method in core.CFG['methods']:
            d=json.loads((core.OUT/f'analysis/details/{ident}_{method}_{seed}_knee.json').read_text(encoding='utf-8'))
            curves.append((method,np.array(d['log_parameters']),max(4,d['diagnostic']['refinement'])))
        grids=[q.reference_grid(c.p,4),*[q.reference_grid(10**p,n) for _,p,n in curves]]
        windows=list(geo.windows)
        lo=min(w['lo'] for w in windows);hi=max(w['hi'] for w in windows)
        if len(windows)>1:windows.append(dict(event='Connecting span',lo=lo,hi=hi))
        fig,axes=plt.subplots(2,len(windows),figsize=(6*len(windows),7),squeeze=False)
        for k,w in enumerate(windows):
            x=np.unique(np.r_[np.linspace(w['lo'],w['hi'],2049),*[g[(g>=w['lo'])&(g<=w['hi'])] for g in grids]])
            f=10**x;z,_=c.response(f)
            axes[0,k].plot(f/1000,20*np.log10(abs(z)),'k--',lw=2.2,label='Target',zorder=20)
            axes[1,k].plot(f/1000,np.rad2deg(np.angle(z)),'k--',lw=2.2,label='Target',zorder=20)
            for method,p,_ in curves:
                z,_=c.response(f,10**p)
                axes[0,k].plot(f/1000,20*np.log10(abs(z)),lw=1.15,label=LABELS[method])
                axes[1,k].plot(f/1000,np.rad2deg(np.angle(z)),lw=1.15,label=LABELS[method])
            axes[0,k].set_title(f"{ident} / {w['event']} / seed {seed}")
            axes[0,k].set_ylabel('Magnitude (dB ohm)');axes[1,k].set(xlabel='Frequency (kHz)',ylabel='Phase (degree)')
            for row in axes:
                row[k].grid(alpha=.2);row[k].ticklabel_format(axis='x',style='plain',useOffset=False)
        axes[0,0].legend(fontsize=8,ncol=2);fig.suptitle('MAG/PHA knee / generation 300 (fallbacks flagged in data)',fontsize=12)
        fig.tight_layout();fig.savefig(out/f'{ident}_local.png',dpi=140);plt.close(fig)
        if batch=='family' and ident in ('case1','ladder8'):
            x=np.unique(np.r_[np.linspace(lo,hi,2049),*[g[(g>=lo)&(g<=hi)] for g in grids]]);f=10**x
            fig=plt.figure(figsize=(12,8));ax=fig.add_subplot(projection='3d');z,_=c.response(f)
            ax.plot(f/1000,z.real,z.imag,'k--',lw=2,label='Target')
            for method,p,_ in curves:
                z,_=c.response(f,10**p);ax.plot(f/1000,z.real,z.imag,lw=1,label=LABELS[method])
            ax.set(xlabel='Frequency (kHz)',ylabel='Re Z (ohm)',zlabel='Im Z (ohm)',title=f'{ident}: MAG/PHA knee / seed {seed}')
            ax.view_init(25,-60);ax.set_box_aspect((1,1,1),zoom=.8);ax.legend(fontsize=8,loc='upper left');fig.subplots_adjust(left=.05,right=.87,bottom=.14,top=.9)
            fig.savefig(out/f'{ident}_3d.png',dpi=140);plt.close(fig)
    if batch=='family':
        fig,axes=plt.subplots(2,4,figsize=(16,8));ident='ladder8'
        for ax,method in zip(axes.flat,core.CFG['methods']):
            s=json.loads((core.OUT/f'selections/{ident}_{method}_{seed}.json').read_text())
            n=np.array(s['normalized']);ax.scatter(n[:,0],n[:,1],s=9,alpha=.65)
            k=s['unique_indices'].index(s['selected_index']);ax.scatter(*n[k],s=180,marker='*',c='red',zorder=5)
            ax.plot([0,1],[1,0],'k--',alpha=.4);ax.set(title=LABELS[method]+' / '+s['reason'],xlabel='Normalized MAG',ylabel='Normalized PHA');ax.grid(alpha=.2)
        axes.flat[-1].axis('off');fig.suptitle(f'Per-method MAG/PHA fronts / {ident} / fixed seed {seed}');fig.tight_layout();fig.savefig(out/'fronts.png',dpi=130);plt.close(fig)
    print('Plots complete: '+batch,flush=True)

if __name__=='__main__':main(sys.argv[1])

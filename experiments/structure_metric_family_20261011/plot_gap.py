"""A fixed, existing seed: show the segment omitted by local auxiliary windows."""
from inventory import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    data=core.target('case1');wins=sorted(core.Geometry(data).windows,key=lambda w:w['lo'])
    lo,hi=wins[0]['hi'],wins[1]['lo'];curves=[];grids=[]
    for method in core.CFG['methods']:
        d=json.loads((SOURCE/f'analysis/details/case1_{method}_42000_primary.json').read_text(encoding='utf-8'))
        p=10**np.asarray(d['log_parameters']);curves.append((method,p));grids.append(data['quadrature'].reference_grid(p,8))
    x=np.unique(np.r_[np.linspace(lo,hi,2049),*[g[(g>=lo)&(g<=hi)] for g in grids]])
    f=10**x;z,_=data['circuit'].response(f)
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
    axes[0].plot(f/1000,20*np.log10(abs(z)),'k--',lw=2.2,label='Target')
    axes[1].plot(f/1000,np.rad2deg(np.angle(z)),'k--',lw=2.2,label='Target')
    labels=['MAG/PHA','Original max','A: event mean','B: mean + chord','C: integral path']
    for (method,p),label in zip(curves,labels):
        z,_=data['circuit'].response(f,p)
        axes[0].plot(f/1000,20*np.log10(abs(z)),label=label,lw=1.2)
        axes[1].plot(f/1000,np.rad2deg(np.angle(z)),label=label,lw=1.2)
    axes[0].set(title='Case1, seed 42000: bridge outside both local auxiliary windows',ylabel='Magnitude (dB ohm)')
    axes[1].set(xlabel='Frequency (kHz)',ylabel='Phase (degree)',ylim=(-95,95))
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2);ax.ticklabel_format(axis='x',useOffset=False,style='plain')
    fig.tight_layout();fig.savefig(OUT/'BRIDGE.png',dpi=150);plt.close(fig)
    print('Existing fixed seed plotted; no optimization')


if __name__=='__main__':main()

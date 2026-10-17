"""Scientific figures from recovered, audited curves; no optimization."""
from pathlib import Path
import json, numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent;DATA=OUT/'retrieved/output/complex_joint_20261012'
DEST=OUT/'analysis/figures';DEST.mkdir(parents=True,exist_ok=True)
seeds=[45006,45002,45004];colors=['#287bb5','#dc7c22','#23924d']
def load(ident,method,seed):
    with np.load(DATA/'jobs'/ident/method/f'seed_{seed}'/'audited_curve.npz') as a:
        return a['x'],a['z'],a['target']
for ident,new,bounds in [('mixed_resonators/base','D',(45000,58000)),('mixed_resonators/lc_reciprocal_8p0','A',(38500,45000))]:
    fig=plt.figure(figsize=(15,8.5));fig.suptitle(f'{ident}: local frequency-labelled complex trajectory | fixed seeds {seeds}',fontsize=12)
    for row,method in enumerate(['MAG_PHA',new]):
        axes=[fig.add_subplot(2,3,row*3+j+1,projection='3d' if j==2 else None) for j in range(3)]
        for k,seed in enumerate(seeds):
            x,z,t=load(ident,method,seed);freq=10**x;mask=(freq>=bounds[0])&(freq<=bounds[1]);f=freq[mask]/1000;z=z[mask];t=t[mask]
            if k==0:
                axes[0].plot(f,t.real,'k--',lw=2,label='Target');axes[1].plot(f,t.imag,'k--',lw=2)
                axes[2].plot(f,t.real,t.imag,'k--',lw=2,label='Target')
            axes[0].plot(f,z.real,color=colors[k],label=str(seed));axes[1].plot(f,z.imag,color=colors[k])
            axes[2].plot(f,z.real,z.imag,color=colors[k],label=str(seed))
        axes[0].set_title(method+' | Re Z');axes[1].set_title(method+' | Im Z');axes[2].set_title(method+' | 3D')
        for ax in axes[:2]:ax.set_xlabel('Frequency / kHz');ax.set_ylabel('Impedance / ohm');ax.grid(alpha=.2)
        axes[2].set_xlabel('f / kHz');axes[2].set_ylabel('Re Z / ohm');axes[2].set_zlabel('Im Z / ohm')
        axes[0].legend(fontsize=8)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(DEST/(ident.replace('/','_')+'_focus.png'),dpi=155);plt.close(fig)
# Case2 final full populations: the selector is retained; plots diagnose its opportunities.
fig,axes=plt.subplots(2,3,figsize=(13,8))
for j,seed in enumerate(seeds):
    for row,method in enumerate(['MAG_PHA','H']):
        d=DATA/'jobs/case2'/method/f'seed_{seed}'
        with np.load(d/'generation_0300.npz') as a:
            f=a['F'][:,:2];sel=int(a['selected_index'])
        axes[row,j].scatter(f[:,0],f[:,1],s=9,alpha=.5,label='All 200 candidates')
        axes[row,j].scatter(*f[sel],marker='*',s=180,color='#d8512d',label='Frozen selection')
        axes[row,j].set_title(f'{method}, seed {seed}');axes[row,j].set_xlabel('MAG MSE / dB^2');axes[row,j].set_ylabel('PHA MSE / rad^2');axes[row,j].grid(alpha=.2)
        axes[row,j].legend(fontsize=7)
fig.suptitle('Case2: saved population and unchanged MAG/PHA selection',fontsize=12)
fig.tight_layout(rect=(0,0,1,.96));fig.savefig(DEST/'case2_population.png',dpi=150);plt.close(fig)
print('Saved local joint-trajectory and population figures')

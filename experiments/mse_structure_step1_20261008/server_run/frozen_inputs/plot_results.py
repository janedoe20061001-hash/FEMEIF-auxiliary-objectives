"""Fixed-seed actual selected curves; no averaging impedance curves."""
from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'frozen'));sys.path.insert(0,str(HERE/'toolkit'))
from model import Model
from adaptive import spectra
from refined_grid import improved_dense_grid
RAW=HERE/'server_run'/'retrieved';OUT=HERE.parents[1]/'output'/'mse_structure_step1_20261008';FIG=OUT/'figures'
def main():
    FIG.mkdir(exist_ok=True,parents=True);m=Model('case1');cfg=json.loads((HERE/'CONFIG.json').read_text());params={}
    colors={'original':'#2776b8','adaptive':'#ce6730'}
    for seed in cfg['plot_seeds']:
        curves=[]
        for arm in ['original','adaptive']:
            for g in [300,1000]:
                with np.load(RAW/'runs'/arm/f'seed_{seed}'/f'generation_{g:03}.npz') as d:
                    ix=int(d['selected_index']);p=10**d['log_parameters'][ix];curves.append((arm,g,p));params[f'{seed}:{arm}:{g}']=dict(index=ix,parameters=p.tolist(),training_mse=d['optimization_objective'][ix].tolist())
        f=improved_dense_grid(m.record,np.asarray([p for _,_,p in curves]),4);target=spectra(f,m.record['target_parameters'])[0]
        fig,axes=plt.subplots(2,3,figsize=(15,7),layout='constrained')
        for col,(lo,hi,title) in enumerate([(100,1e6,'Whole range'),(25160,25295,'Narrow peak near 25.23 kHz'),(1e5,3.1623e5,'Adjacent structures: 100-316 kHz')]):
            mask=(f>=lo)&(f<=hi)
            for row in [0,1]:
                y=20*np.log10(abs(target)) if row==0 else np.rad2deg(np.angle(target));axes[row,col].plot(f[mask],y[mask],color='black',lw=1.8,ls='--',label='Target',zorder=10)
                for arm,g,p in curves:
                    z=spectra(f[mask],p)[0];y=20*np.log10(abs(z)) if row==0 else np.rad2deg(np.angle(z));axes[row,col].plot(f[mask],y,color=colors[arm],lw=1.1,ls=':' if g==300 else '-',label=f'{arm}, {g} generations')
                if col!=1:axes[row,col].set_xscale('log')
                axes[row,col].set_xlim(lo,hi);axes[row,col].grid(alpha=.2)
                axes[row,col].set_ylabel('Magnitude (dB)' if row==0 else 'Phase (degree)')
                if row==1:axes[row,col].set_xlabel('Frequency (Hz)')
            axes[0,col].set_title(title)
        axes[0,0].legend(fontsize=8);fig.suptitle(f'Case1 | fixed seed {seed} | representative: min(MAG + 25.12 PHA)',fontsize=12);fig.savefig(FIG/f'seed_{seed}.png',dpi=155);plt.close(fig)
    chosen=[(seed,np.array(params[f'{seed}:adaptive:1000']['parameters'])) for seed in cfg['plot_seeds']]
    f=improved_dense_grid(m.record,np.asarray([p for _,p in chosen]),4);target=spectra(f,m.record['target_parameters'])[0]
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for col,(lo,hi,title) in enumerate([(23000,26000,'Valley / peak pair: 23.66 / 25.23 kHz'),(70000,225000,'Adjacent peaks and valleys: 70-225 kHz')]):
        mask=(f>=lo)&(f<=hi)
        for row in [0,1]:
            y=20*np.log10(abs(target)) if row==0 else np.rad2deg(np.angle(target));axes[row,col].plot(f[mask],y[mask],color='black',lw=1.8,ls='--',label='Target',zorder=10)
            for (seed,p),color in zip(chosen,['#2776b8','#ce6730','#448f67']):
                z=spectra(f[mask],p)[0];y=20*np.log10(abs(z)) if row==0 else np.rad2deg(np.angle(z));axes[row,col].plot(f[mask],y,color=color,lw=1.3,label=f'seed {seed}')
            if col==1:axes[row,col].set_xscale('log')
            axes[row,col].set_xlim(lo,hi);axes[row,col].grid(alpha=.2);axes[row,col].set_ylabel('Magnitude (dB)' if row==0 else 'Phase (degree)')
            if row==1:axes[row,col].set_xlabel('Frequency (Hz)')
        axes[0,col].set_title(title)
    axes[0,0].legend(fontsize=9);fig.suptitle('Case1 | adaptive, 1000 generations | fixed seeds; no averaging');fig.savefig(FIG/'structure_focus.png',dpi=155);plt.close(fig)
    (OUT/'PLOT_PARAMETERS.json').write_text(json.dumps(params,indent=2),encoding='utf8')
    if not (OUT/'TRAJECTORY.csv').exists():return
    trajectory=[]
    import csv
    with (OUT/'TRAJECTORY.csv').open(encoding='utf-8-sig') as f:trajectory=list(csv.DictReader(f))
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ax,key,label in zip(axes.flat,['amplitude_rmse_db','phase_rmse_deg','amp_incomplete','phase_bad_bands'],['Magnitude RMSE (dB)','Phase RMSE (degree)','Incomplete amplitude events / seed','Phase bands above 15 degree / seed']):
        for arm in ['original','adaptive']:
            for rule in ['min_wsmse','balanced_minmax']:
                rr=[r for r in trajectory if r['arm']==arm and r['rule']==rule];ax.plot([int(r['generation']) for r in rr],[float(r[key]) for r in rr],color=colors[arm],ls='-' if rule=='min_wsmse' else '--',label=f'{arm}, '+('weighted selection' if rule=='min_wsmse' else 'balanced selection'))
        ax.set_ylabel(label);ax.set_xlabel('Generation');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Mean of per-seed diagnostic scores (100 seeds); curves are not averaged');fig.savefig(FIG/'trajectory.png',dpi=155);plt.close(fig)
    (OUT/'PLOT_PARAMETERS.json').write_text(json.dumps(params,indent=2),encoding='utf8')
if __name__=='__main__':main()

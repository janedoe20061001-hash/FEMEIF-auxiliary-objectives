"""Small fixed explanatory set; never used to choose final optimization representatives."""
import importlib.util,sys
from pathlib import Path
_spec=importlib.util.spec_from_file_location('physical_plot_base',Path(__file__).with_name('run.py'))
b=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=b;_spec.loader.exec_module(b)
import numpy as np,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=b.OUT/'figures';OUT.mkdir(exist_ok=True)

def record(t,batch,method,seed):
    d=json.loads((b.OUT/'details'/f'{t}.json').read_text(encoding='utf-8'))
    return next(r for r in d['representatives'] if (r['batch'],r['method'],r['seed'])==(batch,method,seed))

def curves(t,batch,methods,seed):
    data=b.core.target(t);c=data['circuit'];q=data['quadrature']
    p=[10**np.array(record(t,batch,m,seed)['log_parameters']) for m in methods]
    x=np.unique(np.r_[q.reference_grid(c.p,8),*[q.reference_grid(v,8) for v in p]])
    z=[c.response(10**x,v)[0] for v in [c.p,*p]]
    return data,x,z

def decorate(ax,xlabel,ylabel):
    ax.set(xlabel=xlabel,ylabel=ylabel);ax.grid(alpha=.25)

def main():
    d=b.simple('simple_series');c=d['circuit'];f=np.geomspace(1e4,2.6e4,3001)
    fig,aa=plt.subplots(1,2,figsize=(11,4))
    for s in (.25,1,4):
        p=c.p.copy();p[1]*=s;p[2]/=s;z,_=c.response(f,p)
        aa[0].plot(f/1000,20*np.log10(abs(z)),label=f'L x {s:g}; C / {s:g}')
        aa[1].plot(f/1000,np.rad2deg(np.angle(z)),label=f'Q = {10*s:g}')
    for a in aa:a.axvline(1/(2*np.pi*np.sqrt(.001*1e-7))/1000,color='k',ls='--',alpha=.6);a.legend(fontsize=9)
    decorate(aa[0],'Frequency (kHz)','Magnitude (dB ohm)');decorate(aa[1],'Frequency (kHz)','Phase (degree)')
    fig.suptitle('Analytic series RLC: same resonance frequency and valley level, different sharpness')
    fig.tight_layout();fig.savefig(OUT/'simple_separation.png',dpi=140);plt.close(fig)
    data,x,z=curves('control0p25','revision',['MAG_PHA','MAG_PHA_MAX'],42002)
    fig,aa=plt.subplots(1,2,figsize=(11,4));labels=['Target','MAG/PHA baseline','G knee (42002)']
    for zz,label in zip(z,labels):
        aa[0].plot(10**x/1000,20*np.log10(abs(zz)),label=label);aa[1].plot(10**x/1000,np.rad2deg(np.angle(zz)),label=label)
    for a in aa:a.set_xlim(9.5,10.5);a.legend(fontsize=9)
    decorate(aa[0],'Frequency (kHz)','Magnitude (dB ohm)');decorate(aa[1],'Frequency (kHz)','Phase (degree)')
    fig.suptitle('Same phase and nearly same resonance; wrong absolute impedance level')
    fig.tight_layout();fig.savefig(OUT/'scale_failure.png',dpi=140);plt.close(fig)
    data,x,z=curves('mixed0p25','family',['MAG_PHA','MAG_PHA_INTEGRAL'],43000)
    fig=plt.figure(figsize=(14,4.3));a=fig.add_subplot(131);bb=fig.add_subplot(132);cc=fig.add_subplot(133,projection='3d')
    labels=['Target','MAG/PHA baseline','C knee (43000)'];colors=['k','tab:blue','tab:orange']
    for zz,label,color in zip(z,labels,colors):
        a.plot(10**x/1000,np.rad2deg(np.angle(zz)),label=label,color=color)
        bb.plot(10**x/1000,np.rad2deg(np.angle(zz)),label=label,color=color)
        mask=(10**x>=9000)&(10**x<=12500)
        cc.plot(10**x[mask]/1000,zz[mask].real,zz[mask].imag,color=color,label=label)
    a.set_xscale('log');a.set_xlim(1,300);bb.set_xlim(9,12.5)
    for ax in (a,bb):decorate(ax,'Frequency (kHz)','Phase (degree)')
    a.set_title('Full-band phase lobes');bb.set_title('Matched zeros do not fix lobe height');a.legend(fontsize=8)
    cc.set(xlabel='Frequency (kHz)',ylabel='Re Z (ohm)',zlabel='Im Z (ohm)',title='Raw local 3D trajectory');cc.view_init(25,-55)
    fig.tight_layout();fig.savefig(OUT/'mixed_phase_lobes.png',dpi=140);plt.close(fig)
    data,x,z=curves('three0p25','family',['MAG_PHA','MAG_PHA_BRIDGE'],43000)
    fig,aa=plt.subplots(1,2,figsize=(11,4));labels=['Target','MAG/PHA baseline','B knee (43000)']
    for zz,label in zip(z,labels):
        aa[0].plot(10**x/1000,20*np.log10(abs(zz)),label=label);aa[1].plot(10**x/1000,np.rad2deg(np.angle(zz)),label=label)
    for a in aa:a.set_xlim(27,33);a.legend(fontsize=8)
    decorate(aa[0],'Frequency (kHz)','Magnitude (dB ohm)');decorate(aa[1],'Frequency (kHz)','Phase (degree)')
    aa[1].axhline(0,c='k',ls='--',lw=.8);aa[1].set_ylim(-90,5)
    fig.suptitle('Extra phase lobes stay capacitive: a zero-crossing check cannot see them')
    fig.tight_layout();fig.savefig(OUT/'three_extra_lobes.png',dpi=140);plt.close(fig)
    b.dump(b.OUT/'PLOT_SELECTION.json',dict(simple='analytic controlled example',
           examples=[dict(target='control0p25',batch='revision',seed=42002,methods=['MAG_PHA','MAG_PHA_MAX'],reason='previously identified knee failure'),
                     dict(target='mixed0p25',batch='family',seed=43000,methods=['MAG_PHA','MAG_PHA_INTEGRAL'],reason='fixed first seed; explain missing phase lobes'),
                     dict(target='three0p25',batch='family',seed=43000,methods=['MAG_PHA','MAG_PHA_BRIDGE'],reason='fixed first seed; explain extra capacitive lobes')],
           all_final_representatives_retained=True,averaged_curves=False))

if __name__=='__main__':main()

from reselect import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(batch):
    core,_=setup(batch);records=[]
    for path in sorted((core.OUT/'analysis/details').glob('*_knee.json')):
        d=json.loads(path.read_text(encoding='utf-8'));s=d['summary']
        if not s['extra_amp']:continue
        data=core.target(s['target']);p=10**np.array(d['log_parameters']);q=data['quadrature'];c=data['circuit']
        x=q.reference_grid(p,max(4,d['diagnostic']['refinement']));z,_=c.response(10**x,p)
        events=core.nd.events(x,z,data['config'])
        found=[e for e in events if e['event_id'] in d['diagnostic']['extra_events']]
        assert len(found)==s['extra_amp']
        records.append(dict(batch=batch,target=s['target'],method=s['method'],seed=s['seed'],extra_events=found))
    dump(core.OUT/'EXTRA_EVENTS.json',records)
    if batch!='revision':return
    source=OUT/'source_populations'/batch
    if not source.exists():source=ROOT/'output'/BATCHES[batch]/'batch'
    data=core.target('control0p25');c=data['circuit'];q=data['quadrature']
    fig,axes=plt.subplots(2,2,figsize=(13,9));diagnostics=[]
    for column,seed in enumerate((42001,42002)):
        method='MAG_PHA_MAX'
        s=json.loads((core.OUT/f'selections/control0p25_{method}_{seed}.json').read_text())
        with np.load(source/f'control0p25/{method}/seed_{seed}/generation_0300.npz') as z:
            f=z['objectives'];params=z['log_parameters'];front=np.array(s['unique_indices']);n=np.array(s['normalized']);ki=s['selected_index']
            ax=axes[0,column];ax.scatter(n[:,0],n[:,1],s=30);ax.scatter(*n[s['unique_indices'].index(ki)],c='red',marker='*',s=170);ax.plot([0,1],[1,0],'k--',alpha=.5)
            ax.set(title=f'Control / G / seed {seed} / {s["reason"]}',xlabel='Normalized MAG',ylabel='Normalized PHA');ax.grid(alpha=.2)
            d=json.loads((core.OUT/f'analysis/details/control0p25_MAG_PHA_{seed}_knee.json').read_text())
            grid=q.reference_grid(10**params[ki],4);freq=10**grid;target,_=c.response(freq);fit,_=c.response(freq,10**params[ki]);base,_=c.response(freq,10**np.array(d['log_parameters']))
            ax=axes[1,column];ax.plot(freq,20*np.log10(abs(target)),'k--',label='Target');ax.plot(freq,20*np.log10(abs(base)),label='MAG/PHA knee');ax.plot(freq,20*np.log10(abs(fit)),label='G knee' if s['reason']=='convex_knee' else 'G fallback');ax.set_xscale('log');ax.set(xlabel='Frequency (Hz)',ylabel='Magnitude (dB ohm)');ax.grid(alpha=.2);ax.legend()
            diagnostics.append(dict(seed=seed,selected_objectives=f[ki].tolist(),front_min=s['minimum'],front_max=s['maximum'],reason=s['reason'],display='two failure examples chosen for explanation, not randomly representative',rms_magnitude_db=float(np.sqrt(f[ki,0])),rms_phase_deg=float(np.rad2deg(np.sqrt(f[ki,1])))))
    fig.tight_layout();fig.savefig(core.OUT/'figures/control_selection_failures.png',dpi=140);plt.close(fig)
    dump(core.OUT/'CONTROL_SELECTION_FAILURES.json',diagnostics)

if __name__=='__main__':main(sys.argv[1])

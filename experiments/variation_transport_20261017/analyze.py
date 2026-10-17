"""Posthoc explanation and static scientific figures; frozen scores unchanged."""
from metric import *
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2,'savefig.dpi':160})
def loadcsv(path):return list(csv.DictReader(path.open(encoding='utf-8-sig')))
def save(fig,name):fig.savefig(FIG/name,bbox_inches='tight');plt.close(fig)

def main():
    rows=loadcsv(OUT/'DIRECTIONS.csv');old=loadcsv(ROOT/'output/geometry_followup_20261016/PAIR_DIRECTIONS.csv');control=loadcsv(OUT/'CONTROLLED.csv')
    newmain=[r for r in rows if r['metric']=='JV'];bank=json.loads((ROOT/'output/geometry_followup_20261016/HOLDOUT_BANK.json').read_text())['selected'];pairs={p['pair_id']:p for p in bank}
    comparison=[]
    for metric in ['J0','ARX_S_L','Local_ARX','Root','B','JM','JC','JV']:
        rr=[r for r in (rows if metric in NAMES else old) if r['metric']==metric]
        dd=[r for r in rr if r['scope'] in ['dev','old_development']];hh=[r for r in rr if r['scope'] in ['held','old_regression']]
        comparison.append(dict(metric=metric,dev=sum(r['direction']=='correct' for r in dd),regression=sum(r['direction']=='correct' for r in hh),missing=sum(r['direction']=='correct' and r['family']=='missing_structure' for r in hh),distortion=sum(r['direction']=='correct' and r['family']=='existing_distortion' for r in hh),extra=sum(r['direction']=='correct' and r['family']=='extra_structure' for r in hh)))
    transitions={}
    for metric in ['J0','ARX_S_L','Root','B']:
        prev={r['pair_id']:r for r in old if r['metric']==metric and r['scope']=='held'}
        current=[r for r in newmain if r['scope']=='old_regression']
        transitions[metric]=dict(repaired=[r['pair_id'] for r in current if r['direction']=='correct' and prev[r['pair_id']]['direction']!='correct'],newly_wrong=[r['pair_id'] for r in current if r['direction']!='correct' and prev[r['pair_id']]['direction']=='correct'])
    identical=all(len({next(r['direction'] for r in rows if r['pair_id']==p['pair_id'] and r['scope']==p['scope'] and r['metric']==m) for m in NAMES})==1 for p in newmain)
    failures=[];cache={}
    for r in newmain:
        if r['scope']!='old_regression' or r['direction']=='correct':continue
        p=pairs[r['pair_id']];tid=p['target'];label=p['region']
        if tid not in cache:cache[tid]=Variation(tid)
        m=cache[tid];w=m.window(label);sides={}
        for side in ['better','worse']:
            pp=10**np.array(p[side]['log_parameters']);e=m.g.panel_edges(w,m.g.modes(pp),1);t,wt=nodes(e,64);z=m.response(w,t,pp)[0];truth=m.response(w,t,m.c.p)[0];delta=z-truth;offset=wt@delta
            j0=float(wt@abs(delta)**2);shape=float(wt@abs(delta-offset)**2);assert abs(j0-shape-abs(offset)**2)<1e-9*max(1,j0)
            rec=json.loads((OUT/'scores'/f"{p['pair_id']}_{side}.json").read_text())['windows'][label]
            sides[side]=dict(J0=j0,constant_offset_real=float(offset.real),constant_offset_imag=float(offset.imag),constant_offset_squared=float(abs(offset)**2),centered_point_error=shape,event=next(v for v in p[side]['events'] if v['event_id']==label),scores=rec['scores'])
        failures.append(dict(pair_id=p['pair_id'],target=tid,region=label,family=p['family'],**sides,
                             worse_has_larger_constant_but_smaller_centered=bool(sides['worse']['constant_offset_squared']>sides['better']['constant_offset_squared'] and sides['worse']['centered_point_error']<sides['better']['centered_point_error'])))
    dump(OUT/'POSTHOC_ANALYSIS.json',dict(scope='Posthoc explanation only; no formula or label changes',comparison=comparison,transitions=transitions,all_JM_JC_JV_directions_identical=identical,failures=failures))
    # Single factors, each loss in its original fixed scale.
    fig,axes=plt.subplots(1,3,figsize=(12,3.5));grid=[r for r in control if r['label']=='P_control' and r['family']=='controlled_grid']
    specs=[('amplitude',lambda r:float(r['offset'])==0 and float(r['q_ratio'])==1,'R / target R'),('offset',lambda r:float(r['amplitude'])==1 and float(r['q_ratio'])==1,'Center offset / target bandwidth'),('q_ratio',lambda r:float(r['amplitude'])==1 and float(r['offset'])==0,'Q / target Q')]
    for ax,(key,filt,title) in zip(axes,specs):
        rr=sorted([r for r in grid if filt(r)],key=lambda r:float(r[key]));x=[float(r[key]) for r in rr]
        for metric in NAMES:ax.plot(x,[float(r[metric]) for r in rr],'-o',ms=3,label=metric)
        ax.set_xlabel(title);ax.set_ylabel('Loss')
        if key=='q_ratio':ax.set_xscale('log',base=2)
    axes[0].legend();fig.suptitle('Controlled parallel RLC: separate amplitude, position and Q');fig.tight_layout();save(fig,'single_factors.png')
    # All 36 old pairs, not a new validation set.
    names=['J0','ARX_S_L','Root','B','JM','JC','JV'];ids=[('dev',f'pair_{i:03}') for i in range(12)]+[('held',f'held_{i:03}') for i in range(24)];mat=[]
    for scope,pid in ids:
        line=[]
        for metric in names:
            source=rows if metric in NAMES else old;wanted=('old_development' if scope=='dev' else 'old_regression') if metric in NAMES else scope
            r=next(r for r in source if r['pair_id']==pid and r['scope']==wanted and r['metric']==metric);line.append({'correct':1,'wrong':-1}.get(r['direction'],0))
        mat.append(line)
    fig,ax=plt.subplots(figsize=(8,10));ax.imshow(mat,cmap=ListedColormap(['#c35656','#dddddd','#397b60']),vmin=-1,vmax=1,aspect='auto');ax.set_xticks(range(len(names)),names);ax.set_yticks(range(len(ids)),['D '+p if s=='dev' else 'R '+p for s,p in ids],fontsize=8);ax.axhline(11.5,color='black');ax.grid(False);ax.set_title('Existing pairs: green agrees with local label; red disagrees');fig.tight_layout();save(fig,'directions.png')
    # Fixed representative: first existing-distortion failure in source order.
    chosen=next(p for p in failures if p['family']=='existing_distortion');p=pairs[chosen['pair_id']];m=cache[p['target']];w=m.window(p['region'])
    pp=[m.c.p]+[10**np.array(p[s]['log_parameters']) for s in ['better','worse']];e=np.unique(np.concatenate([m.polygrid(w,q,32) for q in pp]));x=w['x'][0]+e*(w['x'][-1]-w['x'][0]);zz=[m.response(w,e,q)[0]*w['scale'] for q in pp]
    fig=plt.figure(figsize=(12,8));a=fig.add_subplot(221,projection='3d');b=fig.add_subplot(222);c=fig.add_subplot(223);d=fig.add_subplot(224)
    for name,z,color in zip(['Target','Better local label','Worse local label'],zz,['black','#26744b','#b74747']):
        a.plot(x,z.real,z.imag,color=color,label=name);b.plot(z.real,z.imag,color=color,label=name);c.plot(x,20*np.log10(abs(z)),color=color);d.plot(x,z.real,color=color,label=name+' Re');d.plot(x,z.imag,'--',color=color,label=name+' Im')
    a.set(xlabel='log10 f / Hz',ylabel='Re Z / ohm',zlabel='Im Z / ohm');b.set(xlabel='Re Z / ohm',ylabel='Im Z / ohm');c.set(xlabel='log10 f / Hz',ylabel='Magnitude / dB ohm');d.set(xlabel='log10 f / Hz',ylabel='Impedance / ohm');b.legend(fontsize=8);d.legend(fontsize=7,ncol=2)
    fig.suptitle(chosen['pair_id']+' '+p['target']+' '+p['region']+': JV disagrees with the local event label');fig.tight_layout();save(fig,'failure_trajectory.png')
    # Explicit constant-offset contribution: exact L2 identity, not new candidate metric.
    fig,axes=plt.subplots(1,2,figsize=(11,4));x=np.arange(len(failures));width=.35
    for ax,key,title in zip(axes,['constant_offset_squared','centered_point_error'],['Squared mean complex offset','Point error after subtracting mean offset']):
        ax.bar(x-width/2,[p['better'][key] for p in failures],width,label='Better local label',color='#26744b');ax.bar(x+width/2,[p['worse'][key] for p in failures],width,label='Worse local label',color='#b74747');ax.set_xticks(x,[p['pair_id'] for p in failures],rotation=25);ax.set_title(title);ax.set_ylabel('Normalized squared impedance')
    axes[0].legend();fig.suptitle('Posthoc diagnostic of all five JV regression failures');fig.tight_layout();save(fig,'offset_decomposition.png')
    fig,ax=plt.subplots(figsize=(8,3.5));benchmark=json.loads((OUT/'BENCHMARK.json').read_text());ax.bar(range(12),[r['ratio'] for r in benchmark['rows']],color='#577b94');ax.axhline(1.25,color='#b74747',label='Predeclared 1.25x gate');ax.set(xlabel='Fixed timing candidate',ylabel='JV / compiled ARX time',title='Same window and tolerance; implementation-specific cost');ax.legend();fig.tight_layout();save(fig,'cost.png')
    print(json.dumps(dict(comparison=comparison,transitions=transitions,identical_directions=identical,constant_offset_tradeoff_count=sum(f['worse_has_larger_constant_but_smaller_centered'] for f in failures)),ensure_ascii=False),flush=True)
if __name__=='__main__':main()

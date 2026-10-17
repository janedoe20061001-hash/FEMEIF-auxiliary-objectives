"""Cohort tables and scientific figures for the fixed local diagnosis."""
from diagnose import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'output/valley_process_20261017';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2,'savefig.dpi':160})
def save(fig,name):fig.savefig(FIG/name,bbox_inches='tight');plt.close(fig)
def main():
    data=json.loads((OUT/'DIAGNOSTICS.json').read_text());r=data['records'];targets={q['target']:q for q in r if q['source'] is None};candidates=[q for q in r if q['source'] is not None];lookup={q['id']:q for q in r};profiles=np.load(OUT/'PROFILES.npz');rows=[]
    for q in candidates:
        target=targets[q['target']];p=q['levels'][-1]['core'];t=target['levels'][-1]['core'];old=q['old_V3_label'];cw=q['matched_valley_widths'];tw=target['matched_valley_widths'];levels=cw['levels'] if cw and cw.get('resolved') else {}
        row=dict(id=q['id'],target=q['target'],original_pair_region=q['source']['original_region'],old_matched=old['matched'],old_complete=old['complete'],old_height_delta_db=old['height_delta_db'],old_width_ratio=old['width_ratio'],old_shift_decades=old['shift_decades'],core_arc80_ratio=p['arc80_width_decades']/t['arc80_width_decades'],core_length_ratio=p['length_ohm']/t['length_ohm'],core_endpoint_efficiency=p['endpoint_efficiency'],target_core_endpoint_efficiency=t['endpoint_efficiency'],core_radial_sequence=' '.join(e['kind'] for e in p['radial_extrema']),target_core_radial_sequence=' '.join(e['kind'] for e in t['radial_extrema']),max_anchor_error_ohm=max(a['complex_error_ohm'] for a in q['anchors']),width1_ratio=levels['1']['width_decades']/tw['levels']['1']['width_decades'] if levels.get('1') else None,width3_ratio=levels['3']['width_decades']/tw['levels']['3']['width_decades'] if levels.get('3') else None,width3_right_left_asymmetry=levels['3']['asymmetry_right_over_left'] if levels.get('3') else None,numerical_passed=q['numerical_passed'])
        rows.append(row)
    csvout(OUT/'COHORT_TABLE.csv',rows)
    complete=[q for q in rows if q['old_complete']];summary=dict(cohort_candidates=len(rows),targets=2,old_V3_complete=len(complete),old_V3_matched=sum(q['old_matched'] for q in rows),old_complete_arc80_ratio_range=[min(q['core_arc80_ratio'] for q in complete),max(q['core_arc80_ratio'] for q in complete)],old_complete_width1_ratio_range=[min(q['width1_ratio'] for q in complete if q['width1_ratio'] is not None),max(q['width1_ratio'] for q in complete if q['width1_ratio'] is not None)],old_complete_length_ratio_range=[min(q['core_length_ratio'] for q in complete),max(q['core_length_ratio'] for q in complete)],all_numerical_passed=data['numerical_passed']==len(r),no_relabeling=True,no_new_auxiliary_score=True,no_new_optimization=True)
    dump(OUT/'SUMMARY.json',summary)
    # Targets: absolute port paths and radial/tangential progression.
    fig=plt.figure(figsize=(13,7.5))
    for i,(tid,t) in enumerate(targets.items()):
        x=profiles[t['id']+'__x'];z=profiles[t['id']+'__z'];d=profiles[t['id']+'__dz_dx'];lo,hi=t['fixed_core'];inside=(x>=lo)&(x<=hi);xx=x[inside];zz=z[inside];dd=d[inside];u=(xx-lo)/(hi-lo)
        a=fig.add_subplot(2,3,3*i+1,projection='3d');a.plot(xx,zz.real,zz.imag,color='black');a.set(xlabel='log10 f / Hz',ylabel='Re Z / ohm',zlabel='Im Z / ohm',title=tid)
        b=fig.add_subplot(2,3,3*i+2);b.plot(zz.real,zz.imag,color='black');e=next(e for e in t['levels'][-1]['core']['radial_extrema'] if e['kind']=='valley');b.scatter([e['real_ohm']],[e['imag_ohm']],c='#b74747',label=f"Minimum: X={e['imag_ohm']:.1f} ohm");b.set(xlabel='Re Z / ohm',ylabel='Im Z / ohm');b.legend(fontsize=8)
        c=fig.add_subplot(2,3,3*i+3);rad=np.real(zz.conj()*dd)/abs(zz);tan=np.imag(zz.conj()*dd)/abs(zz);c.plot(u,rad,label='Radial speed');c.plot(u,tan,label='Tangential speed');c.axhline(0,color='gray',lw=.8);c.set(xlabel='Fixed target core coordinate',ylabel='ohm / decade');c.legend(fontsize=8)
    fig.suptitle('Target V3: a magnitude minimum need not cross Im Z = 0');fig.tight_layout();save(fig,'target_processes.png')
    # Source-ordered example, no alignment or per-candidate scaling.
    item=lookup['held_003_better'];target=targets[item['target']];lo,hi=target['fixed_core'];fig=plt.figure(figsize=(12,8));axes=[fig.add_subplot(221),fig.add_subplot(222),fig.add_subplot(223),fig.add_subplot(224,projection='3d')]
    for q,name,col in [(target,'Target','black'),(item,'Old event rule: complete','#26744b')]:
        x=profiles[q['id']+'__x'];z=profiles[q['id']+'__z'];d=profiles[q['id']+'__dz_dx'];mask=(x>=lo)&(x<=hi);x=x[mask];z=z[mask];d=d[mask];u=(x-lo)/(hi-lo);arc=np.r_[0,np.cumsum(abs(np.diff(z)))];arc/=arc[-1]
        axes[0].plot(x,20*np.log10(abs(z)),color=col,label=name);axes[1].plot(u,arc,color=col,label=name);axes[2].plot(u,np.real(z.conj()*d)/abs(z),color=col,label=name);axes[3].plot(x,z.real,z.imag,color=col,label=name)
    axes[0].set(xlabel='log10 f / Hz',ylabel='Magnitude / dB ohm',title='Same fixed target core');axes[1].set(xlabel='Fixed target core coordinate',ylabel='Fraction of complex path length',title='The movement is concentrated in a narrow interval');axes[2].set_yscale('symlog',linthresh=100);axes[2].set(xlabel='Fixed target core coordinate',ylabel='Radial speed / ohm per decade',title='Symlog display: inbound and outbound motion');axes[3].set(xlabel='log10 f / Hz',ylabel='Re Z / ohm',zlabel='Im Z / ohm',title='A narrow additional excursion');axes[0].legend(fontsize=8);axes[1].legend(fontsize=8);fig.suptitle('held_003 positive side: 3 dB width passes while the process is different');fig.tight_layout();save(fig,'representative_process.png')
    # All candidates, including those originally chosen for other structures.
    fig,ax=plt.subplots(2,1,figsize=(13,7),sharex=True);x=np.arange(len(rows));colors=['#26744b' if q['old_complete'] else '#b48b53' if q['old_matched'] else '#999999' for q in rows]
    for a,key,title in [(ax[0],'core_arc80_ratio','80% arc-length frequency span / target span'),(ax[1],'core_length_ratio','Complex path length / target path length')]:
        a.scatter(x,[q[key] for q in rows],c=colors);a.axhline(1,color='black',ls='--',lw=1);a.set_yscale('log');a.set_ylabel(title);a.axvline(15.5,color='gray')
    ax[1].set_xticks(x,[q['id'].replace('held_','').replace('_better','+').replace('_worse','-') for q in rows],rotation=60,fontsize=8);ax[1].set_xlabel('Original pair and side: + / - are source labels only');fig.suptitle('All 32 old candidates at V3: green old-complete, brown matched, gray unmatched');fig.tight_layout();save(fig,'all_candidates.png')
    # Every frozen V3-positive side previously reported as a JV failure.
    selected=['held_000_better','held_003_better','held_009_better','held_010_better','held_011_better'];rr=[next(q for q in rows if q['id']==ident) for ident in selected];xx=np.arange(5);wid=.25;fig,axes=plt.subplots(1,2,figsize=(12,4))
    for off,key,label,col in [(-wid,'width1_ratio','1 dB magnitude width','#5287a7'),(0,'width3_ratio','3 dB magnitude width','#c19a4a'),(wid,'core_arc80_ratio','80% complex-motion span','#26744b')]:axes[0].bar(xx+off,[q[key] for q in rr],wid,label=label,color=col)
    axes[0].axhline(1,color='black',ls='--');axes[0].set_yscale('log');axes[0].set_ylabel('Candidate / target');axes[0].set_xticks(xx,[q['id'].replace('_better','+') for q in rr],rotation=20);axes[0].legend(fontsize=8)
    axes[1].bar(xx,[q['width3_right_left_asymmetry'] for q in rr],color='#777777');axes[1].set_yscale('log');axes[1].set_ylabel('3 dB right extent / left extent');axes[1].set_xticks(xx,[q['id'].replace('_better','+') for q in rr],rotation=20);fig.suptitle('Different widths answer different questions; none is a new pass threshold');fig.tight_layout();save(fig,'width_comparison.png')
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

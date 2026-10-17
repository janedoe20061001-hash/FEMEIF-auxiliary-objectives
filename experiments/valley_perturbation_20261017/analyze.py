"""Intervention-response checks and limitations; no fitted labels or weights."""
from physical import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'output/valley_perturbation_20261017';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2,'savefig.dpi':160})
def save(fig,name):fig.savefig(FIG/name,bbox_inches='tight');plt.close(fig)
def main():
    plan=json.loads((OUT/'PLAN.json').read_text(encoding='utf-8'));targets={t['id']:t for t in plan['targets']};records=[json.loads((OUT/'records'/f"{t['id']}.json").read_text(encoding='utf-8')) for t in plan['tasks']];items={q['id']:q for q in plan['tasks']};rows=[]
    for r in records:
        if not r['numerical_passed']:continue
        for scope in ['core','window']:
            q=r['levels'][-1]['scopes'][scope];width=targets[r['target']]['core'][1]-targets[r['target']]['core'][0]
            rows.append(dict(id=r['id'],target=r['target'],family=r['family'],scope=scope,meta=json.dumps(r['meta'],sort_keys=True),state_error=r['anchors']['mean_error_over_target_scale'],length_ratio=q['length_ratio'],arc80_ratio=q['arc80_width']/q['target_arc80_width'] if q['arc80_width'] is not None else None,cdf_L1=q['cdf_L1'],endpoint_efficiency=q['endpoint_efficiency'],sequence=' '.join(e['kind'] for e in q['radial_extrema']),boundary_roots=len(q['boundary_extrema']),q50_error_widths=(q['arc_quantiles'][1]-q['target_arc_quantiles'][1])/width if q['arc_quantiles'] is not None else None,zero_motion=q['zero_motion'],undefined_tangent_anchors=sum(v is None for v in r['anchors']['tangent_cosines'])))
    csvout(OUT/'ALL_DIAGNOSTICS.csv',rows)
    offsets=[r for r in rows if r['family']=='offset'];series=[r for r in rows if r['family']=='Q' and r['target']=='series_Q10'];extra=[r for r in rows if r['family']=='extra' and r['scope']=='core'];background=[r for r in rows if r['family']=='background' and r['scope']=='core'];shiftturns=[];state_turns=[]
    for tid in targets:
        for scope in ['core','window']:
            for sign in [-1,1]:
                sr=sorted([r for r in rows if r['target']==tid and r['family']=='shift' and r['scope']==scope and sign*items[r['id']]['meta']['shift_widths']>0],key=lambda r:abs(items[r['id']]['meta']['shift_widths']))
                for a,b in zip(sr,sr[1:]):
                    if a['cdf_L1'] is not None and b['cdf_L1'] is not None and b['cdf_L1']<a['cdf_L1']-1e-4-1e-3*abs(b['cdf_L1']):shiftturns.append(dict(target=tid,scope=scope,from_id=a['id'],to_id=b['id'],from_offset=items[a['id']]['meta']['shift_widths'],to_offset=items[b['id']]['meta']['shift_widths'],from_cdf=a['cdf_L1'],to_cdf=b['cdf_L1']))
                    if scope=='core' and b['state_error']<a['state_error']-1e-4-1e-3*abs(b['state_error']):state_turns.append(dict(target=tid,from_id=a['id'],to_id=b['id'],from_offset=items[a['id']]['meta']['shift_widths'],to_offset=items[b['id']]['meta']['shift_widths'],from_state=a['state_error'],to_state=b['state_error']))
    summary=dict(tasks=len(records),numerical_passed=sum(r['numerical_passed'] for r in records),family_counts={f:sum(r['family']==f for r in records) for f in sorted(set(r['family'] for r in records))},analytic_identity_checks=sum(r['analytic_identity']['applicable'] for r in records),analytic_identity_max_relative=max(r['analytic_identity'].get('max_relative',0) for r in records),offset_max_length_ratio_error=max(abs(r['length_ratio']-1) for r in offsets),offset_max_cdf=max(r['cdf_L1'] for r in offsets),offset_max_state_reference_error=max(abs(r['state_error']-items[r['id']]['meta']['relative_offset']) for r in offsets),series_Q_max_cdf=max(r['cdf_L1'] for r in series),series_Q_max_arc80_ratio_error=max(abs(r['arc80_ratio']-1) for r in series),series_Q_max_length_ratio_error=max(abs(r['length_ratio']-items[r['id']]['meta']['factor']) for r in series),extra_same_single_valley_sequence=sum(r['sequence']=='valley' for r in extra),extra_total=len(extra),extra_sequence_counts={tid:{seq:sum(r['sequence']==seq and r['target']==tid for r in extra) for seq in sorted(set(r['sequence'] for r in extra if r['target']==tid))} for tid in targets},background=background,cdf_shift_reversals=shiftturns,nodal_modes={mode:sum(r['nodal']['mode']==mode for r in records) for mode in sorted(set(r['nodal']['mode'] for r in records))},max_polygon_level=max(r['levels'][-1]['n'] for r in records),developed_new_score=False,optimized=False)
    dump(OUT/'INTERVENTION_CHECKS.json',summary)
    summary['state_shift_reversals']=state_turns
    qrows=[]
    for r in rows:
        if r['family']!='Q' or r['scope']!='core':continue
        k=items[r['id']]['meta']['factor'];width=np.diff(targets[r['target']]['core'])[0];physical_width=2*np.arcsinh(np.sinh(np.log(10)*width/2)/k)/np.log(10)
        qrows.append(dict(target=r['target'],factor=k,physical_3dB_width_ratio=float(physical_width/width),arc80_ratio=r['arc80_ratio'],length_ratio=r['length_ratio'],cdf_L1=r['cdf_L1']))
    summary['physical_Q_widths']=qrows;dump(OUT/'INTERVENTION_CHECKS.json',summary)
    # Q changes: quantities that remain unchanged are as important as responses.
    fig,axes=plt.subplots(1,3,figsize=(13,3.7));cols=['#345d7e','#b26933','#327654']
    for tid,col in zip(['series_Q10','notch_Q20','notch_Q2000'],cols):
        factor=lambda r:items[r['id']]['meta'].get('factor',1.)
        rr=sorted([r for r in rows if r['target']==tid and r['family'] in ['Q','baseline'] and r['scope']=='core'],key=factor);x=[factor(r) for r in rr]
        for ax,key,title in zip(axes,['length_ratio','arc80_ratio','cdf_L1'],['Path length / target','80% motion span / target','Normalized path CDF discrepancy']):ax.plot(x,[r[key] for r in rr],'-o',label=tid,color=col,ms=4);ax.set_xscale('log',base=2);ax.set(xlabel='L multiplier = inverse C multiplier',ylabel=title)
    qr=[r for r in qrows if r['target']=='series_Q10'];axes[1].plot([r['factor'] for r in qr],[r['physical_3dB_width_ratio'] for r in qr],'k--',label='Series RLC: actual 3 dB width');axes[1].legend(fontsize=7);axes[0].set_yscale('log',base=2);axes[1].set_yscale('log',base=2);axes[0].legend(fontsize=8);fig.suptitle('Physical Q interventions: a motion quantile is not a universal valley width');fig.tight_layout();save(fig,'Q_interventions.png')
    # Real translation cannot be detected by derivative/arc descriptors alone.
    tid='mixed_resonators/base';target=Physical(targets[tid]['model']);case=next(r for r in records if r['target']==tid and r['family']=='offset' and r['meta']['relative_offset']==.5);c=Physical(items[case['id']]['model']);lo,hi=targets[tid]['core'];x=grid(c,target,lo,hi,256)
    fig=plt.figure(figsize=(12,3.8));ax1=fig.add_subplot(131);ax2=fig.add_subplot(132);ax3=fig.add_subplot(133)
    for model,name,color in [(target,'Target','black'),(c,'Added series R','#b26933')]:
        stat,profile=scope_stats(model,target,x,targets[tid]['scale']);ax1.plot(profile['z'].real,profile['z'].imag,label=name,color=color);ax2.plot((x-lo)/(hi-lo),profile['cdf'],label=name,color=color)
    ax1.set(xlabel='Re Z / ohm',ylabel='Im Z / ohm',title='Same path, shifted real impedance');ax1.legend(fontsize=8);ax2.set(xlabel='Fixed core coordinate',ylabel='Fraction of complex path length',title='Arc distributions overlap')
    for ti,col in zip(targets,['black','#888888','#345d7e','#b26933','#327654']):
        rr=sorted([r for r in rows if r['target']==ti and r['family']=='offset' and r['scope']=='core'],key=lambda r:items[r['id']]['meta']['relative_offset']);xx=[items[r['id']]['meta']['relative_offset'] for r in rr];ax3.plot(xx,[r['state_error'] for r in rr],'-o',ms=3,color=col,label=ti)
    ax3.set(xlabel='Added R / target center magnitude',ylabel='Fixed-anchor state error',title='State anchors retain the translation');fig.suptitle('Offset intervention: state and movement are different requirements');fig.tight_layout();save(fig,'offset_blindspot.png')
    # Choose the first retained source-order example whose sequence is unchanged.
    chosen=next((r for r in extra if r['sequence']=='valley'),None)
    if chosen is not None:
        item=items[chosen['id']];t=targets[chosen['target']];c=Physical(item['model']);ref=Physical(t['model']);lo,hi=t['core'];x=grid(c,ref,lo,hi,256);fig=plt.figure(figsize=(12,7.5));axes=[fig.add_subplot(221),fig.add_subplot(222),fig.add_subplot(223),fig.add_subplot(224,projection='3d')]
        for model,name,col in [(ref,'Target','black'),(c,'Added physical RLC branch','#b26933')]:
            q,p=scope_stats(model,ref,x,t['scale']);u=(x-lo)/(hi-lo);z=p['z'];axes[0].plot(z.real,z.imag,color=col,label=name);axes[1].plot(u,p['cdf'],color=col,label=name);axes[2].plot(u,20*np.log10(abs(z)),color=col,label=name);axes[3].plot(x,z.real,z.imag,color=col,label=name)
        axes[0].set(xlabel='Re Z / ohm',ylabel='Im Z / ohm');axes[0].legend(fontsize=8);axes[1].set(xlabel='Fixed core coordinate',ylabel='Fraction of complex path length');axes[2].set(xlabel='Fixed core coordinate',ylabel='Magnitude / dB ohm');axes[3].set(xlabel='log10 f / Hz',ylabel='Re Z / ohm',zlabel='Im Z / ohm');fig.suptitle(chosen['id']+': the single-valley sequence can remain unchanged after adding a branch');fig.tight_layout();save(fig,'extra_sequence_blindspot.png')
        dump(OUT/'REPRESENTATIVE_EXTRA.json',dict(scope='Posthoc illustration: first fixed source-order unchanged sequence; not selection for scoring',item=item,diagnostics=chosen))
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    for tid,col in zip(['series_Q10','notch_Q20','mixed_resonators/base'],['#345d7e','#b26933','#327654']):
        for scope,style in [('core','-'),('window','--')]:
            shift=lambda r:items[r['id']]['meta'].get('shift_widths',0.)
            rr=sorted([r for r in rows if r['target']==tid and r['family'] in ['shift','baseline'] and r['scope']==scope],key=shift);x=[shift(r) for r in rr]
            ax[0].plot(x,[r['cdf_L1'] for r in rr],style+'o',color=col,ms=3,label=tid+' '+scope);ax[1].plot(x,[r['state_error'] for r in rr],style+'o',color=col,ms=3)
    ax[0].set(xlabel='Exact frequency shift / target core width',ylabel='Normalized path CDF discrepancy');ax[1].set(xlabel='Exact frequency shift / target core width',ylabel='Fixed-anchor state error');ax[0].legend(fontsize=7,ncol=2);fig.suptitle('Global frequency translations viewed through two fixed intervals');fig.tight_layout();save(fig,'shift_and_window.png')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['background','extra_sequence_counts','cdf_shift_reversals','physical_Q_widths']}),flush=True);print('Q x8:',json.dumps([r for r in qrows if r['factor']==8]),flush=True)
if __name__=='__main__':main()

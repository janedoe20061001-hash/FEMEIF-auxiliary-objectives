"""Separate control truth checks from the observable matching routine."""
from correspondence import *
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'output/valley_attribution_20261017';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2,'savefig.dpi':160})
def save(fig,name):fig.savefig(FIG/name,bbox_inches='tight');plt.close(fig)
def csvread(path):return list(csv.DictReader(path.open(encoding='utf-8-sig')))

def main():
    plan=json.loads((OUT/'PLAN.json').read_text(encoding='utf-8'));targets={t['id']:t for t in plan['targets']};items={t['id']:t for t in plan['tasks']};records=[json.loads((OUT/'records'/(item['id']+'.json')).read_text(encoding='utf-8')) for item in plan['tasks']];rows=[];candidate_rows=[];control_checks=[]
    oldinfo={q['id']:q for q in csvread(ROOT/'output/valley_process_20261017/COHORT_TABLE.csv')};physicalinfo={q['id']:q for q in csvread(ROOT/'output/valley_perturbation_20261017/ALL_DIAGNOSTICS.csv') if q['scope']=='core'}
    oldraw={q['id']:q for q in json.loads((ROOT/'output/valley_process_20261017/DIAGNOSTICS.json').read_text(encoding='utf-8'))['records'] if q['source'] is not None}
    for r in records:
        item=items[r['id']];q=r['levels'][-1];src=oldinfo.get(item.get('source_id')) if r['cohort']=='old_candidate' else physicalinfo.get(item.get('source_id')) if r['cohort']=='physical_regression' else None
        row=dict(id=r['id'],cohort=r['cohort'],target=r['target'],core_occupancy=q['occupancy'],core_candidates=q['core_valley_count'],boundary_candidates=q['boundary_valley_count'],search_candidates=q['search_valley_count'],signature_consistent=q['signature_consistent_count'],signature_relation=q['signature_relation'],signature_center_location=q['signature_center_location'],old_V3_complete=src.get('old_complete') if src else None,prior_anchor_state_error=src.get('state_error') if src else None,prior_core_length_ratio=src.get('core_length_ratio',src.get('length_ratio')) if src else None,numerical_passed=r['numerical_passed'])
        if r['cohort']=='old_candidate':
            raw=oldraw[item['source_id']];row['prior_anchor_state_error']=float(np.mean([x['complex_error_ohm'] for x in raw['anchors']])/targets[r['target']]['scale']);row['prior_core_peak_count']=sum(x['kind']=='peak' for x in raw['levels'][-1]['core']['radial_extrema'])
        else:row['prior_core_peak_count']=None
        rows.append(row)
        for c in q['candidates']:
            candidate_rows.append(dict(id=r['id'],cohort=r['cohort'],target=r['target'],candidate_index=c['index'],x=c['root']['x'],frequency_hz=c['root']['frequency_hz'],location=c['location'],offset_core_widths=c['frequency_offset_core_widths'],real_ohm=c['root']['real_ohm'],imag_ohm=c['root']['imag_ohm'],signature_state_max=c['max_normalized_state_residual'],signature_derivative_max=c['max_normalized_derivative_residual'],signature_consistent=c['translation_signature_consistent']))
        if r['cohort']=='new_control':
            expected=plan['expected_new_controls'][r['id']];actual=q['signature_center_x'];position_error=abs(actual-expected['center_x']) if actual is not None else None
            if expected['kind']=='boundary':
                independent=('left_boundary' if expected['boundary']=='left' else 'right_boundary') if expected['epsilon']==0 else 'in_core' if (expected['boundary']=='left' and expected['epsilon']>0) or (expected['boundary']=='right' and expected['epsilon']<0) else 'outside_core'
            else:
                t=targets[r['target']];assert min(abs(expected['center_x']-bound) for bound in t['core'])>POSITION_TOL
                independent='in_core' if t['core'][0]<expected['center_x']<t['core'][1] else 'outside_core'
            assert independent==expected['location']
            passed=r['numerical_passed'] and q['signature_consistent_count']==1 and position_error<=POSITION_TOL and q['signature_center_location']==independent
            control_checks.append(dict(id=r['id'],target=r['target'],kind=expected['kind'],expected_x=expected['center_x'],observed_x=actual,position_error_decades=position_error,expected_location=independent,observed_location=q['signature_center_location'],consistent_candidates=q['signature_consistent_count'],passed=bool(passed)))
    csvout(OUT/'ATTRIBUTION_TABLE.csv',rows);csvout(OUT/'ALL_VALLEY_CANDIDATES.csv',candidate_rows);csvout(OUT/'CONTROL_CHECKS.csv',control_checks)
    groups={cohort:dict(records=sum(r['cohort']==cohort for r in rows),occupancy=dict(Counter(r['core_occupancy'] for r in rows if r['cohort']==cohort)),relations=dict(Counter(r['signature_relation'] for r in rows if r['cohort']==cohort))) for cohort in ['new_control','physical_regression','old_candidate']}
    background=[r for r in rows if r['cohort']=='physical_regression' and items[r['id']]['source_family']=='background'];old=[r for r in rows if r['cohort']=='old_candidate'];focus=[r for r in old if any(r['id'].startswith('old_'+pid+'_') for pid in ['held_000','held_003','held_009','held_010','held_011'])]
    summary=dict(total_records=len(rows),numerical_passed=sum(r['numerical_passed'] for r in rows),new_controls=len(control_checks),new_controls_passed=sum(r['passed'] for r in control_checks),control_max_position_error=max((r['position_error_decades'] for r in control_checks if r['position_error_decades'] is not None),default=None),control_expected_locations=dict(Counter(r['expected_location'] for r in control_checks)),groups=groups,background=background,focus_five_pairs=focus,strict_translation_consistency_is_not_general_recovery=True,old_labels_unchanged=True,optimization_started=False)
    dump(OUT/'ANALYSIS.json',summary)
    # Known control metadata appears only in this verification step.
    fig,axes=plt.subplots(1,2,figsize=(11,4));colors=['#426580','#6c8a3d','#b26933','#815a94','#3d8d89']
    for tid,col in zip(targets,colors):
        rr=[r for r in control_checks if r['target']==tid and r['observed_x'] is not None];t=targets[tid];w=t['core'][1]-t['core'][0];xx=[(r['expected_x']-t['center'])/w for r in rr];yy=[(r['observed_x']-t['center'])/w for r in rr];axes[0].scatter(xx,yy,s=18,color=col,label=tid);axes[1].scatter(range(len(rr)),[r['position_error_decades'] for r in rr],color=col,s=18)
    axes[0].plot([-2,2],[-2,2],'k--',lw=1);axes[0].set(xlabel='Known displacement / target core width',ylabel='Signature-matched displacement / target core width');axes[0].legend(fontsize=7);axes[1].axhline(POSITION_TOL,color='#b74747',ls='--',label='Numerical position tolerance');axes[1].set_yscale('symlog',linthresh=1e-15);axes[1].set(xlabel='Control index within target',ylabel='Absolute log-frequency position error');axes[1].legend(fontsize=8);fig.suptitle('Fresh translations and boundary checks: algorithm does not read displacement');fig.tight_layout();save(fig,'control_verification.png')
    # All old roots remain visible. No nearest-root forced assignment.
    fig,ax=plt.subplots(figsize=(11,8));target_order=[r['id'] for r in old]
    for j,r in enumerate(old):
        t=targets[r['target']];w=t['core'][1]-t['core'][0];center=t['center'];left=(t['core'][0]-center)/w;right=(t['core'][1]-center)/w
        ax.plot([left,right],[j,j],color='#dddddd',lw=8,zorder=1)
        cc=[c for c in candidate_rows if c['id']==r['id']]
        if cc:ax.scatter([c['offset_core_widths'] for c in cc],[j]*len(cc),c=['#26744b' if c['location']=='in_core' else '#b26933' if c['location'].endswith('boundary') else '#687582' for c in cc],s=25,zorder=2)
    ax.axvline(0,color='black',ls='--',lw=.8);ax.set_yticks(range(len(old)),[r['id'].replace('old_held_','').replace('_better','+').replace('_worse','-') for r in old],fontsize=8);ax.invert_yaxis();ax.set(xlim=(-4.05,4.05),xlabel='Valley position relative to target / target core width',title='All 32 old candidates: every valley in the fixed search interval');fig.tight_layout();save(fig,'old_candidates_positions.png')
    # Representative conflict: source pair 003, positive has extra core process; negative is outside.
    tid='mixed_resonators/base';t=targets[tid];ref=Physical(t['model']);left,right=t['core'];lo,hi=t['center']-1.5*(right-left),t['center']+1.5*(right-left);fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ident,name,col in [(None,'Target','black'),('old_held_003_better','Original positive side','#26744b'),('old_held_003_worse','Original negative side','#b74747')]:
        c=ref if ident is None else Physical(items[ident]['model']);x=grid(c,ref,lo,hi,128);z,_=rx(c,x);u=(x-t['center'])/(right-left);axes[0].plot(u,20*np.log10(abs(z)),color=col,label=name)
        rr=next((r for r in records if r['id']==ident),None)
        if rr:
            cc=rr['levels'][-1]['candidates'];axes[1].scatter([v['frequency_offset_core_widths'] for v in cc],[name]*len(cc),color=col)
    axes[0].axvspan((left-t['center'])/(right-left),(right-t['center'])/(right-left),color='#dddddd',alpha=.4);axes[0].set(xlabel='Frequency offset / target core width',ylabel='Magnitude / dB ohm',title='Target core shaded; source labels kept');axes[0].legend(fontsize=8);axes[1].axvspan((left-t['center'])/(right-left),(right-t['center'])/(right-left),color='#dddddd',alpha=.4);axes[1].axvline(0,color='black',ls='--');axes[1].set(xlim=(-1.5,1.5),xlabel='Valley-root offset / target core width',title='Occupancy is not shape recovery');fig.tight_layout();save(fig,'pair003_occupancy.png')
    print(json.dumps(summary,default=lambda v:v.tolist()),flush=True)
if __name__=='__main__':main()

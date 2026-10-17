"""Analyze frozen observable sections without tuning thresholds or old labels."""
from shape import *
from collections import Counter
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    plan=load(OUT/'PLAN.json');targets={t['id']:t for t in plan['targets']};items={v['id']:v for v in plan['tasks']};rr=[load(OUT/'records'/(i['id']+'.json')) for i in plan['tasks']];refs=load(OUT/'TARGET_SECTIONS.json');rows=[];summaryrows=[];controls=[]
    for r in rr:
        qs=r['levels'][-1]['candidates'];bounded=[q for q in qs if q['status']=='bounded'];distinct=[]
        for q in bounded:
            if not any(max(abs(np.array(q['bounds'])-other))<=1e-9 for other in distinct):distinct.append(q['bounds'])
        summaryrows.append(dict(id=r['id'],target=r['target'],family=r['family'],cohort=r['cohort'],candidate_count=len(qs),bounded_count=len(bounded),distinct_bounded_sections=len(distinct),isolated_sections=sum(q.get('isolated_valley',False) for q in bounded),no_valley=not qs,numerical_passed=r['numerical_passed']))
        for i,q in enumerate(qs):rows.append(dict(id=r['id'],target=r['target'],family=r['family'],cohort=r['cohort'],candidate_index=i,status=q['status'],sequence=','.join(q.get('sequence',[])),**q.get('diagnostics',{})))
        if r.get('control'):controls.append(dict(id=r['id'],target=r['target'],family=r['family'],**r['control']))
    csvout(OUT/'CANDIDATES.csv',rows);csvout(OUT/'RECORDS.csv',summaryrows);csvout(OUT/'CONTROL_CHECKS.csv',controls)
    identities=[r for r in rr if r.get('identity') and r['identity'].get('applicable')];new=[r for r in rr if r['cohort']=='new'];extra=[r for r in summaryrows if r['family']=='extra'];old=[r for r in summaryrows if r['cohort']=='old'];offset=[r for r in summaryrows if r['family']=='offset'];boundedextra=[r for r in rows if r['family']=='extra' and r['status']=='bounded']
    a=dict(records=len(rr),numerical_passed=sum(r['numerical_passed'] for r in rr),controls=len(controls),controls_passed=sum(c['passed'] for c in controls),max_control_G=max(c['G'] for c in controls),max_control_length_ratio_error=max(abs(c['length_ratio']-1) for c in controls),max_control_width_ratio_error=max(c['width_ratio_error'] for c in controls),identities=len(identities),identities_passed=sum(r['identity']['passed'] for r in identities),nodal_records=sum(r.get('nodal') is not None for r in rr),nodal_modes=dict(Counter(r['nodal']['mode'] for r in new if r.get('nodal'))),extra_records=len(extra),extra_records_with_isolated_section=sum(r['isolated_sections']>0 for r in extra),extra_records_with_nonisolated_section=sum(any(q['status']=='bounded' and not q['isolated_valley'] for q in load(OUT/'records'/(r['id']+'.json'))['levels'][-1]['candidates']) for r in extra),extra_records_multiple_distinct_sections=sum(r['distinct_bounded_sections']>1 for r in extra),extra_min_G=min(r['G'] for r in boundedextra),extra_min_G_record=min(boundedextra,key=lambda r:r['G']),offset_records=len(offset),offset_records_bounded=sum(r['bounded_count']>0 for r in offset),old_records=len(old),old_bounded_records=sum(r['bounded_count']>0 for r in old),old_no_valley_records=sum(r['no_valley'] for r in old),old_unbounded_records=sum(not r['no_valley'] and r['bounded_count']==0 for r in old),old_bounded_G_range=[min(r['G'] for r in rows if r['cohort']=='old' and r['status']=='bounded'),max(r['G'] for r in rows if r['cohort']=='old' and r['status']=='bounded')],old_status_counts=dict(Counter(r['status'] for r in rows if r['cohort']=='old')),levels=dict(Counter(str([l['n'] for l in r['levels']]) for r in rr)),old_labels_unchanged=True,structural_acceptance_threshold_defined=False,optimization_started=False)
    dump(OUT/'ANALYSIS.json',a)
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2,'savefig.dpi':160})
    def save(fig,name):fig.tight_layout();fig.savefig(figdir/name,bbox_inches='tight');plt.close(fig)
    tid='notch_Q20';t=targets[tid];ref=refs[tid];W=t['core'][1]-t['core'][0];fig,ax=plt.subplots(1,2,figsize=(11,4));curves=[('Target',ref,'black','-')]
    for k,col in [(.3,'#426580'),(3.3,'#b26933')]:
        item=next(i for i in plan['tasks'] if i['target']==tid and i['family']=='Q' and i['meta']['factor']==k);r=next(r for r in rr if r['id']==item['id']);curves.append((f'Q factor {k}',r['levels'][-1]['candidates'][0],col,'--' if k==.3 else ':'))
    for label,q,col,ls in curves:
        ax[0].plot(np.array(q['profile']['real'])/t['scale'],np.array(q['profile']['imag'])/t['scale'],color=col,ls=ls,lw=2,label=label)
        ax[1].plot(np.linspace(0,1,257),(np.array(q['profile']['x'])-q['center'])/W,color=col,ls=ls,lw=2,label=label)
    ax[0].set(xlabel='Re Z / target S',ylabel='Im Z / target S',title='Same ordered complex path');ax[0].legend();ax[1].set(xlabel='Normalized cumulative arc length u',ylabel='Centered log-frequency / target W',title='Different frequency development');save(fig,'Q_geometry_and_frequency.png')
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for fam,col in [('shift','#426580'),('Q','#6c8a3d'),('offset','#b26933'),('extra','#815a94')]:
        subset=[r for r in rows if r['family']==fam and r['status']=='bounded'];ax[0].scatter([r['G'] for r in subset],[r['P'] for r in subset],s=20,alpha=.6,label=fam,color=col);ax[1].scatter([r['G'] for r in subset],[r['width_ratio'] for r in subset],s=20,alpha=.6,label=fam,color=col)
    ax[0].set(xlabel='Ordered complex-state difference G',ylabel='Centered frequency difference P',title='Components describe different changes');ax[0].legend();ax[1].set(xlabel='Ordered complex-state difference G',ylabel='Absolute-radius section width ratio',title='All bounded candidates; no best-root selection');ax[1].set_yscale('log');save(fig,'components_by_family.png')
    tid='mixed_resonators/base';t=targets[tid];W=t['core'][1]-t['core'][0];fig,ax=plt.subplots(1,2,figsize=(12,4));item=items['old_held_003_better'];r=next(r for r in rr if r['id']==item['id']);q=r['levels'][-1]['candidates'][0]
    for model,label,col in [(t['model'],'Target','black'),(item['model'],'Old 003 original positive','#26744b')]:
        c=Physical(model);xp=grid(c,Physical(t['model']),t['center']-1.1*W,t['center']+1.1*W,256);z=rx(c,xp)[0];ax[0].plot((xp-t['center'])/W,20*np.log10(abs(z)),label=label,color=col)
    ax[0].axhline(20*np.log10(t['scale'])+3,color='#815a94',ls='--',label='Fixed target radius');ax[0].axvspan((q['bounds'][0]-t['center'])/W,(q['bounds'][1]-t['center'])/W,color='#b26933',alpha=.5,label='Candidate selected section');ax[0].set(xlabel='Frequency offset / target W',ylabel='Magnitude / dB ohm',title='A bounded section can be extremely narrow');ax[0].legend(fontsize=8)
    for v,label,col in [(refs[tid],'Target','black'),(q,'Candidate bounded section','#26744b')]:ax[1].plot(np.array(v['profile']['real'])/t['scale'],np.array(v['profile']['imag'])/t['scale'],label=label,color=col)
    ax[1].set(xlabel='Re Z / target S',ylabel='Im Z / target S',title='Endpoint definition does not establish mode identity');ax[1].legend(fontsize=8);save(fig,'old003_section_limit.png')
    # A single observed valley does not exclude an added physical branch.
    selected=a['extra_min_G_record'];t=targets[selected['target']];r=next(v for v in rr if v['id']==selected['id']);q=r['levels'][-1]['candidates'][selected['candidate_index']];fig,ax=plt.subplots(1,2,figsize=(11,4))
    for v,label,col in [(refs[t['id']],'Target','black'),(q,'Added branch: still one radial valley','#815a94')]:
        ax[0].plot(np.array(v['profile']['real'])/t['scale'],np.array(v['profile']['imag'])/t['scale'],color=col,label=label)
        ax[1].plot(np.linspace(0,1,257),(np.array(v['profile']['x'])-v['center'])/(t['core'][1]-t['core'][0]),color=col,label=label)
    ax[0].set(xlabel='Re Z / target S',ylabel='Im Z / target S',title=f"{selected['id']}: nonzero path change despite one valley");ax[0].legend(fontsize=8);ax[1].set(xlabel='Normalized cumulative arc length u',ylabel='Centered log-frequency / target W',title=f"G={selected['G']:.4f}, P={selected['P']:.4f}; no acceptance threshold");save(fig,'extra_single_valley.png')
    print(json.dumps(a,ensure_ascii=False),flush=True)
if __name__=='__main__':main()

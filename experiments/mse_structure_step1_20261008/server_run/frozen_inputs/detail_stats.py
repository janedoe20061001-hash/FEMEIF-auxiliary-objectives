from pathlib import Path
import sys,json
import numpy as np
from summarize import read,OUT,RAW
def main():
    ee=read('stage_events.csv');fe=read('front_events.csv');stage=read('stage_diagnostics.csv');bands=read('stage_bands.csv');out=[]
    for arm in ['original','adaptive']:
        for g in [300,1000]:
            for rule in ['min_wsmse','balanced_minmax']:
                rr=[r for r in ee if r['arm']==arm and r['generation']==g and r['rule']==rule];p={r['seed']:r for r in rr if r['event_id']=='P3'};v={r['seed']:r for r in rr if r['event_id']=='V2'}
                rec=dict(arm=arm,generation=g,rule=rule,p3_and_v2_both_missing=sum(not p[s]['matched'] and not v[s]['matched'] for s in p),p3_v2_missing_status_disagrees=sum(p[s]['matched']!=v[s]['matched'] for s in p),p3_and_v2_both_complete=sum(p[s]['complete'] and v[s]['complete'] for s in p))
                candidates=[r for r in fe if r['arm']==arm and r['generation']==g and r['event_id'] in ['P3','V2']];groups={}
                for r in candidates:groups.setdefault((r['seed'],r['index']),{})[r['event_id']]=r['complete']
                rec['front_any_single_curve_both_p3_v2_complete']=len({s for (s,i),d in groups.items() if d['P3'] and d['V2']})
                sr=[r for r in stage if r['arm']==arm and r['generation']==g and r['rule']==rule];rec['seeds_any_amplitude_incomplete']=sum(r['amp_incomplete']>0 for r in sr);rec['seeds_any_amplitude_missing']=sum(r['amp_missed']>0 for r in sr)
                rec['events']={}
                for event in ['P1','P3','P4','P5','V2','V3','V4']:
                    er=[r for r in rr if r['event_id']==event];rec['events'][event]=dict(width_more_than_twice=sum(r['width_log2_ratio'] is not None and r['width_log2_ratio']>1 for r in er),width_less_than_half=sum(r['width_log2_ratio'] is not None and r['width_log2_ratio']<-1 for r in er),matched_width_unavailable=sum(r['matched'] and r['width_log2_ratio'] is None for r in er),height_more_than_3db_low=sum(r['height_delta_db'] is not None and r['height_delta_db']<-3 for r in er),height_more_than_3db_high=sum(r['height_delta_db'] is not None and r['height_delta_db']>3 for r in er))
                out.append(rec)
    (OUT/'DETAILS.json').write_text(json.dumps(out,indent=2),encoding='utf8')
    runs=json.loads((RAW/'analysis/RUN_SUMMARIES.json').read_text());cost=[]
    for arm in ['original','adaptive']:
        r=[v for v in runs if v['arm']==arm];q=sum(v['model_frequency_queries'] for v in r);cost.append(dict(arm=arm,runs=len(r),candidate_evaluations=sum(v['evaluations'] for v in r),model_frequency_queries=q,mean_frequency_queries_per_candidate=q/sum(v['evaluations'] for v in r),median_run_seconds=float(np.median([v['seconds'] for v in r])),mean_run_seconds=float(np.mean([v['seconds'] for v in r])),maximum_points=max(v['max_adaptive_points'] for v in r),maximum_layers=max(v['max_adaptive_layers'] for v in r)))
    (OUT/'COST.json').write_text(json.dumps(cost,indent=2),encoding='utf8');print(json.dumps({'details':[r for r in out if r['arm']=='adaptive' and r['generation']==1000],'cost':cost},ensure_ascii=False))
if __name__=='__main__':main()

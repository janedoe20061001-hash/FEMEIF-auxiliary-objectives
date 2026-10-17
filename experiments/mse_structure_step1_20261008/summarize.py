"""Summaries of frozen classifications, without changing any selected solution."""
from pathlib import Path
import csv,json,sys,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
RAW=HERE/'server_run'/'retrieved';OUT=HERE.parents[1]/'output'/'mse_structure_step1_20261008'
CONDS=[('original',300),('adaptive',300),('original',1000),('adaptive',1000)]
RULES=['min_wsmse','balanced_minmax']
METRICS=['amplitude_rmse_db','phase_rmse_deg','amp_missed','amp_incomplete','amp_position_bad','amp_height_bad','amp_3db_width_bad','phase_bad_bands','worst_band_phase_rmse_deg','amp_missed_prom_0p5','amp_missed_prom_2p0']
def read(name):
    with (RAW/'analysis'/name).open(encoding='utf-8-sig',newline='') as f:
        result=[]
        for r in csv.DictReader(f):
            for k,v in r.items():
                if v=='True':r[k]=True
                elif v=='False':r[k]=False
                elif not v:r[k]=None
                else:
                    try:r[k]=float(v)
                    except ValueError:pass
            result.append(r)
        return result
def write(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def dump(name,value):(OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def filt(rows,arm,g,rule=None):return [r for r in rows if r['arm']==arm and r['generation']==g and (rule is None or r['rule']==rule)]
def main():
    OUT.mkdir(exist_ok=True,parents=True)
    stage=read('stage_diagnostics.csv');ee=read('stage_events.csv');bands=read('stage_bands.csv');front=read('front_diagnostics.csv');fe=read('front_events.csv');fb=read('front_bands.csv')
    validation=json.loads((RAW/'analysis'/'VALIDATION.json').read_text());assert validation['runs']==200 and validation['representative_rows']==20000 and validation['full_population_individuals_recomputed']==80000
    overall=[];rng=np.random.default_rng(2026100802);boot=rng.integers(0,100,(10000,100));paired=[]
    for rule in RULES:
        values={}
        for arm,g in CONDS:
            rr=sorted(filt(stage,arm,g,rule),key=lambda r:r['seed']);assert len(rr)==100 and [r['seed'] for r in rr]==list(range(2300,2400))
            values[arm,g]=rr
            for k in METRICS:
                x=np.array([r[k] for r in rr]);overall.append(dict(arm=arm,generation=g,rule=rule,metric=k,mean=float(x.mean()),median=float(np.median(x)),q25=float(np.quantile(x,.25)),q75=float(np.quantile(x,.75)),seeds_nonzero=int((x>0).sum())))
        for before,after in [(('original',300),('adaptive',300)),(('original',1000),('adaptive',1000)),(('original',300),('original',1000)),(('adaptive',300),('adaptive',1000))]:
            for k in METRICS:
                a=np.array([r[k] for r in values[before]]);b=np.array([r[k] for r in values[after]]);delta=b-a;ci=np.quantile(delta[boot].mean(axis=1),[.025,.975]);paired.append(dict(rule=rule,before_arm=before[0],before_generation=before[1],after_arm=after[0],after_generation=after[1],metric=k,mean_change=float(delta.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),improved_seeds=int((delta<0).sum()),equal_seeds=int((delta==0).sum()),worse_seeds=int((delta>0).sum())))
    event_summary=[];front_summary=[];tradeoffs=[]
    event_ids=[r['event_id'] for r in json.loads((RAW/'analysis'/'TARGET_EVENTS.json').read_text())]
    for arm,g in CONDS:
        fer=filt(fe,arm,g);frr=filt(front,arm,g);byfront={(int(r['seed']),int(r['index'])):r for r in frr}
        for rule in RULES:
            er=filt(ee,arm,g,rule);rep={int(r['seed']):r for r in filt(stage,arm,g,rule)}
            for event in event_ids:
                rr=[r for r in er if r['event_id']==event];fr=[r for r in fer if r['event_id']==event];assert len(rr)==100
                rec=dict(arm=arm,generation=g,rule=rule,event_id=event,frequency_hz=rr[0]['target_frequency_hz'],target_width_3db_decades=rr[0]['target_width_3db_decades'],missing_seeds=sum(not r['matched'] for r in rr),complete_seeds=sum(r['complete'] for r in rr),position_bad_seeds=sum(r['position_bad'] for r in rr),height_bad_seeds=sum(r['height_bad'] for r in rr),width_bad_seeds=sum(r['width_bad'] for r in rr))
                event_summary.append(rec);anymatched={int(r['seed']) for r in fr if r['matched']};anycomplete={int(r['seed']) for r in fr if r['complete']};failed={int(r['seed']) for r in rr if not r['complete']}
                front_summary.append(dict(**rec,front_any_matched_seeds=len(anymatched),front_any_complete_seeds=len(anycomplete),representative_incomplete_but_front_complete=len(failed&anycomplete)))
                for seed in sorted(failed&anycomplete):
                    candidates=[byfront[seed,int(r['index'])] for r in fr if int(r['seed'])==seed and r['complete']]
                    best=min(candidates,key=lambda r:r['train_mag']+25.12*r['train_phase']);r=rep[seed]
                    tradeoffs.append(dict(arm=arm,generation=g,rule=rule,event_id=event,seed=seed,diagnostic_index=int(best['index']),representative_train_mag=r['train_mag'],representative_train_phase=r['train_phase'],alternative_train_mag=best['train_mag'],alternative_train_phase=best['train_phase'],representative_amp_rmse=r['amplitude_rmse_db'],representative_phase_rmse=r['phase_rmse_deg'],alternative_amp_rmse=best['amplitude_rmse_db'],alternative_phase_rmse=best['phase_rmse_deg']))
    band_summary=[];frontjoint=[]
    for arm,g in CONDS:
        br=filt(bands,arm,g);fbr=filt(fb,arm,g);fr=filt(front,arm,g);fer=filt(fe,arm,g)
        for rule in RULES:
            for band in range(1,9):
                rr=[r for r in br if r['rule']==rule and r['band']==band];anygood={int(r['seed']) for r in fbr if r['band']==band and r['phase_rmse_deg']<=15};bad={int(r['seed']) for r in rr if r['phase_rmse_deg']>15}
                band_summary.append(dict(arm=arm,generation=g,rule=rule,band=band,lo_hz=rr[0]['lo_hz'],hi_hz=rr[0]['hi_hz'],phase_bad_seeds=len(bad),mean_phase_rmse_deg=float(np.mean([r['phase_rmse_deg'] for r in rr])),front_any_good_seeds=len(anygood),representative_bad_but_front_good=len(bad&anygood)))
        erby={};brby={}
        for r in fer:erby.setdefault((int(r['seed']),int(r['index'])),[]).append(r)
        for r in fbr:brby.setdefault((int(r['seed']),int(r['index'])),[]).append(r)
        ampall=set();phaall=set();both=set()
        for r in fr:
            key=int(r['seed']),int(r['index']);a=all(v['complete'] for v in erby[key]);b=all(v['phase_rmse_deg']<=15 for v in brby[key]);assert len(erby[key])==11 and len(brby[key])==8
            if a:ampall.add(key[0])
            if b:phaall.add(key[0])
            if a and b:both.add(key[0])
        frontjoint.append(dict(arm=arm,generation=g,seeds_any_one_curve_all_amplitude_events_complete=len(ampall),seeds_any_one_curve_all_phase_bands_good=len(phaall),seeds_any_one_curve_both=len(both),front_rows=len(fr)))
    write('OVERALL.csv',overall);write('PAIRED.csv',paired);write('EVENTS.csv',event_summary);write('FRONT_EVENTS.csv',front_summary);write('FRONT_TRADEOFFS.csv',tradeoffs);write('BANDS.csv',band_summary);write('FRONT_JOINT.csv',frontjoint)
    trajectory=[]
    for arm in ['original','adaptive']:
        for rule in RULES:
            for g in range(20,1001,20):
                rr=filt(stage,arm,g,rule);trajectory.append(dict(arm=arm,generation=g,rule=rule,**{k:float(np.mean([r[k] for r in rr])) for k in METRICS}))
    write('TRAJECTORY.csv',trajectory)
    dump('SUMMARY.json',dict(validation=validation,overall=overall,paired=paired,events=event_summary,front_events=front_summary,bands=band_summary,front_joint=frontjoint))
    print(json.dumps(dict(overall=overall,events=event_summary,front_joint=frontjoint),ensure_ascii=False))
if __name__=='__main__':main()

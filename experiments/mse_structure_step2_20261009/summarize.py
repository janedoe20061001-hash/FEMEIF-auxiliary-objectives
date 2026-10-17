from common import *
def csvread(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def average(rows,key):return float(np.mean([float(r[key]) for r in rows if r[key] not in ['',None]]))
def paired_interval(values):
    v=np.asarray(values,float);rng=np.random.default_rng(2026100902);boot=v[rng.integers(0,len(v),(10000,len(v)))].mean(axis=1)
    return dict(mean=float(v.mean()),lower=float(np.quantile(boot,.025)),upper=float(np.quantile(boot,.975)),n=len(v))
def wilson(k,n):
    z=1.959963984540054;p=k/n;d=1+z*z/n;center=(p+z*z/(2*n))/d;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(center-half),float(center+half)]
def main():
    cfg=json.loads((HERE/'CONFIG.json').read_text());allrows=csvread(OUT/'analysis/REPRESENTATIVES.csv');events=csvread(OUT/'analysis/EVENTS.csv');regions=csvread(OUT/'analysis/REGIONAL_ERRORS.csv');front=csvread(OUT/'analysis/FRONT.csv');sensitivity=csvread(OUT/'analysis/THRESHOLD_SENSITIVITY.csv')
    summary=[];attainment=[]
    for s in cfg['run_scales']:
        for rule in ['min_wsmse','balanced_minmax']:
            rr=[r for r in allrows if float(r['scale'])==s and int(r['generation'])==300 and r['rule']==rule]
            ee=[e for e in events if float(e['scale'])==s and int(e['generation'])==300 and e['rule']==rule]
            cc={e:[r for r in ee if r['event_id']==e] for e in ['P3','V2']}
            for seed in cfg['seeds']:
                stages=[r for r in allrows if float(r['scale'])==s and r['rule']==rule and int(r['seed'])==seed]
                hit=[int(r['generation']) for r in stages if r['joint_complete']=='True'];final=next(r for r in stages if int(r['generation'])==300)
                attainment.append(dict(scale=s,seed=seed,rule=rule,first_saved_generation=min(hit) if hit else None,ever_complete=bool(hit),final_complete=final['joint_complete']=='True',lost_by_final=bool(hit) and final['joint_complete']!='True'))
            reg=[r for r in regions if float(r['scale'])==s and r['rule']==rule and int(r['generation'])==300 and r['window']=='broad' and r['weighting']=='common_logf']
            ff=[r for r in front if float(r['scale'])==s]
            count=sum(r['joint_complete']=='True' for r in rr)
            row=dict(scale=s,rule=rule,seeds=len(rr),joint_complete=count,joint_failure=len(rr)-count,joint_complete_wilson95=wilson(count,len(rr)),P3_complete=sum(r['P3_complete']=='True' for r in rr),V2_complete=sum(r['V2_complete']=='True' for r in rr),P3_missing=sum(r['P3_matched']!='True' for r in rr),V2_missing=sum(r['V2_matched']!='True' for r in rr),mean_train_mag=average(rr,'train_mag'),mean_train_phase=average(rr,'train_phase'),mean_amp_rmse_db=average(rr,'amplitude_rmse_db'),mean_phase_rmse_deg=average(rr,'phase_rmse_deg'),mean_broad_phase_rmse_deg=average(reg,'local_phase_rmse_deg'),front_any_joint_complete_seeds=len({int(r['seed']) for r in ff if r['joint_complete']=='True'}),front_any_P3_complete_seeds=len({int(r['seed']) for r in ff if r['P3_complete']=='True'}),ever_joint_complete=sum(r['ever_complete'] for r in attainment if r['scale']==s and r['rule']==rule))
            for e in ['P3','V2']:
                matched=[r for r in cc[e] if r['matched']=='True'];row[e+'_matched_n']=len(matched)
                for field in ['height_delta_db','shift_decades','width_log2_ratio']:row[e+'_mean_'+field]=average(matched,field) if matched and any(r[field] for r in matched) else None
                row[e+'_mean_absolute_height_delta_db']=float(np.mean([abs(float(r['height_delta_db'])) for r in matched])) if matched else None
            summary.append(row)
    paired=[]
    for rule in ['min_wsmse','balanced_minmax']:
        for s in [.25,8]:
            for key in ['joint_complete','P3_complete','V2_complete','P3_matched','V2_matched','amplitude_rmse_db','phase_rmse_deg','train_mag','train_phase']:
                diff=[]
                for seed in cfg['seeds']:
                    a=next(r for r in allrows if float(r['scale'])==s and int(r['seed'])==seed and int(r['generation'])==300 and r['rule']==rule);b=next(r for r in allrows if float(r['scale'])==1 and int(r['seed'])==seed and int(r['generation'])==300 and r['rule']==rule)
                    diff.append(int(a[key]=='True')-int(b[key]=='True') if key.endswith(('complete','matched')) else float(a[key])-float(b[key]))
                paired.append(dict(rule=rule,scale=s,baseline_scale=1,measure=key,**paired_interval(diff)))
    ss=[]
    for s in cfg['run_scales']:
        for rule in ['min_wsmse','balanced_minmax']:
            for lim in [.01,.02,.03]:
                r=[x for x in sensitivity if float(x['scale'])==s and x['rule']==rule and float(x['shift_limit'])==lim]
                ss.append(dict(scale=s,rule=rule,shift_limit=lim,joint_complete=sum(x['joint_complete']=='True' for x in r),P3_complete=sum(x['P3_complete']=='True' for x in r),V2_complete=sum(x['V2_complete']=='True' for x in r)))
    dump(OUT/'SUMMARY.json',summary);dump(OUT/'PAIRED.json',paired);csvout(OUT/'ATTAINMENT.csv',attainment);csvout(OUT/'SENSITIVITY_SUMMARY.csv',ss)
    rec=json.loads((HERE/'batch/summaries.json').read_text());status=json.loads((HERE/'batch/status.json').read_text());dump(OUT/'COST.json',dict(wall_seconds=status['seconds'],summed_run_seconds=sum(r['seconds'] for r in rec),candidate_evaluations=sum(r['evaluations'] for r in rec),model_frequency_pair_queries=sum(r['model_frequency_queries'] for r in rec),max_points=max(r['max_adaptive_points'] for r in rec),max_layers=max(r['max_adaptive_layers'] for r in rec),workers=4))
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

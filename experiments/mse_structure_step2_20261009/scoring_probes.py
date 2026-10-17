"""Exploratory direct scoring, separate from the preregistered optimization contrasts."""
from common import *
import time
def main():
    cfg=json.loads((HERE/'CONFIG.json').read_text());names=['R2_half','R2_double','branch_width_half','branch_width_double','branch_center_0p1pct_up','branch_center_1pct_up','R2_to_lower_bound']
    dump(OUT/'SCORING_PROBE_PROTOCOL.json',dict(status='exploratory diagnostic added while pilot running; not a preregistered primary test',scales=cfg['target_scales'],changes=names,other_15_parameters='held at their true target values; oracle scoring diagnostic only',purpose='separate objective penalties from optimizer outcomes',network_confound='all network effects rechecked; height/width/position/valley can couple'))
    rows=[];start=time.perf_counter()
    for s in cfg['target_scales']:
        m=Target(s);cat=target_catalog(m);a=evaluator(m);p0=np.array(m.record['target_parameters']);pars=[]
        for name in names:
            p=p0.copy()
            if name.startswith('R2_'):p[3]=p[3]*({'R2_half':.5,'R2_double':2}.get(name,1/p[3]))
            elif name.startswith('branch_width'):
                k=.5 if name.endswith('half') else 2;p[4]*=k;p[5]/=k
            else:
                k=1.001 if '0p1' in name else 1.01;p[4]/=k;p[5]/=k
            assert np.all(np.log10(p)>=m.lower) and np.all(np.log10(p)<=m.upper);pars.append(p)
        F=a.score(np.log10(pars))
        for name,p,score in zip(names,pars,F):
            references=[a.reference(np.log10(p),improved_dense_grid(m.record,p[None,:],lev))[0] for lev in [2,4,8]]
            strict=np.maximum([1e-8,1e-10],.001*references[-1]);assert np.all(abs(references[1]-references[2])<=strict)
            tol=np.maximum([.01,np.deg2rad(.5)**2],.005*references[-1]);assert np.all(abs(score-references[-1])<=tol)
            diag,ee,_=diagnostic(m,p,cat);assert diag['grid_stable'];ev={e['event_id']:e for e in ee};row=dict(scale=s,probe=name,train_mag=float(score[0]),train_phase=float(score[1]),reference_mag=float(references[-1][0]),reference_phase=float(references[-1][1]),relative_mag_score_error=float(abs(score[0]-references[-1][0])/references[-1][0]),relative_phase_score_error=float(abs(score[1]-references[-1][1])/references[-1][1]),amplitude_rmse_db=diag['amplitude_rmse_db'],phase_rmse_deg=diag['phase_rmse_deg'],joint_complete=ev['P3']['complete'] and ev['V2']['complete'])
            for e in ['P3','V2']:
                for k in ['matched','height_delta_db','shift_decades','width_log2_ratio']:row[e+'_'+k]=ev[e][k]
            rows.append(row)
    csvout(OUT/'SCORING_PROBES.csv',rows);dump(OUT/'SCORING_PROBE_VALIDATION.json',dict(passed=True,rows=len(rows),seconds=time.perf_counter()-start,independent_integrals='levels 2,4,8; level4/8 relative convergence .1% with absolute floors MAG 1e-8,PHA 1e-10',all_parameters_within_bounds=True,max_relative_mag_score_error=max(r['relative_mag_score_error'] for r in rows),max_relative_phase_score_error=max(r['relative_phase_score_error'] for r in rows)))
    print(json.dumps([r for r in rows if r['scale'] in [.25,1,8] and r['probe'] in ['R2_half','R2_to_lower_bound']],ensure_ascii=False,indent=2))
if __name__=='__main__':main()

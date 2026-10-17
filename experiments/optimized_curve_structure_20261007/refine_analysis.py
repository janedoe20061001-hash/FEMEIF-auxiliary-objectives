"""Numerical grid repair only; reuse validated individuals and objectives."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,csv,json,shutil,time
import numpy as np
from model import Model
from diagnostics import diagnose
from refined_grid import improved_dense_grid
from analyze import csvout,dump,sha,summarize,GROUPS
from run_analysis import loadcsv
HERE=Path(__file__).resolve().parent

def refine_one(job):
    r,source,rows,start_index,end_index=job;m=Model(r['id']);new=[];bands=[];refinements=[];matches=[]
    with np.load(Path(source)/'selected'/f'{r["id"]}.npz') as p:x=p['log_parameters'];seeds=p['seeds'];sample_f=p['frequency'];sample_target=p['target'];sample_curves=p['curves']
    old={(int(a['seed']),a['group']):a for a in rows}
    for si in range(start_index,end_index):
        seed=seeds[si]
        grids=[improved_dense_grid(r,10**x[si],v) for v in [1,2]]
        targets=[m.network(f,np.asarray(r['target_parameters']))[0] for f in grids];zs=[m.network(f,10**x[si]) for f in grids]
        for gi,group in enumerate(GROUPS):
            row=dict(old[int(seed),group]);coarse,_,_=diagnose(grids[0],targets[0],zs[0][gi]);fine,bb,mm=diagnose(grids[1],targets[1],zs[1][gi])
            keys=['amplitude_rmse_db','phase_rmse_deg','mean_distance_logweighted_ohm','max_amplitude_error_db','max_phase_error_deg'];changes={k:abs(fine[k]-coarse[k]) for k in keys}
            dc=max(changes['amplitude_rmse_db']/max(fine['amplitude_rmse_db'],.1),changes['phase_rmse_deg']/max(fine['phase_rmse_deg'],.5),changes['mean_distance_logweighted_ohm']/max(fine['mean_distance_logweighted_ohm'],1e-8))
            ed=sum(abs(fine[k]-coarse[k]) for k in ['amp_missed','amp_extra','phase_missed','phase_extra','cross_missed','cross_extra'])
            stable=dc<=.02 and ed==0
            base={k:row[k] for k in ['circuit','family','condition','seed','group']}
            refinements.append(dict(**base,coarse_points=len(grids[0]),fine_points=len(grids[1]),max_relative_integral_change_with_floors=dc,event_count_discrepancy=ed,stable=stable,**changes))
            fine.update(dense_minus_sampled_amplitude_rmse_db=fine['amplitude_rmse_db']-(row['amplitude_rmse_db']-row['dense_minus_sampled_amplitude_rmse_db']),dense_minus_sampled_phase_rmse_deg=fine['phase_rmse_deg']-(row['phase_rmse_deg']-row['dense_minus_sampled_phase_rmse_deg']),dense_extra_fit_amp_events=fine['amp_fit_events']-(row['amp_fit_events']-row['dense_extra_fit_amp_events']),dense_extra_target_amp_events=fine['amp_target_events']-(row['amp_target_events']-row['dense_extra_target_amp_events']))
            row.update(fine);row['dense_points']=len(grids[1]);row['grid_stable']=stable;new.append(row);bands.extend(dict(**base,grid='analytic_dense',**b) for b in bb);matches.extend(dict(**base,**m) for m in mm)
            xx=np.log10(grids[1]);sx=np.log10(sample_f)
            ta=20*np.log10(abs(targets[1]));fa=20*np.log10(abs(zs[1][gi]))
            row['target_sampling_gap_max_db']=float(np.max(abs(ta-np.interp(xx,sx,20*np.log10(abs(sample_target))))))
            row['fit_sampling_gap_max_db']=float(np.max(abs(fa-np.interp(xx,sx,20*np.log10(abs(sample_curves[si,gi]))))))
    return r['id']+'_'+str(start_index),new,bands,refinements,matches

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,default=8);a=ap.parse_args()
    source=a.input.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);assert json.loads((source/'VALIDATION.json').read_text())['passed']
    records=json.loads((HERE/'circuits.json').read_text(encoding='utf8'));old=loadcsv(source/'per_seed_diagnostics.csv');rr=[r for r in records if r['target_parameters'] is not None]
    jobs=[(r,str(source),[v for v in old if v['circuit']==r['id']],i,min(i+10,r['seed_count'])) for r in rr for i in range(0,r['seed_count'],10)];rows=[r for r in old if r['circuit']=='case2'];bands=[r for r in loadcsv(source/'band_diagnostics.csv') if r['circuit']=='case2'];matches=[r for r in loadcsv(source/'feature_matches.csv') if r['circuit']=='case2'];refinements=[];start=time.time();done=0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(refine_one,j) for j in jobs]):
            ident,new,bb,rf,mm=f.result();rows+=new;bands+=bb;refinements+=rf;matches+=mm;done+=1
            dump(out/'status.json',dict(status='running',chunks_completed=done,chunks_total=len(jobs),analytic_circuits=len(rr),seconds=time.time()-start));print(json.dumps(dict(refined=done,chunk=ident,seconds=time.time()-start)),flush=True)
    order={r['id']:i for i,r in enumerate(records)}
    rows.sort(key=lambda r:(order[r['circuit']],int(r['seed']),GROUPS.index(r['group'])))
    assert len(rows)==4640
    means,pairs=summarize(rows)
    for name,v in [('per_seed_diagnostics',rows),('band_diagnostics',bands),('grid_refinement',refinements),('feature_matches',matches),('group_means',means),('paired_comparisons',pairs)]:csvout(out/(name+'.csv'),v)
    shutil.copytree(source/'selected',out/'selected');shutil.copy2(source/'dual_selection_sensitivity.csv',out/'dual_selection_sensitivity.csv')
    previous=json.loads((source/'SUMMARY.json').read_text());unstable=sum(not r['stable'] for r in refinements)
    previous.update(means=means,paired_comparisons=pairs,first_grid_unstable_rows=previous['grid_unstable_rows'],grid_unstable_rows=unstable,grid_refinement_rows=len(refinements),grid_method='unclipped rational modes + exact RLC centers + geometric tails');dump(out/'SUMMARY.json',previous)
    v=json.loads((source/'VALIDATION.json').read_text());v.update(grid_unstable_rows=unstable,first_grid_validation_sha256=sha(source/'VALIDATION.json'),grid_method=previous['grid_method'],refined_grid_sha256=sha(HERE/'refined_grid.py'),refinement_sha256=sha(__file__));dump(out/'VALIDATION.json',v)
    dump(out/'status.json',dict(status='complete',chunks_completed=done,chunks_total=len(jobs),analytic_circuits=len(rr),seconds=time.time()-start));print(json.dumps(dict(passed=True,rows=4640,unstable=unstable,seconds=time.time()-start)))
if __name__=='__main__':main()

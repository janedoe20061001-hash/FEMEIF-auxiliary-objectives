"""Paired scalar-only optimization; own-loss minimum selected in each final population."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import time
import traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from model import Model
HERE=Path(__file__).resolve().parent
COLS=['magnitude_mse_db2','phase_mse_rad2','wsmse','mean_distance_ohm']
GROUPS=['wsmse','mean_distance']
def dump(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Circuit(Problem):
    def __init__(self,m,col):
        self.m=m;self.col=col;super().__init__(n_var=len(m.lower),n_obj=1,xl=m.lower,xu=m.upper)
    def _evaluate(self,x,out,*a,**kw):out['F']=self.m.score(x)[:,self.col,None]
class Trace(Callback):
    def __init__(self,m,col):super().__init__();self.m=m;self.col=col;self.rows=[]
    def notify(self,alg):
        ix=int(np.argmin(alg.pop.get('F')[:,0]));metrics=self.m.score(alg.pop.get('X')[ix])[0]
        self.rows.append(dict(generation=int(alg.n_gen),evaluations=int(alg.evaluator.n_eval),selected_index=ix,
                             **{k:float(v) for k,v in zip(COLS,metrics)}))
def run_one(ident,seed,group,directory,smoke=False):
    m=Model(ident);r=m.record;pop=12 if smoke else r['population'];gen=3 if smoke else r['generations'];col=2 if group=='wsmse' else 3
    d=Path(directory);d.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    x0=np.random.default_rng(seed).uniform(m.lower,m.upper,(pop,len(m.lower)));np.save(d/'initial_log_parameters.npy',x0)
    alg=NSGA2(pop_size=pop,n_offsprings=pop,sampling=x0.copy(),crossover=SBX(prob=.9,eta=1),
        mutation=PM(prob=1.,prob_var=r['mutation_variable_probability'],eta=1),eliminate_duplicates=False)
    alg.tournament_type='comp_by_rank_and_crowding';trace=Trace(m,col)
    result=minimize(Circuit(m,col),alg,('n_gen',gen),seed=seed,callback=trace,verbose=False,save_history=False)
    x,F=result.pop.get('X'),result.pop.get('F');metrics=m.score(x);np.testing.assert_allclose(F[:,0],metrics[:,col],rtol=1e-12,atol=1e-10)
    ix=int(np.argmin(F[:,0]));np.savez_compressed(d/'final_population.npz',log_parameters=x,parameters=10**x,optimization_objective=F,
        cross_metrics=metrics,metric_names=np.asarray(COLS),selected_index=ix,selected_log_parameters=x[ix],selected_metrics=metrics[ix])
    with (d/'history.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(trace.rows[0]));w.writeheader();w.writerows(trace.rows)
    record=dict(status='complete',circuit=ident,seed=seed,group=group,n_obj=1,objective_column=col,population=pop,generations=gen,
        evaluations=int(result.algorithm.evaluator.n_eval),initial_population_sha256=hashlib.sha256(x0.tobytes()).hexdigest(),
        selected_index=ix,elapsed_seconds=time.perf_counter()-start,**{k:float(v) for k,v in zip(COLS,metrics[ix])})
    assert record['evaluations']==pop*gen and len(trace.rows)==gen
    dump(d/'summary.json',record);return record
def job(j):
    ident,seed,group,d=j
    try:return run_one(ident,seed,group,d)
    except Exception:
        error=dict(status='failed',circuit=ident,seed=seed,group=group,traceback=traceback.format_exc())
        Path(d).mkdir(parents=True,exist_ok=True);dump(Path(d)/'FAILED.json',error);return error
def smoke(out):
    out.mkdir(parents=True,exist_ok=False);checks=[]
    for ident in ['case1','case2','rc_series_c1','rlc_bridge_c2','three_parallel_units_c3']:
        for group in GROUPS:run_one(ident,777,group,out/ident/group,True)
        np.testing.assert_array_equal(np.load(out/ident/GROUPS[0]/'initial_log_parameters.npy'),np.load(out/ident/GROUPS[1]/'initial_log_parameters.npy'))
        checks.append(ident)
    dump(out/'checks.json',dict(passed=True,circuits=checks,scalar_objectives=True,paired_initialization=True,three_generations=True))
def batch(out,workers):
    out.mkdir(parents=True,exist_ok=False);records=json.loads((HERE/'circuits.json').read_text(encoding='utf-8'))
    from importlib.metadata import version
    code={str(p.relative_to(HERE)):sha(p) for p in [HERE/'run.py',HERE/'model.py',HERE/'prepare.py',HERE/'PROTOCOL.md',HERE/'circuits.json',HERE/'PREPARATION_CHECKS.json',*list((HERE/'frozen').rglob('*'))] if p.is_file() and '__pycache__' not in str(p)}
    dump(out/'manifest.json',dict(source_sha256=code,python=platform.python_version(),versions={k:version(k) for k in ['numpy','scipy','pymoo','moocore','matplotlib']},
        workers=workers,n_obj=1,groups=GROUPS,planned_runs=2320,planned_evaluations=90120000))
    jobs=[(r['id'],seed,group,str(out/r['id']/group/f'seed_{seed}')) for r in records for seed in range(r['seed_start'],r['seed_start']+r['seed_count']) for group in GROUPS]
    start=time.time();done=0;fails=[]
    def status(state='running'):dump(out/'status.json',dict(status=state,completed=done,failed=len(fails),total=len(jobs),elapsed_seconds=time.time()-start,failures=fails))
    status()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(job,j) for j in jobs]):
            r=f.result()
            if r['status']=='complete':done+=1
            else:fails.append(r)
            status()
            if (done+len(fails))%20==0:print(json.dumps(dict(completed=done,failed=len(fails),last_circuit=r['circuit'],elapsed_seconds=time.time()-start)),flush=True)
    status('failed' if fails else 'complete')
    if fails:raise RuntimeError('Run failures retained in status and per-job records')
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['batch','smoke','worker']);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=28)
    p.add_argument('--circuit');p.add_argument('--group',choices=GROUPS);p.add_argument('--seed',type=int);a=p.parse_args()
    if a.mode=='batch':batch(a.output,a.workers)
    elif a.mode=='smoke':smoke(a.output)
    else:run_one(a.circuit,a.seed,a.group,a.output)
if __name__=='__main__':main()

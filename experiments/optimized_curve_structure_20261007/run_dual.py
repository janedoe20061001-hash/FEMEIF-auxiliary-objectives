"""Add a controlled two-objective arm to the frozen scalar experiment."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
import argparse, csv, hashlib, json, platform, time, traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from model import Model
HERE=Path(__file__).resolve().parent
COLS=['magnitude_mse_db2','phase_mse_rad2','wsmse','mean_distance_ohm']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    p=Path(p); tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf8'); tmp.replace(p)
class Circuit(Problem):
    def __init__(self,m): self.m=m; super().__init__(n_var=len(m.lower),n_obj=2,xl=m.lower,xu=m.upper)
    def _evaluate(self,x,out,*a,**kw): out['F']=self.m.score(x)[:,:2]
class Trace(Callback):
    def __init__(self,m): super().__init__(); self.m=m; self.rows=[]
    def notify(self,alg):
        x=alg.pop.get('X'); metrics=self.m.score(x); ix=int(np.argmin(metrics[:,2]))
        self.rows.append(dict(generation=int(alg.n_gen),evaluations=int(alg.evaluator.n_eval),selected_index=ix,**{k:float(v) for k,v in zip(COLS,metrics[ix])}))
def run_one(ident,seed,directory,smoke=False):
    m=Model(ident); r=m.record; pop=12 if smoke else r['population']; gen=3 if smoke else r['generations']
    d=Path(directory); d.mkdir(parents=True,exist_ok=False); start=time.perf_counter()
    x0=np.random.default_rng(seed).uniform(m.lower,m.upper,(pop,len(m.lower))); np.save(d/'initial_log_parameters.npy',x0)
    alg=NSGA2(pop_size=pop,n_offsprings=pop,sampling=x0.copy(),crossover=SBX(prob=.9,eta=1),mutation=PM(prob=1.,prob_var=r['mutation_variable_probability'],eta=1),eliminate_duplicates=False)
    alg.tournament_type='comp_by_rank_and_crowding'; trace=Trace(m)
    result=minimize(Circuit(m),alg,('n_gen',gen),seed=seed,callback=trace,verbose=False,save_history=False)
    x,F=result.pop.get('X'),result.pop.get('F'); metrics=m.score(x)
    np.testing.assert_allclose(F,metrics[:,:2],rtol=1e-12,atol=1e-10)
    front=NonDominatedSorting().do(F,only_non_dominated_front=True)
    rules={'min_wsmse':metrics[:,2],'balanced_minmax':np.maximum(metrics[:,0],m.alpha*metrics[:,1]),'min_distance':metrics[:,3],'min_magnitude':metrics[:,0],'min_phase':metrics[:,1]}
    selected={name:int(front[np.argmin(value[front])]) for name,value in rules.items()}
    ix=selected['min_wsmse']; assert np.isclose(metrics[ix,2],metrics[:,2].min())
    np.savez_compressed(d/'final_population.npz',log_parameters=x,parameters=10**x,optimization_objective=F,cross_metrics=metrics,front_indices=front,selected_index=ix,selected_log_parameters=x[ix],selected_metrics=metrics[ix],selection_rule_names=np.asarray(list(selected)),selection_indices=np.asarray(list(selected.values())))
    with (d/'history.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(trace.rows[0])); w.writeheader(); w.writerows(trace.rows)
    record=dict(status='complete',circuit=ident,seed=seed,group='dual_mse',n_obj=2,population=pop,generations=gen,evaluations=int(result.algorithm.evaluator.n_eval),initial_population_sha256=hashlib.sha256(x0.tobytes()).hexdigest(),front_size=len(front),selection_indices=selected,elapsed_seconds=time.perf_counter()-start,**{k:float(v) for k,v in zip(COLS,metrics[ix])})
    assert record['evaluations']==pop*gen
    dump(d/'summary.json',record); return record
def job(j):
    ident,seed,d=j
    try: return run_one(ident,seed,d)
    except Exception:
        e=dict(status='failed',circuit=ident,seed=seed,traceback=traceback.format_exc()); Path(d).mkdir(parents=True,exist_ok=True); dump(Path(d)/'FAILED.json',e); return e
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','batch']);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=28);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    if a.mode=='smoke':
        for ident in ['case1','case2','rc_series_c1','rlc_bridge_c2','three_parallel_units_c3']: run_one(ident,777,a.output/ident,True)
        dump(a.output/'checks.json',dict(passed=True,n_obj=2,selection_rules=5,circuits=5));return
    from importlib.metadata import version
    records=json.loads((HERE/'circuits.json').read_text(encoding='utf8'))
    sources=[HERE/'run_dual.py',HERE/'model.py',HERE/'circuits.json',HERE/'PROTOCOL.md',*list((HERE/'frozen').rglob('*'))]
    dump(a.output/'manifest.json',dict(source_sha256={str(p.relative_to(HERE)):sha(p) for p in sources if p.is_file() and '__pycache__' not in str(p)},python=platform.python_version(),versions={k:version(k) for k in ['numpy','scipy','pymoo','moocore']},workers=a.workers,n_obj=2,planned_runs=1160,planned_evaluations=45060000))
    jobs=[(r['id'],s,str(a.output/r['id']/f'seed_{s}')) for r in records for s in range(r['seed_start'],r['seed_start']+r['seed_count'])]
    start=time.time(); done=0; failures=[]
    def status(state='running'): dump(a.output/'status.json',dict(status=state,completed=done,failed=len(failures),total=len(jobs),elapsed_seconds=time.time()-start,failures=failures))
    status()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(job,j) for j in jobs]):
            r=f.result()
            if r['status']=='complete': done+=1
            else: failures.append(r)
            status()
            if (done+len(failures))%20==0: print(json.dumps(dict(completed=done,failed=len(failures),last_circuit=r['circuit'],elapsed_seconds=time.time()-start)),flush=True)
    status('failed' if failures else 'complete')
    if failures: raise RuntimeError('Failures retained')
if __name__=='__main__': main()

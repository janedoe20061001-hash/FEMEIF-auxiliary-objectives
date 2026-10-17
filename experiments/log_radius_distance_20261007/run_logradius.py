"""Literal signed-log-radius transform; same evolutionary settings as baselines."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time,argparse,hashlib,platform,csv,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'frozen_scalar'))
from model import Model
from run import dump,sha
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
COLS=['magnitude_mse_db2','phase_mse_rad2','wsmse','mean_distance_ohm','mean_logradius_distance'];CASES=['case1','case2'];SEEDS=list(range(2026100700,2026100800));ZREF=1.
def coordinates(z):
    a=20*np.log10(abs(z)/ZREF);theta=np.angle(z)
    return np.stack((a*np.sin(theta),a*np.cos(theta)),axis=-1)
def score(m,z):
    z=np.atleast_2d(z);base=m.score_spectra(z)
    residual=coordinates(z)-coordinates(m.target)
    d=np.mean(np.hypot(residual[...,0],residual[...,1]),axis=1)
    return np.column_stack((base,d))
class Objective(Problem):
    def __init__(self,m):self.m=m;super().__init__(n_var=len(m.lower),n_obj=1,xl=m.lower,xu=m.upper)
    def _evaluate(self,x,out,*args,**kw):out['F']=score(self.m,self.m.spectrum(x))[:,4,None]
class Trace(Callback):
    def __init__(self,m):super().__init__();self.m=m;self.rows=[]
    def notify(self,alg):
        ix=int(np.argmin(alg.pop.get('F')[:,0]));v=score(self.m,self.m.spectrum(alg.pop.get('X')[ix]))[0]
        self.rows.append(dict(generation=int(alg.n_gen),evaluations=int(alg.evaluator.n_eval),selected_index=ix,**dict(zip(COLS,v.tolist()))))
def one(circuit,seed,directory,smoke=False):
    m=Model(circuit);r=m.record;pop=12 if smoke else r['population'];gen=3 if smoke else 300;d=Path(directory);d.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    x0=np.random.default_rng(seed).uniform(m.lower,m.upper,(pop,len(m.lower)));np.save(d/'initial_log_parameters.npy',x0)
    alg=NSGA2(pop_size=pop,n_offsprings=pop,sampling=x0.copy(),crossover=SBX(prob=.9,eta=1),mutation=PM(prob=1.,prob_var=r['mutation_variable_probability'],eta=1),eliminate_duplicates=False)
    alg.tournament_type='comp_by_rank_and_crowding';trace=Trace(m)
    res=minimize(Objective(m),alg,('n_gen',gen),seed=seed,callback=trace,verbose=False,save_history=False)
    x,F=res.pop.get('X'),res.pop.get('F');metrics=score(m,m.spectrum(x));np.testing.assert_allclose(F[:,0],metrics[:,4],rtol=1e-12,atol=1e-10)
    ix=int(np.argmin(F[:,0]));np.savez_compressed(d/'final_population.npz',log_parameters=x,parameters=10**x,optimization_objective=F,cross_metrics=metrics,metric_names=np.asarray(COLS),selected_index=ix,selected_log_parameters=x[ix],selected_metrics=metrics[ix])
    with (d/'history.csv').open('w',encoding='utf-8-sig',newline='') as h:w=csv.DictWriter(h,fieldnames=list(trace.rows[0]));w.writeheader();w.writerows(trace.rows)
    sm=dict(status='complete',circuit=circuit,seed=seed,group='logradius_distance',n_obj=1,objective_column=4,population=pop,generations=gen,evaluations=int(res.algorithm.evaluator.n_eval),initial_population_sha256=hashlib.sha256(x0.tobytes()).hexdigest(),selected_index=ix,elapsed_seconds=time.perf_counter()-start,**dict(zip(COLS,metrics[ix].tolist())))
    assert sm['evaluations']==pop*gen;dump(d/'summary.json',sm);return sm
def worker(task):
    try:return one(*task)
    except Exception:
        c,s,d=task;err=dict(status='failed',circuit=c,seed=s,traceback=traceback.format_exc());Path(d).mkdir(parents=True,exist_ok=True);dump(Path(d)/'FAILED.json',err);return err
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','batch']);p.add_argument('--output',required=True,type=Path);p.add_argument('--workers',type=int,default=28);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if a.mode=='smoke':
        for c in CASES:
            m=Model(c);np.testing.assert_allclose(score(m,m.target)[0],np.zeros(5),atol=1e-12);one(c,2026100699,a.output/c,True)
        probe=np.array([1.,np.exp(1j*np.pi/3),10.,-.1],complex);u=coordinates(probe)
        np.testing.assert_allclose(u[0],u[1],atol=1e-12);np.testing.assert_allclose(u[2],u[3],atol=1e-12)
        A=20*np.log10(abs(probe));np.testing.assert_allclose(u[...,0],A*probe.imag/abs(probe),atol=1e-12);np.testing.assert_allclose(u[...,1],A*probe.real/abs(probe),atol=1e-12)
        dump(a.output/'CHECKS.json',dict(passed=True,runs=2,literal_coordinates=True,degeneracy_checks=True));print('Smoke passed',flush=True);return
    from importlib.metadata import version
    files=[HERE/'PROTOCOL.md',HERE/'run_logradius.py',*list((HERE/'frozen_scalar').rglob('*'))];files=[f for f in files if f.is_file() and '__pycache__' not in str(f)]
    dump(a.output/'manifest.json',dict(source_sha256={f.relative_to(HERE).as_posix():sha(f) for f in files},seeds=SEEDS,cases=CASES,group='logradius_distance',reference_ohm=ZREF,planned_runs=200,planned_evaluations=10500000,python=platform.python_version(),versions={k:version(k) for k in ['numpy','scipy','pymoo','moocore']},workers=a.workers))
    done=0;errors=[];start=time.time()
    def status(state):dump(a.output/'status.json',dict(status=state,completed=done,failed=len(errors),total=200,elapsed_seconds=time.time()-start,failures=errors))
    status('running');jobs=[(c,s,str(a.output/c/f'seed_{s}')) for c in CASES for s in SEEDS]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(worker,t) for t in jobs]):
            r=f.result()
            if r['status']=='complete':done+=1
            else:errors.append(r)
            status('running')
            if (done+len(errors))%20==0:print(json.dumps({'completed':done,'failed':len(errors),'elapsed_seconds':time.time()-start}),flush=True)
    status('failed' if errors else 'complete')
    if errors:raise RuntimeError('Failed runs retained; no substitute seeds')
if __name__=='__main__':main()

"""Controlled sampling arms, with original-cell quadrature weights."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,csv,time,argparse,hashlib,platform,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'frozen_scalar'))
from model import Model
from run import dump,sha
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
COLS=['magnitude_mse_db2','phase_mse_rad2','wsmse','mean_distance_ohm']
class SamplingModel:
    def __init__(self,ident,arm):
        self.original=Model(ident);self.record=self.original.record;self.lower=self.original.lower;self.upper=self.original.upper;self.alpha=self.original.alpha
        with np.load(HERE/'grids'/f'{ident}.npz') as p:self.f=p['frequency'];self.w=p['weights'];self.target=p['target_true' if arm=='dense_true' else 'target_interpolated']
        self.amp=20*np.log10(abs(self.target));self.phase=np.angle(self.target)
    def spectrum(self,x):return self.original.network(self.f,10**np.atleast_2d(x))
    def scores(self,z):
        z=np.atleast_2d(z);a=np.square(20*np.log10(abs(z))-self.amp)@self.w;p=np.square(np.angle(z)-self.phase)@self.w;d=abs(z-self.target)@self.w
        return np.column_stack((a,p,a+self.alpha*p,d))
    def score(self,x):return self.scores(self.spectrum(x))
class Objective(Problem):
    def __init__(self,m,g):self.m=m;self.group=g;super().__init__(n_var=len(m.lower),n_obj=2 if g=='dual_mse' else 1,xl=m.lower,xu=m.upper)
    def _evaluate(self,x,out,*a,**kw):
        v=self.m.score(x);out['F']=v[:,:2] if self.group=='dual_mse' else v[:,2 if self.group=='wsmse' else 3,None]
class Trace(Callback):
    def __init__(self,m,g):super().__init__();self.m=m;self.group=g;self.rows=[]
    def notify(self,alg):
        x=alg.pop.get('X');v=self.m.score(x) if self.group=='dual_mse' else None
        ix=int(np.argmin(v[:,2] if v is not None else alg.pop.get('F')[:,0]));val=v[ix] if v is not None else self.m.score(x[ix])[0]
        self.rows.append(dict(generation=int(alg.n_gen),evaluations=int(alg.evaluator.n_eval),selected_index=ix,**dict(zip(COLS,val.tolist()))))
def one(c,seed,arm,g,directory,smoke=False):
    m=SamplingModel(c,arm);r=m.record;pop=12 if smoke else r['population'];gen=3 if smoke else 300;d=Path(directory);d.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    x0=np.random.default_rng(seed).uniform(m.lower,m.upper,(pop,len(m.lower)));np.save(d/'initial_log_parameters.npy',x0)
    alg=NSGA2(pop_size=pop,n_offsprings=pop,sampling=x0.copy(),crossover=SBX(prob=.9,eta=1),mutation=PM(prob=1.,prob_var=r['mutation_variable_probability'],eta=1),eliminate_duplicates=False);alg.tournament_type='comp_by_rank_and_crowding';trace=Trace(m,g)
    res=minimize(Objective(m,g),alg,('n_gen',gen),seed=seed,callback=trace,verbose=False,save_history=False);x,F=res.pop.get('X'),res.pop.get('F');v=m.score(x)
    want=v[:,:2] if g=='dual_mse' else v[:,2 if g=='wsmse' else 3,None];np.testing.assert_allclose(F,want,rtol=1e-12,atol=1e-8)
    front=NonDominatedSorting().do(F,only_non_dominated_front=True);ix=int(front[np.argmin(v[front,2])]) if g=='dual_mse' else int(np.argmin(F[:,0]));balanced=int(front[np.argmin(np.maximum(v[front,0],m.alpha*v[front,1]))]) if g=='dual_mse' else ix
    original=m.original.score(x)
    np.savez_compressed(d/'final_population.npz',log_parameters=x,parameters=10**x,optimization_objective=F,training_metrics=v,original_metrics=original,front_indices=front,selected_index=ix,balanced_index=balanced,selected_log_parameters=x[ix],selected_metrics=v[ix])
    with (d/'history.csv').open('w',encoding='utf-8-sig',newline='') as h:w=csv.DictWriter(h,fieldnames=list(trace.rows[0]));w.writeheader();w.writerows(trace.rows)
    record=dict(status='complete',circuit=c,seed=seed,arm=arm,group=g,n_obj=F.shape[1],population=pop,generations=gen,evaluations=int(res.algorithm.evaluator.n_eval),initial_population_sha256=hashlib.sha256(x0.tobytes()).hexdigest(),selected_index=ix,balanced_index=balanced,front_size=len(front),training_points=len(m.f),elapsed_seconds=time.perf_counter()-start,training_metrics=dict(zip(COLS,v[ix].tolist())),original_metrics=dict(zip(COLS,original[ix].tolist())))
    assert record['evaluations']==pop*gen;dump(d/'summary.json',record);return record
def worker(t):
    try:return one(*t)
    except Exception:
        c,s,a,g,d=t;err=dict(status='failed',circuit=c,seed=s,arm=a,group=g,traceback=traceback.format_exc());Path(d).mkdir(parents=True,exist_ok=True);dump(Path(d)/'FAILED.json',err);return err
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','batch']);p.add_argument('--output',required=True,type=Path);p.add_argument('--workers',type=int,default=28);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);config=json.loads((HERE/'CONFIG.json').read_text())
    prep=json.loads((HERE/'PREPARATION.json').read_text());assert prep['passed']
    for name,h in prep['sha256'].items():assert sha(HERE/name)==h,name
    if a.mode=='smoke':
        for c in ['case1','two_parallel_units_c3','rlc_bridge_c2']:
            for arm in config['arms']:
                m=SamplingModel(c,arm);np.testing.assert_allclose(m.scores(m.target),np.zeros((1,4)),atol=1e-12)
                for g in config['groups']:one(c,2026100611,arm,g,a.output/c/arm/g,True)
        dump(a.output/'CHECKS.json',dict(passed=True,runs=18,target_zero_loss=True,groups=3,arms=2));print('18 smoke runs passed',flush=True);return
    from importlib.metadata import version
    dump(a.output/'manifest.json',dict(source_sha256=prep['sha256'],config=config,python=platform.python_version(),versions={k:version(k) for k in ['numpy','scipy','pymoo','moocore']},workers=a.workers))
    tasks=[(r['circuit'],s,arm,g,str(a.output/r['circuit']/arm/g/f'seed_{s}')) for r in config['records'] for s in r['seeds'] for arm in config['arms'] for g in config['groups']];assert len(tasks)==config['planned_runs'];done=0;failed=[];start=time.time()
    def status(state):dump(a.output/'status.json',dict(status=state,completed=done,failed=len(failed),total=len(tasks),elapsed_seconds=time.time()-start,failures=failed))
    status('running')
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(worker,t) for t in tasks]):
            r=f.result()
            if r['status']=='complete':done+=1
            else:failed.append(r)
            status('running')
            if (done+len(failed))%30==0:print(json.dumps(dict(completed=done,failed=len(failed),elapsed_seconds=time.time()-start)),flush=True)
    status('failed' if failed else 'complete')
    if failed:raise RuntimeError('Failures retained; no replacement seeds')
if __name__=='__main__':main()

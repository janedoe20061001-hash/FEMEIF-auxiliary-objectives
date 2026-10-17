import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,time,hashlib,argparse,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'frozen'))
from model import Model
from adaptive import Adaptive
def dump(p,d):
    p=Path(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf8');q.replace(p)
class Objective(Problem):
    def __init__(self,m,arm):
        self.m=m;self.arm=arm
        with np.load(HERE/'target_grid.npz') as d:self.a=Adaptive(m.record,d['frequency'])
        super().__init__(n_var=len(m.lower),n_obj=2,xl=m.lower,xu=m.upper)
    def _evaluate(self,x,out,*args,**kwargs):out['F']=self.m.score(x)[:,:2] if self.arm=='original' else self.a.score(x)
class Trace(Callback):
    def __init__(self,m,arm,seed,d):super().__init__();self.m=m;self.arm=arm;self.seed=seed;self.d=d;self.history=[];self.stages=[];self.xx=[];self.ff=[];self.replay=False
    def notify(self,alg):
        x,F=alg.pop.get('X'),alg.pop.get('F');front=NonDominatedSorting().do(F,only_non_dominated_front=True);ix=int(front[np.argmin(F[front,0]+self.m.alpha*F[front,1])]);bx=int(front[np.argmin(np.maximum(F[front,0],self.m.alpha*F[front,1]))]);g=int(alg.n_gen)
        self.history.append(dict(generation=g,evaluations=int(alg.evaluator.n_eval),metrics=F[ix].tolist(),balanced_metrics=F[bx].tolist(),front_size=len(front)))
        if g%20==0:self.stages.append(g);self.xx.append(x[[ix,bx]].copy());self.ff.append(F[[ix,bx]].copy())
        if g in [300,1000]:
            old=self.m.score(x)
            if self.arm=='original' and g==300:
                with np.load(HERE/'old300'/f'{self.seed}.npz') as p:
                    np.testing.assert_array_equal(x,p['log_parameters']);np.testing.assert_allclose(old,p['metrics'],rtol=1e-12,atol=1e-10);assert ix==int(p['selected_index']);assert bx==int(p['balanced_index'])
                self.replay=True
            np.savez_compressed(self.d/f'generation_{g:03}.npz',log_parameters=x,optimization_objective=F,original_metrics=old,front_indices=front,selected_index=ix,balanced_index=bx)
def one(job):
    arm,seed,root,generations,pop=job;d=Path(root)/arm/f'seed_{seed}';d.mkdir(parents=True,exist_ok=False)
    try:
        start=time.perf_counter();m=Model('case1');x0=np.random.default_rng(seed).uniform(m.lower,m.upper,(pop,len(m.lower)));np.save(d/'initial_log_parameters.npy',x0)
        objective=Objective(m,arm);alg=NSGA2(pop_size=pop,n_offsprings=pop,sampling=x0.copy(),crossover=SBX(prob=.9,eta=1),mutation=PM(prob=1.,prob_var=m.record['mutation_variable_probability'],eta=1),eliminate_duplicates=False);alg.tournament_type='comp_by_rank_and_crowding'
        trace=Trace(m,arm,seed,d);result=minimize(objective,alg,('n_gen',generations),seed=seed,callback=trace,verbose=False,save_history=False)
        np.savez_compressed(d/'trajectory.npz',generations=np.asarray(trace.stages),log_parameters=np.asarray(trace.xx),training_metrics=np.asarray(trace.ff),rules=np.asarray(['min_wsmse','balanced_minmax']))
        dump(d/'history.json',trace.history);a=result.algorithm.problem.a
        rec=dict(status='complete',arm=arm,seed=seed,generations=generations,population=pop,evaluations=int(result.algorithm.evaluator.n_eval),initial_sha256=hashlib.sha256(x0.tobytes()).hexdigest(),original300_replay_exact=trace.replay,seconds=time.perf_counter()-start,model_frequency_queries=a.queries if arm=='adaptive' else int(result.algorithm.evaluator.n_eval)*len(m.f),max_adaptive_points=a.max_points,max_adaptive_layers=a.max_layers)
        assert rec['evaluations']==generations*pop;dump(d/'summary.json',rec);return rec
    except Exception:
        rec=dict(status='failed',arm=arm,seed=seed,traceback=traceback.format_exc());dump(d/'FAILED.json',rec);return rec
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','batch']);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=24);opt=p.parse_args();opt.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((HERE/'CONFIG.json').read_text());sources=json.loads((HERE/'SOURCES.json').read_text())
    for name,h in sources.items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==h,name
    assert json.loads((HERE/'PREFLIGHT.json').read_text())['passed']
    jobs=[(arm,seed,str(opt.output),1000 if opt.mode=='batch' else 3,200 if opt.mode=='batch' else 12) for seed in (cfg['seeds'] if opt.mode=='batch' else [2026100802]) for arm in cfg['arms']]
    from importlib.metadata import version
    dump(opt.output/'manifest.json',dict(config=cfg,source_sha256=sources,versions={k:version(k) for k in ['numpy','scipy','pymoo','moocore']},workers=opt.workers))
    start=time.perf_counter();records=[]
    with ProcessPoolExecutor(max_workers=opt.workers) as pool:
        for future in as_completed([pool.submit(one,j) for j in jobs]):
            records.append(future.result());done=sum(r['status']=='complete' for r in records);failed=[r for r in records if r['status']=='failed'];dump(opt.output/'status.json',dict(status='running',completed=done,failed=len(failed),total=len(jobs),seconds=time.perf_counter()-start,failures=failed));print(json.dumps(dict(completed=done,failed=len(failed),total=len(jobs))),flush=True)
    failed=[r for r in records if r['status']=='failed'];dump(opt.output/'status.json',dict(status='failed' if failed else 'complete',completed=len(records)-len(failed),failed=len(failed),total=len(jobs),seconds=time.perf_counter()-start,failures=failed));assert not failed
if __name__=='__main__':main()

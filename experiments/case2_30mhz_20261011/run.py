from core import *
import argparse,csv,time,traceback,platform
from concurrent.futures import ProcessPoolExecutor,as_completed
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize

class Objective(Problem):
    def __init__(self,m,method,inverse=False):
        self.m=m;self.method=method;self.inverse=inverse
        super().__init__(n_var=len(m.lower),n_obj=2 if method=='dual_nsga' else 1,xl=m.lower,xu=m.upper)
    def _evaluate(self,x,out,*args,**kw):
        v=self.m.score(x);out['F']=v[:,:2] if self.method=='dual_nsga' else (-1/(v[:,2:3]+1e-12) if self.inverse else v[:,2:3])

def algorithm(m,method,x0):
    kw=dict(pop_size=len(x0),n_offsprings=len(x0),sampling=x0.copy(),crossover=SBX(prob=.9,eta=1),
            mutation=PM(prob=1.,prob_var=m.record['mutation_variable_probability'],eta=1),eliminate_duplicates=False)
    a=GA(**kw) if method=='scalar_ga' else NSGA2(**kw)
    if method!='scalar_ga':a.tournament_type='comp_by_rank_and_crowding'
    return a

class Trace(Callback):
    def __init__(self,m,condition,method,seed,d):
        super().__init__();self.m=m;self.condition=condition;self.method=method;self.seed=seed;self.d=d
        self.history=[];self.stages=[];self.parameters=[];self.values=[];self.reasons=[];self.replay=None
    def notify(self,alg):
        g=int(alg.n_gen);x=alg.pop.get('X');v=self.m.score(x);wi=int(np.argmin(v[:,2]))
        self.history.append(dict(generation=g,evaluations=int(alg.evaluator.n_eval),**dict(zip(METRICS,v[wi].tolist()))))
        if g%20==0:
            ki,reason=knee(v[:,:2]) if self.method=='dual_nsga' else (wi,'scalar_minimum')
            self.stages.append(g);self.parameters.append(x[[ki,wi]].copy());self.values.append(v[[ki,wi]].copy());self.reasons.append(reason)
        if g in [300,1000]:
            np.savez_compressed(self.d/f'generation_{g:04}.npz',log_parameters=x,optimization_objective=alg.pop.get('F'),metrics=v)
            if g==300 and self.condition=='full' and self.method in ['scalar_nsga','dual_nsga']:
                old=ROOT/f'experiments/wsmse_teacher_check_20261011/server_run/retrieved/wsmse_teacher_check_20261011/runs/{self.m.record["id"]}/wsmse/seed_{self.seed}/final_population.npz' if self.method=='scalar_nsga' else ROOT/f'experiments/dual_trajectory_20261008/server_run/retrieved/runs/{self.m.record["id"]}/seed_{self.seed}/generation_300.npz'
                if self.method=='scalar_nsga' and not old.exists():
                    old=ROOT/f'output/wsmse_teacher_check_20261011/runs/{self.m.record["id"]}/wsmse/seed_{self.seed}/final_population.npz'
                with np.load(old) as a:previous=a['log_parameters'];previous_metrics=a['metrics']
                matched=bool(np.allclose(x,previous,rtol=1e-10,atol=1e-10) and np.allclose(v,previous_metrics,rtol=1e-10,atol=1e-9))
                self.replay=dict(matched=matched,max_parameter_difference=float(abs(x-previous).max()),max_metric_difference=float(abs(v-previous_metrics).max()))
                if not matched:
                    assert self.m.record['id']=='case2' and self.method=='dual_nsga' and self.seed in [2334,2365]
                    # A mismatch is evidence, not a reason to discard a seed.
                    # Require the repeat to reproduce its retained first attempt.
                    failed=OUT/f'failed_attempts/full/case2/dual_nsga/seed_{self.seed}/generation_0300.npz'
                    with np.load(failed) as a:np.testing.assert_array_equal(x,a['log_parameters'])

def one(job):
    condition,case,method,seed=job;m=model(case,condition);d=OUT/'runs'/condition/case/method/f'seed_{seed}'
    d.mkdir(parents=True,exist_ok=False)
    try:
        t=time.perf_counter();pop=m.record['population'];x0=np.random.default_rng(seed).uniform(m.lower,m.upper,(pop,len(m.lower)))
        np.save(d/'initial_log_parameters.npy',x0);trace=Trace(m,condition,method,seed,d)
        result=minimize(Objective(m,method),algorithm(m,method,x0),('n_gen',1000),seed=seed,callback=trace,verbose=False,save_history=False)
        np.savez_compressed(d/'trajectory.npz',generations=trace.stages,log_parameters=np.asarray(trace.parameters),metrics=np.asarray(trace.values),knee_reasons=np.asarray(trace.reasons))
        with (d/'history.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(trace.history[0]));w.writeheader();w.writerows(trace.history)
        record=dict(status='complete',condition=condition,case=case,method=method,seed=seed,evaluations=int(result.algorithm.evaluator.n_eval),
                    seconds=time.perf_counter()-t,original300_replay=trace.replay,initial_sha256=hashlib.sha256(x0.tobytes()).hexdigest())
        assert record['evaluations']==pop*1000;dump(d/'summary.json',record);return record
    except Exception:
        record=dict(status='failed',condition=condition,case=case,method=method,seed=seed,traceback=traceback.format_exc());dump(d/'FAILED.json',record);return record


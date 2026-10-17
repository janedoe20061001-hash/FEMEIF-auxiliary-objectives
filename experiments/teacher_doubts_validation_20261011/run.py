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

def preflight():
    checks=[]
    for case in ['case1','case2']:
        m=model(case);x=np.random.default_rng(20261011).uniform(m.lower,m.upper,(5,len(m.lower)))
        # Independent high precision graph solve covers the broad parameter bounds.
        f=m.f[np.unique(np.linspace(0,len(m.f)-1,13,dtype=int))]
        z,c=nodal(m,10**x,f);reference=m.network(f,10**x)
        relative=abs(z-reference)/np.maximum(abs(reference),1e-12)
        np.testing.assert_allclose(z,reference,rtol=2e-7,atol=1e-6)
        np.testing.assert_allclose(m.score(x),metrics(m.spectrum(x),m.target,m.alpha),rtol=1e-12,atol=1e-9)
        if case=='case1':np.testing.assert_allclose(m.score(np.log10(m.record['target_parameters'])),0,atol=1e-12)
        small=np.random.default_rng(888).uniform(m.lower,m.upper,(12,len(m.lower)))
        a=minimize(Objective(m,'scalar_ga'),algorithm(m,'scalar_ga',small),('n_gen',4),seed=888,verbose=False)
        b=minimize(Objective(m,'scalar_ga',True),algorithm(m,'scalar_ga',small),('n_gen',4),seed=888,verbose=False)
        np.testing.assert_array_equal(a.pop.get('X'),b.pop.get('X'))
        checks.append(dict(case=case,nodal_max_relative_error=float(relative.max()),nodal_max_condition=max(c),
                           nodal_decimal_digits=60,loss_independent=True,ga_reciprocal_rank_trajectory_equal=True))
    m=model('case2','paper_band');np.testing.assert_allclose(m.score_spectra(m.target),0,atol=1e-12)
    from importlib.metadata import version
    dump(OUT/'PREFLIGHT.json',dict(passed=True,checks=checks,paper_band_points=len(m.f),paper_band_range=[m.f[0],m.f[-1]],
                                  python=platform.python_version(),versions={k:version(k) for k in ['numpy','scipy','pymoo','moocore','mpmath']}))

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=28);p.add_argument('--preflight',action='store_true');a=p.parse_args()
    preflight()
    if a.preflight:return
    jobs=[('full',c,method,s) for c in ['case1','case2'] for method in ['scalar_nsga','dual_nsga','scalar_ga'] for s in range(2300,2400)]
    jobs += [('paper_band','case2',method,s) for method in ['scalar_ga','dual_nsga'] for s in range(2300,2400)]
    t=time.time();done=0;failed=[]
    def status(state):dump(OUT/'STATUS.json',dict(status=state,total=len(jobs),completed=done,failed=failed,seconds=time.time()-t))
    dump(OUT/'RUN_MANIFEST.json',dict(source_sha256={p.name:sha(p) for p in HERE.iterdir() if p.is_file() and p.suffix in ['.py','.md']},planned_runs=800,planned_evaluations=135000000,workers=a.workers))
    status('running')
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(one,j) for j in jobs]):
            r=future.result()
            if r['status']=='complete':done+=1
            else:failed.append(r)
            status('running')
            if (done+len(failed))%50==0:print(json.dumps(dict(completed=done,failed=len(failed),total=len(jobs))),flush=True)
    status('failed' if failed else 'complete');assert not failed

if __name__=='__main__':main()

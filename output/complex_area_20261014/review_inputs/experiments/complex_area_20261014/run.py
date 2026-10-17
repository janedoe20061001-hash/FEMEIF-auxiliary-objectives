"""One fixed-budget paired NSGA-II job, complete populations and provenance."""
from core import *
from features import Features
import time,traceback

def path_for(root,ident,method,seed):return Path(root)/'jobs'/ident/method/f'seed_{seed}'

def verify(d):
    rec=json.loads((d/'summary.json').read_text(encoding='utf-8'))
    assert rec['status']=='complete' and rec['evaluations']==60000
    for name,digest in rec['files_sha256'].items():assert sha(d/name)==digest,(d,name)
    assert rec['generations_saved']==list(range(20,301,20))
    return rec

def one(job):
    ident,method,seed,root,smoke=job
    d=path_for(root,ident,method,seed);d.mkdir(parents=True,exist_ok=False)
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import Problem
        from pymoo.core.callback import Callback
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.optimize import minimize
        st=time.perf_counter();cpu=time.process_time();data=target(ident)
        q=data['quadrature'];g=Features(data)
        population,generations=(12,2) if smoke else (CFG['population'],CFG['generations'])
        baseline=method in ('MAG_PHA','MAG_PHA_ISLANDS')
        x0=np.random.default_rng(seed).uniform(data['lower'],data['upper'],size=(population,len(data['lower'])))
        np.save(d/'initial.npy',x0)
        class Objective(Problem):
            def __init__(self):
                super().__init__(n_var=len(data['lower']),n_obj=2 if baseline else 3,xl=data['lower'],xu=data['upper'])
                self.q=q;self.g=g
            def _evaluate(self,x,out,*args,**kwargs):
                base=self.q.score(x)
                out['F']=base if baseline else np.column_stack([base,self.g.score(x,method)])
                if not np.isfinite(out['F']).all():raise FloatingPointError('Nonfinite objectives')
        class Trace(Callback):
            def __init__(self):super().__init__();self.history=[];self.saved=[]
            def notify(self,alg):
                x,f=alg.pop.get('X'),alg.pop.get('F');index,selection=knee(f[:,:2]);generation=int(alg.n_gen)
                entry=dict(generation=generation,evaluations=int(alg.evaluator.n_eval),selected_index=index,
                    objective=f[index],reason=selection['reason'])
                self.history.append(entry)
                if generation%CFG['save_every']==0 or generation==generations:
                    np.savez_compressed(d/f'generation_{generation:04}.npz',X=x,F=f,selected_index=index)
                    dump(d/f'selection_{generation:04}.json',selection);self.saved.append(generation)
                    dump(d/'progress.json',entry)
        o=CFG['operators'];alg=NSGA2(pop_size=population,n_offsprings=population,sampling=x0.copy(),
            crossover=SBX(prob=o['sbx_probability'],eta=o['sbx_eta']),
            mutation=PM(prob=o['pm_probability'],prob_var=o['pm_variable_probability'],eta=o['pm_eta']),eliminate_duplicates=False)
        alg.tournament_type='comp_by_rank_and_crowding'
        trace=Trace()
        if method=='MAG_PHA_ISLANDS':
            from types import SimpleNamespace
            from pymoo.core.population import Population
            islands=[]
            for i,initial in enumerate(np.split(x0,4)):
                island=NSGA2(pop_size=len(initial),n_offsprings=len(initial),sampling=initial.copy(),
                    crossover=SBX(prob=o['sbx_probability'],eta=o['sbx_eta']),
                    mutation=PM(prob=o['pm_probability'],prob_var=o['pm_variable_probability'],eta=o['pm_eta']),eliminate_duplicates=False)
                island.tournament_type='comp_by_rank_and_crowding'
                island.setup(Objective(),termination=('n_gen',generations),seed=seed+100000*i,verbose=False)
                islands.append(island)
            for gen in range(1,generations+1):
                for island in islands:island.next()
                combined=SimpleNamespace(pop=Population.merge(*[a.pop for a in islands]),n_gen=gen,
                    evaluator=SimpleNamespace(n_eval=sum(a.evaluator.n_eval for a in islands)))
                trace.notify(combined)
            neval=combined.evaluator.n_eval;qq=q;gg=g
        else:
            res=minimize(Objective(),alg,('n_gen',generations),seed=seed,callback=trace,verbose=False,save_history=False)
            neval=int(res.algorithm.evaluator.n_eval);qq=res.algorithm.problem.q;gg=res.algorithm.problem.g
        assert neval==population*generations
        dump(d/'history.json',trace.history)
        rec=dict(status='complete',target=ident,method=method,seed=seed,smoke=smoke,
            evaluations=neval,seconds=time.perf_counter()-st,cpu_seconds=time.process_time()-cpu,
            initial_sha256=__import__('hashlib').sha256(x0.tobytes()).hexdigest(),base_frequency_queries=qq.queries,
            auxiliary_frequency_queries=gg.queries,quadrature_calls=gg.calls,quadrature_refined=gg.refined,quadrature_max_order=gg.max_order,generations_saved=trace.saved,
            final=trace.history[-1],files_sha256={p.name:sha(p) for p in d.iterdir() if p.suffix in ('.npy','.npz','.json')})
        dump(d/'summary.json',rec);return rec
    except Exception:
        rec=dict(status='failed',target=ident,method=method,seed=seed,traceback=traceback.format_exc())
        dump(d/'FAILED.json',rec);return rec

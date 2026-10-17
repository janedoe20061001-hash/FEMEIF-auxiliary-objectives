from common import *
import argparse,time,traceback,platform
from concurrent.futures import ProcessPoolExecutor,as_completed
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
class Objective(Problem):
    def __init__(self,m):
        self.m=m;self.a=evaluator(m)
        super().__init__(n_var=18,n_obj=2,xl=m.lower,xu=m.upper)
    def _evaluate(self,x,out,*args,**kwargs):out['F']=self.a.score(x)
class Trace(Callback):
    def __init__(self,m,d,last):super().__init__();self.m=m;self.d=d;self.last=last;self.history=[];self.stages=[];self.xx=[];self.ff=[]
    def notify(self,alg):
        x,F=alg.pop.get('X'),alg.pop.get('F');front=NonDominatedSorting().do(F,only_non_dominated_front=True)
        ix=int(front[np.argmin(F[front,0]+self.m.alpha*F[front,1])]);bx=int(front[np.argmin(np.maximum(F[front,0],self.m.alpha*F[front,1]))]);g=int(alg.n_gen)
        self.history.append(dict(generation=g,evaluations=int(alg.evaluator.n_eval),metrics=F[ix].tolist(),balanced_metrics=F[bx].tolist(),front_size=len(front)))
        if g%20==0 or g==self.last:
            self.stages.append(g);self.xx.append(x[[ix,bx]].copy());self.ff.append(F[[ix,bx]].copy())
            np.savez_compressed(self.d/f'generation_{g:03}.npz',log_parameters=x,optimization_objective=F,front_indices=front,selected_index=ix,balanced_index=bx)
            dump(self.d/'progress.json',dict(generation=g,evaluations=int(alg.evaluator.n_eval),metrics=F[ix].tolist()))
def one(job):
    s,seed,root,gens,pop=job;d=Path(root)/label(s)/f'seed_{seed}';d.mkdir(parents=True,exist_ok=False)
    try:
        start=time.perf_counter();m=Target(s);x0=np.random.default_rng(seed).uniform(m.lower,m.upper,(pop,18));np.save(d/'initial_log_parameters.npy',x0)
        obj=Objective(m);alg=NSGA2(pop_size=pop,n_offsprings=pop,sampling=x0.copy(),crossover=SBX(prob=.9,eta=1),mutation=PM(prob=1,prob_var=m.record['mutation_variable_probability'],eta=1),eliminate_duplicates=False);alg.tournament_type='comp_by_rank_and_crowding'
        trace=Trace(m,d,gens);res=minimize(obj,alg,('n_gen',gens),seed=seed,callback=trace,verbose=False,save_history=False)
        np.savez_compressed(d/'trajectory.npz',generations=np.array(trace.stages),log_parameters=np.array(trace.xx),training_metrics=np.array(trace.ff),rules=np.array(['min_wsmse','balanced_minmax']))
        dump(d/'history.json',trace.history);a=res.algorithm.problem.a
        rec=dict(status='complete',scale=s,seed=seed,generations=gens,population=pop,evaluations=int(res.algorithm.evaluator.n_eval),initial_sha256=hashlib.sha256(x0.tobytes()).hexdigest(),seconds=time.perf_counter()-start,model_frequency_queries=a.queries,max_adaptive_points=a.max_points,max_adaptive_layers=a.max_layers)
        assert rec['evaluations']==gens*pop;dump(d/'summary.json',rec);return rec
    except Exception:
        rec=dict(status='failed',scale=s,seed=seed,traceback=traceback.format_exc());dump(d/'FAILED.json',rec);return rec
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','timing','batch']);p.add_argument('--workers',type=int,default=4);opt=p.parse_args()
    cfg=json.loads((HERE/'CONFIG.json').read_text());assert json.loads((HERE/'PREFLIGHT.json').read_text())['passed']
    sources=json.loads((HERE/'INPUT_HASHES.json').read_text())
    for name,h in sources.items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==h,name
    root=HERE/opt.mode;root.mkdir(exist_ok=False)
    scales=cfg['run_scales'];seeds=cfg['seeds'] if opt.mode=='batch' else [2026100903]
    gens,pop=(cfg['generations'],cfg['population']) if opt.mode=='batch' else ((20,200) if opt.mode=='timing' else (3,12))
    from importlib.metadata import version
    code={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'run.py',HERE/'common.py']}
    dump(root/'manifest.json',dict(config=cfg,input_hashes=sources,code_hashes=code,versions={k:version(k) for k in ['numpy','scipy','pymoo','moocore','matplotlib']},python=sys.version,platform=platform.platform(),workers=opt.workers,mode=opt.mode,planned_evaluations=len(scales)*len(seeds)*gens*pop))
    jobs=[(s,seed,str(root),gens,pop) for seed in seeds for s in scales];records=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=opt.workers) as pool:
        for future in as_completed([pool.submit(one,j) for j in jobs]):
            records.append(future.result());done=sum(r['status']=='complete' for r in records);failed=[r for r in records if r['status']=='failed']
            dump(root/'status.json',dict(status='running',completed=done,failed=len(failed),total=len(jobs),seconds=time.perf_counter()-start,failures=failed));print(json.dumps(records[-1]),flush=True)
    failures=[r for r in records if r['status']=='failed'];dump(root/'status.json',dict(status='failed' if failures else 'complete',completed=len(records)-len(failures),failed=len(failures),total=len(jobs),seconds=time.perf_counter()-start,failures=failures));dump(root/'summaries.json',records)
    for seed in seeds:
        init=[np.load(root/label(s)/f'seed_{seed}'/'initial_log_parameters.npy') for s in scales]
        for xx in init[1:]:np.testing.assert_array_equal(init[0],xx)
    assert not failures
if __name__=='__main__':main()

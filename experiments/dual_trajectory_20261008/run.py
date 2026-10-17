"""Replay frozen dual-objective runs and retain every 20th surviving population."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
import sys,json,time,hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'frozen_dual'))
from run_dual import Model,Circuit,NSGA2,SBX,PM,minimize,Callback,NonDominatedSorting,dump

class Snapshots(Callback):
    def __init__(self,m,d): super().__init__();self.m=m;self.d=d;self.rows=[]
    def notify(self,alg):
        x=alg.pop.get('X');v=self.m.score(x);front=NonDominatedSorting().do(v[:,:2],only_non_dominated_front=True)
        ix=int(front[np.argmin(v[front,2])]);bx=int(front[np.argmin(np.maximum(v[front,0],self.m.alpha*v[front,1]))])
        self.rows.append(dict(generation=int(alg.n_gen),evaluations=int(alg.evaluator.n_eval),selected_index=ix,balanced_index=bx,metrics=v[ix].tolist(),balanced_metrics=v[bx].tolist()))
        if alg.n_gen%20==0:
            np.savez_compressed(self.d/f'generation_{alg.n_gen:03}.npz',log_parameters=x,optimization_objective=alg.pop.get('F'),metrics=v,front_indices=front,selected_index=ix,balanced_index=bx)

def run(job):
    ident,seed=job;m=Model(ident);r=m.record;d=HERE/'runs'/ident/f'seed_{seed}';d.mkdir(parents=True,exist_ok=False)
    x0=np.random.default_rng(seed).uniform(m.lower,m.upper,(r['population'],len(m.lower)));np.save(d/'initial_log_parameters.npy',x0)
    alg=NSGA2(pop_size=r['population'],n_offsprings=r['population'],sampling=x0.copy(),crossover=SBX(prob=.9,eta=1),mutation=PM(prob=1.,prob_var=r['mutation_variable_probability'],eta=1),eliminate_duplicates=False)
    alg.tournament_type='comp_by_rank_and_crowding';trace=Snapshots(m,d);t=time.time()
    result=minimize(Circuit(m),alg,('n_gen',300),seed=seed,callback=trace,verbose=False,save_history=False)
    dump(d/'history.json',trace.rows);dump(d/'summary.json',dict(status='complete',circuit=ident,seed=seed,evaluations=int(result.algorithm.evaluator.n_eval),elapsed_seconds=time.time()-t,initial_sha256=hashlib.sha256(x0.tobytes()).hexdigest()))
    return ident,seed

if __name__=='__main__':
    config=json.loads((HERE/'CONFIG.json').read_text());t=time.time()
    jobs=[(c,s) for c in config['cases'] for s in range(2300,2400)]
    with ProcessPoolExecutor(max_workers=28) as pool:
        for c,s in pool.map(run,jobs): print(json.dumps(dict(complete=c,seed=s)),flush=True)
    dump(HERE/'runs/status.json',dict(status='complete',runs=len(jobs),evaluations=10500000,elapsed_seconds=time.time()-t))

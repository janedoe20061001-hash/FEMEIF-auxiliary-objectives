from run import *
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
class Probe(Callback):
    def __init__(self,m,seed,original_callback):
        super().__init__();self.m=m;self.seed=seed;self.original_callback=original_callback;self.differences=[]
    def notify(self,alg):
        g=int(alg.n_gen);x=alg.pop.get('X');v=self.m.score(x)
        if self.original_callback:
            fr=NonDominatedSorting().do(v[:,:2],only_non_dominated_front=True)
            _=int(fr[np.argmin(v[fr,2])]);_=int(fr[np.argmin(np.maximum(v[fr,0],self.m.alpha*v[fr,1]))])
        elif g%20==0:knee(v[:,:2])
        if g%20==0:
            old=ROOT/f'experiments/dual_trajectory_20261008/server_run/retrieved/runs/case2/seed_{self.seed}/generation_{g:03}.npz'
            if old.exists():
                with np.load(old) as a:y=a['log_parameters'];w=a['metrics']
                self.differences.append(dict(g=g,parameter_max=float(abs(x-y).max()),metric_max=float(abs(v-w).max())))
def main():
    rows=[]
    for seed in [2334,2365]:
        for original_callback in [False,True]:
            m=model('case2');x0=np.random.default_rng(seed).uniform(m.lower,m.upper,(150,15));trace=Probe(m,seed,original_callback)
            a=minimize(Objective(m,'dual_nsga'),algorithm(m,'dual_nsga',x0),('n_gen',300),seed=seed,callback=trace,verbose=False)
            rows.append(dict(seed=seed,original_callback=original_callback,differences=trace.differences,minimum=float(m.score(a.pop.get('X'))[:,2].min())))
    dump(OUT/'REPLAY_PROBE.json',rows);print(json.dumps(rows),flush=True)
if __name__=='__main__':main()

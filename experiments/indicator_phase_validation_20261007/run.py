"""Isolated paired NSGA-II validation, reusing frozen operators and reporting code."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

HERE=Path(__file__).resolve().parent
GROUPS=('baseline','old_pair','signed_phase_pair')
OLD=('reactance_crossing_direction_consistency','transition_speed_relative_error')
def dump(p,v):
    p=Path(p); tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(p)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def setup(case):
    sys.path.insert(0,str(HERE/'frozen'/case))
    import experiment as e
    from legacy_objectives import Objectives
    cfg,freq=e.config(HERE/'frozen'/case/'config.json')
    if case=='case1':
        from legacy_objectives import impedance, TARGET, LOWER, UPPER
        target=impedance(freq,TARGET)[0]; network=impedance
    else:
        from case2 import load_target,impedance,LOWER,UPPER
        _,target=load_target(cfg['target_data']); network=impedance
    return e,Objectives,cfg,freq,target,network,LOWER,UPPER
class Evaluation:
    def __init__(self,case,group,env):
        e,Legacy,cfg,self.f,self.target,self.network,self.lower,self.upper=env
        self.group=group
        self.old=Legacy(self.f,aux=() if group=='baseline' else OLD,target_z=self.target)
        self.h=np.diff(np.log10(self.f)); self.slope=np.diff(np.angle(self.target))/self.h
        self.energy=np.sum(self.h*self.slope**2)
        assert self.energy>0
    def spectra(self,z):
        out=self.old.evaluate_spectra(z)
        if self.group=='signed_phase_pair':
            out[:,3]=np.sqrt(np.sum(self.h*(np.diff(np.angle(z),axis=1)/self.h-self.slope)**2,axis=1)/self.energy)
        assert np.isfinite(out).all() and np.all(out>=0)
        return out
    def evaluate(self,x): return self.spectra(self.network(self.f,10**np.atleast_2d(x)))
def run_one(case,group,seed,directory,env,pop=None,gen=None):
    e,_,cfg,freq,*_=env
    cfg=cfg.copy(); cfg['population']=pop or cfg['population']; cfg['generations']=gen or cfg['generations']
    from pymoo.core.problem import Problem
    class Circuit(Problem):
        def __init__(self):
            self.ev=Evaluation(case,group,env)
            super().__init__(n_var=len(self.ev.lower),n_obj=2 if group=='baseline' else 4,xl=self.ev.lower,xu=self.ev.upper)
        def _evaluate(self,x,out,*a,**kw): out['F']=self.ev.evaluate(x)
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    x0=e.initial_population(cfg,seed)
    np.save(directory/'initial_log_parameters.npy',x0)
    alpha=cfg['final_selection']['phase_weight']; trace=e.Trace(alpha)
    alg=e.NSGA2(pop_size=cfg['population'],n_offsprings=cfg['population'],sampling=x0.copy(),
        crossover=e.SBX(prob=cfg['crossover_probability'],eta=cfg['crossover_eta']),
        mutation=e.PM(prob=cfg['mutation_individual_probability'],prob_var=cfg['mutation_variable_probability'],eta=cfg['mutation_eta']),
        eliminate_duplicates=False)
    alg.tournament_type='comp_by_rank_and_crowding'
    start=time.perf_counter()
    result=e.minimize(Circuit(),alg,('n_gen',cfg['generations']),seed=seed,callback=trace,verbose=False,save_history=False)
    x,f=result.pop.get('X'),result.pop.get('F'); selected=e.select_wsmse(f,alpha); ix=selected['wsmse_selected_index']
    np.savez_compressed(directory/'final_population.npz',log_parameters=x,parameters=10**x,objectives=f,
        full_front_indices=e.nd_indices(f),primary_front_indices=e.nd_indices(f[:,:2]),selected_index=ix,
        selected_log_parameters=x[ix],selected_primary_mse=f[ix,:2],selection_alpha=alpha)
    with (directory/'history.csv').open('w',encoding='utf-8-sig',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(trace.rows[0])); w.writeheader(); w.writerows(trace.rows)
    record=dict(status='complete',case=case,group=group,seed=seed,
        objective_names=['magnitude_mse_db2','phase_mse_rad2']+([] if group=='baseline' else [OLD[0],OLD[1] if group=='old_pair' else 'D_phi']),
        initial_population_sha256=hashlib.sha256(x0.tobytes()).hexdigest(),
        evaluations=int(result.algorithm.evaluator.n_eval),generations=len(trace.rows),elapsed_seconds=time.perf_counter()-start,
        **e.stats(f,alpha))
    assert record['evaluations']==cfg['population']*cfg['generations'] and record['generations']==cfg['generations']
    dump(directory/'summary.json',record)
    return record
def smoke(case,out):
    env=setup(case); e,_,cfg,freq,*_=env; cfg=cfg.copy(); cfg.update(population=12,generations=3)
    checks=[]
    for group,aux in [('baseline',()),('old_pair',OLD)]:
        original=out/'original'/group; adapted=out/'adapted'/group
        e.run_one(cfg,freq,aux,999,original,'smoke')
        run_one(case,group,999,adapted,env,12,3)
        with np.load(original/'final_population.npz') as a,np.load(adapted/'final_population.npz') as b:
            np.testing.assert_array_equal(a['log_parameters'],b['log_parameters'])
            np.testing.assert_array_equal(a['objectives'],b['objectives'])
            np.testing.assert_array_equal(a['selected_index'],b['selected_index'])
        checks.append(group+'_exact_frozen_equality')
    ev=Evaluation(case,'signed_phase_pair',env)
    np.testing.assert_allclose(ev.spectra(ev.target[None]),0,atol=1e-12)
    scaled=ev.spectra((1.3*ev.target)[None]); assert scaled[0,3]<1e-12
    old=Evaluation(case,'old_pair',env)
    x=e.initial_population(cfg,998)
    np.testing.assert_array_equal(ev.evaluate(x)[:,:3],old.evaluate(x)[:,:3])
    run_one(case,'signed_phase_pair',999,out/'adapted'/'signed_phase_pair',env,12,3)
    dump(out/'checks.json',dict(case=case,checks=checks+['candidate_target_zero','positive_scale_invariance','first_three_objectives_unchanged','candidate_smoke'],passed=True))
def batch(out,workers):
    out.mkdir(parents=True,exist_ok=False)
    frozen={str(p.relative_to(HERE)):sha(p) for p in (HERE/'frozen').rglob('*') if p.is_file() and '__pycache__' not in str(p)}
    from importlib.metadata import version
    manifest=dict(protocol_sha256=sha(HERE/'PROTOCOL.md'),runner_sha256=sha(__file__),frozen_sha256=frozen,
        versions={k:version(k) for k in ('numpy','scipy','pymoo','moocore','matplotlib')},python=platform.python_version(),
        workers=workers,seeds=list(range(2200,2220)),groups=list(GROUPS),planned_runs=120,planned_evaluations=6300000)
    dump(out/'manifest.json',manifest)
    jobs=[(case,seed,group) for seed in range(2200,2220) for case in ('case1','case2') for group in GROUPS]
    started=time.time(); completed=[]; failures=[]
    def work(job):
        case,seed,group=job; d=out/case/group/f'seed_{seed}'
        log=out/'logs'/f'{case}_{group}_{seed}.log'; log.parent.mkdir(exist_ok=True)
        with log.open('w',encoding='utf-8') as f:
            cp=subprocess.run([sys.executable,str(HERE/'run.py'),'worker','--case',case,'--group',group,'--seed',str(seed),'--output',str(d)],stdout=f,stderr=subprocess.STDOUT)
        if cp.returncode: return dict(case=case,seed=seed,group=group,exit_code=cp.returncode,log=str(log))
        return json.loads((d/'summary.json').read_text())
    def status():
        dump(out/'status.json',dict(status='running',completed=len(completed),failed=len(failures),total=120,elapsed_seconds=time.time()-started,failures=failures))
    status()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(work,j) for j in jobs]):
            r=future.result()
            (completed if r.get('status')=='complete' else failures).append(r); status()
            print(json.dumps(dict(completed=len(completed),failed=len(failures),case=r['case'],group=r['group'],seed=r['seed'])),flush=True)
    dump(out/'status.json',dict(status='complete' if not failures else 'failed',completed=len(completed),failed=len(failures),total=120,elapsed_seconds=time.time()-started,failures=failures))
    if failures: raise RuntimeError('Failed runs retained; inspect status.json')
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['worker','smoke','batch']);p.add_argument('--case',choices=['case1','case2']);p.add_argument('--group',choices=GROUPS)
    p.add_argument('--seed',type=int);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=28);a=p.parse_args()
    if a.mode=='batch': batch(a.output,a.workers)
    elif a.mode=='smoke': smoke(a.case,a.output)
    else: print(json.dumps(run_one(a.case,a.group,a.seed,a.output,setup(a.case))),flush=True)
if __name__=='__main__': main()

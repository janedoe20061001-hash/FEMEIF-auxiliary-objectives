"""Controlled scalar WSMSE check using the published trajectory's frozen source."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import argparse, csv, hashlib, json, platform, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / 'experiments/dual_trajectory_20261008'
FROZEN = BASE / 'frozen_dual'
OLD_SCALAR = ROOT / 'experiments/primary_loss_comparison_20261007/server_run/retrieved/runs'
OLD_DUAL = BASE / 'server_run/retrieved/runs'
OUT = ROOT / 'output/wsmse_teacher_check_20261011'
sys.path.insert(0, str(FROZEN))
from model import Model, response
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize

COLS = ['magnitude_mse_db2', 'phase_mse_rad2', 'wsmse', 'mean_distance_ohm']
SHOW = [2307, 2312, 2321]

def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    temp.replace(path)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def independent_metrics(z, target, alpha):
    z = np.atleast_2d(z)
    a = np.sum(np.square(20 * np.log10(np.abs(z) / np.abs(target))), axis=1) / len(target)
    p = np.sum(np.square(np.arctan2(z.imag, z.real) - np.arctan2(target.imag, target.real)), axis=1) / len(target)
    return np.column_stack([a, p, a + alpha * p, np.abs(z-target).mean(axis=1)])

def preflight():
    sources = json.loads((BASE / 'SOURCES.json').read_text())
    for name, digest in sources.items():
        assert sha(BASE/name) == digest, name
    cases = []
    for ident in ['case1', 'case2']:
        m = Model(ident); r = m.record
        np.testing.assert_allclose(m.score_spectra(m.target), 0, atol=1e-12)
        x = np.random.default_rng(20261011).uniform(m.lower, m.upper, (9, len(m.lower)))
        z = m.spectrum(x)
        np.testing.assert_allclose(z, response(r['tree'], r['parameter_names'], m.f, 10**x), rtol=1e-12, atol=1e-8)
        np.testing.assert_allclose(m.score(x), independent_metrics(z, m.target, m.alpha), rtol=1e-12, atol=1e-10)
        cfg = json.loads((FROZEN / 'frozen' / ident / 'config.json').read_text())
        assert m.alpha == cfg['final_selection']['phase_weight']
        if ident == 'case1':
            np.testing.assert_allclose(m.network(m.f, np.asarray(r['target_parameters']))[0], m.target, rtol=1e-12)
        else:
            data = np.genfromtxt(ROOT/'targetImpedance_case2.csv', delimiter=',', names=True, encoding='utf-8-sig')
            assert len(data) == 201 and sha(ROOT/'targetImpedance_case2.csv') == cfg['target_data']['sha256']
            np.testing.assert_array_equal(m.f, data['frequency'])
            np.testing.assert_allclose(m.target, data['mag']*np.exp(1j*np.deg2rad(data['pha'])), rtol=1e-12)
        cases.append(dict(case=ident,frequency_points=len(m.f),population=r['population'],alpha=m.alpha,
                          target_negative_real_points=int(np.count_nonzero(m.target.real < 0))))
    from importlib.metadata import version
    dump(OUT/'PREFLIGHT.json', dict(passed=True,source_sha256=sources,python=platform.python_version(),
                                  versions={k:version(k) for k in ['numpy','scipy','pymoo','moocore','matplotlib']},
                                  cases=cases,checks=['frozen hashes','target zero','series/parallel network','independent loss','units','CSV hash']))
    print('Preflight passed', flush=True)

class Objective(Problem):
    def __init__(self, model, group):
        self.m, self.group = model, group
        super().__init__(n_var=len(model.lower), n_obj=1 if group == 'wsmse' else 2, xl=model.lower, xu=model.upper)
    def _evaluate(self, x, out, *args, **kwargs):
        scores = self.m.score(x)
        out['F'] = scores[:, 2:3] if self.group == 'wsmse' else scores[:, :2]

class Trace(Callback):
    def __init__(self, model, directory, snapshots):
        super().__init__(); self.m, self.d, self.snapshots = model, directory, snapshots; self.rows=[]
    def notify(self, algorithm):
        x = algorithm.pop.get('X'); v=self.m.score(x); ix=int(np.argmin(v[:,2]))
        self.rows.append(dict(generation=int(algorithm.n_gen),evaluations=int(algorithm.evaluator.n_eval),
                              **dict(zip(COLS, v[ix].tolist()))))
        if self.snapshots and algorithm.n_gen % 20 == 0:
            np.savez_compressed(self.d/f'generation_{algorithm.n_gen:03}.npz',log_parameters=x,
                                optimization_objective=algorithm.pop.get('F'),metrics=v,selected_index=ix)

def run(job):
    ident, seed, group = job
    d=OUT/'runs'/ident/group/f'seed_{seed}';d.mkdir(parents=True,exist_ok=False)
    try:
        m=Model(ident);r=m.record;t=time.perf_counter()
        x0=np.random.default_rng(seed).uniform(m.lower,m.upper,(r['population'],len(m.lower)))
        np.save(d/'initial_log_parameters.npy',x0)
        algorithm=NSGA2(pop_size=r['population'],n_offsprings=r['population'],sampling=x0.copy(),
                        crossover=SBX(prob=.9,eta=1),mutation=PM(prob=1.,prob_var=r['mutation_variable_probability'],eta=1),
                        eliminate_duplicates=False)
        algorithm.tournament_type='comp_by_rank_and_crowding'
        trace=Trace(m,d,seed in SHOW)
        result=minimize(Objective(m,group),algorithm,('n_gen',300),seed=seed,callback=trace,verbose=False,save_history=False)
        x=result.pop.get('X');v=m.score(x);ix=int(np.argmin(v[:,2]))
        np.savez_compressed(d/'final_population.npz',log_parameters=x,metrics=v,selected_index=ix,
                            optimization_objective=result.pop.get('F'))
        with (d/'history.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(trace.rows[0]));writer.writeheader();writer.writerows(trace.rows)
        record=dict(status='complete',case=ident,seed=seed,group=group,selected_index=ix,
                    evaluations=int(result.algorithm.evaluator.n_eval),n_obj=result.pop.get('F').shape[1],
                    elapsed_seconds=time.perf_counter()-t,**dict(zip(COLS,v[ix].tolist())))
        dump(d/'summary.json',record);return record
    except Exception:
        record=dict(status='failed',case=ident,seed=seed,group=group,traceback=traceback.format_exc())
        dump(d/'FAILED.json',record);return record

def batch(workers):
    preflight()
    jobs=[(c,s,'wsmse') for c in ['case1','case2'] for s in range(2300,2400)]
    jobs += [(c,s,'dual') for c in ['case1','case2'] for s in SHOW]
    start=time.time();complete=0;failed=[]
    def status(state):
        dump(OUT/'STATUS.json',dict(status=state,completed=complete,total=len(jobs),failed=failed,elapsed_seconds=time.time()-start))
    status('running')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(run,j) for j in jobs]):
            record=future.result()
            if record['status']=='complete':complete+=1
            else:failed.append(record)
            status('running')
            if complete%20==0:print(f'{complete}/{len(jobs)} complete',flush=True)
    status('failed' if failed else 'complete')
    if failed:raise RuntimeError('Failures retained')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);p.add_argument('--preflight',action='store_true');a=p.parse_args()
    if a.preflight:preflight()
    else:batch(a.workers)

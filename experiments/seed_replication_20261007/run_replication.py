"""Seed-only replication of two frozen scalar-optimization cases."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time,argparse,hashlib,platform
from concurrent.futures import ProcessPoolExecutor,as_completed
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'frozen_scalar'))
from run import run_one,sha,dump
CASES=['case1','case2'];GROUPS=['wsmse','mean_distance'];SEEDS=list(range(2026100700,2026100800))
def worker(task):
    ident,seed,group,directory=task
    try:return run_one(ident,seed,group,directory)
    except Exception:
        import traceback
        p=Path(directory);p.mkdir(parents=True,exist_ok=True)
        err={'status':'failed','circuit':ident,'seed':seed,'group':group,'traceback':traceback.format_exc()};dump(p/'FAILED.json',err);return err
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','batch']);p.add_argument('--output',required=True,type=Path);p.add_argument('--workers',default=28,type=int);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    if a.mode=='smoke':
        import numpy as np
        for ident in CASES:
            for group in GROUPS:run_one(ident,2026100699,group,a.output/ident/group,smoke=True)
            np.testing.assert_array_equal(np.load(a.output/ident/GROUPS[0]/'initial_log_parameters.npy'),np.load(a.output/ident/GROUPS[1]/'initial_log_parameters.npy'))
        dump(a.output/'CHECKS.json',{'passed':True,'runs':4,'formal_seed_overlap':False});print('Smoke passed',flush=True);return
    from importlib.metadata import version
    files=[HERE/'PROTOCOL.md',HERE/'run_replication.py',*list((HERE/'frozen_scalar').rglob('*'))]
    files=[f for f in files if f.is_file() and '__pycache__' not in str(f)]
    manifest={'source_sha256':{str(f.relative_to(HERE)):sha(f) for f in files},'seeds':SEEDS,'cases':CASES,'groups':GROUPS,'python':platform.python_version(),'versions':{k:version(k) for k in ['numpy','scipy','pymoo','moocore']},'workers':a.workers,'planned_runs':400,'planned_evaluations':21000000}
    dump(a.output/'manifest.json',manifest);start=time.time();done=0;failures=[]
    def status(state):dump(a.output/'status.json',{'status':state,'completed':done,'failed':len(failures),'total':400,'elapsed_seconds':time.time()-start,'failures':failures})
    status('running');jobs=[(c,s,g,str(a.output/c/g/f'seed_{s}')) for c in CASES for s in SEEDS for g in GROUPS]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(worker,j) for j in jobs]):
            v=f.result()
            if v['status']=='complete':done+=1
            else:failures.append(v)
            status('running')
            if (done+len(failures))%20==0:print(json.dumps({'completed':done,'failed':len(failures),'elapsed_seconds':time.time()-start}),flush=True)
    status('failed' if failures else 'complete')
    if failures:raise RuntimeError('Failures retained; no replacement seeds')
if __name__=='__main__':main()

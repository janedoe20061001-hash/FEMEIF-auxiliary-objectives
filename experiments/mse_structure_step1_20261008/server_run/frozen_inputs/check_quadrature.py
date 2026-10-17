"""Independent modal-grid integration for every retained adaptive individual."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,json,argparse,time,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'frozen'));sys.path.insert(0,str(HERE/'analysis_toolkit'))
from model import Model
from refined_grid import improved_dense_grid
from adaptive import Adaptive
def one(job):
    seed,runs=job;m=Model('case1');a=Adaptive(m.record,np.load(HERE/'target_grid.npz')['frequency']);errors=[];checked=0;max_ratios=np.zeros(2);start=time.perf_counter()
    for g in [300,1000]:
        with np.load(Path(runs)/'adaptive'/f'seed_{seed}'/f'generation_{g:03}.npz') as data:
            xs=data['log_parameters'];F=data['optimization_objective']
            for index,x in enumerate(xs):
                ref2=a.reference(x,improved_dense_grid(m.record,np.atleast_2d(10**x),2))[0];ref4=a.reference(x,improved_dense_grid(m.record,np.atleast_2d(10**x),4))[0]
                tolerance=np.maximum([.01,np.deg2rad(.5)**2],.005*np.maximum(ref2,ref4));ratio=np.maximum.reduce([abs(F[index]-ref2),abs(F[index]-ref4),abs(ref2-ref4)])/tolerance;max_ratios=np.maximum(max_ratios,ratio);checked+=1
                if np.any(ratio>1):errors.append(dict(seed=seed,generation=g,index=index,training=F[index].tolist(),reference2=ref2.tolist(),reference4=ref4.tolist(),tolerance=tolerance.tolist(),tolerance_ratio=ratio.tolist()))
    return dict(seed=seed,checked=checked,max_tolerance_ratio=max_ratios.tolist(),errors=errors,seconds=time.perf_counter()-start)
def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--output',required=True,type=Path);p.add_argument('--workers',type=int,default=8);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();records=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(one,(s,args.runs)) for s in range(2300,2400)]):
            r=future.result();records.append(r);(args.output/f'seed_{r["seed"]}.json').write_text(json.dumps(r,indent=2));print(json.dumps(dict(completed=len(records),errors=sum(len(r['errors']) for r in records))),flush=True)
    result=dict(passed=not any(r['errors'] for r in records),individuals_checked=sum(r['checked'] for r in records),max_tolerance_ratio=np.max([r['max_tolerance_ratio'] for r in records],axis=0).tolist(),seconds=time.perf_counter()-start,workers=args.workers,errors=[e for r in records for e in r['errors']],source_sha256={f.relative_to(HERE).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in [HERE/'check_quadrature.py',HERE/'adaptive.py',*HERE.glob('analysis_toolkit/**/*.py'),HERE/'frozen/circuits.json',HERE/'target_grid.npz']})
    (args.output/'VALIDATION.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['errors','source_sha256']}));assert result['passed'] and result['individuals_checked']==40000
if __name__=='__main__':main()

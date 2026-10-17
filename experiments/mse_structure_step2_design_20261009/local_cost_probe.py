"""Small local timing probe only; no optimizer, no new fitting experiment."""
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,time,json,platform,importlib.util,statistics
import numpy as np
import scipy
HERE=Path(__file__).resolve().parent;STEP1=HERE.parent/'mse_structure_step1_20261008';sys.path.insert(0,str(STEP1))
import classify_v2 as c
def timed(fn,repeats=3):
    fn();times=[]
    for _ in range(repeats):
        start=time.perf_counter();fn();times.append(time.perf_counter()-start)
    return dict(seconds=times,median_seconds=statistics.median(times))
def main():
    m=c.Model('case1');a=c.Adaptive(m.record,np.load(STEP1/'target_grid.npz')['frequency']);rng=np.random.default_rng(2026100901);x=rng.uniform(m.lower,m.upper,(200,18));old=np.load(STEP1/'old300/2300.npz');p=10**old['log_parameters'][int(old['selected_index'])];catalog=c.target_catalog(m)
    result=dict(date='2026-10-09',scope='CPU timing only; no optimization',python=sys.version,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,pymoo_available=bool(importlib.util.find_spec('pymoo')),native_accelerator_available=False,batch_size=200,original_batch=timed(lambda:m.score(x)),adaptive_python_batch=timed(lambda:a.score_python(x)),representative_diagnostic=timed(lambda:c.diagnostic(m,p,catalog)),max_points=a.max_points,max_layers=a.max_layers)
    (HERE/'LOCAL_COST_PROBE.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))
if __name__=='__main__':main()

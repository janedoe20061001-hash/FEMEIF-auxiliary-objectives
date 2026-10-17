"""Exact verification of vectorized crossing lookup against frozen diagnosis."""
from pathlib import Path
import sys,json,time
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'frozen'));sys.path.insert(0,str(HERE/'toolkit'))
from model import Model,module
from adaptive import spectra
from refined_grid import improved_dense_grid
old=module('old_step1_diagnostics',HERE/'toolkit/diagnostics.py');new=module('new_step1_diagnostics',HERE/'analysis_toolkit/diagnostics.py')
def main():
    m=Model('case1');params=[]
    for seed in range(2300,2400):
        with np.load(HERE/'old300'/f'{seed}.npz') as d:params.append(10**d['log_parameters'][int(d['selected_index'])])
    params.extend(10**np.random.default_rng(2026100802).uniform(m.lower,m.upper,(32,18)));params.extend([m.record['target_parameters'],10**m.lower,10**m.upper]);count=0;t=time.perf_counter()
    for p in params:
        for level in [1,2]:
            f=improved_dense_grid(m.record,np.atleast_2d(p),level);z=spectra(f,p)[0];target=spectra(f,m.record['target_parameters'])[0];x=np.log10(f)
            for curve in [target,z]:
                for prom in [.5,1.,2.]:assert old.events(x,curve,prom)==new.events(x,curve,prom);count+=1
    # All remaining diagnosis functions are unchanged; verify end-to-end too.
    for p in params[:3]:
        f=improved_dense_grid(m.record,np.atleast_2d(p),2);z=spectra(f,p)[0];target=spectra(f,m.record['target_parameters'])[0];assert old.diagnose(f,target,z)==new.diagnose(f,target,z)
    result=dict(passed=True,parameters=len(params),grids_per_parameter=2,event_lists_exact=count,end_to_end_exact=3,seconds=time.perf_counter()-t)
    (HERE/'ANALYSIS_ACCELERATION_CHECK.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))
if __name__=='__main__':main()

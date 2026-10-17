"""Native/reference comparisons on full initial populations, without fitting."""
from pathlib import Path
import sys,json,time
import numpy as np
from global_metrics import GlobalObjectives
from native_backend import native_available
assert native_available,'Formal server checks require the compiled backend'
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'case1'))
from legacy_objectives import impedance as circuit1,TARGET,LOWER as lower1,UPPER as upper1
sys.path.insert(0,str(root/'case2'))
from case2 import impedance as circuit2,load_target,LOWER as lower2,UPPER as upper2

def reference(obj,p):
    result=[]
    for start in range(0,len(p),32):
        delta=p[start:start+32,obj.j]-p[start:start+32,obj.i]
        direction=delta/np.linalg.norm(delta,axis=-1)[...,None]
        result.extend(np.sum(np.sum((direction-obj.directions)**2,axis=-1)*obj.pair_weights,axis=-1)/obj.pair_weights.sum())
    return np.asarray(result)

records=[]
for case in ['case1','case2']:
    if case=='case1':f=np.geomspace(100,1e6,180);target=circuit1(f,TARGET)[0];circuit=circuit1;lo,hi=lower1,upper1;pop=200
    else:
        cfg=json.loads((root/case/'config.json').read_text());f,target=load_target({**cfg['target_data'],'file':str(root/case/cfg['target_data']['file'])})
        circuit=circuit2;lo,hi=lower2,upper2;pop=150
    obj=GlobalObjectives(f,aux=('orientation_logpolar',),target_z=target)
    z=circuit(f,10**np.random.default_rng(0).uniform(lo,hi,(pop,len(lo))));p=obj.path(z)
    begin=time.perf_counter();old=reference(obj,p);old_time=time.perf_counter()-begin
    begin=time.perf_counter();new=obj.orientation_loss(p);new_time=time.perf_counter()-begin
    np.testing.assert_array_equal(old,new)
    np.testing.assert_array_equal(reference(obj,obj.path(target[None,:])),obj.orientation_loss(obj.path(target[None,:])))
    record=dict(case=case,population=pop,frequency_points=len(f),bitwise_equal=True,
        max_abs_difference=float(abs(old-new).max()),reference_seconds=old_time,native_seconds=new_time)
    records.append(record);print(json.dumps(record),flush=True)
(root/'native_equivalence.json').write_text(json.dumps(dict(status='passed',checks=records),indent=2))

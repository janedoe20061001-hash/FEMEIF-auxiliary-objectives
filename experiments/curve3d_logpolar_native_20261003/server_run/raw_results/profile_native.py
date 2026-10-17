"""Compare compiled arithmetic with the frozen NumPy formula, no fitting."""
from pathlib import Path
import sys,ctypes,time,json
import numpy as np
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root/'case1'))
from objectives import Objectives,impedance
f=np.geomspace(100,1e6,180);obj=Objectives(f,aux=('orientation_logpolar',))
x=np.load(root/'case1/runs/orientation_logpolar/seed_000/initial_log_parameters.npy')
p=obj.path(impedance(f,10**x))
lib=ctypes.CDLL(str(root/'orientation_native.so'));fn=lib.direction_terms
ptr=ctypes.c_void_p;integer=ctypes.c_int64
fn.argtypes=[ptr,integer,integer,integer,ptr,ptr,ptr,ptr,ptr];fn.restype=None

def compiled(p):
    result=[]
    for start in range(0,len(p),32):
        a=np.ascontiguousarray(p[start:start+32]);terms=np.empty((len(a),len(obj.i)),order='F')
        fn(a.ctypes.data,len(a),a.shape[1],len(obj.i),obj.i.ctypes.data,obj.j.ctypes.data,
           obj.directions.ctypes.data,obj.pair_weights.ctypes.data,terms.ctypes.data)
        result.extend(np.sum(terms,axis=-1)/obj.pair_weights.sum())
    return np.asarray(result)

start=time.perf_counter();old=obj.orientation_loss(p);oldtime=time.perf_counter()-start
start=time.perf_counter();new=compiled(p);newtime=time.perf_counter()-start
np.testing.assert_allclose(old,new,rtol=1e-14,atol=1e-14)
np.testing.assert_array_equal(old,new)
print(json.dumps(dict(old_seconds=oldtime,new_seconds=newtime,speedup=oldtime/newtime,
     max_abs_difference=float(abs(old-new).max()),bitwise_equal=bool(np.array_equal(old,new)))),flush=True)

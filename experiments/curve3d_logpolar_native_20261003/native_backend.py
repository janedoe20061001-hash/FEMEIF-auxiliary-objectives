"""Compiled direct arithmetic; identical formula, all original pairs retained."""
from pathlib import Path
import ctypes
import numpy as np

binary=Path(__file__).resolve().parent/'orientation_native.so'
library=ctypes.CDLL(str(binary)) if binary.exists() else None
native_available=library is not None
if native_available:
    function=library.direction_terms
    function.argtypes=[ctypes.c_void_p,ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,
                       ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
    function.restype=None

def weighted_terms(p,ii,jj,target,weights):
    p=np.ascontiguousarray(p,dtype=np.float64)
    if native_available:
        output=np.empty((len(p),len(ii)),dtype=np.float64,order='F')
        function(p.ctypes.data,len(p),p.shape[1],len(ii),ii.ctypes.data,jj.ctypes.data,
                 target.ctypes.data,weights.ctypes.data,output.ctypes.data)
        return output
    # Local Windows software checks only. Formal server startup requires native.
    delta=p[:,jj]-p[:,ii]
    direction=delta/np.linalg.norm(delta,axis=-1)[...,None]
    return np.sum((direction-target)**2,axis=-1)*weights

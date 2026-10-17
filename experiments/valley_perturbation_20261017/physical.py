"""Physical models and fixed diagnostics; no objective or optimizer.

Module name is intentionally distinct from the frozen dependency common.py.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,copy,time,datetime
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/arx_light_20261015'))
from core import Circuit,dump,sha,checked_nodal,target
from circuits import leaves
from reference_features import mode_points
sys.path.insert(0,str(ROOT/'experiments/variation_transport_20261017'))
from metric import abs_linear_integral,csvout,nodes
sys.path.insert(0,str(ROOT/'experiments/valley_process_20261017'))
from diagnose import roots
from scipy.optimize import brentq
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'output/valley_perturbation_20261017'

class Physical(Circuit):
    """Also permits R-only/R+L controls, which need no artificial L/C leaves."""
    def __init__(self,record):
        self.ident=record['id'];self.title=self.ident;self.tree=copy.deepcopy(record['tree']);self.values=dict(record['values']);self.names=leaves(self.tree);self.p=np.array([self.values[n] for n in self.names])
        assert len(set(self.names))==len(self.names) and set(self.values)==set(self.names)
        assert np.all(np.isfinite(self.p)) and np.all(self.p>0)

def record(c):return dict(id=c.ident,tree=c.tree,values=dict(zip(c.names,c.p.tolist())))
def rx(c,x):
    z,d=c.response(10**np.asarray(x));return z,d*np.log(10)
def edges(c,ref,lo,hi):
    v=np.unique(np.r_[np.linspace(lo,hi,9),mode_points(c.modes()+ref.modes(),lo,hi)])
    return v
def grid(c,ref,lo,hi,n):
    e=edges(c,ref,lo,hi);return np.unique((e[:-1,None]+np.diff(e)[:,None]*np.linspace(0,1,n+1)).ravel())
def scope_stats(c,ref,x,scale):
    z,d=rx(c,x);r,dr=rx(ref,x);ds=abs(np.diff(z));rs=abs(np.diff(r));L=float(sum(ds));Lt=float(sum(rs));span=x[-1]-x[0];small=1e-12*scale
    cdf=np.r_[0,np.cumsum(ds)]/L if L>small else None;targetcdf=np.r_[0,np.cumsum(rs)]/Lt
    quantile=np.interp([.1,.5,.9],cdf,x) if cdf is not None else None
    rquantile=np.interp([.1,.5,.9],targetcdf,x)
    rr=roots(SimpleNamespace(c=c),c.p,x);boundary=[q for q in rr if min(abs(q['x']-x[0]),abs(q['x']-x[-1]))<=1e-9];interior=[q for q in rr if q not in boundary]
    return dict(length=L,target_length=Lt,length_ratio=L/Lt,endpoint_displacement=float(abs(z[-1]-z[0])),endpoint_efficiency=float(abs(z[-1]-z[0])/L) if L>small else None,arc_quantiles=quantile.tolist() if quantile is not None else None,target_arc_quantiles=rquantile.tolist(),arc80_width=float(quantile[2]-quantile[0]) if quantile is not None else None,target_arc80_width=float(rquantile[2]-rquantile[0]),cdf_L1=abs_linear_integral((x-x[0])/span,cdf-targetcdf) if cdf is not None else None,zero_motion=bool(L<=small),radial_extrema=interior,boundary_extrema=boundary),dict(x=x,z=z,dz_dx=d,target_z=r,target_dz_dx=dr,cdf=cdf,target_cdf=targetcdf)

def gauss_length(c,ref,lo,hi,n,refinement):
    e=edges(c,ref,lo,hi)
    for _ in range(refinement):e=np.unique(np.r_[e,(e[1:]+e[:-1])/2])
    t,w=nodes(e,n);return float(w@abs(rx(c,t)[1]))

def close(a,b):return abs(a-b)<=1e-4+1e-3*abs(b)

def analytic_identity(c,ref,item,targetrec):
    fam=item['family'];meta=item['meta'];center=targetrec['center'];width=targetrec['core'][1]-targetrec['core'][0];xx=np.unique(np.r_[np.linspace(*targetrec['window'],101),np.linspace(2,6,101),center]);r=rx(ref,xx)[0]
    if fam=='baseline':actual=rx(c,xx)[0];expected=r
    elif fam=='shift':actual=rx(c,xx+meta['shift_widths']*width)[0];expected=r
    elif fam=='offset':actual=rx(c,xx)[0];expected=r+meta['delta_R']
    elif fam=='Q' and targetrec['kind']=='notch':
        xnew=center+np.arcsinh(np.sinh(np.log(10)*(xx-center))/meta['factor'])/np.log(10);actual=rx(c,xnew)[0];expected=r
    elif fam=='Q' and targetrec['kind']=='series':actual=rx(c,xx)[0];expected=r.real+1j*meta['factor']*r.imag
    elif fam=='extra':
        f=10**xx;Q=meta['extra_Q'];f0=meta['extra_f0'];R=meta['extra_R'];branch=R/(1+1j*Q*(f/f0-f0/f));actual=rx(c,xx)[0];expected=r+branch
    elif fam=='combined':
        k=meta['Q_factor'];delta=meta['shift_widths']*width
        if targetrec['kind']=='notch':xnew=center+np.arcsinh(np.sinh(np.log(10)*(xx-center))/k)/np.log(10)+delta;expected=r+meta['delta_R']
        else:xnew=xx+delta;expected=r.real+meta['delta_R']+1j*k*r.imag
        actual=rx(c,xnew)[0]
    elif fam=='background':
        p=c.values;expected=np.full(len(xx),p['Rb'],complex)
        if 'Lb' in p:expected+=2j*np.pi*10**xx*p['Lb']
        actual=rx(c,xx)[0]
    else:return dict(applicable=False,reason='Coupled physical intervention; no pure geometric identity claimed')
    difference=abs(actual-expected);ok=np.all(difference<=1e-8+1e-7*abs(expected))
    return dict(applicable=True,passed=bool(ok),max_absolute=float(max(difference)),max_relative=float(max(difference/np.maximum(abs(expected),1e-30))))

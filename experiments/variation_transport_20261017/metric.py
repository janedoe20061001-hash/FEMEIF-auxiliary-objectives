"""Frozen positive/negative projected variation, with frequency labels."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/geometry_followup_20261016'))
from common import target,control_target,Reference,Prefix,nodes,sha,dump,csvout
from scipy.optimize import brentq
OUT=ROOT/'output/variation_transport_20261017'
ANGLES=np.arange(4)*np.pi/4
DIRECTIONS=np.exp(-1j*ANGLES)
NAMES=['JM','JC','JV']

def abs_linear_integral(t,a):
    """Exact integral of absolute value of a piecewise linear function."""
    left,right=a[:-1],a[1:];v=(abs(left)+abs(right))/2
    cross=left*right<0
    v[cross]=(left[cross]**2+right[cross]**2)/(2*(abs(left[cross])+abs(right[cross])))
    return float(np.diff(t)@v)

def poly_features(t,z):
    u=np.real(DIRECTIONS[:,None]*z)
    du=np.diff(u,axis=1)
    # Along each linear segment, Jordan positive/negative variation is linear.
    return np.concatenate([np.column_stack([np.zeros(4),np.cumsum(np.maximum(s*du,0),axis=1)]) for s in [1,-1]],axis=0)

def polygon_loss(t,z,r,den):
    a,b=poly_features(t,z),poly_features(t,r);delta=a-b
    mass=abs(delta[:,-1]);location=np.array([abs_linear_integral(t,d) for d in delta])
    d=np.tile(den,2);jm=float(np.mean(mass/d));jc=float(np.mean(location/d))
    return dict(JM=jm,JC=jc,JV=(jm+jc)/2)

def analytic_checks():
    t=np.array([0.,.2,.3,.4,.6,.7,.8,1.]);z=np.maximum(1-abs(t-.3)/.1,0).astype(complex)
    np.testing.assert_allclose(poly_features(t,z)[:,-1],poly_features(t,z+3+4j)[:,-1],atol=1e-14)
    den=np.ones(4)
    for shift in [.1,.2,.3]:
        q=np.unique(np.r_[0,1,.2,.3,.4,.2+shift,.3+shift,.4+shift])
        a=np.maximum(1-abs(q-.3)/.1,0).astype(complex);b=np.maximum(1-abs(q-(.3+shift))/.1,0).astype(complex)
        v=polygon_loss(q,a,b,den)
        expected=shift*np.mean(abs(np.real(DIRECTIONS)))
        np.testing.assert_allclose(v['JC'],expected,atol=1e-13)
        assert v['JM']<1e-12
    np.testing.assert_allclose(abs_linear_integral(np.array([0.,1.]),np.array([-1.,1.])),.5)
    return dict(piecewise_linear_absolute_integral=True,constant_translation_invariance=True,equal_mass_compact_translation=True)

class Variation:
    def __init__(self,tid):
        self.tid=tid;self.data=control_target() if tid=='controlled' else target(tid)
        self.g=Reference(self.data);self.c=self.g.c;self.windows=self.g.windows;self.fast=None
        if os.name!='nt' and tid!='controlled':
            from features import Features
            self.fast=Features(self.data)
        self.target_cache={};self.den={};self.scale_audit={}
    def response(self,w,t,p):
        lo,hi=w['x'][[0,-1]];f=10**(lo+(hi-lo)*np.asarray(t))
        z,dz=self.fast.response(f,p) if self.fast is not None else self.c.response(f,p)
        return z/w['scale'],dz*np.log(10)*(hi-lo)/w['scale']
    def window(self,label):return next(w for w in self.windows if w['block']['label']==label)
    def polygrid(self,w,p,n):
        e=self.g.panel_edges(w,self.g.modes(p),0)
        return np.unique((e[:-1,None]+np.diff(e)[:,None]*np.linspace(0,1,n+1)).ravel())
    def derivative_edges(self,w,p,ref=0):
        e=self.g.panel_edges(w,self.g.modes(p),ref)
        sample=np.unique((e[:-1,None]+np.diff(e)[:,None]*np.linspace(0,1,9)).ravel())
        roots=[]
        for pp in [p,self.c.p]:
            d=np.real(DIRECTIONS[:,None]*self.response(w,sample,pp)[1])
            for j in range(4):
                for k in np.where(d[j,:-1]*d[j,1:]<0)[0]:
                    roots.append(brentq(lambda t:float(np.real(DIRECTIONS[j]*self.response(w,np.array([t]),pp)[1][0])),sample[k],sample[k+1],xtol=5e-15))
        return np.unique(np.r_[e,roots])
    def gaussian(self,p,label,n=32,ref=0,scales_only=False):
        w=self.window(label);e=self.derivative_edges(w,p,ref);t,wt=nodes(e,n)
        dz=self.response(w,t,p)[1];dr=self.response(w,t,self.c.p)[1]
        features=[];masses=[]
        for d in [dz,dr]:
            v=np.real(DIRECTIONS[:,None]*d)
            pref=[Prefix(e,np.maximum(sign*v[j],0),n) for sign in [1,-1] for j in range(4)]
            features.append(np.array([q(t) for q in pref]));masses.append(np.array([q(np.array([1.]))[0] for q in pref]))
        if scales_only:return masses[1][:4]+masses[1][4:]
        delta=features[0]-features[1];den=np.tile(self.scales(label),2)
        jm=float(np.mean(abs(masses[0]-masses[1])/den));jc=float(np.mean((abs(delta)@wt)/den))
        return dict(JM=jm,JC=jc,JV=(jm+jc)/2)
    def scales(self,label):
        if label not in self.den:
            w=self.window(label);t=self.polygrid(w,self.c.p,512);r=self.response(w,t,self.c.p)[0]
            m=poly_features(t,r)[:,-1];total=m[:4]+m[4:]
            den=np.maximum(total,max(.1*np.sqrt(np.mean(total**2)),1e-4))
            self.den[label]=den
            independent=self.gaussian(self.c.p,label,32,0,True)
            self.scale_audit[label]=dict(total_polygon=total,total_derivative=independent,passed=bool(np.all(abs(total-independent)<=1e-4+1e-3*abs(independent))),den=den)
        return self.den[label]
    def polygon(self,p,label,n):
        w=self.window(label);t=self.polygrid(w,p,n);z=self.response(w,t,p)[0];r=self.response(w,t,self.c.p)[0]
        return polygon_loss(t,z,r,self.scales(label))

def agree(a,b):return all(abs(a[k]-b[k])<=1e-4+1e-3*abs(b[k]) for k in NAMES)

"""New, isolated experiment; original source files remain unchanged."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/arx_light_20261015'))
from core import target,target_ids,CFG,BASECFG,dump,sha,diag,anchors
from reference_features import Features as Reference,mode_points
sys.path.insert(0,str(ROOT/'experiments/arx_structure_diagnosis_20261016'))
from local_geometry import Prefix,nodes,shifted
OUT=ROOT/'output/geometry_followup_20261016'
OLD=ROOT/'output/arx_light_20261015';DATA=OLD/'retrieved/output/arx_light_20261015' if (OLD/'retrieved').exists() else OLD
CONFIG=json.loads((HERE/'config.json').read_text(encoding='utf-8'))

def csvout(path,rows):
    import csv
    if not rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def root_transform(r,delta=.01):
    # Stable algebraic form avoids sqrt cancellation for small r.
    r=np.asarray(r);return r/(np.sqrt(abs(r)+delta)+np.sqrt(delta))

def converged(a,b):return all(abs(a[k]-b[k])<=1e-4+1e-3*abs(b[k]) for k in a)

class ControlCircuit:
    def __init__(self):
        self.R=3000.;self.f0=25000.;self.Q=2000.;self.w0=2*np.pi*self.f0
        self.p=self.params(1,0,1);self.ident='controlled_parallel_RLC'
        # Exact half-power width in log frequency; not a 3.000-dB width.
        r=(np.sqrt(4+1/self.Q**2)+1/self.Q)/2
        self.width=2*np.log10(r)
    def params(self,amplitude,offset,qratio):
        # width is only defined after the truth parameters are initially made.
        w=2*np.pi*self.f0*10**(offset*getattr(self,'width',0.))
        r=self.R*amplitude;q=self.Q*qratio
        return np.array([r,r/(w*q),q/(w*r)])
    def response(self,f,p=None):
        p=self.p if p is None else np.asarray(p)
        f=np.asarray(f);omega=2*np.pi*f;z=np.zeros(f.shape,complex);d=np.zeros_like(z)
        for r,l,c in p.reshape(-1,3):
            b=omega*c-1/(omega*l);v=1/(1/r+1j*b)
            z+=v;d+=-1j*(omega*c+1/(omega*l))*v*v
        return z,d
    def modes(self,p=None):
        v=np.asarray(self.p if p is None else p).reshape(-1,3)
        return [(float(1/(2*np.pi*np.sqrt(l*c))),float(6/(r*np.sqrt(c/l))/np.log(10))) for r,l,c in v]

def control_target():
    from types import SimpleNamespace
    c=ControlCircuit();center=np.log10(c.f0);lo=center-8*c.width;hi=center+8*c.width
    blocks=[dict(label='gap_left',lo=2.,hi=lo,event=False),dict(label='P_control',lo=lo,hi=hi,event=True),dict(label='gap_right',lo=hi,hi=6.,event=False)]
    rec=dict(events=[dict(x=center,kind='peak')],blocks=blocks)
    q=SimpleNamespace(reference_grid=lambda *a:np.unique(np.r_[np.linspace(2,6,2049),mode_points(c.modes(),2,6)]))
    return dict(id=c.ident,circuit=c,record=rec,kind='controlled',quadrature=q)

class Geometry:
    def __init__(self,data):
        self.data=data;self.g=Reference(data);self.c=self.g.c;self.windows=self.g.windows;self.steps=CONFIG['half_steps']
        self.fast=None
        if os.name!='nt' and data['kind']!='controlled':
            from features import Features
            self.fast=Features(data)
        self.lengths=[];self.bden=None
        for w in self.windows:
            e=self.g.panel_edges(w,[],2);t,mass=nodes(e,64);z,dz=self.response(w,t,self.c.p)
            self.lengths.append(Prefix(e,abs(dz),64))
        b2=np.array([p(np.array([1.]))[0]**2 for p in self.lengths]);self.new_area_den=np.maximum(b2,max(.1*np.sqrt(np.mean(b2**2)),1e-4))
        self.local_floors=[];rmsB=[]
        for h in self.steps:
            bb=[];ll=[]
            for j,w in enumerate(self.windows):
                e=shifted(self.g.panel_edges(w,[],2),h);t,m=nodes(e,64);m/=1-2*h
                z,_=self.response(w,np.r_[t-h,t,t+h],self.c.p);zm,zc,zp=np.split(z,3)
                bb.append(float(m@abs(zc-(zm+zp)/2)**2))
                length=self.lengths[j](t+h)-self.lengths[j](t-h);assert min(length)>-1e-10
                ll.append(float(m@length**4))
            self.local_floors.append(max(.1*np.sqrt(np.mean(ll)),1e-4))
            b=np.sqrt(bb);rmsB.append(np.maximum(b,max(.1*np.sqrt(np.mean(b**2)),1e-4)))
        self.bden=np.array(rmsB).T
    def response(self,w,t,p):
        lo,hi=w['x'][[0,-1]];f=10**(lo+(hi-lo)*np.asarray(t))
        z,dz=self.fast.response(f,p) if self.fast is not None else self.c.response(f,p)
        return z/w['scale'],dz*np.log(10)*(hi-lo)/w['scale']
    def local_den(self,j,hi,t):
        h=self.steps[hi];v=self.lengths[j](t+h)-self.lengths[j](t-h)
        assert min(v)>=-1e-10
        return np.maximum(v*v,self.local_floors[hi])
    def block(self,p,label,n=32,refinement=1):
        j=next(i for i,w in enumerate(self.windows) if w['block']['label']==label);w=self.windows[j]
        edges=self.g.panel_edges(w,self.g.modes(p),refinement);t,m=nodes(edges,n)
        z,dz=self.response(w,t,p);r,dr=self.response(w,t,self.c.p)
        z0=self.response(w,np.array([0.]),p)[0][0];r0=self.response(w,np.array([0.]),self.c.p)[0][0]
        az=Prefix(edges,.5*np.imag((z-z0).conj()*dz),n);ar=Prefix(edges,.5*np.imag((r-r0).conj()*dr),n)
        area=float(m@(az(t)-ar(t))**2);point=float(m@abs(z-r)**2)
        out=dict(J0=point,J0_W=point*(w['area_den'][2]/self.new_area_den[j])**2,ARX_L=area/w['area_den'][2]**2,ARX_S_L=area/self.new_area_den[j]**2)
        values={k:[] for k in ['Local_ARX','Root','Root_delta_0p001','Root_delta_0p1','B','D','H','J0_local_weighted']}
        for hi,h in enumerate(self.steps):
            center,mass=nodes(shifted(edges,h),n);mass/=1-2*h
            zz=self.response(w,np.r_[center-h,center,center+h],p)[0];rr=self.response(w,np.r_[center-h,center,center+h],self.c.p)[0]
            zm,zc,zp=np.split(zz,3);rm,rc,rp=np.split(rr,3)
            ac=az(center+h)-az(center-h)-.5*np.imag((zm-z0).conj()*(zp-zm))
            at=ar(center+h)-ar(center-h)-.5*np.imag((rm-r0).conj()*(rp-rm))
            den=self.local_den(j,hi,center);ac/=den;at/=den
            values['Local_ARX'].append(float(mass@(ac-at)**2))
            for name,delta in [('Root',.01),('Root_delta_0p001',.001),('Root_delta_0p1',.1)]:values[name].append(float(mass@(root_transform(ac,delta)-root_transform(at,delta))**2))
            values['B'].append(float(mass@abs(zc-(zm+zp)/2-rc+(rm+rp)/2)**2)/self.bden[j,hi]**2)
            dm,dp=zc-zm,zp-zc;tm,tp=rc-rm,rp-rc
            values['D'].append(float(mass@((abs(dm-tm)**2+abs(dp-tp)**2)/2))/w['increment_den'][hi]**2)
            values['H'].append(float(mass@abs(dm.conj()*dp-tm.conj()*tp)**2)/w['product_den'][hi]**2)
            values['J0_local_weighted'].append(float(mass@(abs(zc-rc)**2/den**2)))
        out.update({k:float(np.mean(v)) for k,v in values.items()});return out
    def polygon(self,p,label,subdivisions=128):
        # Independent direct segment sweep, not the Gaussian prefix implementation.
        j=next(i for i,w in enumerate(self.windows) if w['block']['label']==label);w=self.windows[j]
        edges=self.g.panel_edges(w,self.g.modes(p),0)
        t=np.unique((edges[:-1,None]+np.diff(edges)[:,None]*np.linspace(0,1,subdivisions+1)).ravel())
        z=self.response(w,t,p)[0];r=self.response(w,t,self.c.p)[0]
        def pref(v):return np.r_[0.,np.cumsum(.5*np.imag((v[:-1]-v[0]).conj()*np.diff(v)))]
        az,ar=pref(z),pref(r);loss={k:[] for k in ['Root','B']}
        def interp(q,v):return np.interp(q,t,v.real)+1j*np.interp(q,t,v.imag)
        for hi,h in enumerate(self.steps):
            q=np.unique(np.r_[t,t-h,t+h,h,1-h]);q=q[(q>=h)&(q<=1-h)]
            zm,zc,zp=[interp(q+shift,z) for shift in [-h,0,h]];rm,rc,rp=[interp(q+shift,r) for shift in [-h,0,h]]
            ac=np.interp(q+h,t,az)-np.interp(q-h,t,az)-.5*np.imag((zm-z[0]).conj()*(zp-zm))
            at=np.interp(q+h,t,ar)-np.interp(q-h,t,ar)-.5*np.imag((rm-r[0]).conj()*(rp-rm))
            den=self.local_den(j,hi,q)
            loss['Root'].append(float(np.trapezoid((root_transform(ac/den)-root_transform(at/den))**2,x=q)/(1-2*h)))
            loss['B'].append(float(np.trapezoid(abs(zc-(zm+zp)/2-rc+(rm+rp)/2)**2,x=q)/(1-2*h)/self.bden[j,hi]**2))
        return {k:float(np.mean(v)) for k,v in loss.items()}

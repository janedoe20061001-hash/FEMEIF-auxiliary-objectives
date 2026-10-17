"""Frequency-labelled complex increments, Hermitian products and signed areas."""
from core import np,CFG
from scipy.signal import find_peaks
import ctypes
from pathlib import Path

def average(y,t):
    return np.trapezoid(y,x=t,axis=-1)/(t[-1]-t[0])

def areas(t,z):
    z=np.atleast_2d(z)
    q=np.stack([np.broadcast_to(t,z.shape),z.real,z.imag],axis=-1)
    q=q-q[:,:1,:]
    out=[]
    for i,j in ((0,1),(0,2),(1,2)):
        cross=(q[:,:-1,i]*q[:,1:,j]-q[:,:-1,j]*q[:,1:,i])/2
        out.append(np.column_stack([np.zeros(len(z)),np.cumsum(cross,axis=1)]))
    return np.stack(out,axis=1)

def frechet_python(t,z,ref,eta=.25):
    z=np.atleast_2d(z);n=len(t);prev=np.full((n,len(z)),np.inf)
    for i in range(n):
        cur=np.full_like(prev,np.inf)
        for j in range(n):
            if abs(t[i]-t[j])>eta+1e-12:continue
            d=(t[i]-t[j])**2+abs(z[:,i]-ref[j])**2
            if i==j==0:cur[j]=d
            else:
                v=prev[j].copy()
                if j:v=np.minimum(v,cur[j-1])
                if i and j:v=np.minimum(v,prev[j-1])
                cur[j]=np.maximum(v,d)
        prev=cur
    return np.sqrt(prev[-1])

def frechet(t,z,ref):
    libpath=Path(__file__).with_name('frechet.so')
    if not libpath.exists():return frechet_python(t,z,ref)
    if not hasattr(frechet,'lib'):
        frechet.lib=ctypes.CDLL(str(libpath))
        frechet.lib.batch.argtypes=[ctypes.c_int,ctypes.c_int,*([ctypes.c_void_p]*4),ctypes.c_double]
        frechet.lib.batch.restype=ctypes.c_int
    t=np.ascontiguousarray(t);z=np.ascontiguousarray(np.atleast_2d(z),dtype=np.complex128)
    ref=np.ascontiguousarray(ref,dtype=np.complex128);out=np.empty(len(z))
    code=frechet.lib.batch(len(z),len(t),t.ctypes.data,z.ctypes.data,ref.ctypes.data,out.ctypes.data,.25)
    if code:raise RuntimeError('Native Frechet failed')
    return out

class Features:
    def __init__(self,data):
        self.data=data;self.c=data['circuit'];self.queries=0;self.windows=[]
        self.measured=data['kind']=='measured';cfg=CFG['features']
        for b in data['record']['blocks']:
            if self.measured:
                x=data['quadrature'].x[b['start']:b['end']]
                zz=data['quadrature'].z[b['start']:b['end']]
                steps=cfg['measured_index_steps']
            else:
                x=np.linspace(b['lo'],b['hi'],cfg['points_per_theory_block'])
                zz,_=self.c.response(10**x)
                steps=[round(h*(len(x)-1)) for h in cfg['half_steps_normalized_block']]
            t=(x-x[0])/(x[-1]-x[0]);scale=max(float(max(abs(zz))),cfg['ohm_scale_floor']);z=zz/scale
            w=dict(block=b,x=x,t=t,z=z,scale=scale,steps=steps)
            a=areas(t,z)[0];w['area']=a
            w['area_den']=np.maximum(average(abs(a),t),cfg['area_scale_floor'])
            w['increment_den']=[];w['product_den']=[]
            for k in steps:
                dm=z[k:-k]-z[:-2*k];dp=z[2*k:]-z[k:-k];tt=t[k:-k]
                w['increment_den'].append(max(float(average((abs(dm)+abs(dp))/2,tt)),cfg['increment_scale_floor']))
                w['product_den'].append(max(float(average(abs(dm.conj()*dp),tt)),cfg['product_scale_floor']))
            self.windows.append(w)
        self.event_x=np.array([e['x'] for e in sorted(data['record']['events'],key=lambda e:e['x'])])
        self.event_kind=[e['kind'] for e in sorted(data['record']['events'],key=lambda e:e['x'])]
        if self.measured:self.ptc_x=data['quadrature'].x
        else:
            q=data['quadrature'];self.ptc_x=q.reference_grid(self.c.p,4)

    def components(self,logp,names=('D','H','A'),level=1):
        p=10**np.atleast_2d(logp);values={n:[] for n in names};details=[]
        for w in self.windows:
            if self.measured or level==1:x=w['x'];t=w['t'];ref=w['z'];steps=w['steps']
            else:
                x=np.linspace(w['x'][0],w['x'][-1],128*level+1);t=np.linspace(0,1,len(x))
                ref,_=self.c.response(10**x);ref=ref/w['scale'];steps=[k*level for k in w['steps']]
            z,_=self.c.response(10**x,p);z=z/w['scale'];self.queries+=z.size
            scores={}
            if 'D' in names or 'H' in names:
                dd=[];hh=[]
                for j,k in enumerate(steps):
                    dm=z[:,k:-k]-z[:,:-2*k];dp=z[:,2*k:]-z[:,k:-k]
                    rm=ref[k:-k]-ref[:-2*k];rp=ref[2*k:]-ref[k:-k];tt=t[k:-k]
                    if 'D' in names:dd.append(average((abs(dm-rm)**2+abs(dp-rp)**2)/2,tt)/w['increment_den'][j]**2)
                    if 'H' in names:hh.append(average(abs(dm.conj()*dp-rm.conj()*rp)**2,tt)/w['product_den'][j]**2)
                if dd:scores['D']=np.mean(dd,axis=0)
                if hh:scores['H']=np.mean(hh,axis=0)
            if 'A' in names:
                aa=areas(t,z);ra=w['area'] if level==1 or self.measured else areas(t,ref)[0]
                scores['A']=np.mean(average((aa-ra)**2,t)/w['area_den'][None,:]**2,axis=1)
            if 'C_ref' in names:scores['C_ref']=frechet(t,z,ref)
            for n,v in scores.items():values[n].append(v)
            details.append(dict(label=w['block']['label'],scores={n:v.tolist() for n,v in scores.items()},
                endpoint_error=(abs(z[:,[0,-1]]-ref[[0,-1]])**2).mean(axis=1).tolist()))
        return {n:np.mean(v,axis=0) for n,v in values.items()},details

    def ptc(self,logp,level=1):
        p=10**np.atleast_2d(logp)
        if not len(self.event_x):return np.zeros(len(p))
        x=self.ptc_x if self.measured or level==1 else self.data['quadrature'].reference_grid(p[0],4*level)
        z,_=self.c.response(10**x,p);self.queries+=z.size;out=[]
        zones=np.r_[x[0],(self.event_x[:-1]+self.event_x[1:])/2,x[-1]]
        for row in z:
            amp=20*np.log10(abs(row));events={kind:x[find_peaks(sign*amp,prominence=1.)[0]] for kind,sign in [('peak',1),('valley',-1)]}
            cost=[]
            for i,(e,k) in enumerate(zip(self.event_x,self.event_kind)):
                fx=events[k];fx=fx[(fx>=zones[i])&(fx<=zones[i+1])]
                cost.append(float(np.sum(.41*np.expm1(abs(fx-e)/(x[-1]-x[0])))) if len(fx) else .41)
            out.append(np.mean(cost))
        return np.array(out)

    def score(self,logp,method,level=1):
        if method=='PTC_ref':return self.ptc(logp,level)
        names=method.split('_') if method!='C_ref' else ['C_ref']
        v,_=self.components(logp,names,level)
        return np.mean([v[n] for n in names],axis=0)

    def description(self):
        return [dict(label=w['block']['label'],frequency_hz=(10**w['x'][[0,-1]]).tolist(),points=len(w['x']),scale_ohm=w['scale'],
            increment_den=w['increment_den'],product_den=w['product_den'],area_den=w['area_den'].tolist()) for w in self.windows]

    def anchored_geometry(self,logp,level=8):
        """Independent nonuniform quadrature with candidate anchors; same physical half-steps."""
        if self.measured:return self.components(logp,level=level)[0]
        from core import anchors
        p=10**np.asarray(logp);scores={k:[] for k in ('D','H','A')}
        for w in self.windows:
            lo,hi=w['x'][[0,-1]];span=hi-lo
            extra=anchors(self.c,p,lo,hi,24*level)
            tt=np.unique(np.r_[np.linspace(0,1,128*level+1),(extra[(extra>=lo)&(extra<=hi)]-lo)/span])
            z,_=self.c.response(10**(lo+span*tt),p);ref,_=self.c.response(10**(lo+span*tt));z/=w['scale'];ref/=w['scale']
            aa=areas(tt,z)[0];ra=areas(tt,ref)[0]
            scores['A'].append(np.mean(average((aa-ra)**2,tt)/w['area_den']**2))
            dd=[];hh=[]
            for j,h in enumerate(CFG['features']['half_steps_normalized_block']):
                center=np.unique(np.r_[tt,tt-h,tt+h,h,1-h]);center=center[(center>=h)&(center<=1-h)]
                triplets=np.r_[center-h,center,center+h];freq=10**(lo+span*triplets)
                zz,_=self.c.response(freq,p);rr,_=self.c.response(freq)
                zm,z0,zp=np.split(zz/w['scale'],3);rm,r0,rp=np.split(rr/w['scale'],3)
                dm=z0-zm;dp=zp-z0;drm=r0-rm;drp=rp-r0
                dd.append(float(average((abs(dm-drm)**2+abs(dp-drp)**2)/2,center)/w['increment_den'][j]**2))
                hh.append(float(average(abs(dm.conj()*dp-drm.conj()*drp)**2,center)/w['product_den'][j]**2))
            scores['D'].append(np.mean(dd));scores['H'].append(np.mean(hh))
        return {k:np.array([np.mean(v)]) for k,v in scores.items()}

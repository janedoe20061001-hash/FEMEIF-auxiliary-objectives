"""Case1 quadrature: adapt values; preserve each original log-frequency cell mass."""
import numpy as np
from pathlib import Path
import ctypes

def spectra(f,parameters):
    p=np.atleast_2d(parameters).reshape(-1,6,3);s=2j*np.pi*np.asarray(f)[None,:]
    r=p[:,:,0,None];l=p[:,:,1,None];c=p[:,:,2,None]
    return np.sum(r*s[:,None,:]*l/(r+s[:,None,:]*l+r*s[:,None,:]**2*l*c),axis=1)

class Adaptive:
    def __init__(self,record,base):
        self.r=record;self.p0=np.asarray(record['target_parameters'])
        ori=np.log10(record['frequency_hz']);self.edges=np.r_[ori[0],(ori[:-1]+ori[1:])/2,ori[-1]]
        self.base=np.unique(np.r_[np.log10(base),self.edges,ori]);self.nold=len(ori)
        self.queries=0;self.evaluations=0;self.max_points=0;self.max_layers=0

    def query(self,x,p,rows):
        s=2j*np.pi*10**x;z=np.zeros(len(x),complex);t=np.zeros(len(x),complex)
        for j in range(6):
            r=p[rows,3*j];l=p[rows,3*j+1];c=p[rows,3*j+2]
            z+=r*s*l/(r+s*l+r*s*s*l*c)
            r,l,c=self.p0[3*j:3*j+3];t+=r*s*l/(r+s*l+r*s*s*l*c)
        if not np.isfinite(z).all() or np.any(abs(z)==0):raise FloatingPointError('Invalid model response')
        return np.asarray([20*np.log10(abs(z)),np.angle(z),20*np.log10(abs(t)),np.angle(t)])

    def score(self,log_parameters):
        lib=Path(__file__).with_name('native.so')
        if lib.exists():return self.score_compiled(log_parameters,lib)
        return self.score_python(log_parameters)

    def score_compiled(self,log_parameters,lib):
        p=np.ascontiguousarray(10**np.atleast_2d(log_parameters));n=len(p);v=p.reshape(n,6,3)
        centers=np.log10(1/(2*np.pi*np.sqrt(v[:,:,1]*v[:,:,2])));width=np.clip(6/(v[:,:,0]*np.sqrt(v[:,:,2]/v[:,:,1]))/np.log(10),1e-12,.5)
        extra=np.clip((centers[:,:,None]+width[:,:,None]*np.asarray([-1.,-.25,0.,.25,1.])).reshape(n,-1),self.edges[0],self.edges[-1])
        grid=np.ascontiguousarray(np.sort(np.column_stack([np.tile(self.base,(n,1)),extra]),axis=1))
        out=np.empty((n,2));counts=np.empty(n,dtype=np.int64);depths=np.empty(n,dtype=np.int64)
        if not hasattr(self,'_native'):
            self._native=ctypes.CDLL(str(lib));self._native.adaptive_batch.argtypes=[ctypes.c_int,ctypes.c_int,*([ctypes.c_void_p]*4),ctypes.c_int,*([ctypes.c_void_p]*3)];self._native.adaptive_batch.restype=ctypes.c_int
        code=self._native.adaptive_batch(n,grid.shape[1],p.ctypes.data,self.p0.ctypes.data,grid.ctypes.data,self.edges.ctypes.data,self.nold,out.ctypes.data,counts.ctypes.data,depths.ctypes.data)
        if code:raise RuntimeError(f'Compiled adaptive evaluation failed: {code}')
        self.queries+=int(counts.sum());self.evaluations+=n;self.max_points=max(self.max_points,int(counts.max()));self.max_layers=max(self.max_layers,int(depths.max()))
        return out

    def score_python(self,log_parameters):
        p=10**np.atleast_2d(log_parameters);n=len(p);v=p.reshape(n,6,3)
        fc=1/(2*np.pi*np.sqrt(v[:,:,1]*v[:,:,2]));q=v[:,:,0]*np.sqrt(v[:,:,2]/v[:,:,1])
        # Known candidate branch centers prevent an extreme-Q branch from hiding
        # between all midpoint queries. This does not penalize any structure.
        centers=np.log10(fc);width=np.clip(6/q/np.log(10),1e-12,.5)
        offsets=np.asarray([-1.,-.25,0.,.25,1.])
        extra=np.clip((centers[:,:,None]+width[:,:,None]*offsets).reshape(n,-1),self.edges[0],self.edges[-1])
        xx=np.sort(np.column_stack([np.tile(self.base,(n,1)),extra]),axis=1)
        rows=np.repeat(np.arange(n),xx.shape[1]);x=xx.ravel()
        keep=np.r_[True,(rows[1:]!=rows[:-1])|(x[1:]!=x[:-1])];rows=rows[keep];x=x[keep]
        y=self.query(x,p,rows);allx=[x];allr=[rows];ally=[y]
        ix=np.flatnonzero(rows[:-1]==rows[1:]);ar=rows[ix];a=x[ix];b=x[ix+1];ya=y[:,ix];yb=y[:,ix+1]
        point_counts=np.bincount(rows,minlength=n);layer=0
        while len(ar):
            layer+=1
            if layer>32:raise RuntimeError('Adaptive depth exhausted; candidate not silently accepted')
            mid=(a+b)/2;ym=self.query(mid,p,ar)
            point_counts+=np.bincount(ar,minlength=n)
            if point_counts.max()>10000:raise RuntimeError('Adaptive point budget exhausted; candidate not silently accepted')
            allx.append(mid);allr.append(ar);ally.append(ym)
            err=abs(ym-(ya+yb)/2)
            fail=(err[0]>.1)|(err[2]>.1)|(err[1]>np.deg2rad(.5))|(err[3]>np.deg2rad(.5))
            olda=a[fail];oldb=b[fail];oldm=mid[fail];left=ya[:,fail];right=yb[:,fail];middle=ym[:,fail]
            ar=np.tile(ar[fail],2);a=np.r_[olda,oldm];b=np.r_[oldm,oldb];ya=np.column_stack([left,middle]);yb=np.column_stack([middle,right])
        x=np.concatenate(allx);rows=np.concatenate(allr);y=np.concatenate(ally,axis=1)
        order=np.lexsort((x,rows));x=x[order];rows=rows[order];y=y[:,order]
        edge=np.flatnonzero(rows[:-1]==rows[1:]);dx=x[edge+1]-x[edge]
        cell=np.clip(np.searchsorted(self.edges,(x[edge]+x[edge+1])/2,side='right')-1,0,self.nold-1)
        mass=dx/np.diff(self.edges)[cell]/self.nold
        total=np.bincount(rows[edge],weights=mass,minlength=n);np.testing.assert_allclose(total,1,rtol=1e-10,atol=1e-10)
        da=(y[0]-y[2])**2;dp=(y[1]-y[3])**2
        f=np.column_stack([np.bincount(rows[edge],weights=mass*(v[edge]+v[edge+1])/2,minlength=n) for v in [da,dp]])
        self.queries+=len(x);self.evaluations+=n;self.max_points=max(self.max_points,int(point_counts.max()));self.max_layers=max(self.max_layers,layer)
        return f

    def reference(self,log_parameters,frequency):
        p=10**np.atleast_2d(log_parameters);x=np.unique(np.r_[np.log10(frequency),self.edges]);z=spectra(10**x,p);t=spectra(10**x,self.p0)[0]
        cell=np.clip(np.searchsorted(self.edges,(x[:-1]+x[1:])/2,side='right')-1,0,self.nold-1)
        mass=np.diff(x)/np.diff(self.edges)[cell]/self.nold;w=np.zeros(len(x));w[:-1]+=mass/2;w[1:]+=mass/2
        return np.column_stack([(20*np.log10(abs(z)/abs(t)))**2@w,(np.angle(z)-np.angle(t))**2@w])

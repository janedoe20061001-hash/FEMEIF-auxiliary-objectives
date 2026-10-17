"""Continuous frequency-labelled local RX area; prototype scoring, not optimization."""
from scan import *
from reference_features import Features as Reference,rule
from legacy_features import areas
from numpy.polynomial.legendre import legvander,legint,legval
from functools import lru_cache

@lru_cache(None)
def inverse(n):
    u,_,_=rule(n)
    return np.linalg.inv(legvander(2*u-1,n-1))

class Prefix:
    def __init__(self,edges,values,n):
        self.e=edges;self.width=np.diff(edges)
        coef=inverse(n)@np.asarray(values).reshape(len(self.width),n).T
        self.anti=legint(coef,axis=0)*self.width[None,:]/2
        self.left=legval(-1.,self.anti)
        full=legval(1.,self.anti)-self.left
        self.prefix=np.r_[0.,np.cumsum(full)]
    def __call__(self,t):
        t=np.asarray(t);j=np.clip(np.searchsorted(self.e,t,side='right')-1,0,len(self.width)-1)
        v=2*(t-self.e[j])/self.width[j]-1
        return self.prefix[j]+legval(v,self.anti[:,j],tensor=False)-self.left[j]

def nodes(edges,n):
    u,weights,_=rule(n);width=np.diff(edges)
    return (edges[:-1,None]+width[:,None]*u).ravel(),(width[:,None]*weights).ravel()

def shifted(edges,h):
    e=np.unique(np.r_[edges,edges-h,edges+h,h,1-h])
    return e[(e>=h)&(e<=1-h)]

class LocalGeometry:
    def __init__(self):
        self.data=target('case1');self.g=Reference(self.data);self.c=self.g.c
        self.windows=self.g.windows;self.steps=(1/32,1/16,1/8);self.target_prefix=[]
        pre=json.loads((DATA/'PREFLIGHT.json').read_text())
        desc=next(v for v in pre['checks'] if v['target']=='case1')['scales']
        self.new_den=np.array([v['new_area_den'] for v in desc])
        self.fast=None
        if os.name!='nt':
            from features import Features
            self.fast=Features(self.data)
        for w in self.windows:
            e=self.g.panel_edges(w,[],2);t,wt=nodes(e,64)
            z,dz=self.response(w,t,self.c.p)
            pref=Prefix(e,abs(dz),64)
            e0=self.g.panel_edges(w,[],1);t0,_=nodes(e0,32);_,d0=self.response(w,t0,self.c.p);p0=Prefix(e0,abs(d0),32)
            assert abs(pref(np.array([1.]))[0]-p0(np.array([1.]))[0])<=1e-9+1e-6*abs(pref(np.array([1.]))[0])
            self.target_prefix.append(pref)
        self.floor_by_h=[]
        for h in self.steps:
            totals=[];check=[]
            for j,w in enumerate(self.windows):
                for n,ref,store in [(32,1,check),(64,2,totals)]:
                    e=shifted(self.g.panel_edges(w,[],ref),h);t,wt=nodes(e,n)
                    length=self.target_prefix[j](t+h)-self.target_prefix[j](t-h)
                    assert min(length)>=0
                    store.append(float((wt@length**4)/(1-2*h)))
            a=np.sqrt(np.mean(check));b=np.sqrt(np.mean(totals))
            assert abs(a-b)<=1e-9+1e-5*abs(b),('target_local_scale',h,a,b)
            self.floor_by_h.append(max(.1*b,1e-4))
    def response(self,w,t,p):
        lo,hi=w['x'][[0,-1]];f=10**(lo+(hi-lo)*np.asarray(t))
        z,dz=self.fast.response(f,p) if self.fast is not None else self.c.response(f,p)
        return z/w['scale'],dz*(np.log(10)*(hi-lo)/w['scale'])
    def denominator(self,j,h,t):
        length=self.target_prefix[j](t+h)-self.target_prefix[j](t-h)
        if min(length)<-1e-10:raise RuntimeError('Negative target arclength prefix difference')
        return np.maximum(length**2,self.floor_by_h[self.steps.index(h)])
    def fixed(self,lp,label,n=32,refinement=1):
        j=next(k for k,w in enumerate(self.windows) if w['block']['label']==label);w=self.windows[j]
        p=10**np.asarray(lp);modes=self.g.modes(p);e=self.g.panel_edges(w,modes,refinement);t,wt=nodes(e,n)
        z,dz=self.response(w,t,p);r,dr=self.response(w,t,self.c.p)
        z0,_=self.response(w,np.array([0.]),p);r0,_=self.response(w,np.array([0.]),self.c.p)
        integrand=.5*np.imag((z-z0[0]).conj()*dz-(r-r0[0]).conj()*dr)
        pref=Prefix(e,integrand,n);area=float(wt@pref(t)**2)
        point=float(wt@abs(z-r)**2);old=w['area_den'][2];new=self.new_den[j]
        local=[];weighted=[];hermitian=[]
        for hi,h in enumerate(self.steps):
            outer=shifted(e,h);center,mass=nodes(outer,n);mass=mass/(1-2*h)
            tri=np.r_[center-h,center,center+h]
            zz,_=self.response(w,tri,p);rr,_=self.response(w,tri,self.c.p)
            zm,zc,zp=np.split(zz,3);rm,rc,rp=np.split(rr,3)
            correction=.5*np.imag((zm-z0[0]).conj()*(zp-zm)-(rm-r0[0]).conj()*(rp-rm))
            value=pref(center+h)-pref(center-h)-correction
            den=self.denominator(j,h,center)
            local.append(float(mass@(value/den)**2))
            weighted.append(float(mass@(abs(zc-rc)**2/den**2)))
            dm,dp=zc-zm,zp-zc;tm,tp=rc-rm,rp-rc
            hermitian.append(float(mass@abs(dm.conj()*dp-tm.conj()*tp)**2)/w['product_den'][hi]**2)
        result=dict(ARX_L=area/old**2,ARX_S_L=area/new**2,J0=point,J0_W=point*(old/new)**2,
            Local_ARX=float(np.mean(local)),J0_local_weighted=float(np.mean(weighted)),H=float(np.mean(hermitian)))
        result['Combined']=.5*result['ARX_S_L']+.5*result['Local_ARX']
        return result
    def polygon(self,lp,label,subdivisions=128):
        j=next(k for k,w in enumerate(self.windows) if w['block']['label']==label);w=self.windows[j]
        p=10**np.asarray(lp);e=self.g.panel_edges(w,self.g.modes(p),0)
        t=np.unique(np.r_[(e[:-1,None]+np.diff(e)[:,None]*np.linspace(0,1,subdivisions+1)).ravel(),1.])
        lo,hi=w['x'][[0,-1]];z,_=self.c.response(10**(lo+(hi-lo)*t),p);r,_=self.c.response(10**(lo+(hi-lo)*t));z/=w['scale'];r/=w['scale']
        a=areas(t,z)[0,2]-areas(t,r)[0,2];values=[]
        def interp(q,v):return np.interp(q,t,v.real)+1j*np.interp(q,t,v.imag)
        for h in self.steps:
            center=np.unique(np.r_[t,t-h,t+h,h,1-h]);center=center[(center>=h)&(center<=1-h)]
            zm,zp=interp(center-h,z),interp(center+h,z);rm,rp=interp(center-h,r),interp(center+h,r)
            pref=np.interp(center+h,t,a)-np.interp(center-h,t,a)
            correction=.5*np.imag((zm-z[0]).conj()*(zp-zm)-(rm-r[0]).conj()*(rp-rm))
            loss=((pref-correction)/self.denominator(j,h,center))**2
            values.append(float(np.trapezoid(loss,x=center)/(1-2*h)))
        return float(np.mean(values))

def checks():
    e=np.linspace(0,1,9);t,_=nodes(e,32)
    # z=t+i*t^2 gives global prefix t^3/6 and local area (2h)^3/6.
    pref=Prefix(e,.5*t*t,32);q=np.linspace(0,1,101)
    np.testing.assert_allclose(pref(q),q**3/6,atol=1e-13,rtol=1e-12)
    for h in (1/32,1/16,1/8):
        center=np.linspace(h,1-h,101);a=center-h;b=center+h
        za=a+1j*a*a;zb=b+1j*b*b
        local=pref(b)-pref(a)-.5*np.imag(za.conj()*(zb-za))
        np.testing.assert_allclose(local,(2*h)**3/6,atol=1e-13,rtol=1e-10)

if __name__=='__main__':checks();print('Analytic polynomial / local-prefix identity checks passed')

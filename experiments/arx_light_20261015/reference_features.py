"""Mode-anchored Gauss integration of frequency-labelled complex geometry."""
from core import np,CFG,case1_modes
from legacy_features import Features as Legacy
from numpy.polynomial.legendre import leggauss,legvander,legint,legval
from functools import lru_cache

@lru_cache(None)
def rule(n):
    v,w=leggauss(n);u=(v+1)/2;weights=w/2
    inv=np.linalg.inv(legvander(v,n-1));integ=[]
    for j in range(n):
        c=legint(inv[:,j]);integ.append((legval(v,c)-legval(-1,c))/2)
    return u,weights,np.array(integ).T

def mode_points(modes,lo,hi):
    chunks=[]
    for fc,width in modes:
        center=np.log10(fc);width=max(float(width),1e-12)
        if center<lo-1 or center>hi+1:continue
        extent=max(hi-lo,abs(center-lo),abs(center-hi))
        kmax=max(0,int(np.ceil(np.log2(extent/width))))
        radius=width*2.**np.arange(-6,kmax+1)
        chunks.extend([np.array([center]),center-radius,center+radius])
    if not chunks:return np.array([])
    a=np.concatenate(chunks);return a[(a>lo)&(a<hi)]

class Features:
    def __init__(self,data):
        self.data=data;self.c=data['circuit'];self.legacy=Legacy(data)
        assert data['kind']!='measured','Theoretical integrator must not invent measured data'
        self.windows=self.legacy.windows;self.queries=0;self.calls=0;self.refined=0;self.max_order=0
        self.target_modes=self.modes(self.c.p)
        self.lo=min(w['x'][0] for w in self.windows);self.hi=max(w['x'][-1] for w in self.windows)
        self.global_scale=max(w['scale'] for w in self.windows)
    def modes(self,p):
        return case1_modes(p) if self.data['kind']=='case1' else self.c.modes(p)
    def response(self,t,lo,span,p,scale):
        z,dz=self.c.response(10**(lo+span*t),p);self.queries+=len(t)
        return z/scale,dz*(np.log(10)*span/scale)
    def panel_edges(self,w,modes,refinement):
        lo,hi=w['x'][[0,-1]];span=hi-lo
        edges=np.unique(np.r_[np.linspace(0,1,9),(mode_points(self.target_modes+modes,lo,hi)-lo)/span])
        for _ in range(refinement):edges=np.sort(np.r_[edges,(edges[1:]+edges[:-1])/2])
        return edges
    def block(self,p,w,modes,n,refinement,names):
        lo,hi=w['x'][[0,-1]];span=hi-lo;scale=w['scale'];edges=self.panel_edges(w,modes,refinement)
        u,weights,anti=rule(n);left=edges[:-1];width=np.diff(edges)
        t=(left[:,None]+width[:,None]*u).ravel();wt=(width[:,None]*weights).ravel()
        z,dz=self.response(t,lo,span,p,scale);r,dr=self.response(t,lo,span,self.c.p,scale)
        out={}
        if 'J0' in names:out['J0']=float(wt@abs(z-r)**2)
        if 'A' in names or 'ARX' in names:
            z0,_=self.response(np.array([0.]),lo,span,p,scale);r0,_=self.response(np.array([0.]),lo,span,self.c.p,scale)
            zz=z-z0[0];rr=r-r0[0]
            az=np.array([.5*(t*dz.real-zz.real),.5*(t*dz.imag-zz.imag),.5*np.imag(zz.conj()*dz)])
            ar=np.array([.5*(t*dr.real-rr.real),.5*(t*dr.imag-rr.imag),.5*np.imag(rr.conj()*dr)])
            g=(az-ar).reshape(3,len(width),n)
            full=np.einsum('cij,j,i->ci',g,weights,width)
            prefix=np.cumsum(full,axis=1)-full
            at=prefix[:,:,None]+np.einsum('kj,cij,i->cik',anti,g,width)
            scores=np.sum(at.reshape(3,-1)**2*wt,axis=1)/w['area_den']**2
            if 'A' in names:out['A']=float(np.mean(scores))
            if 'ARX' in names:out['ARX']=float(scores[2])
        if 'D' in names or 'H' in names:
            dd=[];hh=[]
            for j,h in enumerate(CFG['features']['half_steps_normalized_block']):
                e=np.unique(np.r_[edges,edges-h,edges+h,h,1-h]);e=e[(e>=h)&(e<=1-h)]
                widths=np.diff(e);center=(e[:-1,None]+widths[:,None]*u).ravel()
                mass=(widths[:,None]*weights).ravel()/(1-2*h)
                tri=np.r_[center-h,center,center+h]
                v,_=self.response(tri,lo,span,p,scale);v0,_=self.response(tri,lo,span,self.c.p,scale)
                vm,vc,vp=np.split(v,3);rm,rc,rp=np.split(v0,3)
                dm,dp=vc-vm,vp-vc;tm,tp=rc-rm,rp-rc
                if 'D' in names:dd.append(float(mass@((abs(dm-tm)**2+abs(dp-tp)**2)/2))/w['increment_den'][j]**2)
                if 'H' in names:hh.append(float(mass@abs(dm.conj()*dp-tm.conj()*tp)**2)/w['product_den'][j]**2)
            if dd:out['D']=float(np.mean(dd))
            if hh:out['H']=float(np.mean(hh))
        return out
    def fixed(self,lp,names,n=32,refinement=1):
        p=10**np.asarray(lp);modes=self.modes(p)
        if names==('G0',):
            w=dict(x=np.array([self.lo,self.hi]),scale=self.global_scale)
            return np.array([[self.block(p,w,modes,n,refinement,('J0',))['J0']]])
        return np.array([[v[k] for k in names] for v in [self.block(p,w,modes,n,refinement,names) for w in self.windows]])
    def one(self,lp,names):
        cfg=CFG['quadrature'];prev=None
        for refine in range(cfg['max_panel_refinements']+1):
            for n in cfg['orders']:
                now=self.fixed(lp,names,n,refine)
                if prev is not None and np.all(abs(now-prev)<=cfg['atol']+cfg['rtol']*abs(now)):
                    self.calls+=1;self.refined+=int(n>16 or refine>0);self.max_order=max(self.max_order,n)
                    return now.mean(axis=0)
                prev=now
        raise RuntimeError('Unresolved auxiliary quadrature; candidate logp='+repr(np.asarray(lp).tolist()))
    def score(self,logp,method,level=1):
        return np.array([self.one(lp,(method,))[0] for lp in np.atleast_2d(logp)])
    def description(self):return self.legacy.description()

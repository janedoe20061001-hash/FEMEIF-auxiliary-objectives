"""Independent polygon/trapezoidal reference, no analytic area derivative."""
from core import *
from features import Features
from legacy_features import areas,average

def polygon(g,lp,names=('J0','D','H','A','ARX'),subdivisions=128):
    p=10**np.asarray(lp);modes=g.modes(p);rows=[]
    windows=g.windows if names!=('G0',) else [dict(x=np.array([g.lo,g.hi]),scale=g.global_scale)]
    for w in windows:
        lo,hi=w['x'][[0,-1]];span=hi-lo;e=g.panel_edges(w,modes,0)
        t=np.unique(np.r_[(e[:-1,None]+np.diff(e)[:,None]*np.linspace(0,1,subdivisions+1)).ravel(),1.])
        z,_=g.c.response(10**(lo+span*t),p);r,_=g.c.response(10**(lo+span*t));z/=w['scale'];r/=w['scale'];out={}
        out['J0']=out['G0']=float(average(abs(z-r)**2,t))
        if 'A' in names or 'ARX' in names:
            a=areas(t,z)[0];b=areas(t,r)[0]
            v=average((a-b)**2,t)/w['area_den']**2
            out['A']=float(v.mean());out['ARX']=float(v[2])
        if 'D' in names or 'H' in names:
            dd=[];hh=[]
            for j,h in enumerate(CFG['features']['half_steps_normalized_block']):
                center=np.unique(np.r_[t,t-h,t+h,h,1-h]);center=center[(center>=h)&(center<=1-h)]
                tri=np.r_[center-h,center,center+h];v,_=g.c.response(10**(lo+span*tri),p);r,_=g.c.response(10**(lo+span*tri))
                zm,z0,zp=np.split(v/w['scale'],3);rm,r0,rp=np.split(r/w['scale'],3)
                dm,dp=z0-zm,zp-z0;tm,tp=r0-rm,rp-r0
                dd.append(float(average((abs(dm-tm)**2+abs(dp-tp)**2)/2,center))/w['increment_den'][j]**2)
                hh.append(float(average(abs(dm.conj()*dp-tm.conj()*tp)**2,center))/w['product_den'][j]**2)
            out['D']=float(np.mean(dd));out['H']=float(np.mean(hh))
        rows.append([out[n] for n in names])
    return np.array(rows)

def agrees(a,b,strict=False):
    cfg=CFG['quadrature'] if strict else CFG['audit']
    at=cfg['atol'] if strict else cfg['feature_atol'];rt=cfg['rtol'] if strict else cfg['feature_rtol']
    return bool(np.all(abs(np.asarray(a)-np.asarray(b))<=at+rt*abs(np.asarray(b))))

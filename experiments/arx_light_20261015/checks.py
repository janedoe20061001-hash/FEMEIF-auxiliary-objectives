"""Independent polygon area/trapezoidal quadrature; Python circuit evaluator."""
from core import *
from legacy_features import areas,average

def agrees(a,b,strict=False):
    cfg=CFG['quadrature'] if strict else CFG['audit']
    at=cfg['atol'] if strict else cfg['feature_atol'];rt=cfg['rtol'] if strict else cfg['feature_rtol']
    return bool(np.all(abs(np.asarray(a)-np.asarray(b))<=at+rt*abs(np.asarray(b))))

def polygon(g,lp,names,subdivisions=128):
    p=10**np.asarray(lp);modes=g.modes(p);rows=[]
    windows=g.windows if names!=('G0',) else [dict(x=np.array([g.lo,g.hi]),scale=g.global_scale)]
    for k,w in enumerate(windows):
        lo,hi=w['x'][[0,-1]];e=g.panel_edges(w,modes,0)
        t=np.unique(np.r_[(e[:-1,None]+np.diff(e)[:,None]*np.linspace(0,1,subdivisions+1)).ravel(),1.])
        z,_=g.c.response(10**(lo+(hi-lo)*t),p);r,_=g.c.response(10**(lo+(hi-lo)*t));z/=w['scale'];r/=w['scale']
        point=float(average(abs(z-r)**2,t))
        if names==('G0',):rows.append([point]);continue
        a=areas(t,z)[0,2];b=areas(t,r)[0,2];err=float(average((a-b)**2,t))
        old=w['area_den'][2];new=g.scales()[k]
        values=dict(J0=point,J0_W=point*(old/new)**2,ARX_L=err/old**2,ARX_S_L=err/new**2)
        rows.append([values[n] for n in names])
    return np.array(rows)

def target_lengths(g,subdivisions):
    result=[]
    for w in g.windows:
        lo,hi=w['x'][[0,-1]];e=g.panel_edges(w,[],0)
        t=np.unique(np.r_[(e[:-1,None]+np.diff(e)[:,None]*np.linspace(0,1,subdivisions+1)).ravel(),1.])
        z,_=g.c.response(10**(lo+(hi-lo)*t));result.append(float(np.sum(abs(np.diff(z/w['scale'])))))
    return np.array(result)

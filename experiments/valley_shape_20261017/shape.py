"""Observable all-candidate radial sections; geometry and frequency kept separate."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'valley_perturbation_20261017'))
from physical import *
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'output/valley_shape_20261017'

def strict_extrema(c,x):
    rr=roots(SimpleNamespace(c=c),c.p,x)
    z,d=rx(c,x);v=np.real(np.conj(z)*d)/abs(z)
    nonzero=np.flatnonzero(v!=0)
    for i in np.flatnonzero(v==0):
        left=nonzero[nonzero<i];right=nonzero[nonzero>i]
        if len(left) and len(right) and v[left[-1]]*v[right[0]]<0:
            rr+=roots(SimpleNamespace(c=c),c.p,x[[left[-1],right[0]]])
    unique=[]
    for r in sorted(rr,key=lambda q:q['x']):
        if not unique or abs(r['x']-unique[-1]['x'])>1e-13:unique.append(r)
    return unique

def sections(model,t,n):
    c=Physical(model);ref=Physical(t['model']);W=t['core'][1]-t['core'][0]
    lo,hi=t['center']-4*W,t['center']+4*W;h=t['scale']*10**(3/20)
    x=grid(c,ref,lo,hi,n)
    # Include the known target center only as a grid node, never a candidate identity.
    x=np.unique(np.r_[x,t['center']]);rr=strict_extrema(c,x)
    result=[]
    for r in rr:
        if r['kind']!='valley':continue
        q=dict(center=r['x'],root=r,status='valley_above_radius')
        if r['magnitude_ohm']>=h:
            result.append(q);continue
        xx=np.unique(np.r_[x,r['x']]);v=abs(rx(c,xx)[0])-h
        intervals=list(np.where(v[:-1]*v[1:]<0)[0])
        crossing=[brentq(lambda w:float(abs(rx(c,np.array([w]))[0][0])-h),xx[i],xx[i+1],xtol=5e-15) for i in intervals]
        nonzero=np.flatnonzero(v!=0)
        for i in np.flatnonzero(v==0):
            l=nonzero[nonzero<i];rgt=nonzero[nonzero>i]
            if len(l) and len(rgt) and v[l[-1]]*v[rgt[0]]<0:crossing.append(float(xx[i]))
        left=[val for val in crossing if val<r['x']]
        right=[val for val in crossing if val>r['x']]
        if not left or not right:
            q['status']='open_section';result.append(q);continue
        a=max(left);b=min(right)
        xp=np.unique(np.r_[a,xx[(xx>a)&(xx<b)],b]);z=rx(c,xp)[0]
        cs=np.r_[0,np.cumsum(abs(np.diff(z)))];L=float(cs[-1])
        if L<=1e-12*t['scale']:
            q['status']='zero_motion';result.append(q);continue
        u=np.linspace(0,1,257);keep=np.r_[True,np.diff(cs)>0]
        xs=np.interp(u,cs[keep]/L,xp[keep]);zs=np.interp(u,cs[keep]/L,z.real[keep])+1j*np.interp(u,cs[keep]/L,z.imag[keep])
        internal=[v for v in rr if a<v['x']<b]
        g=gauss_length(c,ref,a,b,64,1)
        q.update(status='bounded',bounds=[a,b],width=b-a,length=L,gauss_length=g,gauss_agrees=bool(abs(L-g)<=1e-4*t['scale']+1e-3*g),sequence=[v['kind'] for v in internal],isolated_valley=bool([v['kind'] for v in internal]==['valley']),profile=dict(x=xs.tolist(),real=zs.real.tolist(),imag=zs.imag.tolist()))
        result.append(q)
    return dict(n=n,search=[lo,hi],radius=h,extrema=rr,candidates=result)

def compare(q,ref,t):
    if q['status']!='bounded':return {}
    z=lambda v:np.array(v['profile']['real'])+1j*np.array(v['profile']['imag'])
    x=lambda v:np.array(v['profile']['x'])
    W=t['core'][1]-t['core'][0]
    return dict(G=float(max(abs(z(q)-z(ref)))/t['scale']),length_ratio=q['length']/ref['length'],shift_widths=(q['center']-ref['center'])/W,P=float(max(abs((x(q)-q['center'])-(x(ref)-ref['center'])))/W),width_ratio=q['width']/ref['width'],center_state_error=abs(complex(q['root']['real_ohm'],q['root']['imag_ohm'])-complex(ref['root']['real_ohm'],ref['root']['imag_ohm']))/t['scale'])

def stable(a,b,t):
    if len(a['extrema'])!=len(b['extrema']) or len(a['candidates'])!=len(b['candidates']):return False
    if not all(u['kind']==v['kind'] and abs(u['x']-v['x'])<=1e-9 for u,v in zip(a['extrema'],b['extrema'])):return False
    for u,v in zip(a['candidates'],b['candidates']):
        if u['status']!=v['status']:return False
        if v['status']!='bounded':continue
        if u['sequence']!=v['sequence'] or max(abs(np.array(u['bounds'])-v['bounds']))>1e-9 or not v['gauss_agrees']:return False
        zu=np.array(u['profile']['real'])+1j*np.array(u['profile']['imag']);zv=np.array(v['profile']['real'])+1j*np.array(v['profile']['imag'])
        if max(abs(zu-zv))>1e-4*t['scale']+1e-3*max(abs(zv)):return False
        if abs(u['length']-v['length'])>1e-4*t['scale']+1e-3*v['length']:return False
        if max(abs(np.array(u['profile']['x'])-v['profile']['x']))>(1e-4+1e-3)*v['width']:return False
    return True

def target_section(t,n):
    v=sections(t['model'],t,n)
    q=[r for r in v['candidates'] if abs(r['center']-t['center'])<1e-9]
    assert len(q)==1 and q[0]['status']=='bounded'
    return q[0]

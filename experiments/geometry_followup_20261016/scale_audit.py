"""Independent-order checks on every fixed target scale; no candidate ranking."""
from common import *

def audit():
    results=[]
    for tid in ['controlled']+target_ids():
        m=Geometry(control_target() if tid=='controlled' else target(tid));prefix=[];length_checks=[]
        for j,w in enumerate(m.windows):
            e=m.g.panel_edges(w,[],1);t,wt=nodes(e,32);_,dz=m.response(w,t,m.c.p);p=Prefix(e,abs(dz),32);prefix.append(p)
            q=np.linspace(0,1,101);a=p(q);b=m.lengths[j](q)
            ok=bool(np.all(abs(a-b)<=1e-9+1e-6*abs(b)));length_checks.append(ok)
            if tid=='controlled':
                lo,hi=w['x'][[0,-1]];ff=10**(lo+(hi-lo)*q);r,l,c=m.c.p;omega=2*np.pi*ff;beta=r*(omega*c-1/(omega*l))
                exact=r/w['scale']*(np.arctan(beta)-np.arctan(beta[0]))
                assert np.all(abs(b-exact)<=1e-8+1e-6*abs(exact)),('controlled analytic target length',j)
        B=[];floors=[]
        for h in m.steps:
            bb=[];ll=[]
            for j,w in enumerate(m.windows):
                t,wt=nodes(shifted(m.g.panel_edges(w,[],1),h),32);wt/=1-2*h
                z=m.response(w,np.r_[t-h,t,t+h],m.c.p)[0];zm,zc,zp=np.split(z,3)
                bb.append(float(wt@abs(zc-(zm+zp)/2)**2));length=prefix[j](t+h)-prefix[j](t-h);ll.append(float(wt@length**4))
            b=np.sqrt(bb);B.append(np.maximum(b,max(.1*np.sqrt(np.mean(b*b)),1e-4)));floors.append(max(.1*np.sqrt(np.mean(ll)),1e-4))
        B=np.array(B).T;floors=np.array(floors)
        stored_floor=np.array(m.local_floors)
        bok=bool(np.all(abs(B-m.bden)<=1e-9+1e-5*abs(m.bden)));fok=bool(np.all(abs(floors-stored_floor)<=1e-9+1e-5*abs(stored_floor)))
        results.append(dict(target=tid,target_prefix_passed=all(length_checks),B_scale_passed=bok,area_floor_passed=fok,
            B_scale_32=B.tolist(),B_scale_64=m.bden.tolist(),area_floor_32=floors.tolist(),area_floor_64=m.local_floors))
    r=dict(all_passed=all(v['target_prefix_passed'] and v['B_scale_passed'] and v['area_floor_passed'] for v in results),results=results,common_sha256=sha(HERE/'common.py'))
    dump(OUT/'TARGET_SCALE_AUDIT.json',r);print('Target scale checks:',r['all_passed'],flush=True)
    assert r['all_passed'],'Retain failed target scale checks; do not score candidates'
    return r

if __name__=='__main__':audit()

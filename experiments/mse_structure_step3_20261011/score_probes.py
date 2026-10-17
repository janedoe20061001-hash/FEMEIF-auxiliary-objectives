"""Exploratory score diagnostics; no optimization, metric tuning or extra seeds."""
from core import *

def main():
    data=target("case1");q=data["quadrature"];g=Geometry(data);truth=data["circuit"].p.copy();rows=[]
    probes=[("truth",truth)]
    for name,factor in (("R2_half",.5),("R2_double",2.)):
        p=truth.copy();p[3]*=factor;probes.append((name,p))
    for name,p in probes:
        f=q.score(np.log10(p))[0];a=g.score(np.log10(p),return_events=True)[0]
        rows.append(dict(probe=name,mag=float(f[0]),pha=float(f[1]),event_geometry=a.tolist(),geometry=float(max(a))))
    for shift in [-.02,-.002,-.0002,0.,.0002,.002,.02]:
        p=truth.copy();p[4]*=10**(-2*shift)
        f=q.score(np.log10(p))[0];a=g.score(np.log10(p),return_events=True)[0]
        rows.append(dict(probe="L2_center_shift",branch_center_shift_decades=shift,mag=float(f[0]),pha=float(f[1]),
                         event_geometry=a.tolist(),geometry=float(max(a))))
    dump(OUT/"DIRECT_SCORE_PROBES.json",dict(target="case1",purpose="Direct scoring, no parameter optimization or extra seeds; exploratory diagnostic after protocol freeze; does not tune objective",rows=rows))

if __name__=="__main__":
    main()

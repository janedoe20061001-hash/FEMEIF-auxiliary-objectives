from core import *
import time

def enumeration(t,z,ref,scale,eta,integral=False):
    costs=[]; span=t[-1]-t[0]
    def distance(i,j): return (t[i]-t[j])**2+abs((z[i]-ref[j])/scale)**2
    def visit(i,j,cost):
        if abs(t[i]-t[j])>eta+1e-12: return
        d=distance(i,j)
        if i==j==len(t)-1:
            costs.append(np.sqrt(cost/span) if integral else np.sqrt(cost)); return
        for di,dj in ((1,0),(0,1),(1,1)):
            ii,jj=i+di,j+dj
            if ii>=len(t) or jj>=len(t) or abs(t[ii]-t[jj])>eta+1e-12: continue
            dd=distance(ii,jj)
            measure=((t[ii]-t[i])+(t[jj]-t[j]))/2
            nextcost=cost+(d+dd)*measure/2 if integral else max(cost,dd)
            visit(ii,jj,nextcost)
    visit(0,0,0 if integral else distance(0,0))
    return min(costs)

def check():
    rng=np.random.default_rng(202610111)
    for _ in range(8):
        t=np.array([-1.,-.31,.2,1.]); z=rng.normal(size=4)+1j*rng.normal(size=4); r=rng.normal(size=4)+1j*rng.normal(size=4)
        for integral,fn in ((False,frechet_batch),(True,integral_path_batch)):
            np.testing.assert_allclose(fn(t,z,r,2.,.8)[0],enumeration(t,z,r,2.,.8,integral),rtol=1e-12)
            assert fn(t,r,r,2.,.8)[0]==0
            assert fn(t,r+1,r,2.,.8)[0]>0
    records=[]
    for ident in CFG["targets"]:
        data=target(ident); q,c=data["quadrature"],data["circuit"]
        truth=np.log10(c.p)
        np.testing.assert_allclose(q.score(truth),0,atol=1e-20)
        assert np.all(truth>=data["lower"]) and np.all(truth<=data["upper"])
        x=rng.uniform(data["lower"],data["upper"],(12,len(truth)))
        windows=Geometry(data).description()
        timings={}
        for method in CFG["methods"]:
            geo=Geometry(data,method)
            roundtrip=float(geo.score(truth)[0])
            exact=float(geo.score(truth,physical_parameters=c.p)[0])
            assert exact==0.,(ident,method,exact)
            # Preserve whether the original strict roundtrip gate passes.
            # No change to the optimizer map, objective or final audit tolerances.
            np.testing.assert_allclose(10**truth,c.p,rtol=2e-15,atol=0.)
            begin=time.perf_counter(); values=geo.score(x); seconds=time.perf_counter()-begin
            assert np.isfinite(values).all() and np.all(values>=0)
            timings[method]=dict(seconds_12_candidates=seconds,frequency_queries=geo.queries,exact_physical_identity=exact,log_parameter_roundtrip_score=roundtrip,original_1e12_roundtrip_gate_passed=roundtrip<1e-12)
        g=Geometry(data,"MAG_PHA_MEAN"); a=g.score(x,return_events=True)
        np.testing.assert_allclose(g.score(x),a.mean(axis=-1))
        np.testing.assert_allclose(Geometry(data,"MAG_PHA_MAX").score(x),a.max(axis=-1))
        grid=q.reference_grid(c.p,2); z,dz=c.response(10**grid)
        catalog=nd.events(grid,z,data["config"])
        focus=[r for r in nd.match(data["catalog"],catalog,data["config"])[0] if r["event_id"] in CFG["focus_events"][ident]]
        assert len(focus)==len(CFG["focus_events"][ident]) and all(r["complete"] for r in focus)
        records.append(dict(target=ident,windows=windows,timing=timings))
    # Standalone geometric checks: parallel translation does not change chords;
    # scaling changes their lengths and therefore the combined new score.
    d=target("control0p25"); g=Geometry(d,"MAG_PHA_CHORD"); w=g.windows[0]
    truth=np.log10(d["circuit"].p)
    assert g.chord(w,d["circuit"].p[None,:],w["x"])[0]==0.
    dump(OUT/"PREFLIGHT.json",dict(passed=True,enumerated_path_checks=16,targets=records,source_hashes=hashes()))
    print("Preflight passed: 16 independent path enumerations, all targets and five objective definitions",flush=True)

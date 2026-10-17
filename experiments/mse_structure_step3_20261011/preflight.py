from core import *
import time

def scalar_enumeration(t, z, ref, scale, eta):
    costs = []
    def visit(i, j, worst):
        if abs(t[i]-t[j]) > eta + 1e-12:
            return
        d = np.sqrt((t[i]-t[j])**2 + abs((z[i]-ref[j])/scale)**2)
        worst = max(worst, d)
        if i == j == len(t)-1:
            costs.append(worst)
            return
        for di, dj in ((1,0),(0,1),(1,1)):
            if i+di < len(t) and j+dj < len(t):
                visit(i+di,j+dj,worst)
    visit(0,0,0)
    return min(costs)

def check():
    rng = np.random.default_rng(20261011)
    # Enumerate all paths for tiny independent cases rather than reusing recurrence.
    for _ in range(8):
        t = np.linspace(-1,1,4)
        z = rng.normal(size=4)+1j*rng.normal(size=4)
        ref = rng.normal(size=4)+1j*rng.normal(size=4)
        np.testing.assert_allclose(frechet_batch(t,z,ref,2.,.8)[0], scalar_enumeration(t,z,ref,2.,.8), rtol=1e-12)
        assert frechet_batch(t,ref,ref,2.,.8)[0] == 0
    records = []
    for ident in CFG["targets"]:
        data, started = target(ident), time.perf_counter()
        geo, q, c = Geometry(data), data["quadrature"], data["circuit"]
        truth = np.log10(c.p)
        assert np.all(truth >= data["lower"]) and np.all(truth <= data["upper"])
        np.testing.assert_allclose(q.score(truth), 0, atol=1e-20)
        assert geo.score(truth)[0] < 1e-12
        for w in geo.windows:
            t=(w["x"]-w["center"])/w["width"]
            lower=frechet_batch(t,w["reference"]*.5,w["reference"],w["scale"],CFG["geometry"]["frequency_slack_widths"])[0]
            assert lower >= .5-1e-12
        x=rng.uniform(data["lower"],data["upper"],(12,len(truth)))
        begin=time.perf_counter(); base=q.score(x); base_s=time.perf_counter()-begin
        begin=time.perf_counter(); a=geo.score(x); geo_s=time.perf_counter()-begin
        assert np.isfinite(base).all() and np.isfinite(a).all()
        phase_grid=q.reference_grid(c.p,2)
        z,dz=c.response(10**phase_grid)
        cat=nd.events(phase_grid,z,data["config"])
        focus=[r for r in nd.match(data["catalog"],cat,data["config"])[0] if r["event_id"] in CFG["focus_events"][ident]]
        assert all(r["complete"] for r in focus), focus
        records.append(dict(target=ident,windows=geo.description(),base_12_candidates_seconds=base_s,
                            geometry_12_candidates_seconds=geo_s,seconds=time.perf_counter()-started))
    dump(OUT / "PREFLIGHT.json",dict(passed=True,enumerated_path_checks=8,targets=records,source_hashes=hashes()))
    print(json.dumps(records,ensure_ascii=False),flush=True)

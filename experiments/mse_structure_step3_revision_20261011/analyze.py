"""Independent paired analysis; sampled auxiliary score is audited, not trusted as outcome."""
from core import *
import csv, time
from concurrent.futures import ProcessPoolExecutor, as_completed

class AuditGeometry(Geometry):
    def grid(self,win,p,points=None,anchors=False):
        x=super().grid(win,p,points,anchors)
        if anchors and self.method=='MAG_PHA_CHORD':
            # A resonance in an endpoint Z(t +/- h) occurs at a shifted
            # midpoint, so raw-Z mode anchors alone do not audit the chord.
            extra=[]
            c=self.data['circuit']
            for parameters in (c.p,p[0]):
                for fc,width in c.modes(parameters):
                    center=np.log10(fc)
                    if win['lo']<=center<=win['hi']:
                        for h in CFG['chord']['half_spans_widths']:
                            for sign in (-1,1):
                                extra.extend(center+sign*h*win['width']+np.linspace(-width,width,33))
            a=np.asarray(extra)
            x=np.unique(np.r_[x,a[(a>=win['lo'])&(a<=win['hi'])]])
        return x

def csvout(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def audit_curve(job):
    ident, method, seed, rule, p, train, train_geometry, eligible = job
    detail_path=OUT/f"analysis/details/{ident}_{method}_{seed}_{rule}.json"
    analyzer_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if detail_path.exists():
        cached=json.loads(detail_path.read_text(encoding="utf-8"))
        if (cached.get("analyzer_sha256")==analyzer_hash
                and np.array_equal(cached["log_parameters"],p)
                and np.array_equal(cached.get("training_objective"),train)
                and cached["summary"]["geometry_train"]==train_geometry
                and cached["summary"]["secondary_training_eligible"]==eligible):
            return cached["summary"],[dict(target=ident,method=method,seed=seed,rule=rule,**e) for e in cached["focus"]]
    data = target(ident)
    q, c, geo = data["quadrature"], data["circuit"], AuditGeometry(data,method)
    focus_ids = CFG["focus_events"][ident]
    truth_grid = q.reference_grid(c.p, 4)
    tz, tdz = c.response(10**truth_grid)
    phase_target = nd.phase_events(truth_grid, tz, tdz, data["config"])
    diag, _ = nd.diagnostic(q, p, data["catalog"], phase_target)
    refinements = []
    if not diag["stable"]:
        # Retain failure and original tolerances; increase resolution only.
        refinements.append(dict(levels=[1,2,4,8], diagnostic=diag))
        data["config"]["audit"]["refinements"] = [16,32,64]
        diag, _ = nd.diagnostic(q, p, data["catalog"], phase_target)
        refinements.append(dict(levels=[16,32,64], diagnostic=diag))
    independent = np.array(diag["objective"])
    atol = np.array(data["config"]["audit"]["objective_atol"])
    base_ok = bool(np.all(abs(independent-train) <= atol + .01*abs(independent)))
    focus = [r for r in diag["events"] if r["event_id"] in focus_ids]
    values=[]
    for n in CFG["geometry_audit"]["points"]:
        values.append(dict(points=n, events=geo.score(p,points=n,anchors=True,return_events=True)[0].tolist()))
    previous, last = np.array(values[-2]["events"]), np.array(values[-1]["events"])
    tolerance = CFG["geometry_audit"]["absolute_tolerance"] + CFG["geometry_audit"]["relative_tolerance"]*abs(last)
    geo_stable = bool(np.all(abs(last-previous) <= tolerance))
    independent_geometry=float(geo.aggregate(last))
    geo_training_ok=bool(abs(independent_geometry-train_geometry) <= CFG["geometry_audit"]["absolute_tolerance"] + .01*independent_geometry)
    # Phase RMSE over a fixed span enclosing the focus events, including the
    # connecting segment (which the optimized split-window score does not cover).
    ev=[e for e in data["catalog"] if e["event_id"] in focus_ids]
    lo=min(e["x"]-3*e["width_3db_decades"] for e in ev)
    hi=max(e["x"]+3*e["width_3db_decades"] for e in ev)
    grid=q.reference_grid(10**p,max(4,diag["refinement"]))
    grid=np.unique(np.r_[lo,grid[(grid>lo)&(grid<hi)],hi])
    z,_=c.response(10**grid,10**p)
    ref,_=c.response(10**grid)
    weights=np.diff(grid)/(hi-lo)
    residual=(np.angle(z)-np.angle(ref))**2
    local_phase=float(np.rad2deg(np.sqrt(((residual[:-1]+residual[1:])/2)@weights)))
    rec=dict(target=ident,method=method,seed=seed,rule=rule,
             stable=bool(diag["stable"]),base_score_ok=base_ok,geometry_stable=geo_stable,
             geometry_training_ok=geo_training_ok,mag=float(independent[0]),pha=float(independent[1]),
             train_mag=float(train[0]),train_pha=float(train[1]),geometry_train=float(train_geometry),geometry_reference=independent_geometry,
             focus_joint_ok=all(e["complete"] for e in focus),focus_missing=sum(not e["matched"] for e in focus),
             all_amp_ok=all(e["complete"] for e in diag["events"] if e["width_evaluable"]),
             local_phase_rmse_deg=local_phase,secondary_training_eligible=eligible,
             uniform_amp_rmse_db=diag["amplitude_rmse_db"],uniform_phase_rmse_deg=diag["phase_rmse_deg"])
    detail=dict(summary=rec,log_parameters=p.tolist(),training_objective=train.tolist(),analyzer_sha256=analyzer_hash,
                focus=focus,diagnostic=diag,geometry_refinements=values,
                chord_audit_shifted_endpoint_anchors=(method=='MAG_PHA_CHORD'),
                supplemental_base_refinements=refinements)
    dump(detail_path,detail)
    return rec, [dict(target=ident,method=method,seed=seed,rule=rule,**e) for e in focus]

def main():
    root=OUT/"batch"
    assert json.loads((root/"status.json").read_text())["status"]=="complete"
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    assert manifest["hashes"]==hashes(), "Optimization inputs changed"
    jobs, retention=[],[]
    checked=0
    for ident in CFG["targets"]:
        data=target(ident)
        geo=Geometry(data)
        for seed in CFG["seeds"]:
            baseline=root/ident/"MAG_PHA"/f"seed_{seed}"
            with np.load(baseline/"generation_0300.npz") as a:
                reference_scores=a["objectives"][int(a["primary"])].copy()[:2]
            for method in CFG["methods"]:
                geo=Geometry(data,method)
                d=root/ident/method/f"seed_{seed}"
                info=json.loads((d/"summary.json").read_text())
                assert info["evaluations"]==60000
                assert len(list(d.glob("generation_*.npz")))==15
                for filename,h in info["files_sha256"].items():
                    assert hashlib.sha256((d/filename).read_bytes()).hexdigest()==h
                np.testing.assert_array_equal(np.load(d/"initial.npy"),np.load(baseline/"initial.npy"))
                with np.load(d/"generation_0300.npz") as a:
                    x,f,ix=a["log_parameters"],a["objectives"],int(a["primary"])
                    assert x.shape==(200,len(data["lower"]))
                    assert np.isfinite(x).all() and np.isfinite(f).all()
                    assert np.all(x>=data["lower"]) and np.all(x<=data["upper"])
                    assert ix==int(np.argmin(f[:,0]+25.12*f[:,1]))
                    geometric=geo.score(x)
                    if method!="MAG_PHA":
                        np.testing.assert_allclose(geometric,f[:,2],rtol=1e-12,atol=1e-12)
                    ceiling=reference_scores*1.10+np.array(CFG["selection"]["absolute_floors"])
                    mask=np.all(f[:,:2]<=ceiling,axis=1)
                    possible=np.flatnonzero(mask)
                    choices=[("primary",ix)]
                    if len(possible):
                        common=Geometry(data,"MAG_PHA_MAX").score(x[possible])
                        choices.append(("limited_tradeoff",int(possible[np.argmin(common)])))
                    for rule,index in choices:
                        jobs.append((ident,method,seed,rule,x[index].copy(),f[index,:2].copy(),float(geometric[index]),bool(mask[index])))
                    retention.append(dict(target=ident,method=method,seed=seed,eligible_candidates=int(mask.sum()),
                                          minimum_geometry=float(min(geometric)),primary_geometry=float(geometric[ix])))
                checked+=1
    rows,events=[],[]
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=CFG["workers"]) as pool:
        for future in as_completed([pool.submit(audit_curve,j) for j in jobs]):
            r,e=future.result();rows.append(r);events.extend(e)
            dump(OUT/"analysis/status.json",dict(status="running",completed=len(rows),planned=len(jobs),seconds=time.perf_counter()-start))
    rows.sort(key=lambda r:(r["target"],r["seed"],r["method"],r["rule"]))
    # Independently confirm the restricted selector's allowance after refit.
    for row in rows:
        b=next(r for r in rows if r["target"]==row["target"] and r["seed"]==row["seed"] and r["method"]=="MAG_PHA" and r["rule"]=="primary")
        ceiling=1.10*np.array([b["mag"],b["pha"]])+np.array(CFG["selection"]["absolute_floors"])
        row["secondary_reference_eligible"]=bool(np.all(np.array([row["mag"],row["pha"]])<=ceiling))
    csvout(OUT/"analysis/REPRESENTATIVES.csv",rows)
    csvout(OUT/"analysis/EVENTS.csv",events)
    csvout(OUT/"analysis/RETAINED_SCORES.csv",retention)
    dump(OUT/"INTEGRITY.json",dict(checked_jobs=checked,evaluations=checked*60000,saved_stages=15,
                                   paired_initializations=True,source_hashes_verified=True,checkpoint_hashes_verified=True))
    failures=[r for r in rows if not(r["stable"] and r["base_score_ok"] and r["geometry_stable"] and r["geometry_training_ok"])]
    dump(OUT/"analysis/AUDIT.json",dict(passed=not failures,curves=len(rows),failures=failures,
                                      structure_and_base_score_failures=sum(not(r["stable"] and r["base_score_ok"]) for r in rows),
                                      geometry_sampling_failures=sum(not(r["geometry_stable"] and r["geometry_training_ok"]) for r in rows)))
    dump(OUT/"analysis/status.json",dict(status="complete",completed=len(rows),planned=len(jobs),seconds=time.perf_counter()-start))
    print(json.dumps(dict(curves=len(rows),audit_failures=len(failures)),ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()

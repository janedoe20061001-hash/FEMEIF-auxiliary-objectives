"""Frozen-model adapters and frequency-constrained, banded discrete Frechet score."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import sys, importlib.util, json, hashlib
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "output/mse_structure_step3_revision_20261011"
NESTED = ROOT / "experiments/mse_structure_complex_20261010"
OLD = ROOT / "experiments/mse_structure_step2_20261009"
sys.path.insert(0, str(NESTED))
from circuits import Circuit
from protocol import read_config, targets
from sampling import Quadrature
import diagnostics as nd

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

old_adaptive = load("step3_case1_adaptive", OLD / "adaptive.py")
sys.path.insert(0, str(OLD / "frozen"))
legacy_diag = load("step3_case1_diagnostics", OLD / "analysis_toolkit/diagnostics.py")
saved_diag = sys.modules["diagnostics"]
try:
    sys.modules["diagnostics"] = legacy_diag
    legacy_grid = load("step3_case1_refined_grid", OLD / "analysis_toolkit/refined_grid.py")
finally:
    sys.modules["diagnostics"] = saved_diag

CFG = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                               default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()), encoding="utf-8")
    temp.replace(path)

class CaseQuadrature(Quadrature):
    def __init__(self, circuit, config, record):
        self.c, self.config, self.record = circuit, config, record
        with np.load(OLD / "targets/s1/grid.npz") as a:
            self.adaptive = old_adaptive.Adaptive(record, a["frequency"])
        self.edges = self.adaptive.edges

    def score(self, x):
        return self.adaptive.score(x)

    @property
    def queries(self):
        return self.adaptive.queries

    def reference_grid(self, p=None, refinement=1):
        pp = np.empty((0, len(self.c.p))) if p is None else np.atleast_2d(p)
        return np.unique(np.r_[np.log10(legacy_grid.improved_dense_grid(self.record, pp, refinement)), self.edges])

def target(ident):
    cfg = read_config()
    if ident == "case1":
        record = json.loads((OLD / "targets/s1/record.json").read_text(encoding="utf-8"))
        c = Circuit("case1", "Case1", record["tree"], dict(zip(record["parameter_names"], record["target_parameters"])),
                    ("L2", "C2"), "R2", "R1")
        # Known branch anchors supplement the high-order network representation.
        c.modes = lambda p=None: [(fc, width / np.log(10)) for fc, width in legacy_grid.mode_candidates(record, c.p if p is None else p)]
        q = CaseQuadrature(c, cfg, record)
        # Recompute crossing coordinates with the same independent Case1 grid;
        # old catalogs do not contain the Hz width/crossing fields used by the
        # common diagnostic adapter. Event numbering remains target-specific.
        xx = q.reference_grid(c.p, 4)
        zz, _ = c.response(10**xx)
        catalog = nd.events(xx, zz, cfg)
        lower, upper = np.array(record["lower_log10"]), np.array(record["upper_log10"])
    else:
        key = {"ladder8": "rlc_ladder/lc_reciprocal_8p0", "mixed0p25": "mixed_resonators/lc_reciprocal_0p25",
               "control0p25": "control_rlc/lc_reciprocal_0p25"}[ident]
        t = next(t for t in targets(cfg) if t["id"] == key)
        c, lower, upper = t["circuit"], t["lower"], t["upper"]
        q = Quadrature(c, cfg)
        group = "first" if ident == "control0p25" else "other"
        record = json.loads((ROOT / f"output/mse_structure_complex_20261010_server/retrieved/{group}/inputs_core/{key}/target.json").read_text(encoding="utf-8"))
        catalog = record["events"]
        np.testing.assert_array_equal(c.p, record["circuit"]["parameters"])
    return dict(id=ident, circuit=c, quadrature=q, catalog=catalog, lower=lower, upper=upper, config=cfg)

def frechet_batch(t, z, reference, scale, eta):
    """Same fixed grid for both curves; all monotone path points covered, band constrained."""
    z = np.atleast_2d(z)
    n, batch = len(t), len(z)
    previous = np.full((n, batch), np.inf)
    for i in range(n):
        current = np.full((n, batch), np.inf)
        lo, hi = np.searchsorted(t, [t[i] - eta - 1e-12, t[i] + eta + 1e-12], side="left")
        for j in range(int(lo), min(n, int(hi) + 1)):
            if abs(t[i] - t[j]) > eta + 1e-12:
                continue
            d = (t[i] - t[j]) ** 2 + abs((z[:, i] - reference[j]) / scale) ** 2
            if i == j == 0:
                current[j] = d
            else:
                predecessor = previous[j].copy()
                if j:
                    predecessor = np.minimum(predecessor, current[j-1])
                    if i:
                        predecessor = np.minimum(predecessor, previous[j-1])
                current[j] = np.maximum(predecessor, d)
        previous = current
    result = np.sqrt(previous[-1])
    if not np.isfinite(result).all():
        raise AssertionError("No valid frequency-constrained path")
    return result

class Bottleneck:
    def __init__(self, data):
        self.data = data
        self.queries = 0
        self.windows = []
        settings = CFG["geometry"]
        cat = sorted(data["catalog"], key=lambda e: e["x"])
        for event_id in CFG["focus_events"][data["id"]]:
            e = next(e for e in cat if e["event_id"] == event_id)
            w = e["width_3db_decades"]
            if w is None or w <= 0:
                raise ValueError("Unresolved event cannot define local coordinates")
            ix = cat.index(e)
            lo, hi = max(2., e["x"] - 3*w), min(6., e["x"] + 3*w)
            if ix:
                lo = max(lo, (cat[ix-1]["x"] + e["x"]) / 2)
            if ix + 1 < len(cat):
                hi = min(hi, (cat[ix+1]["x"] + e["x"]) / 2)
            # Include center exactly; all scales fixed from target, never candidate.
            x = np.unique(np.r_[np.linspace(lo, hi, settings["points_per_window"]), e["x"]])
            z, _ = data["circuit"].response(10**x)
            scale = max(float(max(abs(z))), settings["ohm_floor"])
            self.windows.append(dict(event=event_id, center=e["x"], width=w, lo=lo, hi=hi,
                                     scale=scale, x=x, reference=z))

    def score(self, logp, points=None, anchors=False, return_events=False, physical_parameters=None):
        p = 10**np.atleast_2d(logp) if physical_parameters is None else np.atleast_2d(physical_parameters)
        scores = []
        for win in self.windows:
            x = win["x"]
            if points:
                x = np.unique(np.r_[x, np.linspace(win["lo"], win["hi"], points)])
            if anchors:
                if len(p) != 1:
                    raise ValueError("Independent anchor audit evaluates one candidate at a time")
                additions = []
                for fc, w in self.data["circuit"].modes(p[0]):
                    c = np.log10(fc)
                    if win["lo"] <= c <= win["hi"]:
                        additions.extend(c + np.linspace(-w, w, 33))
                a = np.asarray(additions)
                x = np.unique(np.r_[x, a[(a >= win["lo"]) & (a <= win["hi"])]])
            z, _ = self.data["circuit"].response(10**x, p)
            ref, _ = self.data["circuit"].response(10**x)
            self.queries += len(p) * len(x)
            t = (x - win["center"]) / win["width"]
            scores.append(frechet_batch(t, z, ref, win["scale"], CFG["geometry"]["frequency_slack_widths"]))
        a = np.column_stack(scores)
        return a if return_events else a.max(axis=1)

    def description(self):
        return [{k: v for k, v in w.items() if k not in ("x", "reference")} for w in self.windows]

def integral_path_batch(t, z, reference, scale, eta):
    """Trapezoid link cost in the symmetric frequency measure, fixed denominator.

    Every path covers both complete ordered curves. Sum of (dt+ds)/2 is
    always the window span; adding extra path steps cannot dilute the score.
    """
    z=np.atleast_2d(z); n,batch=len(t),len(z)
    previous=np.full((n,batch),np.inf)
    previous_d=np.full((n,batch),np.inf)
    for i in range(n):
        current=np.full((n,batch),np.inf)
        ds=np.full((n,batch),np.inf)
        lo,hi=np.searchsorted(t,[t[i]-eta-1e-12,t[i]+eta+1e-12],side="left")
        for j in range(int(lo),min(n,int(hi)+1)):
            if abs(t[i]-t[j])>eta+1e-12: continue
            d=(t[i]-t[j])**2+abs((z[:,i]-reference[j])/scale)**2
            ds[j]=d
            if i==j==0:
                current[j]=0.
                continue
            alternatives=[]
            if i and np.isfinite(previous[j]).any():
                alternatives.append(previous[j]+(previous_d[j]+d)*(t[i]-t[i-1])/4)
            if j and np.isfinite(current[j-1]).any():
                alternatives.append(current[j-1]+(ds[j-1]+d)*(t[j]-t[j-1])/4)
            if i and j and np.isfinite(previous[j-1]).any():
                alternatives.append(previous[j-1]+(previous_d[j-1]+d)*((t[i]-t[i-1])+(t[j]-t[j-1]))/4)
            if alternatives: current[j]=np.minimum.reduce(alternatives)
        previous,previous_d=current,ds
    result=np.sqrt(previous[-1]/(t[-1]-t[0]))
    if not np.isfinite(result).all(): raise AssertionError("No admissible integral path")
    return result

class Geometry(Bottleneck):
    def __init__(self,data,method="MAG_PHA_MAX"):
        super().__init__(data)
        self.method=method

    def grid(self,win,p,points=None,anchors=False):
        x=win["x"]
        if points: x=np.unique(np.r_[x,np.linspace(win["lo"],win["hi"],points)])
        if anchors:
            assert len(p)==1
            extra=[]
            for fc,w in self.data["circuit"].modes(p[0]):
                center=np.log10(fc)
                if win["lo"]<=center<=win["hi"]: extra.extend(center+np.linspace(-w,w,33))
            a=np.asarray(extra)
            x=np.unique(np.r_[x,a[(a>=win["lo"])&(a<=win["hi"])]])
        return x

    def chord(self,win,p,x):
        scores=[]
        c=self.data["circuit"]
        for h in CFG["chord"]["half_spans_widths"]:
            mids=x[(x-h*win["width"]>=win["lo"])&(x+h*win["width"]<=win["hi"])]
            if len(mids)<2: raise ValueError("Window does not resolve chord scale")
            ends=np.r_[mids-h*win["width"],mids+h*win["width"]]
            z,_=c.response(10**ends,p); ref,_=c.response(10**ends)
            self.queries+=len(p)*len(ends)
            n=len(mids); delta=(z[:,n:]-z[:,:n])/win["scale"]
            rd=(ref[n:]-ref[:n])/win["scale"]
            v=np.stack((np.full_like(delta.real,2*h),delta.real,delta.imag),axis=-1)
            vr=np.stack((np.full_like(rd.real,2*h),rd.real,rd.imag),axis=-1)
            length=np.linalg.norm(v,axis=-1); lr=np.linalg.norm(vr,axis=-1)
            residual=((v/length[...,None]-vr/lr[...,None])**2).sum(axis=-1)+np.log(length/lr)**2
            t=(mids-win["center"])/win["width"]
            scores.append(np.trapezoid(residual,t,axis=-1)/(t[-1]-t[0]))
        return np.sqrt(np.mean(scores,axis=0))

    def aggregate(self,events):
        a=np.asarray(events)
        return a.max(axis=-1) if self.method in ("MAG_PHA","MAG_PHA_MAX") else a.mean(axis=-1)

    def score(self,logp,points=None,anchors=False,return_events=False,physical_parameters=None):
        # Exact physical parameters are used ONLY for diagnostic identity
        # checks; optimization always follows the original logarithmic map.
        p=10**np.atleast_2d(logp) if physical_parameters is None else np.atleast_2d(physical_parameters)
        if self.method in ("MAG_PHA","MAG_PHA_MAX","MAG_PHA_MEAN"):
            a=super().score(logp,points,anchors,return_events=True,physical_parameters=physical_parameters)
        else:
            scores=[]
            for win in self.windows:
                x=self.grid(win,p,points,anchors)
                z,_=self.data["circuit"].response(10**x,p)
                ref,_=self.data["circuit"].response(10**x)
                self.queries+=len(p)*len(x)
                t=(x-win["center"])/win["width"]
                if self.method=="MAG_PHA_CHORD":
                    f=frechet_batch(t,z,ref,win["scale"],CFG["geometry"]["frequency_slack_widths"])
                    scores.append(f+CFG["chord"]["coefficient"]*self.chord(win,p,x))
                elif self.method=="MAG_PHA_INTEGRAL":
                    scores.append(integral_path_batch(t,z,ref,win["scale"],CFG["geometry"]["frequency_slack_widths"]))
                else: raise ValueError(self.method)
            a=np.column_stack(scores)
        return a if return_events else self.aggregate(a)

def hashes():
    files = [HERE / name for name in ("core.py", "run.py", "preflight.py", "protocol.json")]
    files += list(NESTED.glob("*.py")) + [NESTED / "config.json", NESTED / "requirements-lock.txt"]
    files += [OLD / "adaptive.py", OLD / "analysis_toolkit/refined_grid.py", OLD / "analysis_toolkit/diagnostics.py",
              OLD / "analysis_toolkit/frozen/population_source.py", OLD / "frozen/model.py",
              OLD / "frozen/frozen/case1/legacy_objectives.py", OLD / "targets/s1/record.json", OLD / "targets/s1/events.json", OLD / "targets/s1/grid.npz"]
    for group in ("first", "other"):
        files.extend((ROOT / f"output/mse_structure_complex_20261010_server/retrieved/{group}/inputs_core").glob("*/*/target.json"))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

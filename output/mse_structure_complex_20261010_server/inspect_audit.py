"""Post hoc higher-resolution check. Original code, settings and results stay intact."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import copy
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "mse_structure_complex_20261010"))
import numpy as np
from protocol import targets, dump, source_hashes
from sampling import Quadrature
from diagnostics import diagnostic

p = argparse.ArgumentParser()
p.add_argument("--group", choices=["first", "other"], default="first")
group = p.parse_args().group
manifest = json.loads((ROOT / f"batch_{group}/manifest.json").read_text())
cfg = manifest["config"]
assert manifest["hashes"] == source_hashes(cfg)
original_audit = json.loads((ROOT / f"analysis_{group}/AUDIT.json").read_text())
failures = original_audit["failures"]
records = []
for fail in failures:
    t = next(t for t in targets(cfg, "core", [fail["target"].split("/")[0]]) if t["id"] == fail["target"])
    meta = json.loads((ROOT / "inputs_core" / t["id"] / "target.json").read_text())
    with np.load(ROOT / f"batch_{group}" / t["id"] / f'seed_{fail["seed"]}' / f'generation_{fail["generation"]:04}.npz') as a:
        logp = a["log_parameters"][fail["index"]]
        score = a["optimization_objective"][fail["index"]]
        representative = fail["index"] in [int(a["selected_index"]), int(a["balanced_index"])]
    refined_cfg = copy.deepcopy(cfg)
    refined_cfg["audit"]["refinements"] = [8, 16, 32, 64]
    refined, _ = diagnostic(Quadrature(t["circuit"], refined_cfg), logp, meta["events"], meta["phase_events"])
    objective_ok = np.all(abs(np.array(refined["objective"])-score) <= np.array(cfg["audit"]["objective_atol"])+cfg["audit"]["objective_rtol"]*abs(np.array(refined["objective"])))
    levels = []
    for level in [4, 8, 16, 32, 64]:
        level_cfg = copy.deepcopy(cfg)
        level_cfg["audit"]["refinements"] = [level]
        rr, _ = diagnostic(Quadrature(t["circuit"], level_cfg), logp, meta["events"], meta["phase_events"])
        levels.append(dict(level=level, objective=rr["objective"], events=rr["events"], phase_events=rr["phase_events"], extras=rr["extra_events"], phase_extras=rr["extra_phase_events"]))
    def signature(v):
        return ([(e["event_id"],e["matched"],e["complete"]) for e in v["events"]],[(e["event_id"],e["matched"],e["complete"]) for e in v["phase_events"]],v["extras"],v["phase_extras"])
    high = np.array([v["objective"] for v in levels[-2:]])
    atol, rtol = np.array(cfg["audit"]["objective_atol"]), cfg["audit"]["objective_rtol"]
    high_objective_ok = np.all(abs(high-score) <= atol+rtol*abs(high))
    high_converged = np.all(abs(high[1]-high[0]) <= atol+rtol*abs(high[1])) and signature(levels[-2]) == signature(levels[-1])
    accepted = fail["objective_ok"] and not representative and high_objective_ok and high_converged
    records.append(dict(identity=fail, representative=representative, training_objective=score.tolist(), objective_ok=bool(objective_ok), high_objective_ok=bool(high_objective_ok), high_converged=bool(high_converged), accepted=bool(accepted), refined=refined, levels=levels))
passed = original_audit["completed_jobs"] == original_audit["planned_jobs"] and all(r["accepted"] for r in records)
dump(ROOT / f"AUDIT_REFINEMENT_{group}.json", dict(passed=passed, original_audit_sha256=hashlib.sha256((ROOT/f"analysis_{group}/AUDIT.json").read_bytes()).hexdigest(), original_hashes=manifest["hashes"], helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), original_audit_preserved=True, refinements=[4,8,16,32,64], tolerances_unchanged=True, table_scope="Original tables exclude originally unstable candidates. Representatives are unaffected; no historic table or score is rewritten.", records=records))
print(json.dumps(dict(passed=passed, records=[dict(identity=r["identity"], accepted=r["accepted"], representative=r["representative"], high_converged=r["high_converged"], high_objective_ok=r["high_objective_ok"]) for r in records])), flush=True)
if not passed:
    raise SystemExit(1)

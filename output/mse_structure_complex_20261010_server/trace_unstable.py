"""Read-only refinement trace for the remaining unstable nonrepresentative."""
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[name] = "1"
import copy
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"mse_structure_complex_20261010"))
import numpy as np
from protocol import targets,dump
from sampling import Quadrature
from diagnostics import diagnostic
manifest = json.loads((ROOT/"batch_other/manifest.json").read_text())
cfg = manifest["config"]
t = next(t for t in targets(cfg,"core",["three_level"]) if t["id"]=="three_level/lc_reciprocal_8p0")
meta = json.loads((ROOT/"inputs_core"/t["id"]/"target.json").read_text())
with np.load(ROOT/"batch_other"/t["id"]/"seed_33000/generation_0300.npz") as a:
    logp = a["log_parameters"][133]
records = []
for level in [4,8,16,32,64,128,256]:
    cc = copy.deepcopy(cfg)
    cc["audit"]["refinements"]=[level]
    r,_ = diagnostic(Quadrature(t["circuit"],cc),logp,meta["events"],meta["phase_events"])
    records.append(dict(level=level,record=r))
    dump(ROOT/"REFINEMENT_TRACE_other.json",dict(records=records))
    print(json.dumps(dict(level=level,objective=r["objective"],events=r["events"],extra=r["extra_events"],phase=r["phase_events"],phase_extra=r["extra_phase_events"])),flush=True)

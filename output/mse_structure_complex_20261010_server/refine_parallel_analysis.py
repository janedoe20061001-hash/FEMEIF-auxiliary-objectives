"""Resolve grid instability with stricter numerical resolution, preserving old tables."""
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[name] = "1"
import copy
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import zipfile
ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"mse_structure_complex_20261010"))
import numpy as np
import analyze as frozen
from diagnostics import diagnostic
from protocol import dump, targets, source_hashes
from sampling import Quadrature
from parallel_analysis import key

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    started = time.time()
    original = ROOT/"analysis_other_parallel"
    original_audit_path = original/"AUDIT.json"
    audit = json.loads(original_audit_path.read_text())
    assert audit["completed_jobs"]==audit["planned_jobs"]==120
    assert audit["failures"] and all(v["objective_ok"] and not v["structure_stable"] for v in audit["failures"])
    batch,prepared = ROOT/"batch_other",ROOT/"inputs_core"
    manifest = json.loads((batch/"manifest.json").read_text())
    cfg = manifest["config"]
    assert manifest["hashes"]==source_hashes(cfg)
    proof = json.loads((ROOT/"PARALLEL_ANALYSIS.json").read_text())
    assert proof["crosschecks_exact"] and proof["original_hashes"]==manifest["hashes"]
    assert proof["helper_sha256"]==sha(ROOT/"parallel_analysis.py")
    ids = sorted({t["circuit"]["id"] for t in manifest["plan"]["targets"]})
    ts = targets(cfg,manifest["plan"]["profile"],ids)
    byid = {t["id"]:t for t in ts}
    byc = {(t["circuit"].ident,tuple(t["circuit"].p)):t["id"] for t in ts}
    metadata = {t["id"]:json.loads((prepared/t["id"]/"target.json").read_text()) for t in ts}
    arrays,cache,identities,original_records = {},{},{},{}
    strip = {"target","seed","generation","index","on_front","training_objective","audit_passed"}
    with (original/"details.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            identity = (r["target"],r["seed"],r["generation"],r["index"])
            ck = identity[:3]
            if ck not in arrays:
                with np.load(batch/r["target"]/f'seed_{r["seed"]}'/f'generation_{r["generation"]:04}.npz') as a:
                    arrays[ck] = a["log_parameters"].copy()
            k = key(byid[r["target"]]["circuit"],arrays[ck][r["index"]])
            rec = {n:v for n,v in r.items() if n not in strip}
            if k in cache:
                assert cache[k]==rec
            cache[k]=rec
            identities[identity]=k
            original_records[identity]=r
    refined_cfg = copy.deepcopy(cfg)
    refined_cfg["audit"]["refinements"] = [32,64]
    replacement,records,refinement_pairs = {},[],{}
    for failure in audit["failures"]:
        ident = (failure["target"],failure["seed"],failure["generation"],failure["index"])
        k = identities[ident]
        t,meta = byid[ident[0]],metadata[ident[0]]
        if k not in replacement:
            logp = arrays[ident[:3]][ident[3]]
            cc = copy.deepcopy(refined_cfg)
            if ident==("three_level/lc_reciprocal_8p0",33000,300,133):
                cc["audit"]["refinements"]=[128,256]
            rec,_ = diagnostic(Quadrature(t["circuit"],cc),logp,meta["events"],meta["phase_events"])
            replacement[k]=rec
            refinement_pairs[k]=cc["audit"]["refinements"]
        rec = replacement[k]
        training = np.array(original_records[ident]["training_objective"])
        score = np.array(rec["objective"])
        objective_ok = np.all(abs(score-training)<=np.array(cfg["audit"]["objective_atol"])+cfg["audit"]["objective_rtol"]*abs(score))
        accepted = rec["stable"] and objective_ok
        records.append(dict(identity=failure,accepted=bool(accepted),objective_ok=bool(objective_ok),stable_refined=rec["stable"],refinements_used=refinement_pairs[k],original_record=original_records[ident],refined_record=rec))
    resolution = dict(passed=all(r["accepted"] for r in records),original_audit_sha256=sha(original_audit_path),original_analysis="analysis_other_original_grid",refinements=[32,64],exception_refinements=[128,256],tolerances_unchanged=True,optimization_unchanged=True,records=records)
    dump(ROOT/"AUDIT_REFINEMENT_other.json",resolution)
    assert resolution["passed"], "Unresolved candidate: stop without substituting results"
    cache.update(replacement)
    crosschecks = []
    for t in ts:
        qq = [i for i in identities if i[0]==t["id"]]
        for ident in [qq[0],qq[-1]]:
            k,meta = identities[ident],metadata[t["id"]]
            check_cfg = copy.deepcopy(cfg)
            if k in replacement:
                check_cfg["audit"]["refinements"] = refinement_pairs[k]
            check,_ = diagnostic(Quadrature(t["circuit"],check_cfg),arrays[ident[:3]][ident[3]],meta["events"],meta["phase_events"])
            assert check==cache[k], "Cache/fresh recomputation mismatch"
            crosschecks.append(list(ident))
    def cached(q,logp,target,target_phase=None):
        meta = metadata[byc[(q.c.ident,tuple(q.c.p))]]
        assert target==meta["events"] and target_phase==meta["phase_events"]
        return copy.deepcopy(cache[key(q.c,logp)]),None
    frozen.diagnostic = cached
    out = ROOT/"analysis_other_refined"
    new_audit = frozen.analyze(batch,prepared,out,population="all",plots=True)
    assert new_audit["passed"], "Full refined output audit failed"
    resolution.update(final_audit_passed=True,unique_refined_candidates=len(replacement),source_hashes=manifest["hashes"],helper_sha256=sha(Path(__file__)),fresh_crosschecks=crosschecks,elapsed_seconds=time.time()-started)
    dump(ROOT/"AUDIT_REFINEMENT_other.json",resolution)
    state_path = ROOT/"PIPELINE_STATUS.json"
    state = json.loads(state_path.read_text())
    assert not state.get("archive_other_sha256"), "Existing original archive preserved: do not replace"
    # End only redundant postprocessing. All optimization jobs are already complete.
    if state.get("status")=="running":
        assert state["stage"] in ("analyze_other","refine_other")
        parent,child = state["pid"],state.get("child_pid")
        parent_path = Path(f"/proc/{parent}/cmdline")
        if parent_path.exists():
            assert b"resume_pipeline.py" in parent_path.read_bytes()
            os.kill(parent,signal.SIGTERM)
        if child and Path(f"/proc/{child}/cmdline").exists():
            command = Path(f"/proc/{child}/cmdline").read_bytes()
            assert b"analyze.py" in command or b"inspect_audit.py" in command
            os.kill(child,signal.SIGTERM)
            for _ in range(100):
                stat = Path(f"/proc/{child}/stat")
                if not stat.exists() or stat.read_text().split()[2]=="Z":
                    break
                time.sleep(.1)
            else:
                raise RuntimeError("Postprocessing did not stop; original data preserved")
    else:
        assert state.get("status")=="needs_review"
    dump(ROOT/"AUDIT_REFINEMENT_other.json",resolution)
    if (ROOT/"analysis_other").exists():
        (ROOT/"analysis_other").rename(ROOT/"analysis_other_serial_partial")
    original.rename(ROOT/"analysis_other_original_grid")
    out.rename(ROOT/"analysis_other")
    state.pop("traceback",None)
    state.update(status="running",stage="archive_other_refined",child_pid=None,pid=os.getpid(),analysis_workers=24,
                 parallel_analysis_verified=True,original_grid_analysis_preserved=True,other_supplemental_audit_passed=True,updated_unix=time.time())
    dump(state_path,state)
    archive = ROOT/"results_other.zip"
    names = ["batch_other","analysis_other","analysis_other_original_grid","analysis_other_serial_partial","inputs_core","SERVER_ENVIRONMENT.json","VALIDATION.json","PIPELINE_STATUS.json","PARALLEL_ANALYSIS.json","AUDIT_REFINEMENT_other.json","REFINEMENT_TRACE_other.json","parallel_analysis.py","refine_parallel_analysis.py","trace_unstable.py","inspect_audit.py","resume_pipeline.py"]
    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED) as z:
        for name in names:
            p = ROOT/name
            if p.is_dir():
                for file in sorted(p.rglob("*")):
                    if file.is_file():
                        z.write(file,file.relative_to(ROOT).as_posix())
            elif p.exists():
                z.write(p,p.name)
    digest = sha(archive)
    (ROOT/"results_other.sha256").write_text(digest+"\n")
    state.update(status="complete",stage="complete",archive_other=str(archive),archive_other_sha256=digest,finished_unix=time.time(),updated_unix=time.time(),elapsed_seconds=time.time()-state["started_unix"])
    dump(state_path,state)
    print(json.dumps(dict(passed=True,refined_records=len(records),unique_refined_candidates=len(replacement),archive_bytes=archive.stat().st_size)),flush=True)

if __name__=="__main__":
    main()

"""Cache unchanged frozen diagnostics in parallel; preserve the serial audit prefix."""
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[name] = "1"
import copy
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent
CODE = ROOT/"mse_structure_complex_20261010"
sys.path.insert(0,str(CODE))
import numpy as np
import analyze as frozen
from diagnostics import diagnostic as original_diagnostic
from protocol import targets, source_hashes, dump
from sampling import Quadrature

WORKERS = 24
CTX = {}

def key(c, logp):
    return (c.ident,tuple(c.p),tuple(logp))

def init(cfg, ts, metadata):
    global CTX
    CTX = {t["id"]:(Quadrature(t["circuit"],cfg),metadata[t["id"]]) for t in ts}

def compute(task):
    ident, logp = task
    q,meta = CTX[ident]
    rec,_ = original_diagnostic(q,logp,meta["events"],meta["phase_events"])
    return key(q.c,logp),rec

def main():
    start = time.time()
    batch,prepared = ROOT/"batch_other",ROOT/"inputs_core"
    manifest = json.loads((batch/"manifest.json").read_text())
    cfg = manifest["config"]
    assert manifest["hashes"]==source_hashes(cfg)
    ids = sorted({t["circuit"]["id"] for t in manifest["plan"]["targets"]})
    ts = targets(cfg,manifest["plan"]["profile"],ids)
    byid = {t["id"]:t for t in ts}
    metadata = {t["id"]:json.loads((prepared/t["id"]/"target.json").read_text()) for t in ts}
    tasks,lookup = {},{}
    for t in ts:
        for seed in cfg["seeds"]:
            for file in sorted((batch/t["id"]/f"seed_{seed}").glob("generation_*.npz")):
                generation = int(file.stem.split("_")[-1])
                with np.load(file) as a:
                    X = a["log_parameters"]
                    ix = {int(a["selected_index"]),int(a["balanced_index"])}
                    if generation==cfg["generations"]:
                        ix |= set(range(len(X)))
                    for i in sorted(ix):
                        k = key(t["circuit"],X[i])
                        lookup[(t["id"],seed,generation,i)] = k
                        tasks[k] = (t["id"],X[i].copy())
    # Snapshot only complete lines. The serial process continues independently.
    prefix_path = ROOT/"analysis_other/details.jsonl"
    size = prefix_path.stat().st_size
    with prefix_path.open("rb") as f:
        prefix = f.read(size)
    prefix = prefix[:prefix.rfind(b"\n")+1]
    (ROOT/"serial_prefix_snapshot.jsonl").write_bytes(prefix)
    cache = {}
    strip = {"target","seed","generation","index","on_front","training_objective","audit_passed"}
    serial_records = 0
    for line in prefix.splitlines():
        r = json.loads(line)
        k = lookup[(r["target"],r["seed"],r["generation"],r["index"])]
        rec = {n:v for n,v in r.items() if n not in strip}
        if k in cache:
            assert cache[k]==rec, "Serial duplicate diagnostic differs"
        cache[k]=rec
        serial_records += 1
    pending = [v for k,v in tasks.items() if k not in cache]
    print(json.dumps(dict(serial_records=serial_records,unique_tasks=len(tasks),pending=len(pending),workers=WORKERS)),flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS,initializer=init,initargs=(cfg,ts,metadata)) as pool:
        futures = [pool.submit(compute,t) for t in pending]
        for count,f in enumerate(as_completed(futures),1):
            k,rec = f.result()
            cache[k]=rec
            if count%1000==0:
                print(json.dumps(dict(completed=count,pending=len(pending))),flush=True)
    assert set(cache)==set(tasks)
    # Fresh serial recomputation samples both cached and newly parallel results.
    init(cfg,ts,metadata)
    sample = set()
    for t in ts:
        kk = [k for k,v in tasks.items() if v[0]==t["id"]]
        if kk:
            sample.update([kk[0],kk[len(kk)//2],kk[-1]])
    for k in sample:
        _,rec = compute(tasks[k])
        assert rec==cache[k], "Parallel/cache vs fresh frozen diagnostic differs"
    def cached(q,logp,target,target_phase=None):
        m = metadata[tasks[key(q.c,logp)][0]]
        assert target==m["events"] and target_phase==m["phase_events"]
        return copy.deepcopy(cache[key(q.c,logp)]),None
    frozen.diagnostic = cached
    out = ROOT/"analysis_other_parallel"
    audit = frozen.analyze(batch,prepared,out,population="all",plots=True)
    proof = dict(workers=WORKERS,serial_prefix_records=serial_records,serial_prefix_sha256=hashlib.sha256(prefix).hexdigest(),
                 unique_diagnostics=len(tasks),new_parallel_diagnostics=len(pending),fresh_serial_crosschecks=len(sample),
                 crosschecks_exact=True,scientific_function="Unchanged frozen diagnostics.diagnostic",original_hashes=manifest["hashes"],
                 helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),audit_passed=audit["passed"],failures=audit["failures"],elapsed_seconds=time.time()-start)
    dump(ROOT/"PARALLEL_ANALYSIS.json",proof)
    # A fully completed serial pipeline wins; do not overwrite an existing archive.
    state_path = ROOT/"PIPELINE_STATUS.json"
    state = json.loads(state_path.read_text())
    if state.get("archive_other_sha256"):
        print("Original pipeline already archived; parallel result retained separately",flush=True)
        return
    if not audit["passed"]:
        print("Parallel audit has unresolved failures; leave original pipeline untouched",flush=True)
        return
    if state.get("stage")!="analyze_other" or state.get("status")!="running":
        print("Original pipeline changed stage; no takeover",flush=True)
        return
    parent,child = state["pid"],state["child_pid"]
    assert b"resume_pipeline.py" in Path(f"/proc/{parent}/cmdline").read_bytes()
    assert b"analyze.py" in Path(f"/proc/{child}/cmdline").read_bytes()
    # Only terminate redundant postprocessing after replacement passed every audit.
    os.kill(parent,signal.SIGTERM)
    os.kill(child,signal.SIGTERM)
    for _ in range(100):
        p = Path(f"/proc/{child}/stat")
        if not p.exists() or p.read_text().split()[2]=="Z":
            break
        time.sleep(.1)
    else:
        raise RuntimeError("Serial analysis did not stop; keep both directories separate")
    (ROOT/"analysis_other").rename(ROOT/"analysis_other_serial_partial")
    out.rename(ROOT/"analysis_other")
    state.update(status="running",stage="archive_other_parallel",child_pid=None,pid=os.getpid(),
                 analysis_workers=WORKERS,parallel_analysis_verified=True,serial_analysis_preserved=True,updated_unix=time.time())
    dump(state_path,state)
    archive = ROOT/"results_other.zip"
    names = ["batch_other","analysis_other","analysis_other_serial_partial","inputs_core","SERVER_ENVIRONMENT.json","VALIDATION.json","PIPELINE_STATUS.json","PARALLEL_ANALYSIS.json","parallel_analysis.py","inspect_audit.py","resume_pipeline.py"]
    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED) as z:
        for name in names:
            p = ROOT/name
            if p.is_dir():
                for f in sorted(p.rglob("*")):
                    if f.is_file():
                        z.write(f,f.relative_to(ROOT).as_posix())
            elif p.exists():
                z.write(p,p.name)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (ROOT/"results_other.sha256").write_text(digest+"\n")
    state.update(status="complete",stage="complete",archive_other=str(archive),archive_other_sha256=digest,finished_unix=time.time(),updated_unix=time.time(),elapsed_seconds=time.time()-state["started_unix"])
    dump(state_path,state)
    print("PARALLEL_ANALYSIS_COMPLETE",flush=True)

if __name__=="__main__":
    main()

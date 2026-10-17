"""Resume authorized jobs, retaining the original failed audit and supplemental check."""
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import zipfile
import server_pipeline as pipeline

ROOT = Path(__file__).resolve().parent
old = json.loads((ROOT / "PIPELINE_STATUS.json").read_text())
backup = ROOT / "PIPELINE_STATUS_before_resume.json"
if not backup.exists():
    backup.write_text(json.dumps(old, indent=2))
pipeline.STATE.update(old)
pipeline.STATE.update(pid=os.getpid(), resumed_unix=time.time())
pipeline.STATE.pop("traceback", None)

def archive(group):
    names = [f"batch_{group}",f"analysis_{group}","inputs_core","SERVER_ENVIRONMENT.json","VALIDATION.json","PIPELINE_STATUS.json",f"AUDIT_REFINEMENT_{group}.json","inspect_audit.py","resume_pipeline.py","PIPELINE_STATUS_before_resume.json"]
    dest = ROOT / f"results_{group}.zip"
    with zipfile.ZipFile(dest, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name in names:
            p = ROOT / name
            if p.is_dir():
                for f in sorted(p.rglob("*")):
                    if f.is_file():
                        z.write(f,f.relative_to(ROOT).as_posix())
            elif p.is_file():
                z.write(p,p.name)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    (ROOT / f"results_{group}.sha256").write_text(digest+"\n")
    pipeline.state(**{f"archive_{group}":str(dest), f"archive_{group}_sha256":digest})

try:
    pipeline.stage("refine_first", [str(ROOT/"inspect_audit.py"),"--group","first"])
    refinement = json.loads((ROOT / "AUDIT_REFINEMENT_first.json").read_text())
    if not refinement["passed"]:
        raise RuntimeError("First batch supplemental audit unresolved")
    pipeline.state(first_supplemental_audit_passed=True, original_first_audit_preserved=True)
    archive("first")
    pipeline.stage("run_other", ["run.py","run","--circuits",*pipeline.OTHER,"--prepared",str(ROOT/"inputs_core"),"--validation",str(ROOT/"VALIDATION.json"),"--out",str(ROOT/"batch_other"),"--workers","8"])
    try:
        pipeline.stage("analyze_other", ["analyze.py","--batch",str(ROOT/"batch_other"),"--prepared",str(ROOT/"inputs_core"),"--out",str(ROOT/"analysis_other"),"--population","all","--plots"])
    except RuntimeError:
        audit_path = ROOT / "analysis_other/AUDIT.json"
        if not audit_path.exists():
            raise
        audit = json.loads(audit_path.read_text())
        if audit["passed"] or not audit["failures"]:
            raise
        pipeline.stage("refine_other", [str(ROOT/"inspect_audit.py"),"--group","other"])
        pipeline.state(other_supplemental_audit_passed=True, original_other_audit_preserved=True)
    archive("other")
    pipeline.state(status="complete", stage="complete", finished_unix=time.time(), elapsed_seconds=time.time()-pipeline.STATE["started_unix"])
except Exception:
    pipeline.state(status="needs_review", traceback=traceback.format_exc(), child_pid=None)
    raise

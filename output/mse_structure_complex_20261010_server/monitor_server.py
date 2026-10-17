"""Read-only compact status for the experiment pipeline."""
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
report = dict(now_unix=time.time())
status = ROOT / "PIPELINE_STATUS.json"
report["pipeline"] = json.loads(status.read_text()) if status.exists() else None
for group in ("first", "other"):
    batch = ROOT / f"batch_{group}"
    completed, failed, active = [], [], []
    if batch.exists():
        for d in sorted(batch.glob("*/*/seed_*")):
            if (d / "summary.json").exists():
                s = json.loads((d / "summary.json").read_text())
                completed.append(dict(target=s["target"],seed=s["seed"],seconds=s["seconds"]))
            elif (d / "FAILED.json").exists():
                failed.append(json.loads((d / "FAILED.json").read_text()))
            elif (d / "progress.json").exists():
                active.append(dict(job=d.relative_to(batch).as_posix(), **json.loads((d/"progress.json").read_text())))
            else:
                active.append(dict(job=d.relative_to(batch).as_posix(),generation=0))
    report[group] = dict(completed=len(completed),failed=len(failed),active=active,
                         failures=failed[:2],recent_completions=completed[-3:])
    audit = ROOT / f"analysis_{group}/AUDIT.json"
    if audit.exists():
        a = json.loads(audit.read_text())
        report[group]["audit"] = {k:a[k] for k in ("passed","completed_jobs","planned_jobs")}
        report[group]["audit"]["failures"] = len(a["failures"])
    archive = ROOT / f"results_{group}.zip"
    if archive.exists():
        report[group]["archive_bytes"] = archive.stat().st_size
print(json.dumps(report))

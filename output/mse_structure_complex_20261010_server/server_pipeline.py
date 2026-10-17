"""Detached, sequential experiment batches; stop on failed run or audit."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "mse_structure_complex_20261010"
PYTHON = ROOT / ".venv/bin/python"
FIRST = ["user_nested", "control_rlc"]
OTHER = ["rlc_ladder", "loaded_tank", "dual_nested", "three_level", "mixed_resonators"]
STATE = dict(status="starting", pid=os.getpid(), started_unix=time.time(), workers=8,
             first_circuits=FIRST, remaining_circuits=OTHER, planned_total_jobs=168, planned_evaluations=10080000)


def state(**values):
    STATE.update(values)
    STATE["updated_unix"] = time.time()
    tmp = ROOT / "PIPELINE_STATUS.json.tmp"
    tmp.write_text(json.dumps(STATE, indent=2))
    tmp.replace(ROOT / "PIPELINE_STATUS.json")


def stage(name, args):
    state(status="running", stage=name)
    with (ROOT / (name + ".log")).open("w") as log:
        child = subprocess.Popen([str(PYTHON), *args], cwd=CODE, stdout=log, stderr=subprocess.STDOUT,
                                 env={**os.environ, "PYTHONUNBUFFERED":"1", "MPLBACKEND":"Agg"})
        state(child_pid=child.pid, command=[str(PYTHON), *args])
        result = child.wait()
    if result:
        raise RuntimeError(f"{name} failed with code {result}; see {name}.log")
    state(last_successful_stage=name, child_pid=None)


def archive(group):
    names = [f"batch_{group}", f"analysis_{group}", "inputs_core", "SERVER_ENVIRONMENT.json", "VALIDATION.json", "PIPELINE_STATUS.json"]
    destination = ROOT / f"results_{group}.zip"
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name in names:
            p = ROOT / name
            if p.is_dir():
                for file in sorted(p.rglob("*")):
                    if file.is_file():
                        z.write(file,file.relative_to(ROOT).as_posix())
            elif p.is_file():
                z.write(p,p.name)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    (ROOT / f"results_{group}.sha256").write_text(digest+"\n")
    state(**{f"archive_{group}":str(destination), f"archive_{group}_sha256":digest})


def main():
    state(status="waiting_for_dependencies", stage="install")
    install_pid = json.loads((ROOT / "INSTALL_PID.json").read_text())["pid"]
    deadline = time.monotonic()+900
    while time.monotonic()<deadline:
        proc = Path(f"/proc/{install_pid}/stat")
        if not proc.exists() or proc.read_text().split()[2] == "Z":
            break
        time.sleep(2)
    else:
        raise RuntimeError("Dependency installation exceeded 15 minutes")
    stage("dependency_check", ["-m","pip","check"])
    stage("validation", ["test_validation.py", "--report", str(ROOT / "VALIDATION.json")])
    stage("prepare", ["prepare.py", "--out", str(ROOT / "inputs_core")])
    for name, circuits in (("first", FIRST), ("other", OTHER)):
        stage("run_"+name, ["run.py","run", "--circuits",*circuits,"--prepared",str(ROOT/"inputs_core"),
                            "--validation",str(ROOT/"VALIDATION.json"),"--out",str(ROOT/f"batch_{name}"),"--workers","8"])
        stage("analyze_"+name, ["analyze.py","--batch",str(ROOT/f"batch_{name}"),"--prepared",str(ROOT/"inputs_core"),
                                "--out",str(ROOT/f"analysis_{name}"),"--population","all","--plots"])
        archive(name)
    state(status="complete", stage="complete", finished_unix=time.time(), elapsed_seconds=time.time()-STATE["started_unix"])


if __name__ == "__main__":
    try:
        main()
    except Exception:
        state(status="needs_review", traceback=traceback.format_exc(), child_pid=None)
        raise

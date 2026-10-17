"""Durable parallel coordinator for Case2 robustness runs."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
import traceback
import fcntl
import experiment as e


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    e.dump(temporary, data); temporary.replace(path)


def job(cfg, frequency, target, condition, group, seed, path, fingerprint):
    return e.run_one(cfg, frequency, target, condition, group, seed, path, fingerprint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--config", type=Path, default=e.HERE / "config.json")
    parser.add_argument("--output", type=Path, default=e.HERE / "runs")
    args = parser.parse_args()
    if not args.run or args.workers < 1:
        parser.error("Explicit --run and positive --workers required")
    cfg, frequency, target = e.config(args.config)
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    lock = (root / "coordinator.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    mf = e.manifest(cfg, frequency, target)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())["fingerprint"] != mf["fingerprint"]:
            raise ValueError("Existing fingerprint differs")
    else:
        e.dump(manifest_path, mf)
    pending, completed, failed = [], [], []
    for condition in cfg["conditions"]:
        for seed in range(cfg["seeds"]["start"], cfg["seeds"]["start"] + cfg["seeds"]["count"]):
            for group in e.GROUPS:
                path = root / condition["name"] / e.group_name(group) / f"seed_{seed:03d}"
                marker = path / "summary.json"
                if marker.exists():
                    record = json.loads(marker.read_text())
                    if record.get("fingerprint") != mf["fingerprint"] or record.get("status") != "complete":
                        raise ValueError(f"Invalid completion: {path}")
                    completed.append(record)
                elif path.exists():
                    raise ValueError(f"Incomplete directory: {path}")
                else:
                    pending.append((condition, group, seed, path))
    total = len(completed) + len(pending); started = time.time()

    def progress(state):
        atomic_json(root / "progress.json", {
            "state": state, "workers": args.workers, "total_runs": total,
            "completed_runs": len(completed), "failed_runs": len(failed),
            "elapsed_seconds": time.time() - started,
            "completed_evaluations": sum(r["evaluations"] for r in completed),
            "recent_completions": [{"condition": r["condition"], "group": r["group"],
                                    "seed": r["seed"]} for r in completed[-5:]],
            "failures": failed,
        })

    progress("running")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(job, cfg, frequency, target, c, g, s, p,
                               mf["fingerprint"]): (c, g, s)
                   for c, g, s, p in pending}
        for future in as_completed(futures):
            condition, group, seed = futures[future]
            try:
                record = future.result(); completed.append(record)
                print(f"{len(completed)}/{total} {condition['name']} {e.group_name(group)} seed={seed}", flush=True)
            except Exception:
                failed.append({"condition": condition["name"], "group": e.group_name(group),
                               "seed": seed, "error": traceback.format_exc()})
            progress("running")
    progress("failed" if failed else "complete")
    print(f"Batch finished: {len(completed)} complete, {len(failed)} failed", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


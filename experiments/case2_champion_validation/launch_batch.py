"""Durable parallel coordinator for the independent champion validation."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
import traceback

import experiment as e
from summarize import summarize


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    e.dump(temporary, data)
    temporary.replace(path)


def job(cfg, frequency, group, seed, path, fingerprint):
    return e.run_one(cfg, frequency, group, seed, path, fingerprint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--config", type=Path, default=e.HERE / "config.json")
    parser.add_argument("--output", type=Path, default=e.HERE / "runs")
    args = parser.parse_args()
    if not args.run:
        parser.error("No batch started; add --run after inspecting the plan")
    if args.workers < 1:
        parser.error("workers must be positive")
    cfg, frequency = e.config(args.config)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (root / "coordinator.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    mf = e.manifest(cfg, frequency)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())["fingerprint"] != mf["fingerprint"]:
            raise ValueError("Existing batch fingerprint differs")
    else:
        e.dump(manifest_path, mf)
    started = time.time()
    pending, completed, failed = [], [], []
    for seed in range(cfg["seeds"]["start"],
                      cfg["seeds"]["start"] + cfg["seeds"]["count"]):
        for group in e.groups(cfg):
            run_dir = root / e.group_name(group) / f"seed_{seed:03d}"
            marker = run_dir / "summary.json"
            if marker.exists():
                record = json.loads(marker.read_text())
                if (record.get("status") != "complete" or
                        record.get("fingerprint") != mf["fingerprint"] or
                        record.get("seed") != seed or
                        record.get("group") != e.group_name(group) or
                        not (run_dir / "final_population.npz").exists() or
                        not (run_dir / "history.csv").exists()):
                    raise ValueError(f"Invalid existing completion: {run_dir}")
                completed.append(record)
            elif run_dir.exists():
                raise ValueError(f"Incomplete run directory: {run_dir}")
            else:
                pending.append((group, seed, run_dir))
    total = len(pending) + len(completed)

    def progress(state):
        atomic_json(root / "progress.json", {
            "state": state, "pid": os.getpid(), "workers": args.workers,
            "total_runs": total, "completed_runs": len(completed),
            "failed_runs": len(failed), "started_unix": started,
            "updated_unix": time.time(), "elapsed_seconds": time.time() - started,
            "completed_evaluations": sum(r["evaluations"] for r in completed),
            "recent_completions": [
                {"group": r["group"], "seed": r["seed"],
                 "seconds": r["elapsed_seconds"]} for r in completed[-5:]],
            "failures": failed,
        })

    progress("running")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(job, cfg, frequency, group, seed, run_dir,
                               mf["fingerprint"]): (group, seed)
                   for group, seed, run_dir in pending}
        for future in as_completed(futures):
            group, seed = futures[future]
            try:
                record = future.result()
                completed.append(record)
                print(f"{len(completed)}/{total} seed={seed} {e.group_name(group)} "
                      f"{record['elapsed_seconds']:.1f}s", flush=True)
            except Exception:
                error = {"group": e.group_name(group), "seed": seed,
                         "error": traceback.format_exc()}
                failed.append(error)
                print(json.dumps(error), flush=True)
            progress("running")
    if any(r["group"] == "baseline" for r in completed):
        summarize(root)
    progress("failed" if failed else "complete")
    print(f"Batch finished: {len(completed)} complete, {len(failed)} failed", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

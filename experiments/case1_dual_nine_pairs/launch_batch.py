"""Durable parallel batch coordinator. Launch via nohup; progress is atomically saved."""
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name]="1"
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import time
import traceback
import experiment as e
from summarize import summarize
from sources import validate_references


def atomic_json(path,data):
    temp=path.with_suffix(".tmp")
    e.dump(temp,data)
    temp.replace(path)


def job(cfg,freq,group,seed,path,fingerprint):
    return e.run_one(cfg,freq,group,seed,path,fingerprint)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=28)
    parser.add_argument("--config",type=Path,default=e.HERE/"config.json")
    parser.add_argument("--output",type=Path,default=e.HERE/"runs")
    args=parser.parse_args()
    if args.workers<1:
        parser.error("workers must be positive")
    cfg,freq=e.config(args.config)
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    # Advisory lock prevents two launchers scheduling the same run directories.
    import fcntl
    lock=(root/"coordinator.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    mf=e.manifest(cfg,freq)
    path=root/"manifest.json"
    if path.exists():
        if json.loads(path.read_text())["fingerprint"]!=mf["fingerprint"]:
            raise ValueError("Existing batch fingerprint differs")
    else:
        e.dump(path,mf)
    plan=json.loads((e.HERE/"pair_plan.json").read_text())
    e.dump(root/"reference_preflight.json",validate_references(root,plan))
    start=time.time()
    pending=[];completed=[];failed=[]
    for seed in range(cfg["seeds"]["start"],cfg["seeds"]["start"]+cfg["seeds"]["count"]):
        for group in e.groups(cfg):
            rd=root/e.group_name(group)/f"seed_{seed:03d}"
            mark=rd/"summary.json"
            if mark.exists():
                r=json.loads(mark.read_text())
                if (r.get("status")!="complete" or r.get("fingerprint")!=mf["fingerprint"]
                    or r.get("seed")!=seed or r.get("group")!=e.group_name(group)
                    or not (rd/"final_population.npz").exists() or not (rd/"history.csv").exists()):
                    raise ValueError(f"Invalid existing completion {rd}")
                completed.append(r)
            elif rd.exists():
                raise ValueError(f"Incomplete directory must be inspected before resuming: {rd}")
            else:
                pending.append((group,seed,rd))
    total=len(pending)+len(completed)
    def progress(state):
        atomic_json(root/"progress.json",dict(state=state,pid=os.getpid(),workers=args.workers,
            total_runs=total,completed_runs=len(completed),failed_runs=len(failed),
            started_unix=start,updated_unix=time.time(),elapsed_seconds=time.time()-start,
            completed_evaluations=sum(r["evaluations"] for r in completed),
            recent_completions=[dict(group=r["group"],seed=r["seed"],seconds=r["elapsed_seconds"]) for r in completed[-5:]],
            failures=failed))
    progress("running")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(job,cfg,freq,g,s,rd,mf["fingerprint"]):(g,s) for g,s,rd in pending}
        for future in as_completed(futures):
            g,s=futures[future]
            try:
                r=future.result();completed.append(r)
                print(f"{len(completed)}/{total} seed={s} {e.group_name(g)} {r['elapsed_seconds']:.1f}s",flush=True)
            except Exception:
                error=dict(group=e.group_name(g),seed=s,error=traceback.format_exc())
                failed.append(error)
                print(json.dumps(error),flush=True)
            progress("running")
    if any(r["group"]=="baseline" for r in completed) or (root.parent/"reference").exists():
        summarize(root)
    progress("failed" if failed else "complete")
    print(f"Batch finished: {len(completed)} complete, {len(failed)} failed",flush=True)
    if failed:
        raise SystemExit(1)


if __name__=="__main__":
    main()

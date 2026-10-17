"""Default command is PLAN. Only explicit `run` invokes NSGA-II."""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from protocol import read_config, source_hashes, jobs, initial, plan, dump, verify_job
from sampling import Quadrature


def validate_inputs(cfg, profile, ids, prepared, validation):
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    checked = json.loads(validation.read_text(encoding="utf-8"))
    hashes = source_hashes(cfg)
    from importlib.metadata import version
    if checked["versions"] != {n: version(n) for n in checked["versions"]}:
        raise RuntimeError("Dependency versions changed after validation")
    if manifest["status"] != "prepared" or manifest["hashes"] != hashes or not checked["passed"] or checked["hashes"] != hashes:
        raise RuntimeError("Prepared inputs or validation are stale; revalidate and prepare in a new directory")
    expected = {j[0]["id"] for j in jobs(cfg, profile, ids)}
    actual = {str(Path(n).parent).replace("\\", "/") for n in manifest["target_hashes"]}
    if not expected <= actual:
        raise RuntimeError("Prepared target set is incomplete")
    for name, digest in manifest["target_hashes"].items():
        if hashlib.sha256((prepared / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Target modified: {name}")


def one(job):
    t, seed, cfg, root = job
    d = Path(root) / t["id"] / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=False)
    try:
        # Imports inside formal-run path; plan and numerical validation never optimize.
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import Problem
        from pymoo.core.callback import Callback
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.optimize import minimize
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
        start = time.perf_counter()
        q = Quadrature(t["circuit"], cfg)
        x0 = initial(t, cfg, seed)
        np.save(d / "initial_log_parameters.npy", x0)

        class Objective(Problem):
            def __init__(self):
                super().__init__(n_var=len(t["circuit"].names), n_obj=2, xl=t["lower"], xu=t["upper"])
                self.quadrature = q

            def _evaluate(self, x, out, *args, **kwargs):
                out["F"] = self.quadrature.score(x)

        class Trace(Callback):
            def __init__(self):
                super().__init__()
                self.history = []

            def notify(self, alg):
                x, F = alg.pop.get("X"), alg.pop.get("F")
                front = NonDominatedSorting().do(F, only_non_dominated_front=True)
                alpha = cfg["alpha_selection_only"]
                ix = int(front[np.argmin(F[front, 0] + alpha * F[front, 1])])
                bx = int(front[np.argmin(np.maximum(F[front, 0], alpha * F[front, 1]))])
                generation = int(alg.n_gen)
                rec = dict(generation=generation, evaluations=int(alg.evaluator.n_eval),
                           primary=F[ix], balanced=F[bx], front_size=len(front))
                self.history.append(rec)
                if generation % cfg["save_every"] == 0 or generation == cfg["generations"]:
                    np.savez_compressed(d / f"generation_{generation:04}.npz", log_parameters=x, optimization_objective=F,
                                        front_indices=front, selected_index=ix, balanced_index=bx)
                    dump(d / "progress.json", rec)

        ops = cfg["operators"]
        alg = NSGA2(pop_size=cfg["population"], n_offsprings=cfg["population"], sampling=x0.copy(),
                    crossover=SBX(prob=ops["sbx_probability"], eta=ops["sbx_eta"]),
                    mutation=PM(prob=ops["pm_probability"], prob_var=ops["pm_variable_probability"], eta=ops["pm_eta"]),
                    eliminate_duplicates=False)
        alg.tournament_type = "comp_by_rank_and_crowding"
        trace = Trace()
        result = minimize(Objective(), alg, ("n_gen", cfg["generations"]), seed=seed, callback=trace, verbose=False, save_history=False)
        evaluations = int(result.algorithm.evaluator.n_eval)
        if evaluations != cfg["generations"] * cfg["population"]:
            raise AssertionError("Unexpected search budget")
        evaluator = result.algorithm.problem.quadrature
        dump(d / "history.json", trace.history)
        rec = dict(status="complete", target=t["id"], seed=seed, evaluations=evaluations, seconds=time.perf_counter()-start,
                   initial_sha256=hashlib.sha256(x0.tobytes()).hexdigest(), model_frequency_queries=evaluator.queries,
                   max_adaptive_points=evaluator.max_points, max_adaptive_layers=evaluator.max_layers,
                   files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in d.iterdir() if p.suffix in (".npy", ".npz") or p.name == "history.json"})
        dump(d / "summary.json", rec)
        return rec
    except Exception:
        rec = dict(status="failed", target=t["id"], seed=seed, traceback=traceback.format_exc())
        dump(d / "FAILED.json", rec)
        return rec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["plan", "run"], default="plan")
    parser.add_argument("--config")
    parser.add_argument("--profile", choices=["core", "extended"], default="core")
    parser.add_argument("--circuits", nargs="+")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", action="store_true", help="Skip completed jobs only; no mid-generation recovery")
    opt = parser.parse_args()
    cfg = read_config(opt.config)
    planned = plan(cfg, opt.profile, opt.circuits)
    if opt.command == "plan":
        print(json.dumps(dict(profile=opt.profile, circuits=sorted({t["circuit"]["id"] for t in planned["targets"]}),
                              targets=len(planned["targets"]), seeds=cfg["seeds"], jobs=planned["jobs"],
                              evaluations=planned["planned_evaluations"], optimization_started=False), ensure_ascii=False, indent=2))
        return
    if not all((opt.prepared, opt.validation, opt.out)) or opt.workers < 1:
        parser.error("run requires --prepared, --validation, --out and positive --workers")
    validate_inputs(cfg, opt.profile, opt.circuits, opt.prepared, opt.validation)
    from importlib.metadata import version
    manifest = dict(config=cfg, plan=planned, hashes=source_hashes(cfg),
                    prepared_manifest=json.loads((opt.prepared / "manifest.json").read_text(encoding="utf-8")),
                    versions={n: version(n) for n in ("numpy", "scipy", "pymoo", "matplotlib", "moocore")},
                    python=sys.version, platform=platform.platform(), workers=opt.workers)
    recs, pending = [], []
    if opt.resume:
        old = json.loads((opt.out / "manifest.json").read_text(encoding="utf-8"))
        if any(old[k] != manifest[k] for k in ("config", "plan", "hashes", "versions")):
            raise RuntimeError("Resume config/code/versions must match original")
    else:
        opt.out.mkdir(parents=True, exist_ok=False)
        dump(opt.out / "manifest.json", manifest)
    for t, seed in jobs(cfg, opt.profile, opt.circuits):
        d = opt.out / t["id"] / f"seed_{seed}"
        if d.exists():
            completed = d / "summary.json"
            if not completed.exists() or not (d / f'generation_{cfg["generations"]:04}.npz').exists():
                raise RuntimeError(f"Incomplete job preserved: {d}. Use a new output directory to rerun; no RNG checkpoint resume.")
            rec = verify_job(d,cfg)
            x0 = np.load(d / "initial_log_parameters.npy")
            np.testing.assert_array_equal(x0, initial(t, cfg, seed))
            if hashlib.sha256(x0.tobytes()).hexdigest() != rec["initial_sha256"]:
                raise RuntimeError("Completed-job initialization modified")
            recs.append(rec)
        else:
            pending.append((t, seed, cfg, str(opt.out)))
    with ProcessPoolExecutor(max_workers=opt.workers) as pool:
        for future in as_completed([pool.submit(one, job) for job in pending]):
            rec = future.result()
            recs.append(rec)
            dump(opt.out / "status.json", dict(status="running", completed=sum(r["status"] == "complete" for r in recs),
                                              failed=sum(r["status"] == "failed" for r in recs), total=planned["jobs"]))
            print(json.dumps(rec), flush=True)
    dump(opt.out / "summaries.json", recs)
    failed = sum(r["status"] == "failed" for r in recs)
    dump(opt.out / "status.json", dict(status="failed" if failed else "complete", completed=len(recs)-failed, failed=failed, total=planned["jobs"]))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

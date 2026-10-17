from core import *
import time, traceback, argparse, platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def one(job):
    ident, method, seed, generations, population, root = job
    directory = Path(root) / ident / method / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    try:
        data = target(ident)
        q, geo = data["quadrature"], Geometry(data,method)
        x0 = np.random.default_rng(seed).uniform(data["lower"], data["upper"], (population, len(data["lower"])))
        np.save(directory / "initial.npy", x0)
        class Objective(Problem):
            def __init__(self):
                super().__init__(n_var=len(data["lower"]), n_obj=2 if method == "MAG_PHA" else 3,
                                 xl=data["lower"], xu=data["upper"])
            def _evaluate(self, x, out, *args, **kwargs):
                f = q.score(x)
                out["F"] = f if method == "MAG_PHA" else np.column_stack([f, geo.score(x)])
        class Trace(Callback):
            def __init__(self):
                super().__init__()
                self.records = []
            def notify(self, alg):
                x, f = alg.pop.get("X"), alg.pop.get("F")
                # Identical primary selection independent of geometry in both arms.
                ix = int(np.argmin(f[:, 0] + CFG["selection"]["alpha"] * f[:, 1]))
                bx = int(np.argmin(np.maximum(f[:, 0], CFG["selection"]["alpha"] * f[:, 1])))
                g = int(alg.n_gen)
                self.records.append(dict(generation=g, evaluations=int(alg.evaluator.n_eval), primary=f[ix].tolist()))
                if g % CFG["save_every"] == 0 or g == generations:
                    front = NonDominatedSorting().do(f, only_non_dominated_front=True)
                    np.savez_compressed(directory / f"generation_{g:04}.npz", log_parameters=x, objectives=f,
                                        primary=ix, balanced=bx, front=front)
                    dump(directory / "progress.json", self.records[-1])
        ops = data["config"]["operators"]
        alg = NSGA2(pop_size=population, n_offsprings=population, sampling=x0.copy(),
                    crossover=SBX(prob=ops["sbx_probability"], eta=ops["sbx_eta"]),
                    mutation=PM(prob=ops["pm_probability"], prob_var=ops["pm_variable_probability"], eta=ops["pm_eta"]),
                    eliminate_duplicates=False)
        alg.tournament_type = "comp_by_rank_and_crowding"
        trace = Trace()
        result = minimize(Objective(), alg, ("n_gen", generations), seed=seed, callback=trace, verbose=False)
        n = int(result.algorithm.evaluator.n_eval)
        assert n == generations * population
        dump(directory / "history.json", trace.records)
        rec = dict(status="complete", target=ident, method=method, seed=seed, evaluations=n,
                   seconds=time.perf_counter()-start, base_frequency_queries=q.queries, geometry_frequency_queries=geo.queries,
                   initial_sha256=hashlib.sha256(x0.tobytes()).hexdigest(),
                   files_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob("*.np*")})
        dump(directory / "summary.json", rec)
        return rec
    except Exception:
        rec = dict(status="failed", target=ident, method=method, seed=seed, traceback=traceback.format_exc())
        dump(directory / "FAILED.json", rec)
        return rec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preflight", "smoke", "batch"])
    args = parser.parse_args()
    if args.mode == "preflight":
        from preflight import check
        check()
        return
    assert json.loads((OUT / "PREFLIGHT.json").read_text(encoding="utf-8"))["passed"]
    root = OUT / args.mode
    root.mkdir(parents=True, exist_ok=False)
    seeds = CFG["seeds"] if args.mode == "batch" else [2026101101]
    generations, population = (CFG["generations"], CFG["population"]) if args.mode == "batch" else (2, 12)
    frozen = hashes()
    dump(root / "manifest.json", dict(protocol=CFG, hashes=frozen, python=sys.version, platform=platform.platform(),
                                     versions={name: version(name) for name in ("numpy", "scipy", "pymoo", "moocore")},
                                     generations=generations, population=population, seeds=seeds))
    records, start = [], time.perf_counter()
    planned = [(ident, method, seed, generations, population, str(root)) for ident in CFG["targets"]
               for seed in seeds for method in CFG["methods"]]
    # Interleave target/method families so the server has useful independent
    # tasks even when one method is slower. Ordering is frozen before results.
    prediction=json.loads((OUT/"PREFLIGHT.json").read_text(encoding="utf-8"))["predicted_seconds"]
    planned.sort(key=lambda j:(-prediction[j[0]][j[1]],j[2],CFG["methods"].index(j[1])))
    workers=min(CFG["workers"],len(planned),len(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else (os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one, job) for job in planned]):
            records.append(future.result())
            failed = sum(r["status"] != "complete" for r in records)
            dump(root / "status.json", dict(status="running", completed=len(records)-failed, failed=failed,
                                             planned=len(planned), seconds=time.perf_counter()-start))
            print(json.dumps({k:records[-1][k] for k in ("status","target","method","seed")}, ensure_ascii=False), flush=True)
    failed = sum(r["status"] != "complete" for r in records)
    dump(root / "summaries.json", records)
    dump(root / "status.json", dict(status="failed" if failed else "complete", completed=len(records)-failed,
                                     failed=failed, planned=len(planned), seconds=time.perf_counter()-start))
    assert not failed
    assert hashes() == frozen, "Code/config changed during run"
    for ident in CFG["targets"]:
        for seed in seeds:
            initializations = [np.load(root / ident / m / f"seed_{seed}/initial.npy") for m in CFG["methods"]]
            for x0 in initializations[1:]:
                np.testing.assert_array_equal(initializations[0],x0)

if __name__ == "__main__":
    main()

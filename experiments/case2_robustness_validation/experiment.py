"""Noise and missing-frequency robustness validation for the final Case2 method."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from importlib.metadata import version

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
CHAMPION = EXPERIMENTS / "case2_champion_validation"
EXTENDED = EXPERIMENTS / "case2_extended_pairs_validation"
sys.path.insert(0, str(CHAMPION))
sys.path.append(str(EXTENDED))

import numpy as np
import pymoo
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from objectives import Objectives, DEFAULTS
from case2 import LOWER, UPPER, N_VAR, PARAMETER_NAMES, impedance, load_target
from final_selection import select_wsmse

CORE = ("reactance_crossing_direction_consistency", "transition_speed_relative_error")
DIAGNOSTICS = ("characteristic_frequency_shift_error_decades", "DTW", "LMEP")
GROUPS = ((), CORE)


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding="utf-8")


def group_name(group):
    return "baseline" if not group else "I03+II02"


def config(path):
    cfg = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
    if cfg.get("features") != DEFAULTS:
        raise ValueError("Feature constants differ from champion protocol")
    if cfg.get("algorithm") != "NSGA-II" or cfg.get("encoding") != "log10":
        raise ValueError("Expected NSGA-II/log10")
    if cfg.get("selection_objectives") != "all" or cfg.get("report_front") != "wsmse_selected":
        raise ValueError("Invalid selection protocol")
    for key in ("population", "generations"):
        if type(cfg.get(key)) is not int or cfg[key] < 2:
            raise ValueError(f"Invalid {key}")
    seeds = cfg.get("seeds", {})
    if type(seeds.get("start")) is not int or type(seeds.get("count")) is not int or seeds["count"] < 1:
        raise ValueError("Invalid seeds")
    names = [c.get("name") for c in cfg.get("conditions", [])]
    if not names or len(names) != len(set(names)):
        raise ValueError("Conditions must have unique names")
    allowed = {"noise", "stride", "random_gap"}
    for condition in cfg["conditions"]:
        if condition.get("kind") not in allowed:
            raise ValueError("Unknown robustness condition")
        if condition["kind"] == "noise" and (condition.get("sigma_magnitude_db", -1) < 0 or condition.get("sigma_phase_degree", -1) < 0):
            raise ValueError("Invalid noise condition")
        if condition["kind"] == "stride" and condition.get("stride", 0) < 2:
            raise ValueError("Invalid stride condition")
        if condition["kind"] == "random_gap" and not 0 < condition.get("fraction", 0) < .5:
            raise ValueError("Invalid gap condition")
    frequency, target = load_target(cfg["target_data"])
    for condition in cfg["conditions"]:
        observed_frequency, observed_target, _ = observation(frequency, target, condition, seeds["start"])
        for group in GROUPS:
            Objectives(observed_frequency, cfg["features"], group, target_z=observed_target)
    return cfg, frequency, target


def observation(frequency, target, condition, seed):
    frequency = np.asarray(frequency, float)
    target = np.asarray(target, complex)
    rng = np.random.default_rng(1_000_003 + 10_007 * seed +
                                int(hashlib.sha256(condition["name"].encode()).hexdigest()[:8], 16))
    kind = condition["kind"]
    if kind == "noise":
        magnitude_db = 20 * np.log10(np.maximum(abs(target), 1e-300))
        phase = np.angle(target)
        magnitude_db = magnitude_db + rng.normal(0, condition["sigma_magnitude_db"], len(target))
        phase = phase + np.deg2rad(rng.normal(0, condition["sigma_phase_degree"], len(target)))
        observed = 10 ** (magnitude_db / 20) * np.exp(1j * phase)
        index = np.arange(len(frequency))
    elif kind == "stride":
        index = np.arange(0, len(frequency), condition["stride"])
        if index[-1] != len(frequency) - 1:
            index = np.r_[index, len(frequency) - 1]
        observed = target[index]
    else:
        width = max(1, int(round(condition["fraction"] * len(frequency))))
        start = int(rng.integers(1, len(frequency) - width - 1))
        keep = np.ones(len(frequency), bool)
        keep[start:start + width] = False
        index = np.flatnonzero(keep)
        observed = target[index]
    metadata = {
        "condition": condition["name"], "kind": kind,
        "observation_seed": int(seed), "observed_points": int(len(index)),
        "observed_indices_sha256": hashlib.sha256(index.astype(np.int64).tobytes()).hexdigest(),
        "observed_target_sha256": hashlib.sha256(np.asarray(observed, complex).tobytes()).hexdigest(),
    }
    return frequency[index], observed, metadata


def initial_population(cfg, seed):
    return np.random.default_rng(seed).uniform(LOWER, UPPER, (cfg["population"], N_VAR))


def nd_indices(values):
    return NonDominatedSorting().do(values, only_non_dominated_front=True)


def stats(values, alpha):
    full = nd_indices(values)
    primary = nd_indices(values[:, :2])
    return {
        "full_front_size": len(full), "primary_front_size": len(primary),
        "full_front_mean_magnitude_mse": float(values[full, 0].mean()),
        "full_front_mean_phase_mse": float(values[full, 1].mean()),
        "primary_front_mean_magnitude_mse": float(values[primary, 0].mean()),
        "primary_front_mean_phase_mse": float(values[primary, 1].mean()),
        "minimum_magnitude_mse": float(values[:, 0].min()),
        "minimum_phase_mse": float(values[:, 1].min()),
        **select_wsmse(values, alpha),
    }


class RobustProblem(Problem):
    def __init__(self, frequency, target, settings, group):
        super().__init__(n_var=N_VAR, n_obj=2 + len(group), xl=LOWER, xu=UPPER)
        self.objectives = Objectives(frequency, settings, group, target_z=target)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.objectives.evaluate_spectra(impedance(self.objectives.f, 10 ** x))


class Trace(Callback):
    def __init__(self, alpha):
        super().__init__(); self.alpha = alpha; self.rows = []

    def notify(self, algorithm):
        self.rows.append({"generation": int(algorithm.n_gen),
                          "evaluations": int(algorithm.evaluator.n_eval),
                          **stats(algorithm.pop.get("F"), self.alpha)})


def dependency_files():
    return {
        "local/experiment.py": HERE / "experiment.py",
        "local/launch_batch.py": HERE / "launch_batch.py",
        "local/report_results.py": HERE / "report_results.py",
        "shared/unified_objectives.py": CHAMPION / "objectives.py",
        "shared/chinese_objectives.py": EXTENDED / "objectives.py",
        "shared/paper_objectives.py": EXPERIMENTS / "case2_paper5_combinations" / "objectives.py",
        "shared/case2.py": EXTENDED / "case2.py",
        "shared/final_selection.py": EXTENDED / "final_selection.py",
    }


def manifest(cfg, frequency, target):
    record = {
        "config": cfg, "frequency_hz": frequency.tolist(),
        "target_real": target.real.tolist(), "target_imag": target.imag.tolist(),
        "parameter_names": PARAMETER_NAMES, "lower_log10": LOWER.tolist(),
        "upper_log10": UPPER.tolist(),
        "code_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for name, path in dependency_files().items()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pymoo": pymoo.__version__, "scipy": version("scipy"),
                     "moocore": version("moocore")},
    }
    record["fingerprint"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    return record


def run_one(cfg, full_frequency, full_target, condition, group, seed, directory, fingerprint):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir(parents=True)
    frequency, target, obs_meta = observation(full_frequency, full_target, condition, seed)
    x0 = initial_population(cfg, seed)
    np.save(directory / "initial_log_parameters.npy", x0)
    np.savez_compressed(directory / "observation.npz", frequency=frequency,
                        target_real=target.real, target_imag=target.imag)
    alpha = cfg["final_selection"]["phase_weight"]
    problem = RobustProblem(frequency, target, cfg["features"], group)
    trace = Trace(alpha)
    algorithm = NSGA2(
        pop_size=cfg["population"], n_offsprings=cfg["population"], sampling=x0.copy(),
        crossover=SBX(prob=cfg["crossover_probability"], eta=cfg["crossover_eta"]),
        mutation=PM(prob=cfg["mutation_individual_probability"],
                    prob_var=cfg["mutation_variable_probability"], eta=cfg["mutation_eta"]),
        eliminate_duplicates=False)
    algorithm.tournament_type = "comp_by_rank_and_crowding"
    started = time.perf_counter()
    result = minimize(problem, algorithm, ("n_gen", cfg["generations"]), seed=seed,
                      callback=trace, verbose=False, save_history=False)
    x, values = result.pop.get("X"), result.pop.get("F")
    selected = select_wsmse(values, alpha)
    index = selected["wsmse_selected_index"]
    selected_parameters = 10 ** x[index]
    clean = Objectives(full_frequency, cfg["features"], DIAGNOSTICS,
                       target_z=full_target).evaluate_spectra(
                           impedance(full_frequency, selected_parameters))[0]
    np.savez_compressed(directory / "final_population.npz", log_parameters=x,
                        parameters=10 ** x, parameter_names=np.array(PARAMETER_NAMES),
                        objectives=values, selected_index=index,
                        selected_parameters=selected_parameters,
                        selected_training_primary=values[index, :2],
                        selected_clean_metrics=clean,
                        selected_clean_metric_names=np.array(
                            ["magnitude_mse_db2", "phase_mse_rad2", *DIAGNOSTICS]),
                        selection_alpha=alpha)
    with (directory / "history.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace.rows[0]))
        writer.writeheader(); writer.writerows(trace.rows)
    record = {
        "status": "complete", "fingerprint": fingerprint,
        "condition": condition["name"], "group": group_name(group), "seed": seed,
        "objective_names": ["magnitude_mse_db2", "phase_mse_rad2", *group],
        "initial_population_sha256": hashlib.sha256(x0.tobytes()).hexdigest(),
        "evaluations": int(result.algorithm.evaluator.n_eval),
        "generations": len(trace.rows), "elapsed_seconds": time.perf_counter() - started,
        **obs_meta, **stats(values, alpha),
        "clean_magnitude_mse": float(clean[0]), "clean_phase_mse": float(clean[1]),
        "clean_II01": float(clean[2]), "clean_P03_DTW": float(clean[3]),
        "clean_P05_LMEP": float(clean[4]),
    }
    if record["evaluations"] != cfg["population"] * cfg["generations"]:
        raise AssertionError("Evaluation count mismatch")
    dump(directory / "summary.json", record)
    return record


def plan(cfg):
    runs = len(cfg["conditions"]) * len(GROUPS) * cfg["seeds"]["count"]
    return {"conditions": [c["name"] for c in cfg["conditions"]],
            "groups": [group_name(g) for g in GROUPS],
            "seed_start": cfg["seeds"]["start"], "seed_count": cfg["seeds"]["count"],
            "planned_runs": runs,
            "planned_evaluations": runs * cfg["population"] * cfg["generations"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "smoke"])
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--output", type=Path, default=HERE / "smoke")
    args = parser.parse_args()
    cfg, frequency, target = config(args.config)
    if args.command == "plan":
        print(json.dumps(plan(cfg), indent=2)); return
    cfg = dict(cfg); cfg.update(population=8, generations=3)
    mf = manifest(cfg, frequency, target)
    condition = cfg["conditions"][-1]
    for group in GROUPS:
        run_one(cfg, frequency, target, condition, group, cfg["seeds"]["start"],
                args.output / condition["name"] / group_name(group) /
                f"seed_{cfg['seeds']['start']:03d}", mf["fingerprint"])
    print("smoke complete")


if __name__ == "__main__":
    main()


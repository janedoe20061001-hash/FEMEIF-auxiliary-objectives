"""Independent Case2 champion validation on seeds 200--399."""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import platform
import sys
from importlib.metadata import version

import numpy as np
import pymoo

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
EXTENDED = EXPERIMENTS / "case2_extended_pairs_validation"
if str(EXTENDED) not in sys.path:
    sys.path.append(str(EXTENDED))

from objectives import AUX, CATALOG, DEFAULTS, Objectives
from case2 import LOWER, UPPER, N_VAR, PARAMETER_NAMES, load_target
from final_selection import select_wsmse


def _load_base():
    spec = importlib.util.spec_from_file_location(
        "case2_champion_base_experiment", EXTENDED / "experiment.py")
    if spec is None or spec.loader is None:
        raise ImportError(EXTENDED / "experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding="utf-8")


def groups(cfg):
    raw = cfg.get("groups")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Champion groups must be an explicit nonempty list")
    out = []
    for group in raw:
        if (not isinstance(group, list) or len(group) > 4 or
                len(set(group)) != len(group) or any(a not in AUX for a in group)):
            raise ValueError("Invalid champion group")
        out.append(tuple(group))
    if len(out) != len(set(out)) or out[0] != ():
        raise ValueError("Champion groups must be unique and start with baseline")
    return out


def group_name(aux):
    return "+".join(aux) if aux else "baseline"


def config(path):
    cfg = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
    for key in ("population", "generations"):
        if type(cfg.get(key)) is not int or cfg[key] < 2:
            raise ValueError(f"{key} must be an integer >=2")
    seeds = cfg.get("seeds", {})
    if (type(seeds.get("start")) is not int or seeds["start"] < 0 or
            type(seeds.get("count")) is not int or seeds["count"] < 1):
        raise ValueError("Invalid seeds")
    if cfg.get("algorithm") != "NSGA-II" or cfg.get("encoding") != "log10":
        raise ValueError("Expected NSGA-II with log10 encoding")
    for key in ("crossover_probability", "mutation_individual_probability",
                "mutation_variable_probability"):
        if not 0 <= cfg[key] <= 1:
            raise ValueError(f"{key} must be in [0,1]")
    if cfg["crossover_eta"] < 0 or cfg["mutation_eta"] < 0:
        raise ValueError("Operator eta must be nonnegative")
    if cfg.get("features") != DEFAULTS:
        raise ValueError("Feature constants differ from audited definitions")
    if (cfg.get("auxiliary_transform") != "loss" or
            cfg.get("selection_objectives") != "all" or
            cfg.get("report_front") != "wsmse_selected"):
        raise ValueError("Invalid objective/selection protocol")
    fs = cfg.get("final_selection", {})
    if fs.get("method") != "min_wsmse_final_population" or fs.get("tie_break") != "first_index":
        raise ValueError("Invalid final selection protocol")
    select_wsmse(np.zeros((1, 2)), fs["phase_weight"])
    if cfg.get("case") != "case2_paper_figure15":
        raise ValueError("Expected Case2 figure-15 circuit")
    frequency, target = load_target(cfg["target_data"])
    for aux in groups(cfg):
        Objectives(frequency, cfg["features"], aux, "loss",
                   cfg["reciprocal_epsilon"], target)
    return cfg, frequency


def plan(cfg):
    gs = groups(cfg)
    return {
        "status": "plan_only",
        "seed_start": cfg["seeds"]["start"],
        "seed_count": cfg["seeds"]["count"],
        "population": cfg["population"],
        "generations": cfg["generations"],
        "group_count": len(gs),
        "groups": [{"name": group_name(g), "auxiliaries": list(g),
                    "objective_count": 2 + len(g)} for g in gs],
        "planned_runs": len(gs) * cfg["seeds"]["count"],
        "planned_evaluations": len(gs) * cfg["seeds"]["count"] *
                               cfg["population"] * cfg["generations"],
    }


def _dependency_files():
    return {
        "local/experiment.py": HERE / "experiment.py",
        "local/objectives.py": HERE / "objectives.py",
        "local/report_results.py": HERE / "report_results.py",
        "shared/chinese_objectives.py": EXTENDED / "objectives.py",
        "shared/paper_objectives.py": EXPERIMENTS / "case2_paper5_combinations" / "objectives.py",
        "shared/base_experiment.py": EXTENDED / "experiment.py",
        "shared/case2.py": EXTENDED / "case2.py",
        "shared/final_selection.py": EXTENDED / "final_selection.py",
        "shared/summarize.py": EXTENDED / "summarize.py",
        "shared/analysis.py": EXTENDED / "analysis.py",
        "local/launch_batch.py": HERE / "launch_batch.py",
    }


def manifest(cfg, frequency):
    target_frequency, target = load_target(cfg["target_data"])
    if not np.array_equal(frequency, target_frequency):
        raise ValueError("Manifest frequency mismatch")
    code = {name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in _dependency_files().items()}
    record = {
        "config": cfg,
        "frequency_hz": frequency.tolist(),
        "target_real": target.real.tolist(),
        "target_imag": target.imag.tolist(),
        "code_sha256": code,
        "parameter_names": PARAMETER_NAMES,
        "lower_log10": LOWER.tolist(),
        "upper_log10": UPPER.tolist(),
        "target_features": Objectives(frequency, cfg["features"], target_z=target).target_summary(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pymoo": pymoo.__version__, "scipy": version("scipy"),
                     "moocore": version("moocore")},
    }
    record["fingerprint"] = hashlib.sha256(
        json.dumps(record, sort_keys=True).encode()).hexdigest()
    return record


BASE.groups = groups
BASE.group_name = group_name


def run_one(cfg, frequency, aux, seed, directory, fingerprint):
    return BASE.run_one(cfg, frequency, aux, seed, directory, fingerprint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "run", "smoke"])
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--group")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    cfg, frequency = config(args.config)
    if args.command == "plan":
        info = plan(cfg)
        print(json.dumps(info, indent=2))
        return
    if args.command == "smoke":
        cfg = copy.deepcopy(cfg)
        cfg.update(population=12, generations=3, seeds={"start": 200, "count": 1})
        default = HERE / "smoke"
    else:
        default = HERE / "runs"
    if args.group is None or args.seed is None:
        parser.error("run/smoke requires --group and --seed")
    selected = [g for g in groups(cfg) if group_name(g) == args.group]
    if not selected:
        parser.error("--group does not match config")
    if not cfg["seeds"]["start"] <= args.seed < cfg["seeds"]["start"] + cfg["seeds"]["count"]:
        parser.error("--seed outside configured range")
    out = (args.output or default) / group_name(selected[0]) / f"seed_{args.seed:03d}"
    mf = manifest(cfg, frequency)
    print(json.dumps(run_one(cfg, frequency, selected[0], args.seed, out,
                             mf["fingerprint"]), indent=2))


if __name__ == "__main__":
    main()

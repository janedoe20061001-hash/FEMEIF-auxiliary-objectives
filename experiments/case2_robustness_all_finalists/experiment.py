"""Run the nine remaining champion finalists under all robustness conditions."""
from pathlib import Path
import hashlib
import importlib.util
import json

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent / "case2_robustness_validation"


def _load_base():
    spec = importlib.util.spec_from_file_location(
        "case2_robustness_finalists_base", BASE_DIR / "experiment.py")
    if spec is None or spec.loader is None:
        raise ImportError(BASE_DIR / "experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base()

FINALIST_GROUPS = (
    ("reactance_crossing_direction_consistency",),
    ("transition_speed_relative_error",),
    ("reactance_crossing_direction_consistency", "characteristic_frequency_shift_error_decades", "transition_speed_relative_error"),
    ("DTW",),
    ("DTW", "LFP"),
    ("DTW", "LMEP"),
    ("DTW", "LFP", "LMEP"),
    ("PTC", "DTW", "LFP", "LMEP"),
    ("reactance_crossing_direction_consistency", "transition_speed_relative_error", "DTW"),
)
NAMES = {
    FINALIST_GROUPS[0]: "I03",
    FINALIST_GROUPS[1]: "II02",
    FINALIST_GROUPS[2]: "I03+II01+II02",
    FINALIST_GROUPS[3]: "P03",
    FINALIST_GROUPS[4]: "P03+P04",
    FINALIST_GROUPS[5]: "P03+P05",
    FINALIST_GROUPS[6]: "P03+P04+P05",
    FINALIST_GROUPS[7]: "P02+P03+P04+P05",
    FINALIST_GROUPS[8]: "I03+II02+P03",
}


def group_name(group):
    return NAMES[tuple(group)]


def config(path):
    cfg, frequency, target = BASE.config(path)
    configured = tuple(tuple(group) for group in cfg.get("groups", []))
    if configured != FINALIST_GROUPS:
        raise ValueError("Configured finalist groups differ from the fixed protocol")
    for condition in cfg["conditions"]:
        observed_frequency, observed_target, _ = BASE.observation(
            frequency, target, condition, cfg["seeds"]["start"])
        for group in FINALIST_GROUPS:
            BASE.Objectives(observed_frequency, cfg["features"], group,
                            target_z=observed_target)
    return cfg, frequency, target


def dependency_files():
    files = dict(BASE.dependency_files())
    files.update({
        "finalists/experiment.py": HERE / "experiment.py",
        "finalists/launch_batch.py": HERE / "launch_batch.py",
        "finalists/report_results.py": HERE / "report_results.py",
        "shared/robustness_experiment.py": BASE_DIR / "experiment.py",
    })
    return files


def manifest(cfg, frequency, target):
    record = BASE.manifest(cfg, frequency, target)
    record["code_sha256"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in dependency_files().items()
    }
    record["fingerprint"] = hashlib.sha256(
        json.dumps(record, sort_keys=True).encode()).hexdigest()
    return record


BASE.group_name = group_name


def run_one(cfg, frequency, target, condition, group, seed, directory, fingerprint):
    return BASE.run_one(cfg, frequency, target, condition, group, seed,
                        directory, fingerprint)


def plan(cfg):
    runs = len(cfg["conditions"]) * len(FINALIST_GROUPS) * cfg["seeds"]["count"]
    return {"conditions": [c["name"] for c in cfg["conditions"]],
            "groups": [group_name(g) for g in FINALIST_GROUPS],
            "seed_start": cfg["seeds"]["start"],
            "seed_count": cfg["seeds"]["count"],
            "planned_runs": runs,
            "planned_evaluations": runs * cfg["population"] * cfg["generations"]}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan"])
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    args = parser.parse_args()
    cfg, _, _ = config(args.config)
    print(json.dumps(plan(cfg), indent=2))


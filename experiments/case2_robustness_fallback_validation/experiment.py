"""Independent validation of high-noise and gap-specific fallback combinations."""
from pathlib import Path
import hashlib
import importlib.util
import json

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent / "case2_robustness_validation"

spec = importlib.util.spec_from_file_location("case2_fallback_base", BASE_DIR / "experiment.py")
BASE = importlib.util.module_from_spec(spec); spec.loader.exec_module(BASE)

GROUPS = (
    (),
    BASE.CORE,
    ("reactance_crossing_direction_consistency",),
    ("DTW", "LFP"),
    ("DTW", "LMEP"),
)
NAMES = {GROUPS[0]: "baseline", GROUPS[1]: "I03+II02", GROUPS[2]: "I03",
         GROUPS[3]: "P03+P04", GROUPS[4]: "P03+P05"}


def group_name(group): return NAMES[tuple(group)]


def config(path):
    cfg, frequency, target = BASE.config(path)
    if tuple(tuple(g) for g in cfg.get("groups", [])) != GROUPS:
        raise ValueError("Fallback group protocol changed")
    for condition in cfg["conditions"]:
        f, z, _ = BASE.observation(frequency, target, condition, cfg["seeds"]["start"])
        for group in GROUPS: BASE.Objectives(f, cfg["features"], group, target_z=z)
    return cfg, frequency, target


def dependency_files():
    files = dict(BASE.dependency_files())
    files.update({"fallback/experiment.py": HERE / "experiment.py",
                  "fallback/launch_batch.py": HERE / "launch_batch.py",
                  "fallback/report_results.py": HERE / "report_results.py",
                  "shared/robustness_experiment.py": BASE_DIR / "experiment.py"})
    return files


def manifest(cfg, frequency, target):
    record = BASE.manifest(cfg, frequency, target)
    record["code_sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                             for name, path in dependency_files().items()}
    record["fingerprint"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    return record


BASE.group_name = group_name
def run_one(cfg, frequency, target, condition, group, seed, directory, fingerprint):
    return BASE.run_one(cfg, frequency, target, condition, group, seed, directory, fingerprint)


def plan(cfg):
    runs = len(cfg["conditions"]) * len(GROUPS) * cfg["seeds"]["count"]
    return {"conditions": [c["name"] for c in cfg["conditions"]],
            "groups": [group_name(g) for g in GROUPS], "seed_start": cfg["seeds"]["start"],
            "seed_count": cfg["seeds"]["count"], "planned_runs": runs,
            "planned_evaluations": runs * cfg["population"] * cfg["generations"]}


if __name__ == "__main__":
    cfg, _, _ = config(HERE / "config.json"); print(json.dumps(plan(cfg), indent=2))


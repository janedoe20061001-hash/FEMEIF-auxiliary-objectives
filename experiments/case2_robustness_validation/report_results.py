"""Audit robustness runs and produce paired clean-target comparisons."""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import hashlib
import json
import numpy as np
import experiment as e
from objectives import Objectives
from case2 import impedance
from final_selection import select_wsmse


def paired(candidate, baseline, components, resamples=20000):
    delta = np.asarray(candidate) - np.asarray(baseline)
    rng = np.random.default_rng(20260921)
    boot = delta[rng.integers(0, len(delta), (resamples, len(delta)))].mean(axis=1)
    ordinary = np.quantile(boot, [.025, .975], axis=0)
    adjusted = np.quantile(boot, [.025 / components, 1 - .025 / components], axis=0)
    return delta, ordinary, adjusted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(); root = args.directory.resolve()
    mf = json.loads((root / "manifest.json").read_text()); cfg = mf["config"]
    for name, path in e.dependency_files().items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != mf["code_sha256"][name]:
            raise ValueError(f"Code changed: {name}")
    cfg2, full_frequency, full_target = e.config(e.HERE / "config.json")
    if cfg2 != cfg:
        raise ValueError("Config changed")
    seeds = list(range(cfg["seeds"]["start"], cfg["seeds"]["start"] + cfg["seeds"]["count"]))
    expected = {(c["name"], e.group_name(g), s) for c in cfg["conditions"] for g in e.GROUPS for s in seeds}
    seen, records = set(), []
    initial = defaultdict(set)
    for path in sorted(root.glob("*/*/seed_*/summary.json")):
        record = json.loads(path.read_text()); key = (record["condition"], record["group"], record["seed"])
        if key in seen or record["fingerprint"] != mf["fingerprint"]:
            raise ValueError(f"Invalid record: {path}")
        seen.add(key)
        x0 = np.load(path.parent / "initial_log_parameters.npy")
        digest = hashlib.sha256(x0.tobytes()).hexdigest()
        if digest != record["initial_population_sha256"]:
            raise ValueError("Initial hash mismatch")
        initial[(record["condition"], record["seed"])].add(digest)
        condition = next(c for c in cfg["conditions"] if c["name"] == record["condition"])
        frequency, target, metadata = e.observation(full_frequency, full_target, condition, record["seed"])
        with np.load(path.parent / "final_population.npz") as population:
            x, values = population["log_parameters"], population["objectives"]
            group = () if record["group"] == "baseline" else e.CORE
            verified = Objectives(frequency, cfg["features"], group, target_z=target).evaluate_spectra(
                impedance(frequency, 10 ** x))
            np.testing.assert_allclose(verified, values, rtol=1e-11, atol=1e-12)
            selection = select_wsmse(values, cfg["final_selection"]["phase_weight"])
            if selection["wsmse_selected_index"] != int(population["selected_index"]):
                raise ValueError("Selection mismatch")
            clean = Objectives(full_frequency, cfg["features"], e.DIAGNOSTICS,
                               target_z=full_target).evaluate_spectra(
                                   impedance(full_frequency, population["selected_parameters"]))[0]
            np.testing.assert_allclose(clean, population["selected_clean_metrics"], rtol=1e-11, atol=1e-12)
            np.testing.assert_allclose(clean[:2], [record["clean_magnitude_mse"], record["clean_phase_mse"]], rtol=1e-11, atol=1e-12)
        if metadata["observed_target_sha256"] != record["observed_target_sha256"]:
            raise ValueError("Observation mismatch")
        records.append(record)
    if seen != expected or any(len(v) != 1 for v in initial.values()):
        raise ValueError("Missing runs or unmatched paired initialization")
    data = {(r["condition"], r["group"], r["seed"]): r for r in records}
    components = 2 * len(cfg["conditions"])
    output = []
    text = ["# Case2 噪声与缺失频点鲁棒性", "",
            f"完成 {len(records)} 次优化；每个条件 {len(seeds)} 个配对 seeds。优化使用受扰观测，表中误差统一在原始 201 点实测曲线上计算。",
            f"对 {components} 个主效应分量进行 Bonferroni 调整；负变化表示 I-03+II-02 优于双主目标基线。", "",
            "|条件|观测点/扰动|幅值变化%|相位变化%|两项同降seed|幅值95%区间|相位95%区间|严格校正双通过|",
            "|---|---|---:|---:|---:|---|---|---|"]
    for condition in cfg["conditions"]:
        base = np.array([[data[condition["name"], "baseline", s]["clean_magnitude_mse"],
                          data[condition["name"], "baseline", s]["clean_phase_mse"]] for s in seeds])
        core = np.array([[data[condition["name"], "I03+II02", s]["clean_magnitude_mse"],
                          data[condition["name"], "I03+II02", s]["clean_phase_mse"]] for s in seeds])
        delta, ci, adjusted = paired(core, base, components)
        change = 100 * delta.mean(axis=0) / base.mean(axis=0)
        strict = bool(np.all(adjusted[1] < 0))
        example = data[condition["name"], "baseline", seeds[0]]
        detail = (f"σ={condition['sigma_magnitude_db']}dB/{condition['sigma_phase_degree']}°"
                  if condition["kind"] == "noise" else f"{example['observed_points']} points")
        row = {"condition": condition["name"], "kind": condition["kind"],
               "observed_points": example["observed_points"],
               "magnitude_change_pct": float(change[0]), "phase_change_pct": float(change[1]),
               "both_lower_seeds": int(np.all(delta < 0, axis=1).sum()),
               "magnitude_ci_low": float(ci[0, 0]), "magnitude_ci_high": float(ci[1, 0]),
               "phase_ci_low": float(ci[0, 1]), "phase_ci_high": float(ci[1, 1]),
               "magnitude_adjusted_low": float(adjusted[0, 0]), "magnitude_adjusted_high": float(adjusted[1, 0]),
               "phase_adjusted_low": float(adjusted[0, 1]), "phase_adjusted_high": float(adjusted[1, 1]),
               "strict_both": strict}
        output.append(row)
        text.append(f"|{condition['name']}|{detail}|{change[0]:.3f}|{change[1]:.3f}|{row['both_lower_seeds']}/{len(seeds)}|[{ci[0,0]:.4g}, {ci[1,0]:.4g}]|[{ci[0,1]:.4g}, {ci[1,1]:.4g}]|{'是' if strict else '否'}|")
    with (root / "robustness_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0])); writer.writeheader(); writer.writerows(output)
    (root / "RESULTS.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    audit = {"status": "passed", "runs": len(records), "expected_runs": len(expected),
             "same_condition_seed_initial_populations_match": True,
             "observations_reconstructed": True, "final_populations_recomputed": True,
             "clean_metrics_recomputed": True, "selection_recomputed": True,
             "fingerprint": mf["fingerprint"]}
    (root / "completion_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"status": "complete", "runs": len(records),
                      "strict_conditions": sum(r["strict_both"] for r in output)}))


if __name__ == "__main__":
    main()


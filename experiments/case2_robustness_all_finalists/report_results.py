"""Audit all added finalists and compare all ten candidates under robustness."""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import hashlib
import json
import numpy as np
import experiment as e

PAIR_NAME = "I03+II02"
ALL_NAMES = ("I03", "II02", PAIR_NAME, "I03+II01+II02", "P03", "P03+P04",
             "P03+P05", "P03+P04+P05", "P02+P03+P04+P05", "I03+II02+P03")


def paired(candidate, baseline, components, seed, resamples=20000):
    delta = np.asarray(candidate) - np.asarray(baseline)
    rng = np.random.default_rng(seed)
    boot = delta[rng.integers(0, len(delta), (resamples, len(delta)))].mean(axis=1)
    ordinary = np.quantile(boot, [.025, .975], axis=0)
    adjusted = np.quantile(boot, [.025/components, 1-.025/components], axis=0)
    return delta, ordinary, adjusted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--reference", type=Path,
                        default=e.HERE.parent / "case2_robustness_validation" / "runs")
    args = parser.parse_args(); root = args.directory.resolve(); reference = args.reference.resolve()
    mf = json.loads((root / "manifest.json").read_text()); cfg = mf["config"]
    for name, path in e.dependency_files().items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != mf["code_sha256"][name]:
            raise ValueError(f"Code changed: {name}")
    reference_audit = json.loads((reference / "completion_audit.json").read_text())
    if reference_audit.get("status") != "passed" or reference_audit.get("runs") != 1200:
        raise ValueError("Reference robustness batch is not audited")
    cfg2, full_frequency, full_target = e.config(e.HERE / "config.json")
    if cfg2 != cfg:
        raise ValueError("Config changed")
    seeds = list(range(cfg["seeds"]["start"], cfg["seeds"]["start"] + cfg["seeds"]["count"]))
    expected = {(c["name"], e.group_name(g), s) for c in cfg["conditions"] for g in e.FINALIST_GROUPS for s in seeds}
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
        frequency, target, metadata = e.BASE.observation(full_frequency, full_target, condition, record["seed"])
        group = next(g for g in e.FINALIST_GROUPS if e.group_name(g) == record["group"])
        with np.load(path.parent / "final_population.npz") as population:
            x, values = population["log_parameters"], population["objectives"]
            verified = e.BASE.Objectives(frequency, cfg["features"], group, target_z=target).evaluate_spectra(
                e.BASE.impedance(frequency, 10 ** x))
            np.testing.assert_allclose(verified, values, rtol=1e-11, atol=1e-12)
            selection = e.BASE.select_wsmse(values, cfg["final_selection"]["phase_weight"])
            if selection["wsmse_selected_index"] != int(population["selected_index"]):
                raise ValueError("Selection mismatch")
            clean = e.BASE.Objectives(full_frequency, cfg["features"], e.BASE.DIAGNOSTICS,
                                      target_z=full_target).evaluate_spectra(
                                          e.BASE.impedance(full_frequency, population["selected_parameters"]))[0]
            np.testing.assert_allclose(clean, population["selected_clean_metrics"], rtol=1e-11, atol=1e-12)
        if metadata["observed_target_sha256"] != record["observed_target_sha256"]:
            raise ValueError("Observation mismatch")
        baseline_initial = np.load(reference / record["condition"] / "baseline" /
                                   f"seed_{record['seed']:03d}" / "initial_log_parameters.npy")
        if hashlib.sha256(baseline_initial.tobytes()).hexdigest() != digest:
            raise ValueError("Reference/new initial populations differ")
        records.append(record)
    if seen != expected or any(len(v) != 1 for v in initial.values()):
        raise ValueError("Missing runs or unmatched initial populations")

    new_data = {(r["condition"], r["group"], r["seed"]): r for r in records}
    reference_data = {}
    for path in reference.glob("*/*/seed_*/summary.json"):
        r = json.loads(path.read_text())
        if r["group"] in ("baseline", PAIR_NAME):
            reference_data[r["condition"], r["group"], r["seed"]] = r
    rows = []
    global_components = 2 * len(cfg["conditions"]) * len(ALL_NAMES)
    condition_components = 2 * len(ALL_NAMES)
    alpha = cfg["final_selection"]["phase_weight"]
    for condition_index, condition in enumerate(cfg["conditions"]):
        name = condition["name"]
        baseline = np.array([[reference_data[name, "baseline", s]["clean_magnitude_mse"],
                              reference_data[name, "baseline", s]["clean_phase_mse"]] for s in seeds])
        for group_index, group_name in enumerate(ALL_NAMES):
            source = reference_data if group_name == PAIR_NAME else new_data
            candidate = np.array([[source[name, group_name, s]["clean_magnitude_mse"],
                                   source[name, group_name, s]["clean_phase_mse"]] for s in seeds])
            delta, ci, condition_ci = paired(candidate, baseline, condition_components,
                                             20260921 + condition_index * 100 + group_index)
            _, _, global_ci = paired(candidate, baseline, global_components,
                                     20261921 + condition_index * 100 + group_index)
            change = 100 * delta.mean(axis=0) / baseline.mean(axis=0)
            wsmse = candidate[:, 0] + alpha * candidate[:, 1]
            base_wsmse = baseline[:, 0] + alpha * baseline[:, 1]
            rows.append({
                "condition": name, "group": group_name,
                "magnitude_mean": float(candidate[:, 0].mean()),
                "phase_mean": float(candidate[:, 1].mean()),
                "magnitude_change_pct": float(change[0]), "phase_change_pct": float(change[1]),
                "wsmse_change_pct": float(100 * (wsmse.mean() / base_wsmse.mean() - 1)),
                "both_lower_seeds": int(np.all(delta < 0, axis=1).sum()),
                "ordinary_both": bool(np.all(ci[1] < 0)),
                "condition_adjusted_both": bool(np.all(condition_ci[1] < 0)),
                "global_adjusted_both": bool(np.all(global_ci[1] < 0)),
                "magnitude_ci_low": float(ci[0, 0]), "magnitude_ci_high": float(ci[1, 0]),
                "phase_ci_low": float(ci[0, 1]), "phase_ci_high": float(ci[1, 1]),
                "magnitude_condition_adjusted_high": float(condition_ci[1, 0]),
                "phase_condition_adjusted_high": float(condition_ci[1, 1]),
                "magnitude_global_adjusted_high": float(global_ci[1, 0]),
                "phase_global_adjusted_high": float(global_ci[1, 1]),
            })
    with (root / "all_finalists_robustness.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    text = ["# Case2 全部入围组合鲁棒性比较", "",
            "新增 5400 次运行，并复用已审计的 1200 次基线与 I-03+II-02 结果。所有指标均在原始 201 点目标上重新计算。", "",
            "|条件|WSMSE最佳组合|幅值变化%|相位变化%|WSMSE变化%|条件内严格双通过组数|全局严格双通过组数|",
            "|---|---|---:|---:|---:|---:|---:|"]
    for condition in cfg["conditions"]:
        subset = [r for r in rows if r["condition"] == condition["name"]]
        best = min(subset, key=lambda r: r["magnitude_mean"] + alpha * r["phase_mean"])
        text.append(f"|{condition['name']}|{best['group']}|{best['magnitude_change_pct']:.3f}|{best['phase_change_pct']:.3f}|{best['wsmse_change_pct']:.3f}|{sum(r['condition_adjusted_both'] for r in subset)}/{len(subset)}|{sum(r['global_adjusted_both'] for r in subset)}/{len(subset)}|")
    text += ["", "## 全部组合", "",
             "|条件|组合|幅值变化%|相位变化%|WSMSE变化%|两项同降seed|普通双通过|条件内校正双通过|全局校正双通过|",
             "|---|---|---:|---:|---:|---:|---|---|---|"]
    for row in rows:
        text.append(f"|{row['condition']}|{row['group']}|{row['magnitude_change_pct']:.3f}|{row['phase_change_pct']:.3f}|{row['wsmse_change_pct']:.3f}|{row['both_lower_seeds']}/{len(seeds)}|{'是' if row['ordinary_both'] else '否'}|{'是' if row['condition_adjusted_both'] else '否'}|{'是' if row['global_adjusted_both'] else '否'}|")
    (root / "RESULTS.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    audit = {"status": "passed", "new_runs": len(records), "expected_new_runs": len(expected),
             "reference_runs": 1200, "reference_audit_passed": True,
             "same_condition_seed_initial_populations_match_reference": True,
             "observations_reconstructed": True, "final_populations_recomputed": True,
             "clean_metrics_recomputed": True, "selection_recomputed": True,
             "fingerprint": mf["fingerprint"]}
    (root / "completion_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"status": "complete", "new_runs": len(records),
                      "comparisons": len(rows)}))


if __name__ == "__main__":
    main()


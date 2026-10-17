"""Audit all champion populations and generate paired reports."""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import hashlib
import json
import numpy as np

import experiment as e
from objectives import CATALOG, Objectives
from case2 import PARAMETER_NAMES, LOWER, UPPER, load_target, impedance
from final_selection import select_wsmse
from summarize import summarize
from analysis import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    mf = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cfg = mf["config"]
    for name, path in e._dependency_files().items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != mf["code_sha256"][name]:
            raise ValueError(f"Code changed: {name}")
    frequency, target = load_target(cfg["target_data"])
    alpha = cfg["final_selection"]["phase_weight"]
    expected_groups = [e.group_name(g) for g in e.groups(cfg)]
    seeds = range(cfg["seeds"]["start"], cfg["seeds"]["start"] + cfg["seeds"]["count"])
    expected = {(g, seed) for g in expected_groups for seed in seeds}
    seen, records, selected = set(), [], []
    initials = defaultdict(set)
    for path in sorted(root.glob("*/seed_*/summary.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        key = (record["group"], record["seed"])
        if (key in seen or record["fingerprint"] != mf["fingerprint"] or
                record["status"] != "complete"):
            raise ValueError(f"Invalid record: {path}")
        seen.add(key)
        initial = np.load(path.parent / "initial_log_parameters.npy")
        digest = hashlib.sha256(initial.tobytes()).hexdigest()
        if digest != record["initial_population_sha256"]:
            raise ValueError("Initial population hash mismatch")
        initials[record["seed"]].add(digest)
        with np.load(path.parent / "final_population.npz") as population:
            x = population["log_parameters"]
            f = population["objectives"]
            if (x.shape != (cfg["population"], 15) or
                    f.shape != (cfg["population"], len(record["objective_names"]))):
                raise ValueError("Population shape mismatch")
            if np.any(x < LOWER) or np.any(x > UPPER) or not np.isfinite(f).all():
                raise ValueError("Invalid population values")
            verified = Objectives(frequency, cfg["features"], target_z=target).evaluate_spectra(
                impedance(frequency, 10 ** x))
            np.testing.assert_allclose(verified, f[:, :2], rtol=1e-11, atol=1e-12)
            check = select_wsmse(f, alpha)
            index = check["wsmse_selected_index"]
            if index != int(population["selected_index"]):
                raise ValueError("Selected index mismatch")
            row = {"group": record["group"], "seed": record["seed"], **check}
            row.update({name: float(value) for name, value in
                        zip(PARAMETER_NAMES, population["selected_parameters"])})
            selected.append(row)
        records.append(record)
    if seen != expected or any(len(v) != 1 for v in initials.values()):
        raise ValueError("Missing runs or unpaired initial populations")
    rows = summarize(root)
    main_rows = {r["group"]: r for r in rows if r["front"] == "wsmse_selected"}
    with (root / "selected_individuals.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    labels = {"baseline": ("BASE", "双主目标基线")}
    labels.update({name: (ident, title) for ident, name, title in CATALOG})

    def label(group):
        if group == "baseline":
            return labels[group]
        parts = group.split("+")
        return "+".join(labels[p][0] for p in parts), "＋".join(labels[p][1] for p in parts)

    improved = []
    text = [f"# {cfg['experiment_name']}", "",
            f"完成 {len(records)} 次运行；{len(expected_groups)} 组，每组 {cfg['seeds']['count']} seeds，种群 {cfg['population']}，{cfg['generations']} 代。",
            f"每个 seed 从最终种群选择 magnitude MSE + {alpha} × phase MSE 最小的同一个体；WSMSE 不参与进化。", "",
            "|组别|辅助目标|幅值均值|相位均值|幅值变化%|相位变化%|两项同降seed|",
            "|---|---|---:|---:|---:|---:|---:|"]
    for group in expected_groups:
        row = main_rows[group]
        ident, title = label(group)
        if group != "baseline" and row["both_means_lower"]:
            improved.append(group)
        text.append(f"|{ident}|{title}|{row['magnitude_mean']:.6g}|{row['phase_mean']:.6g}|{row['magnitude_change_pct']:.3f}|{row['phase_change_pct']:.3f}|{row['seeds_with_both_lower']}/{cfg['seeds']['count']}|")
    text[2] += f" 两项均值均低于基线：{len(improved)}/{len(expected_groups)-1}。"
    (root / "RESULTS.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    audit = {"status": "passed", "runs": len(records), "expected_runs": len(expected),
             "same_seed_initial_populations_match": True,
             "final_populations_recomputed": True, "selection_recomputed": True,
             "fingerprint": mf["fingerprint"]}
    (root / "completion_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    analyze(records, cfg, root)
    print(json.dumps({"status": "complete", "runs": len(records),
                      "improved": len(improved)}))


if __name__ == "__main__":
    main()


"""Describe selected-parameter stability; this is not a truth-recovery test."""
from pathlib import Path
import argparse
import csv
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
EXTENDED = HERE.parent / "case2_extended_pairs_validation"
if str(EXTENDED) not in sys.path:
    sys.path.append(str(EXTENDED))
from case2 import PARAMETER_NAMES, LOWER, UPPER


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected = manifest["config"]["seeds"]["count"]
    rows = list(csv.DictReader((root / "selected_individuals.csv").open(
        encoding="utf-8-sig")))
    grouped = {}
    for row in rows:
        grouped.setdefault(row["group"], []).append(
            [float(row[name]) for name in PARAMETER_NAMES])
    if not grouped or any(len(values) != expected for values in grouped.values()):
        raise ValueError("Incomplete selected-parameter table")
    baseline = np.log10(np.asarray(grouped["baseline"], float))
    baseline_iqr = np.quantile(baseline, .75, axis=0) - np.quantile(baseline, .25, axis=0)
    summary, detailed = [], []
    span = UPPER - LOWER
    for group, values in grouped.items():
        log_values = np.log10(np.asarray(values, float))
        q25, median, q75 = np.quantile(log_values, [.25, .5, .75], axis=0)
        iqr = q75 - q25
        near = (log_values <= LOWER + .01 * span) | (log_values >= UPPER - .01 * span)
        finite_ratio = np.divide(iqr, baseline_iqr, out=np.full_like(iqr, np.nan),
                                 where=baseline_iqr > 0)
        summary.append({
            "group": group, "seeds": len(log_values),
            "median_parameter_log10_iqr": float(np.median(iqr)),
            "max_parameter_log10_iqr": float(np.max(iqr)),
            "parameters_with_lower_iqr_than_baseline": int(np.sum(iqr < baseline_iqr)),
            "median_iqr_ratio_to_baseline": float(np.nanmedian(finite_ratio)),
            "boundary_near_rate": float(near.mean()),
            "seeds_with_any_boundary_near": int(near.any(axis=1).sum()),
        })
        for index, name in enumerate(PARAMETER_NAMES):
            detailed.append({
                "group": group, "parameter": name,
                "log10_q25": float(q25[index]), "log10_median": float(median[index]),
                "log10_q75": float(q75[index]), "log10_iqr": float(iqr[index]),
                "iqr_ratio_to_baseline": float(finite_ratio[index]),
                "boundary_near_rate": float(near[:, index].mean()),
            })
    with (root / "parameter_stability.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    with (root / "parameter_stability_by_parameter.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detailed[0]))
        writer.writeheader(); writer.writerows(detailed)
    text = ["# Case2 冠军复赛：选解参数稳定性", "",
            "IQR 在 log10 参数空间计算；越小只表示跨 seed 更集中，不代表更接近真实参数。",
            "Case2 没有真实元件参数，因此本报告不能用于参数恢复准确性结论。", "",
            "|组别|log10 IQR中位数|最大log10 IQR|15项中比基线更集中|IQR比值中位数|边界附近比例|至少一项近边界的seed|",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        text.append(
            f"|{row['group']}|{row['median_parameter_log10_iqr']:.4f}|"
            f"{row['max_parameter_log10_iqr']:.4f}|"
            f"{row['parameters_with_lower_iqr_than_baseline']}/15|"
            f"{row['median_iqr_ratio_to_baseline']:.3f}|"
            f"{100*row['boundary_near_rate']:.2f}%|"
            f"{row['seeds_with_any_boundary_near']}/{row['seeds']}|")
    (root / "PARAMETER_STABILITY.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "groups": len(summary),
                      "records": len(rows)}))


if __name__ == "__main__":
    main()


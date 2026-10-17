"""Compare final populations in the two-primary-objective plane without WSMSE weights."""
import argparse
import csv
from io import BytesIO
import json
from pathlib import Path
import tarfile

import numpy as np

from objectives import CATALOG


def primary_front(points):
    points = np.unique(np.asarray(points, float), axis=0)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all() or np.any(points <= 0):
        raise ValueError("Expected positive finite two-objective points")
    weak = np.all(points[:, None, :] <= points[None, :, :], axis=2)
    strict = np.any(points[:, None, :] < points[None, :, :], axis=2)
    dominated = np.any(weak & strict, axis=0)
    front = points[~dominated]
    return front[np.lexsort((front[:, 1], front[:, 0]))]


def coverage(a, b):
    """Fraction of B weakly dominated by at least one point in A."""
    return float(np.mean(np.any(np.all(a[:, None, :] <= b[None, :, :], axis=2), axis=0)))


def multiplicative_epsilon(a, b):
    """Smallest multiplicative factor by which A weakly covers all of B."""
    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("Multiplicative epsilon requires strictly positive errors")
    return float(np.max(np.min(np.max(a[:, None, :] / b[None, :, :], axis=2), axis=0)))


def load_fronts(archive_path):
    fronts = {}
    manifests = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            parts = Path(member.name).parts
            if member.isfile() and parts[-1] == "manifest.json" and parts[0] in ("runs", "reference"):
                manifests[parts[0]] = json.loads(archive.extractfile(member).read())
            if not (member.isfile() and member.name.endswith("/final_population.npz")):
                continue
            if parts[0] not in ("runs", "reference"):
                continue
            group, seed_text = parts[-3], parts[-2]
            seed = int(seed_text.removeprefix("seed_"))
            with np.load(BytesIO(archive.extractfile(member).read())) as data:
                fronts[group, seed] = primary_front(data["objectives"][:, :2])
    if set(manifests) != {"runs", "reference"}:
        raise ValueError("Archive must contain run and reference manifests")
    return fronts, manifests


def bootstrap_interval(values, seed=20260921, resamples=20000):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    return np.quantile(values[indices].mean(axis=1), [0.025, 0.975]).tolist()


def compare_pair(fronts, group, reference, seeds):
    rows = []
    for seed in seeds:
        a, b = fronts[group, seed], fronts[reference, seed]
        cab, cba = coverage(a, b), coverage(b, a)
        eab, eba = multiplicative_epsilon(a, b), multiplicative_epsilon(b, a)
        rows.append(dict(
            group=group, reference_group=reference, seed=seed,
            candidate_front_size=len(a), reference_front_size=len(b),
            coverage_candidate_over_reference=cab,
            coverage_reference_over_candidate=cba,
            coverage_advantage=cab-cba,
            epsilon_candidate_to_reference=eab,
            epsilon_reference_to_candidate=eba,
            log_epsilon_advantage=float(np.log(eba/eab)),
            candidate_completely_covers_reference=eab <= 1.0,
            reference_completely_covers_candidate=eba <= 1.0,
            candidate_min_magnitude=float(a[:, 0].min()),
            reference_min_magnitude=float(b[:, 0].min()),
            candidate_min_phase=float(a[:, 1].min()),
            reference_min_phase=float(b[:, 1].min()),
        ))
    return rows


def summarize(rows):
    coverage_values = np.array([r["coverage_advantage"] for r in rows])
    epsilon_values = np.array([r["log_epsilon_advantage"] for r in rows])
    cm = np.array([r["candidate_min_magnitude"] for r in rows])
    rm = np.array([r["reference_min_magnitude"] for r in rows])
    cp = np.array([r["candidate_min_phase"] for r in rows])
    rp = np.array([r["reference_min_phase"] for r in rows])
    return dict(
        group=rows[0]["group"], reference_group=rows[0]["reference_group"], seeds=len(rows),
        candidate_front_size_mean=float(np.mean([r["candidate_front_size"] for r in rows])),
        reference_front_size_mean=float(np.mean([r["reference_front_size"] for r in rows])),
        coverage_candidate_over_reference_mean=float(np.mean([r["coverage_candidate_over_reference"] for r in rows])),
        coverage_reference_over_candidate_mean=float(np.mean([r["coverage_reference_over_candidate"] for r in rows])),
        coverage_advantage_mean=float(coverage_values.mean()),
        coverage_advantage_ci95=bootstrap_interval(coverage_values),
        coverage_win_seeds=int(np.sum(coverage_values > 0)),
        coverage_tie_seeds=int(np.sum(coverage_values == 0)),
        epsilon_candidate_to_reference_geomean=float(np.exp(np.mean(np.log([r["epsilon_candidate_to_reference"] for r in rows])))),
        epsilon_reference_to_candidate_geomean=float(np.exp(np.mean(np.log([r["epsilon_reference_to_candidate"] for r in rows])))),
        log_epsilon_advantage_mean=float(epsilon_values.mean()),
        log_epsilon_advantage_ci95=bootstrap_interval(epsilon_values),
        epsilon_win_seeds=int(np.sum(epsilon_values > 0)),
        candidate_complete_cover_seeds=int(np.sum([r["candidate_completely_covers_reference"] for r in rows])),
        reference_complete_cover_seeds=int(np.sum([r["reference_completely_covers_candidate"] for r in rows])),
        magnitude_endpoint_change_pct=float(100*(cm.mean()/rm.mean()-1)),
        phase_endpoint_change_pct=float(100*(cp.mean()/rp.mean()-1)),
    )


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    fronts, manifests = load_fronts(args.archive)
    run_cfg = manifests["runs"]["config"]
    seeds = list(range(run_cfg["seeds"]["start"], run_cfg["seeds"]["start"]+run_cfg["seeds"]["count"]))
    groups = sorted({group for group, _ in fronts})
    expected = {(group, seed) for group in groups for seed in seeds}
    if set(fronts) != expected or "baseline" not in groups or len(groups) != 14:
        raise ValueError("Expected 14 complete groups x 100 paired seeds")
    baseline_rows = []
    for group in groups:
        if group != "baseline": baseline_rows.extend(compare_pair(fronts, group, "baseline", seeds))
    baseline_summary = [summarize([r for r in baseline_rows if r["group"] == group]) for group in groups if group != "baseline"]
    constituent_rows = []
    for group in groups:
        members = set(group.split("+"))
        if len(members) < 2: continue
        for reference in groups:
            ref = set(reference.split("+"))
            if reference != "baseline" and ref < members:
                constituent_rows.extend(compare_pair(fronts, group, reference, seeds))
    constituent_summary = [summarize([r for r in constituent_rows if (r["group"],r["reference_group"]) == key])
                           for key in sorted({(r["group"],r["reference_group"]) for r in constituent_rows})]
    write_csv(args.output/"pareto_seedwise.csv", baseline_rows)
    write_csv(args.output/"pareto_summary.csv", baseline_summary)
    write_csv(args.output/"pareto_constituent_seedwise.csv", constituent_rows)
    write_csv(args.output/"pareto_constituent_summary.csv", constituent_summary)
    labels = {name: ident for ident, name, _ in CATALOG}
    label = lambda group: "+".join(labels[x] for x in group.split("+"))
    combos = [r for r in baseline_summary if "+" in r["group"]]
    text = ["# 不使用WSMSE权重的双主目标前沿比较", "",
        "每个seed只用幅值MSE和相位MSE重新提取最终种群的非支配前沿；辅助目标和17.38均不参与评价。重复目标点先去重。",
        "C(A,B)是A前沿弱支配的B前沿点比例；覆盖优势为C(A,B)-C(B,A)，正数偏向辅助组。乘法epsilon优势为log[Iε(B,A)/Iε(A,B)]，正数同样偏向辅助组。epsilon不使用两主目标权重，且对目标单位的正比例缩放不敏感。",
        "95%区间按100个配对seed整体重采样20000次；这是统计区间，不是幅值—相位偏好参数。端点变化分别使用每个前沿的最小幅值和最小相位，两个端点通常不是同一个个体。", "",
        "|组合|C(组合,基线)|C(基线,组合)|覆盖优势[95%区间]|覆盖胜seed|epsilon优势[95%区间]|epsilon胜seed|组合完全覆盖|基线完全覆盖|幅值端点变化%|相位端点变化%|",
        "|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|"]
    for r in combos:
        c=r["coverage_advantage_ci95"]; e=r["log_epsilon_advantage_ci95"]
        text.append(f"|{label(r['group'])}|{r['coverage_candidate_over_reference_mean']:.3f}|{r['coverage_reference_over_candidate_mean']:.3f}|{r['coverage_advantage_mean']:.3f} [{c[0]:.3f},{c[1]:.3f}]|{r['coverage_win_seeds']}/100|{r['log_epsilon_advantage_mean']:.3f} [{e[0]:.3f},{e[1]:.3f}]|{r['epsilon_win_seeds']}/100|{r['candidate_complete_cover_seeds']}/100|{r['reference_complete_cover_seeds']}/100|{r['magnitude_endpoint_change_pct']:.2f}|{r['phase_endpoint_change_pct']:.2f}|")
    both_positive=sum(r["coverage_advantage_ci95"][0]>0 and r["log_epsilon_advantage_ci95"][0]>0 for r in combos)
    text += ["", f"八个组合中，{sum(r['coverage_advantage_mean']>0 and r['log_epsilon_advantage_mean']>0 for r in combos)}/8在覆盖优势和epsilon优势的均值上都优于baseline；{both_positive}/8的两个95%区间下界也都高于零。",
             "前沿覆盖率仍受最终种群在前沿上的采样密度影响，因此同时报告更偏向最坏覆盖关系的乘法epsilon；不以简单的前沿坐标平均值作为主结论。",
             "组合相对各组成项的完整结果见pareto_constituent_summary.csv。"]
    (args.output/"PARETO_WEIGHT_FREE.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    meta=dict(status="complete",archive=str(args.archive),groups=len(groups),seeds=len(seeds),
              populations=len(fronts),comparisons_to_baseline=len(baseline_summary),
              constituent_comparisons=len(constituent_summary),uses_wsmse_weight=False)
    (args.output/"pareto_audit.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(meta,ensure_ascii=False))


if __name__ == "__main__": main()

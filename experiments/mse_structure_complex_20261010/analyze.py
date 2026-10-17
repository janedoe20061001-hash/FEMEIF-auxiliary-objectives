"""Independent curve checks; backend compares all seeds, plots a fixed few only."""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from protocol import targets, dump, source_hashes, verify_job
from sampling import Quadrature
from diagnostics import diagnostic


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def population_check(t, seed, generation, evaluated, cfg):
    """Seed is the statistical unit, not each of the 200 correlated candidates."""
    valid = {i:r for i,(r,audit) in evaluated.items() if audit and r["stable"]}
    def joint(r):
        ee = [a for a in r["events"] if a["width_evaluable"]]
        return all(a["complete"] for a in ee) if ee else None
    outcomes = [joint(r) for r in valid.values()]
    aggregate = dict(target=t["id"], seed=seed, generation=generation, checked_candidates=len(evaluated), valid_candidates=len(valid),
                     any_joint_amp_complete=any(v for v in outcomes if v is not None) if any(v is not None for v in outcomes) else None,
                     any_joint_phase_complete=any(all(a["complete"] for a in r["phase_events"]) for r in valid.values())
                          if any(r["phase_events"] for r in valid.values()) else None)
    rule = cfg["close_scores"]
    candidates = sorted(valid)
    similarities = {threshold:None for threshold in rule["thresholds"]}
    similar_count = {threshold:0 for threshold in rule["thresholds"]}
    for pos, i in enumerate(candidates):
        a = valid[i]
        if pos+1 == len(candidates):
            continue
        others = candidates[pos+1:]
        fs = np.array([valid[j]["objective"] for j in others])
        fa = np.array(a["objective"])
        dist = np.max(abs(fs-fa)/np.maximum(np.minimum(fs,fa),rule["floors"]),axis=1)
        for threshold in similarities:
            similar_count[threshold] += int(np.count_nonzero(dist<=threshold))
        for j,delta in zip(others,dist):
            if delta > max(similarities):
                continue
            b = valid[j]
            different = False
            for ea,eb in zip(a["events"],b["events"]):
                if ea["event_id"] != eb["event_id"]:
                    raise AssertionError("Target event order differs within population")
                if ea["matched"] != eb["matched"]:
                    different = True
                if ea["matched"] and eb["matched"]:
                    different |= abs(ea["height_delta_db"]-eb["height_delta_db"]) >= rule["height_difference_db"]
                    wa,wb = ea["width_ratio"],eb["width_ratio"]
                    if wa and wb:
                        different |= max(wa/wb,wb/wa) >= rule["width_ratio_between_fits"]
            if different:
                for threshold in similarities:
                    old = similarities[threshold]
                    if delta <= threshold and (old is None or delta < old["max_relative_score_difference"]):
                        similarities[threshold] = dict(index_a=i,index_b=j,max_relative_score_difference=float(delta))
    pairs = [dict(target=t["id"],seed=seed,generation=generation,threshold=threshold,
                  similar_score_pairs=similar_count[threshold],has_meaningful_shape_difference=pair is not None,
                  index_a=pair["index_a"] if pair else None,index_b=pair["index_b"] if pair else None,
                  max_relative_score_difference=pair["max_relative_score_difference"] if pair else None) for threshold,pair in similarities.items()]
    return aggregate,pairs


def plot_stage(out, t, g, curves, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    # One common plotting grid includes the target and ALL selected candidate anchors.
    q = Quadrature(t["circuit"], cfg)
    x = np.unique(np.concatenate([q.reference_grid(10**p, 2) for _, p in curves]))
    z0, _ = t["circuit"].response(10**x)
    axes[0].plot(10**x, 20*np.log10(abs(z0)), "k--", label="Target")
    axes[1].plot(10**x, np.rad2deg(np.angle(z0)), "k--", label="Target")
    exports = [np.column_stack([10**x, z0.real, z0.imag, 20*np.log10(abs(z0)), np.angle(z0)])]
    for seed, p in curves:
        z, _ = t["circuit"].response(10**x, 10**p)
        axes[0].plot(10**x, 20*np.log10(abs(z)), label=f"seed {seed}")
        axes[1].plot(10**x, np.rad2deg(np.angle(z)), label=f"seed {seed}")
        exports.append(np.column_stack([10**x,z.real,z.imag,20*np.log10(abs(z)),np.angle(z)]))
    axes[0].set(ylabel="Magnitude (dB ohm)", title=f'{t["id"]}, generation {g}')
    axes[1].set(ylabel="Phase (degree)", xlabel="Frequency (Hz)")
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    path = out / t["id"]
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / f"stage_{g:04}.png", dpi=160)
    plt.close(fig)
    np.savez_compressed(path / f"stage_{g:04}_3d_data.npz", columns=np.array(["f_hz","ReZ_ohm","ImZ_ohm","magnitude_db","phase_rad"]),
                        target=exports[0], curves=np.array(exports[1:]), seeds=np.array([seed for seed,_ in curves]))


def analyze(batch, prepared, out, population="all", all_stages=False, plots=False):
    manifest = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
    cfg, planned = manifest["config"], manifest["plan"]
    if manifest["hashes"] != source_hashes(cfg):
        raise RuntimeError("Analysis code/config changed from frozen run")
    expected_inputs = manifest["prepared_manifest"]["target_hashes"]
    for name, digest in expected_inputs.items():
        if hashlib.sha256((prepared / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Prepared target modified: {name}")
    ids = sorted({t["circuit"]["id"] for t in planned["targets"]})
    ts = targets(cfg, planned["profile"], ids)
    out.mkdir(parents=True, exist_ok=False)
    all_rows, event_rows, phase_rows, local_rows, band_rows, coverage, failures, curves = [], [], [], [], [], [], [], {}
    population_rows, similar_pairs = [], []
    with (out / "details.jsonl").open("w", encoding="utf-8") as detail:
        for t in ts:
            meta = json.loads((prepared / t["id"] / "target.json").read_text(encoding="utf-8"))
            q = Quadrature(t["circuit"], cfg)
            for seed in cfg["seeds"]:
                d = batch / t["id"] / f"seed_{seed}"
                try:
                    verify_job(d,cfg)
                    ok, reason = True, ""
                except (OSError, ValueError, KeyError, RuntimeError) as err:
                    ok, reason = False, str(err)
                coverage.append(dict(target=t["id"], seed=seed, completed=ok, reason=reason))
                if not ok:
                    continue
                for file in sorted(d.glob("generation_*.npz")):
                    g = int(file.stem.split("_")[-1])
                    with np.load(file) as a:
                        X, F, front = a["log_parameters"], a["optimization_objective"], a["front_indices"]
                        indices = {"primary": int(a["selected_index"]), "balanced": int(a["balanced_index"])}
                    selected = set(indices.values())
                    if g == cfg["generations"] or all_stages:
                        selected |= set(range(len(X))) if population == "all" else set(front.tolist()) if population == "front" else set()
                    evaluated = {}
                    for index in sorted(selected):
                        rec, _ = diagnostic(q, X[index], meta["events"], meta["phase_events"])
                        audit = np.all(abs(np.array(rec["objective"]) - F[index]) <= np.array(cfg["audit"]["objective_atol"]) + cfg["audit"]["objective_rtol"] * abs(np.array(rec["objective"])))
                        identity = dict(target=t["id"], seed=seed, generation=g, index=index, on_front=bool(index in front))
                        detail.write(json.dumps(dict(**identity, training_objective=F[index].tolist(), audit_passed=bool(audit), **rec), allow_nan=False)+"\n")
                        evaluated[index] = rec, bool(audit)
                        if not audit or not rec["stable"]:
                            failures.append(dict(**identity, objective_ok=bool(audit), structure_stable=rec["stable"]))
                    if g == cfg["generations"] or all_stages:
                        aggregate,pairs = population_check(t,seed,g,evaluated,cfg)
                        population_rows.append(aggregate)
                        similar_pairs.extend(pairs)
                    for rule, index in indices.items():
                        rec, audit = evaluated[index]
                        ident = dict(target=t["id"], seed=seed, generation=g, rule=rule)
                        evaluable = [r for r in rec["events"] if r["width_evaluable"]]
                        all_rows.append(dict(**ident, stable=rec["stable"], audit_passed=audit,
                             mag=rec["objective"][0], pha=rec["objective"][1], amp_rmse_db=rec["amplitude_rmse_db"], phase_rmse_deg=rec["phase_rmse_deg"],
                             amp_events=len(rec["events"]), width_evaluable=len(evaluable), missing=sum(not r["matched"] for r in rec["events"]),
                             incomplete=sum(not r["complete"] for r in evaluable), extra=len(rec["extra_events"]),
                             phase_missing=sum(not r["matched"] for r in rec["phase_events"]), phase_incomplete=sum(not r["complete"] for r in rec["phase_events"]),
                             joint_amp_complete=all(r["complete"] for r in evaluable) if evaluable else None))
                        event_rows.extend(dict(**ident, **r) for r in rec["events"])
                        phase_rows.extend(dict(**ident, **r) for r in rec["phase_events"])
                        local_rows.extend(dict(**ident, **r) for r in rec["local"])
                        band_rows.extend(dict(**ident, **r) for r in rec["bands"])
                    if plots and seed in cfg["plot_seeds"] and g in (20, 100, cfg["generations"]):
                        curves.setdefault((t["id"], g), []).append((seed, X[indices["primary"]].copy()))
    for name, rows in [("representatives",all_rows),("events",event_rows),("phase_events",phase_rows),("local",local_rows),("bands",band_rows),("coverage",coverage),
                       ("population",population_rows),("similar_scores_different_shapes",similar_pairs)]:
        write_csv(out / f"{name}.csv", rows)
    groups = []
    for target in sorted({r["target"] for r in all_rows}):
        for rule in ("primary","balanced"):
            rows = [r for r in all_rows if r["target"] == target and r["rule"] == rule and r["generation"] == cfg["generations"]]
            valid = [r for r in rows if r["stable"] and r["audit_passed"]]
            rates = [r["joint_amp_complete"] for r in valid if r["joint_amp_complete"] is not None]
            groups.append(dict(target=target, rule=rule, seeds_completed=len(rows), seeds_valid=len(valid), seeds_planned=len(cfg["seeds"]),
                          mag_mean=float(np.mean([r["mag"] for r in valid])) if valid else None,
                          pha_mean=float(np.mean([r["pha"] for r in valid])) if valid else None,
                          joint_amp_complete_rate=float(np.mean(rates)) if rates else None))
    write_csv(out / "summary.csv", groups)
    # Paired outcomes ONLY within a topology: keep every seed, no curve averaging.
    paired = []
    lookup = {(r["target"],r["seed"],r["rule"]):r for r in all_rows if r["generation"] == cfg["generations"]}
    for t in ts:
        if t["intervention"] == "base":
            continue
        for seed in cfg["seeds"]:
            for rule in ("primary", "balanced"):
                a, b = lookup.get((t["base"].ident+"/base",seed,rule)), lookup.get((t["id"],seed,rule))
                valid = a is not None and b is not None and a["audit_passed"] and b["audit_passed"] and a["stable"] and b["stable"]
                paired.append(dict(target=t["id"],seed=seed,rule=rule,valid=bool(valid),
                              delta_mag=b["mag"]-a["mag"] if valid else None, delta_pha=b["pha"]-a["pha"] if valid else None,
                              caution="Global deltas compare different targets; structural comparisons require base_event_correspondence.json."))
    write_csv(out / "paired_global_deltas.csv", paired)
    # Whole retained population coverage; no claim about unevaluated generations.
    audit_summary = dict(passed=not failures and all(r["completed"] for r in coverage), failures=failures,
                         completed_jobs=sum(r["completed"] for r in coverage), planned_jobs=len(coverage),
                         population_scope=population, population_all_saved_stages=all_stages,
                         caveat="No significance claim; 8 seeds are exploratory. Missing jobs and failed numerical audits are disclosed.")
    dump(out / "AUDIT.json", audit_summary)
    table = "\n|目标|选解|完成/计划seed|核验通过seed|MAG均值|PHA均值|幅值结构全部合格的比例|\n|---|---|---:|---:|---:|---:|---:|\n"
    for r in groups:
        fmt = lambda v: "不适用" if v is None else f"{v:.5g}"
        table += f'|{r["target"]}|{r["rule"]}|{r["seeds_completed"]}/{r["seeds_planned"]}|{r["seeds_valid"]}|{fmt(r["mag_mean"])}|{fmt(r["pha_mean"])}|{fmt(r["joint_amp_complete_rate"])}|\n'
    (out / "REPORT.md").write_text("# 嵌套电路第二步结果\n\n"
        "这次仍用幅值、相位两项整体误差寻找参数，另外检查曲线的峰、谷和相位转变。\n\n"
        f"已完成 {audit_summary['completed_jobs']}/{len(coverage)} 个任务；数值核验异常 {len(failures)} 条。"
        "只有核验通过的 seed 纳入汇总，完整性请先看 AUDIT.json。\n\n"
        "summary.csv 是每个 seed 各选一条后的分数均值。曲线没有取平均。events.csv 看丢失、位置、高度和宽度；"
        "phase_events.csv 看相位转变；local.csv 看局部误差占总分多少；bands.csv 检查没有明显峰谷的频段。\n\n"
        "details.jsonl 还包括所选范围内的种群候选，可以判断结论是否只来自代表解选择。"
        "population.csv 记录该 seed 被检查的候选中是否存在结构全部合格的解；"
        "similar_scores_different_shapes.csv 检查两项总分接近时，局部形状是否仍差很多。\n\n"
        "幅值结构全部合格只对有可计算 3 dB 宽度的目标事件评价；不适用保留为空，不能当成成功或失败。"
        "相位转变单独评价。三维曲线数据为 f、Re Z、Im Z；如有图片，只叠加预先固定的少数 seed。\n\n"
        "**这些是探索性结果。** 不同电路的元件数、参数空间和实际峰谷不同，不能把结果直接归因于拓扑复杂度，"
        "也不能据此宣布 MSE 在理论上无法拟合。下一步要先看哪类缺陷在不同条件下反复出现，再决定辅助指标。\n"+table,
        encoding="utf-8")
    if plots:
        byid = {t["id"]:t for t in ts}
        for (ident,g), vv in curves.items():
            plot_stage(out, byid[ident], g, vv, cfg)
    return audit_summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--population", choices=["reps","front","all"], default="all")
    p.add_argument("--all-stages", action="store_true")
    p.add_argument("--plots", action="store_true")
    opt = p.parse_args()
    result = analyze(opt.batch,opt.prepared,opt.out,opt.population,opt.all_stages,opt.plots)
    print(json.dumps(dict(passed=result["passed"],numerical_failures=len(result["failures"]),completed_jobs=result["completed_jobs"])))
    if not result["passed"]:
        raise SystemExit(1)

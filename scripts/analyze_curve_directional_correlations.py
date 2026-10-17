"""Post-hoc correlation audit; never changes or reruns the frozen experiments.

Use common baseline populations for fair metric comparisons. Individuals within
a population are NOT independent experimental repeats. Report run-level median
correlations, IQR, and run-bootstrap intervals, with pooled values separately.
"""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr

WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE / "experiments/curve3d_directional_20261003"
RAW = ROOT / "server_run/raw_results"
OUT = ROOT / "correlations"
INPUTS = {}


def tracked(path):
    INPUTS[str(path.relative_to(WORKSPACE)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, tracked(path))
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


gm = module("correlation_global_metrics", ROOT / "global_metrics.py")
c1 = module("correlation_case1", RAW / "case1/legacy_objectives.py")
c2 = module("correlation_case2", RAW / "case2/case2.py")
CATALOG = gm.CATALOG[:5]
PRIMARY = ("magnitude_mse_db2", "phase_mse_rad2")
SEEDS = list(range(200, 300))
PER_RUN = []
SUMMARY = []
SELECTED_SUMMARY = []
PRIMARY_RELATIONS = []
MAX_PRIMARY_DIFFERENCE = 0.0
MAX_AUX_DIFFERENCE = 0.0
MAX_CORRELATION_DIFFERENCE = {"pearson": 0.0, "spearman": 0.0}
BOOTSTRAP_REPLICATES = 4000


def coefficients(x, y):
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite correlation input")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"pearson": None, "spearman": None}
    result = {"pearson": float(pearsonr(x, y).statistic),
              "spearman": float(spearmanr(x, y).statistic)}
    # Independently check both statistics, including average ranks for ties.
    def manual_pearson(a, b):
        a, b = a - np.mean(a), b - np.mean(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    def manual_rank(a):
        _, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
        ends = np.cumsum(counts)
        return ((ends - counts + 1 + ends) / 2)[inverse]
    independent = {"pearson": manual_pearson(x, y),
                   "spearman": manual_pearson(manual_rank(x), manual_rank(y))}
    for kind in result:
        difference = abs(result[kind] - independent[kind])
        if difference > 2e-12:
            raise AssertionError("Independent correlation calculation disagrees")
        MAX_CORRELATION_DIFFERENCE[kind] = max(MAX_CORRELATION_DIFFERENCE[kind], difference)
    return result


def audit_objectives(new, old):
    global MAX_PRIMARY_DIFFERENCE, MAX_AUX_DIFFERENCE
    if new.shape != old.shape or not np.allclose(new, old, rtol=2e-10, atol=1e-10):
        raise AssertionError("Recomputed objectives differ from frozen results")
    MAX_PRIMARY_DIFFERENCE = max(MAX_PRIMARY_DIFFERENCE, float(np.max(np.abs(new[:, :2] - old[:, :2]))))
    if new.shape[1] > 2:
        MAX_AUX_DIFFERENCE = max(MAX_AUX_DIFFERENCE, float(np.max(np.abs(new[:, 2:] - old[:, 2:]))))


def summarize(case, context, metric_id, metric, label, populations):
    """Each population belongs to one independent seed; keep these strata."""
    for primary_index, primary in enumerate(PRIMARY):
        local = []
        for seed, values in populations:
            row = dict(case=case, context=context, metric_id=metric_id,
                       metric=metric, label=label, primary=primary,
                       seed=seed, individuals=len(values),
                       **coefficients(values[:, 2], values[:, primary_index]))
            PER_RUN.append(row)
            local.append(row)
        all_values = np.concatenate([v for _, v in populations])
        pooled = coefficients(all_values[:, 2], all_values[:, primary_index])
        rng_seed = int(hashlib.sha256(f"{case}/{context}/{metric}/{primary}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(rng_seed)
        row = dict(case=case, context=context, metric_id=metric_id, metric=metric,
                   label=label, primary=primary, runs=len(populations),
                   individuals=len(all_values), pooled_pearson=pooled["pearson"],
                   pooled_spearman=pooled["spearman"])
        for kind in ("pearson", "spearman"):
            values = np.array([r[kind] for r in local if r[kind] is not None])
            row[f"{kind}_valid_runs"] = len(values)
            if len(values):
                median, q25, q75 = np.quantile(values, [.5, .25, .75])
                estimates = np.median(values[rng.integers(0, len(values), (BOOTSTRAP_REPLICATES, len(values)))], axis=1)
                lo, hi = np.quantile(estimates, [.025, .975])
                row.update({f"{kind}_median": float(median), f"{kind}_q25": float(q25),
                            f"{kind}_q75": float(q75), f"{kind}_median_bootstrap_lo": float(lo),
                            f"{kind}_median_bootstrap_hi": float(hi)})
            else:
                for suffix in ("median", "q25", "q75", "median_bootstrap_lo", "median_bootstrap_hi"):
                    row[f"{kind}_{suffix}"] = None
        SUMMARY.append(row)


def common_values(evaluators, z):
    output = [e.evaluate_spectra(z) for e in evaluators]
    for value in output[1:]:
        np.testing.assert_array_equal(value[:, :2], output[0][:, :2])
    return np.column_stack([output[0][:, :2]] + [v[:, 2] for v in output])


def selected_correlations(case, context, metric_id, name, label, values):
    assert len(values) == len(SEEDS)
    values = np.asarray(values)
    for k, primary in enumerate(PRIMARY):
        SELECTED_SUMMARY.append(dict(case=case, context=context, metric_id=metric_id,
            metric=name, label=label, primary=primary, independent_runs=len(values),
            **coefficients(values[:, 2], values[:, k])))


def process(case):
    cfg = json.loads(tracked(RAW / case / "config.json").read_text(encoding="utf-8"))
    assert cfg["seeds"] == {"start": 200, "count": 100}
    if case == "case1":
        frequency = np.geomspace(cfg["frequency"]["min_hz"], cfg["frequency"]["max_hz"], cfg["frequency"]["points"])
        model = c1.impedance
        target = model(frequency, c1.TARGET)[0]
    else:
        tracked(RAW / case / cfg["target_data"]["file"])
        frequency, target = c2.load_target({**cfg["target_data"], "file": str(RAW / case / cfg["target_data"]["file"])})
        model = c2.impedance
    evaluators = [gm.GlobalObjectives(frequency, aux=(name,), target_z=target) for _, name, _ in CATALOG]
    common = {"common_initial_population": [], "common_baseline_final_population": []}
    selected_baseline = []
    for seed in SEEDS:
        folder = RAW / case / "runs/baseline" / f"seed_{seed}"
        info = json.loads(tracked(folder / "summary.json").read_text(encoding="utf-8"))
        assert info["status"] == "complete" and info["seed"] == seed
        initial = np.load(tracked(folder / "initial_log_parameters.npy"), allow_pickle=False)
        with np.load(tracked(folder / "final_population.npz"), allow_pickle=False) as final:
            points = final["log_parameters"]
            stored = final["objectives"]
            selected_index = int(final["selected_index"])
            assert selected_index == info["wsmse_selected_index"]
        assert initial.shape[0] == points.shape[0] == cfg["population"]
        initial_values = common_values(evaluators, model(frequency, 10 ** initial))
        final_values = common_values(evaluators, model(frequency, 10 ** points))
        audit_objectives(final_values[:, :2], stored)
        common["common_initial_population"].append((seed, initial_values))
        common["common_baseline_final_population"].append((seed, final_values))
        selected_baseline.append(final_values[selected_index])
    selected_baseline = np.asarray(selected_baseline)
    for context, populations in common.items():
        for seed, values in populations:
            PRIMARY_RELATIONS.append(dict(case=case, context=context, seed=seed,
                individuals=len(values), **coefficients(values[:, 0], values[:, 1])))
    for index, (metric_id, name, label) in enumerate(CATALOG):
        selected_correlations(case, "common_baseline_selected_per_run", metric_id, name, label,
                              selected_baseline[:, [0, 1, 2 + index]])
        for context, populations in common.items():
            summarize(case, context, metric_id, name, label,
                      [(s, v[:, [0, 1, 2 + index]]) for s, v in populations])
        populations = []
        selected_own = []
        for seed in SEEDS:
            folder = RAW / case / "runs" / name / f"seed_{seed}"
            info = json.loads(tracked(folder / "summary.json").read_text(encoding="utf-8"))
            assert info["status"] == "complete" and info["objective_names"] == list(PRIMARY) + [name]
            with np.load(tracked(folder / "final_population.npz"), allow_pickle=False) as data:
                values = evaluators[index].evaluate_spectra(model(frequency, 10 ** data["log_parameters"]))
                audit_objectives(values, data["objectives"])
                selected_index = int(data["selected_index"])
                assert selected_index == info["wsmse_selected_index"]
                selected_own.append(values[selected_index])
            populations.append((seed, values))
        summarize(case, "own_auxiliary_final_population", metric_id, name, label, populations)
        selected_correlations(case, "own_auxiliary_selected_per_run", metric_id, name, label, selected_own)
    # A four-objective experiment has TWO auxiliary variables, not a synthetic score.
    combination = gm.GlobalObjectives(frequency, aux=(gm.AREA, gm.LENGTH), target_z=target)
    populations = []
    for seed in SEEDS:
        folder = RAW / case / "runs" / gm.COMBINATION / f"seed_{seed}"
        info = json.loads(tracked(folder / "summary.json").read_text(encoding="utf-8"))
        assert info["status"] == "complete" and info["objective_names"] == list(PRIMARY) + [gm.AREA, gm.LENGTH]
        with np.load(tracked(folder / "final_population.npz"), allow_pickle=False) as data:
            values = combination.evaluate_spectra(model(frequency, 10 ** data["log_parameters"]))
            audit_objectives(values, data["objectives"])
        populations.append((seed, values))
    for index, (metric_id, name, label) in enumerate(CATALOG[:2]):
        summarize(case, "two_auxiliary_final_population", metric_id, name, label,
                  [(s, v[:, [0, 1, 2 + index]]) for s, v in populations])
    print(f"{case}: completed common, own, and combined population correlations", flush=True)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(context, kind="spearman", intervals=False):
    lines = ["| 指标 | Case1：幅值 MSE | Case1：相位 MSE | Case2：幅值 MSE | Case2：相位 MSE |",
             "|---|---:|---:|---:|---:|"]
    for metric_id, name, label in CATALOG:
        cells = []
        for case in ("case1", "case2"):
            for primary in PRIMARY:
                row = next(r for r in SUMMARY if r["context"] == context and r["case"] == case and r["metric"] == name and r["primary"] == primary)
                value = f"{row[f'{kind}_median']:+.3f}"
                if intervals:
                    value += f" [{row[f'{kind}_q25']:+.3f}, {row[f'{kind}_q75']:+.3f}]"
                cells.append(value)
        lines.append(f"| {metric_id} {label.replace('对照：', '')} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def selected_table(kind):
    lines = ["| 指标 | Case1：幅值 MSE | Case1：相位 MSE | Case2：幅值 MSE | Case2：相位 MSE |",
             "|---|---:|---:|---:|---:|"]
    for metric_id, name, label in CATALOG:
        cells = []
        for case in ("case1", "case2"):
            for primary in PRIMARY:
                row = next(r for r in SELECTED_SUMMARY if r["context"] == "common_baseline_selected_per_run" and r["case"] == case and r["metric"] == name and r["primary"] == primary)
                cells.append(f"{row[kind]:+.3f}")
        lines.append(f"| {metric_id} {label.replace('对照：', '')} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def primary_table():
    lines = ["| 样本 | Case1 Pearson / Spearman | Case2 Pearson / Spearman |",
             "|---|---:|---:|"]
    for context, label in [("common_initial_population", "初始随机种群"),
                           ("common_baseline_final_population", "基线最终种群")]:
        cells = []
        for case in ("case1", "case2"):
            rows = [r for r in PRIMARY_RELATIONS if r["case"] == case and r["context"] == context]
            cells.append(" / ".join(f"{np.median([r[k] for r in rows]):+.3f}" for k in ("pearson", "spearman")))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    tracked(Path(__file__).resolve())
    for case in ("case1", "case2"):
        process(case)
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "per_run_correlations.csv", PER_RUN)
    write_csv(OUT / "correlation_summary.csv", SUMMARY)
    write_csv(OUT / "selected_per_run_correlations.csv", SELECTED_SUMMARY)
    write_csv(OUT / "primary_pair_per_run.csv", PRIMARY_RELATIONS)
    document = dict(schema_version=1, experiment_protocol="curve3d_directional_20261003_v1",
                    seeds=SEEDS, metrics=[dict(id=i, metric=n, label=l) for i, n, l in CATALOG],
                    primary_objectives=list(PRIMARY), all_losses_lower_is_better=True,
                    runs_per_case=100, individuals_per_run={"case1": 200, "case2": 150},
                    calculations="Pearson on original losses; Spearman on average ranks; no log transform; final populations retain every individual and any duplicate",
                    primary_summary="median of 100 within-run correlations; IQR; percentile bootstrap of independent runs",
                    bootstrap_replicates=BOOTSTRAP_REPLICATES,
                    pooled_values="descriptive only; individuals in the same run are not independent replicates; no pooled p-values",
                    normalization="unaltered frozen target-only normalization; all original frequencies retained",
                    recomputation_max_absolute_difference={"primary": MAX_PRIMARY_DIFFERENCE, "auxiliary": MAX_AUX_DIFFERENCE},
                    independent_correlation_check_max_difference=MAX_CORRELATION_DIFFERENCE,
                    independent_correlation_check="centered-vector inner products; independent average ranks from unique values and tie counts",
                    input_sha256=INPUTS, summary=SUMMARY, selected_per_run_correlations=SELECTED_SUMMARY,
                    primary_pair_per_run=PRIMARY_RELATIONS)
    (OUT / "correlation_results.json").write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    report = """# 最新三维曲线指标与两个主目标的相关系数

## 计算对象和读法

使用 20261003 方向与组合优化正式实验：每个 Case 有 100 个独立随机种子（200—299），Case1 每个种群 200 个候选解，Case2 每个种群 150 个候选解。没有重新优化，没有调整公式、相位处理、归一化或数据点。

两个主目标是**幅值 MSE（dB²）**和**相位 MSE（rad²）**；五个辅助指标也是误差，全部越小越好。正相关表示辅助误差大时主误差通常也大；负相关表示在该批候选解中存在相反变化的倾向。相关系数不是误差下降百分比，也不代表因果效果。

为了公平比较，主表把五个指标计算在**相同的双主目标基线最终种群**上。每个种子分别算 Pearson 与 Spearman，再汇总 100 个种子的中位数，不将种群个体当作独立实验重复。Spearman 使用平均秩处理并列值；不对误差额外取对数。没有用任意相关性门槛决定指标去留。

Pearson 衡量线性关系，较容易受极端误差影响；Spearman 衡量排序是否一致。以下主表均为“运行内相关系数的中位数”；它们不是把所有个体混在一起得到的单个相关系数。每 Case 每指标对应 20,000 / 15,000 个基线最终候选解。

## 一、共同基线最终种群：Spearman

""" + table("common_baseline_final_population", "spearman", True) + """

方括号为 100 次运行内相关系数的第 25%—75% 分位数，不是置信区间。

## 二、共同基线最终种群：Pearson

""" + table("common_baseline_final_population", "pearson", True) + """

## 三、每次运行最终选中一个解：跨运行相关性

沿用原实验的 WSMSE 选解规则，先从每次双主目标基线运行选中一个解，再在这相同的 100 个解上重算五个指标。这里每个样本来自一个独立随机种子，回答“不同运行拟合得好不好，与辅助误差是否相关”。这是跨运行的单个相关系数，**不是运行内相关系数的中位数**；也不等于加入该辅助指标后的实验收益。

### Spearman

""" + selected_table("spearman") + """

### Pearson

""" + selected_table("pearson") + """

## 四、为什么基线最终种群内会出现负相关？

先检查两个主误差之间的运行内相关系数中位数：

""" + primary_table() + """

双主目标搜索后的候选解已经经过选择，两个主误差存在明显权衡。在这样的种群里，一个辅助指标与相位误差正相关、与幅值误差负相关，不能直接说明它在所有候选参数中都“损害幅值”。这是条件于这批搜索结果的关联，相关性不是指标固有的固定属性。下面进一步比较初始和辅助优化后的样本。

## 五、初始随机种群：Spearman（检查搜索前的关系）

""" + table("common_initial_population", "spearman") + """

初始种群与基线最终种群是同一批种子。两者反映不同搜索阶段，不能直接把相关性的改变解释为某个指标导致的改善。

## 六、加入该辅助指标后的最终种群：Spearman

""" + table("own_auxiliary_final_population", "spearman") + """

本表每一行来自不同的辅助目标实验组，因此适合看“该指标在自己的搜索结果中如何变化”，不适合取代共同基线表做公平筛选。多目标选择会改变候选解的分布，也可能使误差之间出现负相关。

“累计面积＋累计弧长”是两个辅助目标，不是一个新标量指标；其最终种群中的两个指标分别计算，结果保存在明细文件的 `two_auxiliary_final_population` 条目。

## 七、用于筛选时要注意

- 与两个主误差均正相关，只说明同一批候选解中的误差排序较一致，不能证明加入优化后必然改善；还要结合之前的配对正式实验结果。
- 若主要关联相位而对幅值很弱或为负，不能把它解释成同时照顾两个主目标的指标。
- 高相关可能表示信息重复；低相关也可能表示无关信息或在 Pareto 候选解中存在权衡。不能单凭相关性高低决定目标好坏。
- Case1、Case2 分别分析；初始、基线最终、辅助优化后最终分别分析；不混合不同阶段和不同问题。
- 当前只保存了初始和最终的个体级数据；没有中间代每个候选解的完整轨迹，不能据此判断全过程相关性。
- 没有报告把全部种群个体视为独立样本的显著性 p 值。CSV 附带运行内中位数的 4,000 次运行级 bootstrap 95% 区间，以及全体个体混合的描述性相关系数。

## 可复算文件

- `correlation_summary.csv`：各情境的运行内中位数、四分位数、bootstrap 区间、混合 Pearson / Spearman。
- `per_run_correlations.csv`：每个种子、每个指标、每个主目标的相关系数。
- `selected_per_run_correlations.csv`：基线／各辅助实验每次选中一个解，跨 100 次独立运行的相关系数。
- `primary_pair_per_run.csv`：两个主误差之间的运行内相关系数，帮助识别搜索后的权衡。
- `correlation_results.json`：完整结果、计算口径、输入文件 SHA-256 和重算一致性校验。
- 复算脚本：`scripts/analyze_curve_directional_correlations.py`。
"""
    (OUT / "README.md").write_text(report, encoding="utf-8")
    (WORKSPACE / "output/最新三维指标_主目标相关性分析_20261003.md").write_text(report, encoding="utf-8")
    print(table("common_baseline_final_population", "spearman"), flush=True)
    print(table("common_baseline_final_population", "pearson"), flush=True)
    print(selected_table("spearman"), flush=True)
    print(selected_table("pearson"), flush=True)
    print(primary_table(), flush=True)
    print(json.dumps(document["recomputation_max_absolute_difference"]), flush=True)


if __name__ == "__main__":
    main()

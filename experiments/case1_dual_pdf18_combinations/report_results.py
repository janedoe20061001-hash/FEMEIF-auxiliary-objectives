"""Audit and report one WSMSE-selected final individual per seed; archive raw outputs."""
import argparse
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np
from final_selection import select_wsmse
from objectives import CATALOG
from summarize import summarize


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("directory",type=Path)
    parser.add_argument("--no-archive",action="store_true")
    args=parser.parse_args();root=args.directory
    manifest=json.loads((root/"manifest.json").read_text())
    cfg=manifest["config"];alpha=cfg["final_selection"]["phase_weight"]
    reference=root.parent/"reference"
    ref_manifest=json.loads((reference/"manifest.json").read_text())
    assert {k:v for k,v in cfg.items() if k!="groups"}=={k:v for k,v in ref_manifest["config"].items() if k!="groups"}
    assert manifest["frequency_hz"]==ref_manifest["frequency_hz"]
    for name in ("objectives.py","final_selection.py","requirements-lock.txt"):
        assert manifest["code_sha256"][name]==ref_manifest["code_sha256"][name]
    for name in ("numpy","pymoo","scipy","moocore"):
        assert manifest["versions"][name]==ref_manifest["versions"][name]
    rows=summarize(root)
    main_rows=[r for r in rows if r["front"]=="wsmse_selected"]
    selected=[];initials=defaultdict(set);seen=set()
    for source,source_manifest in ((root,manifest),(reference,ref_manifest)):
      for path in sorted(source.glob("*/seed_*/summary.json")):
        record=json.loads(path.read_text())
        assert record["status"]=="complete" and record["fingerprint"]==source_manifest["fingerprint"]
        assert record["evaluations"]==cfg["population"]*cfg["generations"]
        key=(record["group"],record["seed"]);assert key not in seen;seen.add(key)
        initials[record["seed"]].add(record["initial_population_sha256"])
        initial=np.load(path.parent/"initial_log_parameters.npy")
        assert hashlib.sha256(initial.tobytes()).hexdigest()==record["initial_population_sha256"]
        with np.load(path.parent/"final_population.npz") as data:
            assert data["objectives"].shape==(cfg["population"],len(record["objective_names"]))
            check=select_wsmse(data["objectives"],alpha)
            assert check["wsmse_selected_index"]==int(data["selected_index"])
            for name,value in check.items():
                assert np.isclose(record[name],value,rtol=1e-12,atol=1e-14)
            assert check["wsmse_selected_index"] in data["primary_front_indices"]
            np.testing.assert_allclose(data["selected_primary_mse"],
                                       data["objectives"][check["wsmse_selected_index"],:2],rtol=0,atol=0)
            np.testing.assert_array_equal(data["selected_parameters_R_L_C"],
                                          data["parameters_R_L_C"][check["wsmse_selected_index"]])
            r=dict(group=record["group"],seed=record["seed"],**check)
            for block,params in enumerate(data["selected_parameters_R_L_C"],1):
                r.update({f"{name}{block}":float(value) for name,value in zip(["R","L","C"],params)})
        selected.append(r)
    start=cfg["seeds"]["start"];count=cfg["seeds"]["count"]
    groups=["baseline"]+[r[1] for r in CATALOG]
    if cfg["groups"]!="single_auxiliary":
        groups=["+".join(g) if g else "baseline" for g in cfg["groups"]]
    combo_groups=groups[:]
    singles=list(dict.fromkeys(a for g in cfg["groups"] for a in g))
    groups=["baseline"]+singles+combo_groups
    expected={(g,s) for g in groups for s in range(start,start+count)}
    assert seen==expected
    assert all(len(v)==1 for v in initials.values())
    with (root/"selected_individuals.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(selected[0]));w.writeheader();w.writerows(selected)
    audit=dict(status="passed",runs=len(selected),expected_runs=len(expected),
        final_selection_alpha=alpha,selected_individuals_verified_against_full_populations=True,
        same_seed_initial_populations_match=True,source_fingerprint=manifest["fingerprint"])
    audit.update(new_runs=len(combo_groups)*count,reused_runs=(len(singles)+1)*count,reference_fingerprint=ref_manifest["fingerprint"])
    (root/"completion_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    by={r["group"]:r for r in main_rows}
    labels={"baseline":("BASE","双主目标基线")}
    labels.update({name:(ident,title) for ident,name,title in CATALOG})
    for g in combo_groups:
        labels[g]=("+".join(labels[a][0] for a in g.split("+")),"＋".join(labels[a][1] for a in g.split("+")))
    ordered=[g for g in groups if g in by]
    improved=[g for g in ordered if g!="baseline" and by[g]["both_means_lower"]]
    text=["# Case I：三个候选辅助目标的两两与三指标组合结果","",
          f"新增 {len(combo_groups)*count} 次运行，复用 baseline 和三个单指标的 {4*count} 次运行；每组 {count} seeds，种群 {cfg['population']}，{cfg['generations']} 代。",
          f"最终选解：每个 seed 从最后一代完整种群中取幅值 MSE + {alpha} × 相位 MSE 最小的一个个体；相同分数取最早行。",
          "两个 MSE 来自同一个被选中的个体，再跨 seed 等权平均。不是分别取幅值/相位最优，也不是前沿均值。",
          "WSMSE 只参与只读统计，不参与 NSGA-II 交配或幸存选择；进化仍使用两个独立主目标及该组辅助损失。",
          f"两项跨 seed 均值都低于配对基线的组合：{sum(g in improved for g in combo_groups)} / {len(combo_groups)}。",
          "这是描述性结果，未做统计显著性检验；WSMSE 下降不保证其两个分量都下降。",
          "",
          "|编号|辅助指标|幅值 MSE 均值|相位 MSE 均值|WSMSE 均值|幅值变化 %|相位变化 %|WSMSE变化 %|两项同降seed数|",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for g in ordered:
        r=by[g];ident,title=labels[g]
        text.append(f"|{ident}|{title}|{r['magnitude_mean']:.6g}|{r['phase_mean']:.6g}|{r['wsmse_mean']:.6g}|"
                    f"{r['magnitude_change_pct']:.3f}|{r['phase_change_pct']:.3f}|{r['wsmse_change_pct']:.3f}|{r['seeds_with_both_lower']}/{count}|")
    text+=["","## 组合与组成它的单指标及两指标组合比较","",
           "|组合|比较对象|幅值均值变化 %|相位均值变化 %|两项均值同降|","|---|---|---:|---:|---|"]
    comparisons=[]
    for g in combo_groups:
        for other in singles+combo_groups:
            if set(other.split("+")) < set(g.split("+")):
                a,b=by[g],by[other]
                da=100*(a["magnitude_mean"]/b["magnitude_mean"]-1)
                dp=100*(a["phase_mean"]/b["phase_mean"]-1)
                comparisons.append(dict(group=g,reference_group=other,magnitude_change_pct=da,phase_change_pct=dp,both_means_lower=da<0 and dp<0))
                text.append(f"|{labels[g][0]}|{labels[other][0]}|{da:.3f}|{dp:.3f}|{'是' if da<0 and dp<0 else '否'}|")
    with (root/"constituent_comparisons.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(comparisons[0]));w.writeheader();w.writerows(comparisons)
    text+=["","![各指标相对基线变化](comparison.png)","",
           "完整结果：paired_summary.csv 同时保留 wsmse_selected、primary_front、full_front 三种口径；selected_individuals.csv 保留每个 seed 实际选中的参数和误差。",
           "默认使用重新构造的 180 点对数均匀网格；中文 PDF 原 401 点坐标缺失，归一化尺度与原文不同。不能将本结果称为原 401 点实验的精确复现。",
           "选解权重在正式实验前固定为 25.12，未按结果调整。本次没有重跑英文论文五项的组合实验。"]
    (root/"RESULTS.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    use=[g for g in ordered if g!="baseline"]
    fig,axes=plt.subplots(1,3,figsize=(14,8),layout="constrained",sharey=True)
    for ax,key,title in zip(axes,["magnitude_change_pct","phase_change_pct","wsmse_change_pct"],
                            ["Magnitude MSE","Phase MSE","WSMSE (alpha=25.12)"]):
        values=[by[g][key] for g in use]
        ax.barh(range(len(use)),values,color=["#32815b" if v<0 else "#b45046" for v in values])
        ax.set_yticks(range(len(use)),[labels[g][0] for g in use])
        ax.axvline(0,color="black",lw=.7);ax.set_xscale("symlog",linthresh=10)
        ax.set_title(title);ax.set_xlabel("Change vs paired baseline [%]");ax.grid(axis="x",alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle("One final WSMSE-selected individual per seed; mean over seeds\nGreen: lower; red: higher. Descriptive results, not significance tests.")
    fig.savefig(root/"comparison.png",dpi=180);plt.close(fig)
    if not args.no_archive:
        archive=root.parent/"raw_results.tar.gz"
        with tarfile.open(archive,"w:gz",compresslevel=1) as tar:
            tar.add(root,arcname="runs")
            tar.add(reference,arcname="reference")
        digest=hashlib.sha256(archive.read_bytes()).hexdigest()
        (root.parent/"raw_results.sha256").write_text(digest+"  raw_results.tar.gz\n",encoding="utf-8")
    print(json.dumps(dict(status="complete",runs=len(selected),both_mean_improvement_groups=improved)))


if __name__=="__main__":
    main()

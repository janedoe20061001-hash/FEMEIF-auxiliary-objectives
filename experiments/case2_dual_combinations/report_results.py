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
from case2 import PARAMETER_NAMES, LOWER, UPPER, load_target, impedance
from objectives import Objectives
from stability import analyze
from constituents import analyze_constituents
from prepare_reference import validate_reference, reference_groups
from cross_case import compare as compare_cases


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("directory",type=Path)
    parser.add_argument("--no-archive",action="store_true")
    args=parser.parse_args();root=args.directory
    manifest=json.loads((root/"manifest.json").read_text())
    cfg=manifest["config"];alpha=cfg["final_selection"]["phase_weight"]
    for name,digest in manifest["code_sha256"].items():
        assert hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==digest, name
    freq,target=load_target(cfg["target_data"])
    np.testing.assert_array_equal(freq,manifest["frequency_hz"])
    np.testing.assert_allclose(target.real,manifest["target_real"],rtol=1e-14,atol=1e-14)
    np.testing.assert_allclose(target.imag,manifest["target_imag"],rtol=1e-14,atol=1e-14)
    reference=root.parent/"reference"
    ref_manifest=validate_reference(reference,cfg,manifest)
    rows=summarize(root)
    main_rows=[r for r in rows if r["front"]=="wsmse_selected"]
    selected=[];initials=defaultdict(set);seen=set()
    for source,source_manifest in ((root,manifest),(reference,ref_manifest)):
      for path in sorted(source.glob("*/seed_*/summary.json")):
        record=json.loads(path.read_text())
        assert record["status"]=="complete" and record["fingerprint"]==source_manifest["fingerprint"]
        assert record["evaluations"]==cfg["population"]*cfg["generations"]
        assert record["generations"]==cfg["generations"]
        key=(record["group"],record["seed"]);assert key not in seen;seen.add(key)
        initials[record["seed"]].add(record["initial_population_sha256"])
        initial=np.load(path.parent/"initial_log_parameters.npy")
        assert initial.shape==(cfg["population"],15)
        assert hashlib.sha256(initial.tobytes()).hexdigest()==record["initial_population_sha256"]
        with np.load(path.parent/"final_population.npz") as data:
            assert data["objectives"].shape==(cfg["population"],len(record["objective_names"]))
            assert data["log_parameters"].shape==(cfg["population"],15)
            assert np.isfinite(data["objectives"]).all() and (data["objectives"]>=0).all()
            assert np.all(data["log_parameters"]>=LOWER) and np.all(data["log_parameters"]<=UPPER)
            assert float(data["selection_alpha"])==alpha
            np.testing.assert_array_equal(data["parameter_names"],PARAMETER_NAMES)
            np.testing.assert_allclose(data["parameters"],10**data["log_parameters"],rtol=1e-14)
            np.testing.assert_array_equal(data["selected_parameters"],data["parameters"][int(data["selected_index"])] )
            evaluator=Objectives(freq,cfg["features"],target_z=target)
            verified=evaluator.evaluate_spectra(impedance(freq,data["parameters"]))
            np.testing.assert_allclose(verified,data["objectives"][:,:2],rtol=1e-11,atol=1e-12)
            check=select_wsmse(data["objectives"],alpha)
            assert check["wsmse_selected_index"]==int(data["selected_index"])
            for name,value in check.items():
                assert np.isclose(record[name],value,rtol=1e-12,atol=1e-14)
            assert check["wsmse_selected_index"] in data["primary_front_indices"]
            np.testing.assert_allclose(data["selected_primary_mse"],
                                       data["objectives"][check["wsmse_selected_index"],:2],rtol=0,atol=0)
            r=dict(group=record["group"],seed=record["seed"],**check)
            r.update({name:float(value) for name,value in zip(PARAMETER_NAMES,data["selected_parameters"])})
        selected.append(r)
    start=cfg["seeds"]["start"];count=cfg["seeds"]["count"]
    groups=["baseline"]+[r[1] for r in CATALOG]
    if cfg["groups"]!="single_auxiliary":
        groups=["+".join(g) if g else "baseline" for g in cfg["groups"]]
    combo_groups=groups[:]
    groups=reference_groups(cfg)+combo_groups
    expected={(g,s) for g in groups for s in range(start,start+count)}
    assert seen==expected
    assert all(len(v)==1 for v in initials.values())
    with (root/"selected_individuals.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(selected[0]));w.writeheader();w.writerows(selected)
    audit=dict(status="passed",runs=len(selected),expected_runs=len(expected),
        final_selection_alpha=alpha,selected_individuals_verified_against_full_populations=True,
        same_seed_initial_populations_match=True,source_fingerprint=manifest["fingerprint"],
        new_runs=len(combo_groups)*count,reused_runs=len(reference_groups(cfg))*count,reference_fingerprint=ref_manifest["fingerprint"])
    (root/"completion_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    by={r["group"]:r for r in main_rows}
    labels={"baseline":("BASE","双主目标基线")}
    labels.update({name:(ident,title) for ident,name,title in CATALOG})
    for g in combo_groups:
        labels[g]=("+".join(labels[a][0] for a in g.split("+")),"＋".join(labels[a][1] for a in g.split("+")))
    ordered=[g for g in groups if g in by]
    improved=[g for g in ordered if g!="baseline" and by[g]["both_means_lower"]]
    text=["# Case II：Case1两套三指标集合的组合迁移检验","",
          f"新增 {len(combo_groups)*count} 次，复用 {len(reference_groups(cfg))*count} 次；每组 {count} seeds，种群 {cfg['population']}，{cfg['generations']} 代。",
          f"最终选解：每个 seed 从最后一代完整种群中取幅值 MSE + {alpha} × 相位 MSE 最小的一个个体；相同分数取最早行。",
          "两个 MSE 来自同一个被选中的个体，再跨 seed 等权平均。不是分别取幅值/相位最优，也不是前沿均值。",
          "WSMSE 只参与只读统计，不参与 NSGA-II 交配或幸存选择；进化仍使用两个独立主目标及该组辅助损失。",
          f"两项跨 seed 均值都低于配对基线的组合：{sum(g in improved for g in combo_groups)} / {len(combo_groups)}。",
          "均值表是描述性结果；配对置信区间和敏感性检查见 STABILITY.md。WSMSE 下降不保证两个分量都下降。",
          "",
          "|编号|辅助指标|幅值 MSE 均值|相位 MSE 均值|WSMSE 均值|幅值变化 %|相位变化 %|WSMSE变化 %|两项同降seed数|",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for g in ordered:
        r=by[g];ident,title=labels[g]
        text.append(f"|{ident}|{title}|{r['magnitude_mean']:.6g}|{r['phase_mean']:.6g}|{r['wsmse_mean']:.6g}|"
                    f"{r['magnitude_change_pct']:.3f}|{r['phase_change_pct']:.3f}|{r['wsmse_change_pct']:.3f}|{r['seeds_with_both_lower']}/{count}|")
    text+=["","![各指标相对基线变化](comparison.png)","",
           "完整结果：paired_summary.csv 同时保留 wsmse_selected、primary_front、full_front 三种口径；selected_individuals.csv 保留每个 seed 实际选中的参数和误差。",
           "使用用户提供CSV的全部201点（40 Hz–110 MHz），不平滑、不插值。论文使用96点且图示频段较窄，因此这是论文预算和电路上的双主目标扩展，不是论文原实验的精确复现。",
           "选解权重按论文Case2固定为17.38。18项采用中文PDF原公式，包括数量阈值按目标事件数封顶的规则；名称中的5-of-6不是对Case2事件数量的预设。"]
    if manifest["target_features"]["target_turns"]==2:
        text += ["", "当前原公式仅检出2个目标转折（均在低频端）。数量阈值封顶后，I-01与I-05等价，I-02与II-11等价；这四组不能算作四个相互独立的指标证据。本轮保留全部18个标签，不据此调整特征定义。"]
    (root/"RESULTS.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    use=[g for g in ordered if g!="baseline"]
    fig,axes=plt.subplots(1,3,figsize=(14,8),layout="constrained",sharey=True)
    for ax,key,title in zip(axes,["magnitude_change_pct","phase_change_pct","wsmse_change_pct"],
                            ["Magnitude MSE","Phase MSE","WSMSE (alpha=17.38)"]):
        values=[by[g][key] for g in use]
        ax.barh(range(len(use)),values,color=["#32815b" if v<0 else "#b45046" for v in values])
        ax.set_yticks(range(len(use)),[labels[g][0] for g in use])
        ax.axvline(0,color="black",lw=.7);ax.set_xscale("symlog",linthresh=10)
        ax.set_title(title);ax.set_xlabel("Change vs paired baseline [%]");ax.grid(axis="x",alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle("One final WSMSE-selected individual per seed; mean over seeds\nGreen: lower; red: higher. Descriptive results, not significance tests.")
    fig.savefig(root/"comparison.png",dpi=180);plt.close(fig)
    settings=json.loads((Path(__file__).parent/"analysis_config.json").read_text())
    formal=cfg["population"]==150 and cfg["generations"]==300 and count==100
    analysis_cfg=dict(cfg,groups=[[] if g=="baseline" else g.split("+") for g in groups])
    stage="Case2 两套三指标集合迁移" if formal else "smoke / 仅软件验证"
    analyze(selected,analysis_cfg,settings,root,stage)
    analyze_constituents(selected,analysis_cfg,settings,root,stage)
    if formal:
        compare_cases(root,main_rows)
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

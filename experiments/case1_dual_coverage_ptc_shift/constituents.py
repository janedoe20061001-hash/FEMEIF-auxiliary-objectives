"""Paired comparisons against component objectives; shared by both report stages."""
import csv
import json
from pathlib import Path
import numpy as np
from objectives import CATALOG
from stability import paired_stats,KEYS


def analyze_constituents(records,config,settings,output,stage):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    groups=["+".join(g) if g else "baseline" for g in config["groups"]]
    seeds=list(range(config["seeds"]["start"],config["seeds"]["start"]+config["seeds"]["count"]))
    comparisons=[(g,b) for g in groups if g!="baseline" for b in groups
                 if b!="baseline" and set(b.split("+"))<set(g.split("+"))]
    if not comparisons or len(groups)!=len(set(groups)):raise ValueError("Invalid comparison plan")
    if len(groups)==8 and settings["constituent_family_components"]!=2*len(comparisons):
        raise ValueError("Prespecified constituent comparison family differs")
    data={}
    for r in records:
        key=(r["group"],r["seed"])
        if key in data:raise ValueError("Duplicate group/seed")
        if r["wsmse_selected_alpha"]!=config["final_selection"]["phase_weight"]:
            raise ValueError("Different selection weights")
        data[key]=[r[k] for k in KEYS]
    if set(data)!={(g,s) for g in groups for s in seeds}:
        raise ValueError("Missing or unexpected paired seeds")
    arrays={g:np.asarray([data[g,s] for s in seeds],float) for g in groups}
    local=dict(settings,familywise_components=2*len(comparisons))
    rows=[];findings=[]
    for g,b in comparisons:
        pair=paired_stats(arrays[g],arrays[b],local)
        for row in pair:row.update(group=g,reference_group=b)
        rows.extend(pair)
        findings.append(dict(group=g,reference_group=b,
            both_means_lower=all(r["mean_difference"]<0 for r in pair),
            both_ci95_below_zero=all(r["ci95_high"]<0 for r in pair),
            both_adjusted_ci_below_zero=all(r["family_ci_high"]<0 for r in pair)))
    with (output/"constituent_stability.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    meta=dict(stage=stage,records=len(records),comparisons=len(comparisons),
              component_intervals=2*len(comparisons),settings=local,findings=findings)
    (output/"constituent_stability.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    labels={name:ident for ident,name,title in CATALOG}
    label=lambda name:"+".join(labels[a] for a in name.split("+"))
    conf=100*(1-(1-local["confidence_level"])/local["familywise_components"])
    text=["# 组合相对组成项的稳定性","",f"阶段：{stage}；seed {seeds[0]}–{seeds[-1]}。",
          "差值为组合误差减去组成项误差，负数表示组合更好。每个 seed 的两项误差来自同一个固定 WSMSE 选中个体。",
          f"成对 bootstrap {local['bootstrap_resamples']} 次，固定 RNG seed {local['bootstrap_seed']}；报告单项95%区间，以及对全部{2*len(comparisons)}项误差比较的 Bonferroni 调整区间（每项约{conf:.3f}%）。区间覆盖率为近似值。",
          "旧数据上的候选经过挑选，因此区间只用于探索性诊断，不能消除选择偏差。区间跨零是证据不足，不证明组合无效。组成项比较与相对baseline的主判断分别报告。",
          "", "|组合|比较对象|误差|均值变化 %|配对均值差|95%区间|调整区间|去掉最好5%后的均值差|",
          "|---|---|---|---:|---:|---|---|---:|"]
    for r in rows:
        pct="NA" if r["mean_change_pct"] is None else f"{r['mean_change_pct']:.3f}"
        text.append(f"|{label(r['group'])}|{label(r['reference_group'])}|{'幅值' if r['metric']=='magnitude' else '相位'}|{pct}|{r['mean_difference']:.6g}|[{r['ci95_low']:.6g}, {r['ci95_high']:.6g}]|[{r['family_ci_low']:.6g}, {r['family_ci_high']:.6g}]|{r['remove_best_mean_difference']:.6g}|")
    text+=["","敏感性删减只作诊断，不改变正式均值；两项误差的删减seed可能不同。",
           "编号：I-02 转折区域覆盖5-of-6；I-06 中文PDF峰谷一致性（最近匹配、均值）；II-01 局部特征频移误差。"]
    (output/"CONSTITUENTS.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    return meta

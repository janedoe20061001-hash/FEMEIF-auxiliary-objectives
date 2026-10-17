"""Generic paired-seed bootstrap analysis for Case2 protocol batches."""
from pathlib import Path
import csv
import json
import numpy as np
from objectives import CATALOG

KEYS=("wsmse_selected_magnitude_mse","wsmse_selected_phase_mse")


def paired(candidate, baseline, resamples=20000, rng_seed=20260921, components=2):
    candidate=np.asarray(candidate,float);baseline=np.asarray(baseline,float)
    if candidate.shape!=baseline.shape or candidate.ndim!=2 or candidate.shape[1]!=2:
        raise ValueError("paired arrays must be n x 2")
    delta=candidate-baseline;n=len(delta)
    rng=np.random.default_rng(rng_seed)
    index=rng.integers(0,n,(resamples,n))
    boot=delta[index].mean(axis=1)
    tail=.025
    ci=np.quantile(boot,[tail,1-tail],axis=0)
    adjusted=np.quantile(boot,[tail/components,1-tail/components],axis=0)
    rows=[]
    for j,metric in enumerate(("magnitude","phase")):
        order=np.sort(delta[:,j]);k=max(1,int(np.floor(.05*n)))
        rows.append(dict(metric=metric,n=n,candidate_mean=float(candidate[:,j].mean()),
            reference_mean=float(baseline[:,j].mean()),mean_difference=float(delta[:,j].mean()),
            mean_change_pct=float(100*delta[:,j].mean()/baseline[:,j].mean()),
            lower_seeds=int((delta[:,j]<0).sum()),both_lower_seeds=int(np.all(delta<0,axis=1).sum()),
            ci95_low=float(ci[0,j]),ci95_high=float(ci[1,j]),
            adjusted_ci_low=float(adjusted[0,j]),adjusted_ci_high=float(adjusted[1,j]),
            remove_best_5pct_mean_difference=float(order[k:].mean())))
    return rows


def analyze(records,cfg,output):
    output=Path(output);groups=["+".join(g) if g else "baseline" for g in __import__('experiment').groups(cfg)]
    seeds=range(cfg["seeds"]["start"],cfg["seeds"]["start"]+cfg["seeds"]["count"])
    data={(r["group"],r["seed"]):[r[k] for k in KEYS] for r in records}
    expected={(g,s) for g in groups for s in seeds}
    if set(data)!=expected:raise ValueError("missing/unexpected paired records")
    arrays={g:np.asarray([data[g,s] for s in seeds]) for g in groups}
    comparisons=[(g,"baseline") for g in groups if g!="baseline"]
    components=2*len(comparisons)
    rows=[]
    for group,ref in comparisons:
        for row in paired(arrays[group],arrays[ref],components=components):
            row.update(group=group,reference_group=ref);rows.append(row)
    with (output/"paired_stability.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

    labels={name:ident for ident,name,title in CATALOG}
    label=lambda g:"BASE" if g=="baseline" else "+".join(labels[x] for x in g.split("+"))
    text=["# 配对 seed 稳定性分析","",
          f"实验：{cfg['experiment_name']}；seed {min(seeds)}–{max(seeds)}；每组 {len(list(seeds))} 次。",
          f"对每个非基线组与同 seed 基线做 {20000} 次配对 bootstrap；{components} 项误差使用 Bonferroni 调整区间。负差值表示改善。","",
          "|组别|指标|均值变化%|均值差|95%区间|调整区间|改善seed|去掉最好5%后的均值差|",
          "|---|---|---:|---:|---|---|---:|---:|"]
    for r in rows:
        text.append(f"|{label(r['group'])}|{'幅值' if r['metric']=='magnitude' else '相位'}|{r['mean_change_pct']:.3f}|{r['mean_difference']:.6g}|[{r['ci95_low']:.6g}, {r['ci95_high']:.6g}]|[{r['adjusted_ci_low']:.6g}, {r['adjusted_ci_high']:.6g}]|{r['lower_seeds']}/{r['n']}|{r['remove_best_5pct_mean_difference']:.6g}|")
    (output/"STABILITY.md").write_text("\n".join(text)+"\n",encoding="utf-8")

    constituent=[]
    group_set=set(groups)
    subset_pairs=[]
    for group in groups:
        members=set(group.split("+")) if group!="baseline" else set()
        if len(members)<2:continue
        for ref in groups:
            other=set(ref.split("+")) if ref!="baseline" else set()
            if other and other<members:subset_pairs.append((group,ref))
    c_components=max(2,2*len(subset_pairs))
    for group,ref in subset_pairs:
        for row in paired(arrays[group],arrays[ref],components=c_components):
            row.update(group=group,reference_group=ref);constituent.append(row)
    if constituent:
        with (output/"constituent_stability.csv").open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(constituent[0]));w.writeheader();w.writerows(constituent)
        ctext=["# 组合相对组成项的稳定性","",f"共 {len(subset_pairs)} 对组合—组成项比较、{2*len(subset_pairs)} 项误差；负差值表示组合更好。","",
               "|组合|组成项|指标|均值变化%|均值差|95%区间|调整区间|",
               "|---|---|---|---:|---:|---|---|"]
        for r in constituent:
            ctext.append(f"|{label(r['group'])}|{label(r['reference_group'])}|{'幅值' if r['metric']=='magnitude' else '相位'}|{r['mean_change_pct']:.3f}|{r['mean_difference']:.6g}|[{r['ci95_low']:.6g}, {r['ci95_high']:.6g}]|[{r['adjusted_ci_low']:.6g}, {r['adjusted_ci_high']:.6g}]|")
        (output/"CONSTITUENTS.md").write_text("\n".join(ctext)+"\n",encoding="utf-8")
    meta=dict(experiment_name=cfg["experiment_name"],records=len(records),groups=len(groups),
              seed_start=cfg["seeds"]["start"],seed_count=cfg["seeds"]["count"],
              baseline_components=components,constituent_comparisons=len(subset_pairs))
    (output/"analysis.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return meta

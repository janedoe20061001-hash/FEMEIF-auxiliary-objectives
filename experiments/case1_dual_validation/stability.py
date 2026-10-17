"""Paired-seed uncertainty and sensitivity; no optimizer is invoked here."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np
from objectives import CATALOG

HERE=Path(__file__).resolve().parent
KEYS=("wsmse_selected_magnitude_mse","wsmse_selected_phase_mse")


def paired_stats(candidate,baseline,settings):
    a,b=np.asarray(candidate,float),np.asarray(baseline,float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=2 or len(a)<2:
        raise ValueError("Expected matched seed rows with two metrics")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite metrics")
    n=len(a);d=a-b
    rng=np.random.default_rng(settings["bootstrap_seed"])
    indices=rng.integers(0,n,size=(settings["bootstrap_resamples"],n))
    boot=d[indices].mean(axis=1)  # One index resamples the whole paired seed.
    tail=(1-settings["confidence_level"])/2
    ordinary=np.quantile(boot,[tail,1-tail],axis=0)
    family=np.quantile(boot,[tail/settings["familywise_components"],1-tail/settings["familywise_components"]],axis=0)
    k=max(1,int(np.floor(n*settings["remove_best_fraction"])))
    trim=int(np.floor(n*settings["trim_fraction"]))
    rows=[]
    for j,metric in enumerate(("magnitude","phase")):
        delta=d[:,j];ordered=np.sort(delta);mean=float(delta.mean())
        gains=np.maximum(-delta,0);gain_total=float(gains.sum())
        loo=(delta.sum()-delta)/(n-1)
        rows.append(dict(metric=metric,n=n,candidate_mean=float(a[:,j].mean()),
            baseline_mean=float(b[:,j].mean()),mean_difference=mean,
            mean_change_pct=float(100*mean/b[:,j].mean()) if b[:,j].mean()>0 else None,
            median_difference=float(np.median(delta)),
            lower_seeds=int((delta<0).sum()),equal_seeds=int((delta==0).sum()),
            both_lower_seeds=int(np.all(d<0,axis=1).sum()),
            ci95_low=float(ordinary[0,j]),ci95_high=float(ordinary[1,j]),
            family_ci_low=float(family[0,j]),family_ci_high=float(family[1,j]),
            trimmed_mean_difference=float(ordered[trim:n-trim].mean()),
            removed_best_seeds=k,remove_best_mean_difference=float(ordered[k:].mean()),
            leave_one_out_min=float(loo.min()),leave_one_out_max=float(loo.max()),
            top_gain_share=float(np.sort(gains)[-k:].sum()/gain_total) if gain_total>0 else None))
    return rows


def analyze(records,config,settings,output,stage):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    groups=["+".join(g) if g else "baseline" for g in config["groups"]]
    if len(groups)!=8 or len(set(groups))!=8 or groups[0]!="baseline":
        raise ValueError("The locked analysis requires eight groups")
    seeds=list(range(config["seeds"]["start"],config["seeds"]["start"]+config["seeds"]["count"]))
    expected={(g,s) for g in groups for s in seeds}
    data={}
    for r in records:
        key=(r["group"],r["seed"])
        if key in data: raise ValueError(f"Duplicate group/seed {key}")
        if r["wsmse_selected_alpha"]!=config["final_selection"]["phase_weight"]:
            raise ValueError("Selection alpha mismatch")
        data[key]=[r[k] for k in KEYS]
    if set(data)!=expected: raise ValueError("Missing, unexpected or unpaired seeds")
    arrays={g:np.array([data[g,s] for s in seeds]) for g in groups}
    rows=[]
    for group in groups[1:]:
        refs=["baseline"]+[g for g in groups[1:] if set(g.split("+"))<set(group.split("+"))]
        for ref in refs:
            for row in paired_stats(arrays[group],arrays[ref],settings):
                row.update(group=group,reference_group=ref,
                           family_adjustment_applies=ref=="baseline")
                rows.append(row)
    with (output/"paired_stability.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    primary=[r for r in rows if r["reference_group"]=="baseline"]
    labels={name:ident for ident,name,title in CATALOG}
    label=lambda g:"+".join(labels[a] for a in g.split("+"))
    findings={}
    for g in groups[1:]:
        pair=[r for r in primary if r["group"]==g]
        findings[g]=dict(both_means_lower=all(r["mean_difference"]<0 for r in pair),
            both_marginal_ci_below_zero=all(r["ci95_high"]<0 for r in pair),
            both_family_ci_below_zero=all(r["family_ci_high"]<0 for r in pair),
            both_means_lower_after_removing_best=all(r["remove_best_mean_difference"]<0 for r in pair))
    meta=dict(stage=stage,settings=settings,seed_start=seeds[0],seed_end=seeds[-1],
              records=len(data),findings=findings,
              interpretation="Discovery intervals are exploratory after candidate selection; bootstrap coverage is approximate.")
    (output/"stability.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    text=["# 配对 seed 稳定性分析","",
          f"数据阶段：{stage}；seed {seeds[0]}–{seeds[-1]}；每组 {len(seeds)} 次运行。",
          "每个 seed 的候选组误差减去同 seed 的 baseline 误差，负数表示改善。幅值单位 dB²，相位单位 rad²。两个误差始终来自同一个 WSMSE 选中个体。",
          f"按 seed 成对重采样 {settings['bootstrap_resamples']} 次，两项指标使用相同重采样索引；固定分析随机种子 {settings['bootstrap_seed']}。",
          "报告百分位 bootstrap 的单项 95% 区间，以及覆盖七组×两项共 14 个比较的 Bonferroni 调整区间（每项约 99.643%）。后者旨在控制整组判断的误报，但 bootstrap 覆盖率是近似的。",
          "旧 seed 用于挑选了这三个指标，因此旧数据上的区间只作探索性诊断，不能代替新 seed 的独立验证。区间跨零表示证据不足，不表示证明没有效果。",
          "",
          "|组别|指标|均值变化 %|配对均值差|单项95%区间|14项调整区间|改善seed数|",
          "|---|---|---:|---:|---|---|---:|"]
    for r in primary:
        pct="NA" if r["mean_change_pct"] is None else f"{r['mean_change_pct']:.2f}"
        text.append(f"|{label(r['group'])}|{'幅值' if r['metric']=='magnitude' else '相位'}|{pct}|{r['mean_difference']:.6g}|[{r['ci95_low']:.6g}, {r['ci95_high']:.6g}]|[{r['family_ci_low']:.6g}, {r['family_ci_high']:.6g}]|{r['lower_seeds']}/{r['n']}|")
    text+=["","## 少数 seed 的影响","",
           "下面均为敏感性诊断，不替代完整 100 seed 均值，也不是删掉不利结果重新定结论。每项误差分别执行删减，因此两项可能删去不同 seed。",
           "截尾均值各去掉差值两端 5%；去掉最好 5% 则删除改善最大的 seed；逐个删除的最大均值差检查单个 seed 是否足以改变方向。",
           "最大5%收益占比的分母是所有有改善 seed 的正收益总和，不是净均值改善；不存在正收益时记 NA。",
           "",
           "|组别|指标|两端5%截尾均值差|去掉最好5%后的均值差|逐个删除的最大均值差|最大5%收益占比|",
           "|---|---|---:|---:|---:|---:|"]
    for r in primary:
        share="NA" if r["top_gain_share"] is None else f"{100*r['top_gain_share']:.1f}%"
        text.append(f"|{label(r['group'])}|{'幅值' if r['metric']=='magnitude' else '相位'}|{r['trimmed_mean_difference']:.6g}|{r['remove_best_mean_difference']:.6g}|{r['leave_one_out_max']:.6g}|{share}|")
    text+=["","paired_stability.csv 还包含组合相对组成项的配对比较。这些比较属于探索性诊断，不能套用针对 baseline 的 14 项整体判断。",
           "编号：I-03 电抗过零方向一致性；II-01 局部特征频移误差；II-02 转变速度相对误差。"]
    (output/"STABILITY.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    return meta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    digest=hashlib.file_digest(args.archive.open("rb"),"sha256").hexdigest()
    expected=args.archive.with_name("raw_results.sha256").read_text().split()[0]
    if digest!=expected: raise ValueError("Archive checksum mismatch")
    cfg=json.loads((HERE/"config.json").read_text())
    cfg["seeds"]=dict(start=0,count=100)
    records=[]
    with tarfile.open(args.archive) as tar:
        manifests={name:json.load(tar.extractfile(name+"/manifest.json")) for name in ("runs","reference")}
        for mf in manifests.values():
            if {k:v for k,v in mf["config"].items() if k!="groups"}!={k:v for k,v in cfg.items() if k!="groups"}:
                raise ValueError("Archived experimental settings differ")
            for name in ("objectives.py","final_selection.py","requirements-lock.txt"):
                if mf["code_sha256"][name]!=hashlib.sha256((HERE/name).read_bytes()).hexdigest():
                    raise ValueError("Archived objective or selection definitions differ")
        for member in tar:
            if member.isfile() and member.name.endswith("/summary.json"):
                origin=member.name.split("/")[0];r=json.load(tar.extractfile(member))
                if r["status"]!="complete" or r["fingerprint"]!=manifests[origin]["fingerprint"]:
                    raise ValueError("Archive record provenance mismatch")
                records.append(r)
    meta=analyze(records,cfg,json.loads((HERE/"analysis_config.json").read_text()),args.output,"discovery / 旧数据探索")
    (args.output/"source.json").write_text(json.dumps(dict(archive=str(args.archive.resolve()),sha256=digest,manifests=manifests),indent=2),encoding="utf-8")
    print(json.dumps(meta["findings"],indent=2))


if __name__=="__main__":
    main()

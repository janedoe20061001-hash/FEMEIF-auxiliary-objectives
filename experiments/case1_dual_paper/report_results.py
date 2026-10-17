"""Descriptive paired-seed report; do not infer statistical significance."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from summarize import summarize
from objectives import AUX


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("directory",type=Path)
    args=parser.parse_args();root=args.directory
    rows=summarize(root)
    primary=[r for r in rows if r["front"]=="full_front"]
    by={r["group"]:r for r in primary}
    progress=json.loads((root/"progress.json").read_text()) if (root/"progress.json").exists() else {}
    manifest=json.loads((root/"manifest.json").read_text())
    passing=[r for r in primary if r["group"]!="baseline" and r["complete"] and r["both_means_lower"]]
    triples=[]
    for group in itertools.combinations(AUX,3):
        subsets=[tuple(g) for n in (2,3) for g in itertools.combinations(group,n)]
        names=["+".join(g) for g in subsets]
        if all(name in by and by[name]["complete"] and by[name]["both_means_lower"] for name in names):
            triples.append("+".join(group))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,7),layout="constrained")
    palette={1:"#2878b5",2:"#47a76a",3:"#d69424",4:"#976cc5",5:"#bb4455"}
    positive=[]
    for r in primary:
        if r["group"]=="baseline":
            continue
        x=r["magnitude_mean"]/r["baseline_magnitude_mean"]
        y=r["phase_mean"]/r["baseline_phase_mean"]
        if x<=0 or y<=0:
            continue
        positive.append((x,y))
        n=r["group"].count("+")+1
        ax.scatter(x,y,color=palette[n],s=55)
        ax.annotate(str(len(positive)),(x,y),xytext=(4,4),textcoords="offset points",fontsize=8)
        r["plot_id"]=len(positive)
    ax.axvline(1,color="gray",ls="--");ax.axhline(1,color="gray",ls="--")
    ax.set_xscale("log");ax.set_yscale("log")
    ax.set_xlabel("Magnitude MSE / paired baseline (lower is better)")
    ax.set_ylabel("Phase MSE / paired baseline (lower is better)")
    ax.set_title("Case I: mean over seeds of full-front mean MSE\nBoth improve only when x < 1 and y < 1; descriptive comparison")
    for n,color in palette.items():
        ax.scatter([],[],color=color,label=f"{n} auxiliary objective(s)")
    ax.legend();ax.grid(alpha=.2)
    fig.savefig(root/"comparison.png",dpi=170);plt.close(fig)
    lines=[
        "# Case I 双主目标正式实验结果",
        "",
        f"任务状态：{progress.get('state','unknown')}；完成 {progress.get('completed_runs','unknown')} 次，失败 {progress.get('failed_runs','unknown')} 次。",
        f"配置：每组 {manifest['config']['seeds']['count']} seeds，种群 {manifest['config']['population']}，{manifest['config']['generations']} 代。",
        "",
        "主口径：每个 seed 的完整目标空间最终非支配前沿上，对两项主目标分别求均值，然后跨 seed 等权平均，与同 seed 的纯双主目标基线比较。",
        "百分比为两组均值之比减一；负值表示降低。此处是描述性统计，不代表统计显著性，也不代表每个 seed 都改善。",
        "",
        f"完整配对且两项均值都降低的辅助组：{len(passing)} / 31。",
        "其自身三项组合和三个两项子组合均满足两项均值降低的三元集合：" + ("；".join(triples) if triples else "未发现。"),
        "",
        "![相对基线的两项主目标](comparison.png)",
        "",
        "|图编号|辅助组|配对 seeds|幅值 MSE 均值|相位 MSE 均值|幅值变化 %|相位变化 %|同 seed 两项都下降次数|",
        "|---|---|---|---|---|---|---|---|"]
    for r in primary:
        lines.append(f"|{r.get('plot_id','—')}|{r['group']}|{r['paired_seeds']}|{r['magnitude_mean']:.6g}|{r['phase_mean']:.6g}|{r['magnitude_change_pct']:.3f}|{r['phase_change_pct']:.3f}|{r['seeds_with_both_lower']}|")
    lines+=["","仅按双主目标重筛的前沿结果另见 paired_summary.csv 中 primary_front 行，不与上述 full_front 混用。",
        "频率默认是重新构造的 180 点对数均匀网格；原文频率坐标缺失。原始辅助项使用误差形式，不使用倒数形式；具体实现约定见代码 README。",
        "各 seed 的完整种群、参数、目标、逐代历史均保留在对应分组目录。manifest.json 保存实际配置与代码/环境指纹。"]
    (root/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(dict(groups=len(primary),both_mean_improvements=len(passing),
                          triple_sets_with_all_pairs_improving=triples),ensure_ascii=False))


if __name__=="__main__":
    main()

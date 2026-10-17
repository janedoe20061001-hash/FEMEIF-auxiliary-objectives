"""Re-report saved paired summaries; this script does NOT run any optimization."""
from pathlib import Path
import csv
import hashlib
import itertools
import json

HERE=Path(__file__).resolve().parent


def main():
    root=HERE/"server_run";source=root/"paired_summary.csv"
    with source.open(encoding="utf-8-sig") as fh:
        rows=[r for r in csv.DictReader(fh) if r["front"]=="primary_front"]
    assert len(rows)==32
    assert all(r["complete"]=="True" and int(r["paired_seeds"])==100 for r in rows)
    by={r["group"]:r for r in rows}
    order=["SS","PTC","DTW","LFP","LMEP"]
    def passes(g):
        r=by["+".join(g)]
        return float(r["magnitude_change_pct"])<0 and float(r["phase_change_pct"])<0
    triples=[g for g in itertools.combinations(order,3)
             if all(passes(s) for n in [2,3] for s in itertools.combinations(g,n))]
    with (root/"primary_front_summary.csv").open("w",encoding="utf-8-sig",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    both=sum(passes(g.split("+")) for g in by if g!="baseline")
    lines=["# 论文五项辅助目标：双主目标前沿汇总","",
        "**复用已完成的 3,200 次正式运行，没有重新运行优化。**",
        "每一代的亲本选择和父子代合并后的幸存选择仍使用全部优化目标；仅在最终种群上按幅值 MSE、相位 MSE 重新筛选非支配前沿。",
        "先对各 seed 的该前沿求两项均值，再跨 100 seeds 等权平均，与同 seed 的纯双主目标基线比较。单项最小值不是一个共同解，也不是本表的统计量。",
        "",f"31 个辅助组中，{both} 个组的两项均值均下降。",
        "其三项组合和全部三个两项子组合均满足两项均值下降的三元集合："+ "；".join("+".join(g) for g in triples)+"。",
        "",
        "**这是描述性均值筛选，不代表所有 seed 同时改善，未做显著性检验。前沿均值受解集分布、数量和覆盖范围影响，也不等价于一个前沿整体支配另一个前沿。**",
        "这些结果不要求三元集合中的每一个单指标也有效，更不要求组合优于所有单指标；判据统一是相对纯双主目标基线的两项均值都下降。",
        "",
        "|辅助组|幅值 MSE 变化 %|相位 MSE 变化 %|两项同降 seed 数 / 100|",
        "|---|---:|---:|---:|"]
    for g in [()]+[s for n in range(1,6) for s in itertools.combinations(order,n)]:
        name="+".join(g) if g else "baseline";r=by[name]
        lines.append(f"|{name}|{float(r['magnitude_change_pct']):.3f}|{float(r['phase_change_pct']):.3f}|{r['seeds_with_both_lower']}|")
    lines+=["","## 两个三元集合的逐项检查",""]
    for g in triples:
        lines+=["### "+" + ".join(g),"",
                "|组合|幅值变化 %|相位变化 %|","|---|---:|---:|"]
        for n in [2,3]:
            for s in itertools.combinations(g,n):
                r=by["+".join(s)]
                lines.append(f"|{r['group']}|{float(r['magnitude_change_pct']):.3f}|{float(r['phase_change_pct']):.3f}|")
        lines.append("")
    lines+=["## 实验边界","",
        "本批次是 NSGA-II、100 seeds、200 个体、300 代，使用重新构造的 180 点对数均匀频率网格，辅助项使用误差形式。它不是原作者 GrEA / WSMSE 实验的精确复现。",
        "完整目标前沿的原始汇总仍在 RESULTS.md 保留；本报告只改变读取已保存结果的统计口径，没有修改任何运行或目标值。"]
    (root/"PRIMARY_FRONT_RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    record=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        report_front="primary_front",optimization_rerun=False,groups=32,paired_seeds=100,
        both_means_improve=both,triple_sets=["+".join(g) for g in triples])
    (root/"primary_front_report_audit.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
    print(json.dumps(record,indent=2))


if __name__=="__main__":
    main()

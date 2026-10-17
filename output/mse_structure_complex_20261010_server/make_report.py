"""Plain-language local report from verified, retrieved experiment tables."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "REPORT.md"
NAMES = {"user_nested":"你提出的嵌套电路", "control_rlc":"单并联 RLC 对照", "rlc_ladder":"两级 RLC 梯形网络",
         "loaded_tank":"带负载的谐振网络", "dual_nested":"双嵌套支路", "three_level":"三级 RLC 网络", "mixed_resonators":"混合谐振支路"}


def rows(p):
    if not p.exists():
        return []
    with p.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def fmt(v):
    return "不适用" if v in (None, "") else f"{float(v):.5g}"


def make_report(snapshot=None):
    snapshot = snapshot or json.loads((ROOT/"LAST_STATUS.json").read_text(encoding="utf-8"))
    pipeline = snapshot.get("pipeline") or {}
    done = sum(snapshot.get(g,{}).get("completed",0) for g in ("first","other"))
    text = "# 嵌套电路第二步实验：进展与结果\n\n"
    text += f"2026年10月10日启动。当前服务器阶段：`{pipeline.get('stage','尚未读取')}`；已完成 {done}/168 个 seed 任务。\n\n"
    text += "这次还是让程序同时减小幅值和相位的整体误差，观察哪些峰、谷和相位转变仍恢复不好。没有加入辅助指标，没有更新网站。\n\n"
    text += "先运行你提出的电路与单并联 RLC 对照，再运行另外五类嵌套电路。每类 3 个目标、每目标 8 个 seed、每 seed 300 代、种群 200；全部预算为 10,080,000 次参数候选评价。\n\n"
    text += "生成目标时，指定 L 乘 0.25/1/8、C 除以同一倍率。拟合时允许所有元件参数改变。嵌套结构的实际峰宽、位置与相邻结构可能一起改变，倍率本身不当作峰宽。\n\n"
    if pipeline.get("status") == "needs_review":
        text += "**流程遇到问题并已停止推进后续批次。** 当前不能把未完成或未通过核验的数据写成正常实验结论。原因记录在 LAST_STATUS.json 和服务器日志中。\n\n"
    summary, reps, events, phase, populations, pairs, audits, locations = [],[],[],[],[],[],[],{}
    for group in ("first","other"):
        base = ROOT/"retrieved"/group
        analysis = base/f"analysis_{group}"
        if not analysis.exists():
            continue
        audit_path = analysis/"AUDIT.json"
        if not audit_path.exists():
            continue
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audits.append((group,audit))
        if not audit["passed"]:
            supplement_path = base/f"AUDIT_REFINEMENT_{group}.json"
            supplement = json.loads(supplement_path.read_text(encoding="utf-8")) if supplement_path.exists() else {}
            resolved = supplement.get("passed",False) and supplement.get("original_audit_sha256") == hashlib.sha256(audit_path.read_bytes()).hexdigest() and len(supplement.get("records",[])) == len(audit["failures"]) and all(r.get("accepted") and not r.get("representative") for r in supplement.get("records",[]))
            if not resolved:
                text += f"{group} 批次独立核验未通过，请先检查 AUDIT.json。\n\n"
                continue
            text += f"{group} 批次有 {len(audit['failures'])} 条非代表候选在原检查网格下结构判定不稳定；加密至32、64级后结构判定一致，分数按原容差复算通过。原始 AUDIT.json 与补充复核记录均保留。下面的代表曲线统计未受影响，种群比较仍保守排除原来不稳定的候选。\n\n"
        summary += rows(analysis/"summary.csv")
        reps += rows(analysis/"representatives.csv")
        events += rows(analysis/"events.csv")
        phase += rows(analysis/"phase_events.csv")
        populations += rows(analysis/"population.csv")
        pairs += rows(analysis/"similar_scores_different_shapes.csv")
        for r in summary:
            ident = r["target"].split("/")[0]
            if (analysis/ident).exists():
                locations[ident] = analysis
    if not summary:
        text += "**目前还没有已回收且通过独立核验的结果。** 本文件会在首批和后续批次回收后更新；下列研究问题仍待数据回答。\n\n"
    else:
        text += "## 先看整体分数与结构是否一起变好\n\n"
        text += "MAG 是 dB 幅值平均平方误差，PHA 是弧度相位平均平方误差，两项越小越好。每个 seed 按固定规则选一条代表曲线，再对这些曲线的分数取平均；曲线本身不平均。\n\n"
        text += "表中‘幅值结构全部合格’只评价能定义 3 dB 宽度的目标峰谷：频率位置差≤0.02个十倍频程、高度差≤3 dB、宽度比在0.5～2之间。相位转变另外检查。没有可算宽度的目标事件时标为不适用。\n\n"
        text += "|电路与条件|有效seed/计划|MAG均值 / dB²|PHA均值 / rad²|幅值结构全部合格的比例|\n|---|---:|---:|---:|---:|\n"
        for r in summary:
            if r["rule"] != "primary":
                continue
            ident,condition = r["target"].split("/")
            text += f'|{NAMES.get(ident,ident)} / {condition}|{r["seeds_valid"]}/{r["seeds_planned"]}|{fmt(r["mag_mean"])}|{fmt(r["pha_mean"])}|{fmt(r["joint_amp_complete_rate"])}|\n'
        text += "\n‘有效 seed’同时通过分数复算和结构判断稳定性检查。最终只按 MAG+25.12×PHA 选代表的表列在这里；另一种平衡选解规则同样保存在 summary.csv，避免把选解偏好误认为优化器没有找到结构。\n\n"
        valid = {(r["target"],r["seed"]) for r in reps if r["generation"]=="300" and r["rule"]=="primary" and r["stable"]=="True" and r["audit_passed"]=="True"}
        final_ev = [r for r in events if r["generation"]=="300" and r["rule"]=="primary" and (r["target"],r["seed"]) in valid]
        text += "## 具体是哪些峰谷没有恢复\n\n"
        text += "下面每个目标挑出合格率最低的两个可计算宽度的事件，作为后续查曲线的入口。这是从结果中筛出的线索，不代替完整逐 seed 数据，也不是预先证明某种结构最难。峰谷编号只在该目标内使用。\n\n"
        text += "|目标|事件|目标频率 / Hz|目标实际3 dB宽度 / Hz|漏失次数 / 有效seed|合格次数 / 有效seed|\n|---|---|---:|---:|---:|---:|\n"
        for target in sorted({r["target"] for r in final_ev}):
            buckets = {}
            for r in final_ev:
                if r["target"]==target and r["width_evaluable"]=="True":
                    buckets.setdefault(r["event_id"],[]).append(r)
            worst = sorted(buckets,key=lambda key:sum(r["complete"]=="True" for r in buckets[key])/len(buckets[key]))[:2]
            for event_id in worst:
                rr = buckets[event_id]
                text += f'|{target}|{event_id}|{fmt(rr[0]["target_frequency_hz"])}|{fmt(rr[0]["target_width_hz"])}|{sum(r["matched"]=="False" for r in rr)}/{len(rr)}|{sum(r["complete"]=="True" for r in rr)}/{len(rr)}|\n'
        text += "\n这是目标实际峰宽，不是给 L、C 设定的倍率；每一行都来自对应目标曲线的两侧 3 dB 交点。无法定义宽度的浅峰谷仍在 events.csv 检查匹配和位置，没有填成0。\n\n"
        text += "## 是代表解没恢复，还是保留的整个种群都没恢复\n\n"
        text += "|目标|末代检查的seed|种群中存在幅值结构全合格的seed|两项分数差≤5%却有明显形状差异的seed|\n|---|---:|---:|---:|\n"
        for target in sorted({r["target"] for r in populations}):
            pp = [r for r in populations if r["target"]==target and r["generation"]=="300"]
            qq = [r for r in pairs if r["target"]==target and r["generation"]=="300" and r["threshold"]=="0.05"]
            available = [r for r in pp if r["any_joint_amp_complete"]!=""]
            hits = f'{sum(r["any_joint_amp_complete"]=="True" for r in available)}/{len(available)}' if available else "不适用"
            text += f'|{target}|{len(pp)}|{hits}|{sum(r["has_meaningful_shape_difference"]=="True" for r in qq)}/{len(qq)}|\n'
        text += "\n‘形状差异’指漏失状态不同、同一目标事件拟合高度差至少6 dB、或拟合宽度相差至少2倍。只检查末代保留的200条候选，不代表所有曾经产生或可能产生的曲线。统计单位仍是 seed。\n\n"
        text += "## 中途有没有进步\n\n|目标|代数|全局MAG均值|全局PHA均值|未合格幅值事件数均值|漏失相位转变数均值|\n|---|---:|---:|---:|---:|---:|\n"
        for target in sorted({r["target"] for r in reps}):
            for generation in ("20","100","300"):
                rr = [r for r in reps if r["target"]==target and r["generation"]==generation and r["rule"]=="primary" and r["stable"]=="True" and r["audit_passed"]=="True"]
                if rr:
                    text += f'|{target}|{generation}|{fmt(fmean(float(r["mag"]) for r in rr))}|{fmt(fmean(float(r["pha"]) for r in rr))}|{fmt(fmean(float(r["incomplete"]) for r in rr))}|{fmt(fmean(float(r["phase_missing"]) for r in rr))}|\n'
        text += "\n这张表平均的是数值统计，不是把不同 seed 的阻抗曲线拼成一条。局部误差对总分的贡献还在 local.csv；没有明显事件的区域可看 bands.csv。\n\n"
        text += "## 只看少量实际曲线\n\n以下图各叠加固定三个 seed 的代表解与目标，选择基准目标300代展示；其他阶段和条件保留在文件夹里。\n\n"
        for ident in ("user_nested","rlc_ladder","three_level"):
            if ident in locations:
                image = locations[ident]/ident/"base"/"stage_0300.png"
                if image.exists():
                    text += f'### {NAMES[ident]}\n\n![{NAMES[ident]}]({image.resolve().as_posix()})\n\n'
    text += "## 这一轮能回答到哪里\n\n"
    text += "先看困难是否在不同电路中反复出现，再检查目标实际位置、宽度、突出程度和相邻结构怎样变化。单靠三档倍率不能把原因唯一归到峰宽；不同拓扑元件数不同，也不能直接归到‘复杂度’。8个seed是探索规模，不据此宣布普遍规律。\n\n"
    text += "理论模型的真参数仍在搜索范围内，存在零误差解。当前结果描述规定算法和300代预算下的恢复情况；不能写成‘MSE理论上永远无法优化这些结构’。之后才根据反复出现的缺陷，选择一个围绕三维曲线的辅助指标。\n\n"
    text += "始终保留四个问题：哪类结构反复难恢复；哪个条件使它更难；针对性指标能否改善；改善是否能推广、代价是什么。\n"
    from interpret_results import interpret
    interpret(snapshot)
    text += f"\n先读 [研究结果：通俗解读]({(ROOT/'研究结果_通俗解读.md').as_posix()})；完整电路、参数和统计见 [FINDINGS.md]({(ROOT/'FINDINGS.md').as_posix()})；运行问题与影响见 [ISSUES.md]({(ROOT/'ISSUES.md').as_posix()})。\n"
    REPORT.write_text(text,encoding="utf-8")
    return REPORT


if __name__ == "__main__":
    print(make_report())

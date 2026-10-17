"""Evidence-backed plain-language interpretation of retrieved, audited tables."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parent
NAMES = {"user_nested":"所提出的嵌套电路", "control_rlc":"单并联 RLC 对照", "rlc_ladder":"两级 RLC 梯形网络", "loaded_tank":"带负载的谐振网络", "dual_nested":"双嵌套支路", "three_level":"三级 RLC 网络", "mixed_resonators":"混合谐振支路"}

def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def accepted(base, group):
    audit_path = base/f"analysis_{group}/AUDIT.json"
    if not audit_path.exists():
        return False
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["passed"]:
        return True
    p = base/f"AUDIT_REFINEMENT_{group}.json"
    if not p.exists():
        return False
    v = json.loads(p.read_text(encoding="utf-8"))
    return (v.get("passed") and v.get("original_audit_sha256")==hashlib.sha256(audit_path.read_bytes()).hexdigest()
            and len(v["records"])==len(audit["failures"])
            and all(r["accepted"] and not r["representative"] for r in v["records"]))

def num(value):
    return None if value in ("", None) else float(value)

def fmt(value):
    return "无法定义" if value is None else f"{value:.4g}"

def interpret(snapshot):
    tables = {name:[] for name in ("representatives","events","phase_events","local","population","similar_scores_different_shapes")}
    target_meta, locations, groups = {}, {}, []
    for group in ("first","other"):
        base = ROOT/"retrieved"/group
        if not accepted(base,group):
            continue
        groups.append(group)
        analysis = base/f"analysis_{group}"
        for name in tables:
            tables[name] += read_csv(analysis/f"{name}.csv")
        for target_path in (base/"inputs_core").glob("*/*/target.json"):
            v = json.loads(target_path.read_text(encoding="utf-8"))
            target_meta[v["id"]] = v
        for target in {r["target"] for r in tables["representatives"]}:
            if (analysis/target).exists():
                locations[target] = analysis
    reps = tables["representatives"]
    valid = {(r["target"],r["seed"],r["generation"],r["rule"]) for r in reps if r["stable"]=="True" and r["audit_passed"]=="True"}
    def usable(r, rule="primary", generation="300"):
        return r["rule"]==rule and r["generation"]==generation and (r["target"],r["seed"],r["generation"],r["rule"]) in valid
    metrics, event_evidence, phase_evidence, costs = [], [], [], []
    for target in sorted({r["target"] for r in reps}):
        rr = [r for r in reps if r["target"]==target and usable(r)]
        if not rr:
            continue
        ee = [r for r in tables["events"] if r["target"]==target and usable(r)]
        pe = [r for r in tables["phase_events"] if r["target"]==target and usable(r)]
        pp = [r for r in tables["population"] if r["target"]==target and r["generation"]=="300"]
        bp = [r for r in reps if r["target"]==target and usable(r,"balanced")]
        amp_available = [r for r in rr if r["joint_amp_complete"]!=""]
        phase_seeds = {r["seed"] for r in pe}
        phase_ok = sum(all(v["complete"]=="True" for v in pe if v["seed"]==seed) for seed in phase_seeds)
        amp_population = [r for r in pp if r["any_joint_amp_complete"]!=""]
        phase_population = [r for r in pp if r["any_joint_phase_complete"]!=""]
        for eid in sorted({r["event_id"] for r in pe}):
            ev = [r for r in pe if r["event_id"]==eid]
            matched = [r for r in ev if r["matched"]=="True"]
            shifts = [float(r["shift_decades"]) for r in matched]
            phase_evidence.append(dict(target=target,event=eid,frequency_hz=float(ev[0]["target_frequency_hz"]),seeds=len(ev),missing=sum(r["matched"]=="False" for r in ev),complete=sum(r["complete"]=="True" for r in ev),position_bad=sum(abs(v)>.02 for v in shifts),width_bad=sum(not .5<=float(r["width_ratio"])<=2 for r in matched),slope_bad=sum(not .5<=float(r["slope_ratio"])<=2 for r in matched),min_shift=min(shifts) if shifts else None,max_shift=max(shifts) if shifts else None))
        metrics.append(dict(target=target,seeds=len(rr),amp_evaluable_seeds=len(amp_available),amp_ok=sum(r["joint_amp_complete"]=="True" for r in amp_available),
                            phase_evaluable_seeds=len(phase_seeds),phase_ok=phase_ok,
                            any_amp_ok=sum(r["any_joint_amp_complete"]=="True" for r in amp_population),amp_population_seeds=len(amp_population),
                            any_phase_ok=sum(r["any_joint_phase_complete"]=="True" for r in phase_population),phase_population_seeds=len(phase_population),
                            checked_candidates=sum(int(r["checked_candidates"]) for r in pp),valid_candidates=sum(int(r["valid_candidates"]) for r in pp),
                            balanced_amp_ok=sum(r["joint_amp_complete"]=="True" for r in bp if r["joint_amp_complete"]!=""),
                            mag_mean=fmean(float(r["mag"]) for r in rr),pha_mean=fmean(float(r["pha"]) for r in rr)))
        for eid in sorted({r["event_id"] for r in ee}):
            ev = [r for r in ee if r["event_id"]==eid]
            m = next(v for v in target_meta[target]["events"] if v["event_id"]==eid)
            loc = [r for r in tables["local"] if r["target"]==target and r["event_id"]==eid and usable(r)]
            matched = [r for r in ev if r["matched"]=="True"]
            ar = [num(r["amp_fraction"]) for r in loc if num(r["amp_fraction"]) is not None]
            pr = [num(r["pha_fraction"]) for r in loc if num(r["pha_fraction"]) is not None]
            event_evidence.append(dict(target=target,event=eid,kind=m["kind"],frequency_hz=m["frequency_hz"],width_hz=m["width_3db_hz"],width_decades=m["width_3db_decades"],height_db=m["height_db"],prominence_db=m["prominence_db"],
                                       seeds=len(ev),missing=sum(r["matched"]=="False" for r in ev),complete=sum(r["complete"]=="True" for r in ev) if m["width_3db_decades"] is not None else None,
                                       position_bad=sum(abs(float(r["shift_decades"]))>.02 for r in matched),height_bad=sum(abs(float(r["height_delta_db"]))>3 for r in matched),
                                       width_bad=sum(num(r["width_ratio"]) is None or not .5<=float(r["width_ratio"])<=2 for r in matched) if m["width_3db_decades"] is not None else None,
                                       amp_fraction_mean=fmean(ar) if ar else None,pha_fraction_mean=fmean(pr) if pr else None))
    done = sum(snapshot.get(g,{}).get("completed",0) for g in ("first","other"))
    finished = done==168 and set(groups)=={"first","other"} and (snapshot.get("pipeline") or {}).get("status")=="complete"
    status_text = "全部优化、核验和回收已完成。" if finished else "后续数据进入后会更新。"
    text = f"# 这批复杂电路的结果，应该怎样理解\n\n已完成优化 {done}/168 个 seed；以下解释只使用已回收且通过原始或补充核验的批次：{', '.join(groups) or '暂无'}。{status_text}\n\n"
    text += f"建议先读 [研究结果：通俗解读]({(ROOT/'研究结果_通俗解读.md').as_posix()})，那里用少量局部图解释主要发现；本文保留完整电路、参数和逐条件统计。\n\n"
    text += "## 先把本轮任务说清楚\n\n我们现在没有让程序学习新指标，而是只给它两项要求：整条曲线的幅值平均平方误差（MAG）和相位平均平方误差（PHA）尽量小。然后另外检查：峰谷有没有、位置是否对、宽度是否对，以及相位有没有在正确位置转弯。\n\n每个目标用8次独立的随机起点（seed）拟合。一个seed里有200条末代候选；主表按事先固定的 MAG＋25.12×PHA 选一条。这里的25.12只负责挑展示和统计用的代表，并没有把双目标优化改成单目标。图中是几条真实结果各画各的，曲线不平均。\n\n"
    if not metrics:
        text += "目前先不写困难结构结论，等待完整核验数据。\n"
    else:
        all_amp_good = [m for m in metrics if m["amp_evaluable_seeds"] and m["amp_ok"]==m["amp_evaluable_seeds"]]
        bad_amp = [m for m in metrics if m["amp_evaluable_seeds"] and m["amp_ok"]<m["amp_evaluable_seeds"]]
        bad_phase = [m for m in metrics if m["phase_evaluable_seeds"] and m["phase_ok"]<m["phase_evaluable_seeds"]]
        text += f"## 目前最直接的结果\n\n已核验 {len(metrics)} 个目标条件，其中 {len(all_amp_good)} 个条件的代表曲线在全部有效seed中都恢复了可评价的幅值峰谷；{len(bad_amp)} 个条件有幅值结构不合格；{len(bad_phase)} 个条件有相位转变不合格。这是在当前阈值下的计数，幅值合格不等于相位合格，也不等于每个点都完全重合。\n\n"
        if not bad_amp:
            text += "**这部分数据目前没有支持‘这些电路的幅值峰谷普遍很难拟合’。** 这是需要保留的结果，不能因为研究想找困难，就把小残差也说成漏峰。\n\n"
        text += "|目标|代表曲线幅值全合格|代表曲线相位全合格|种群中有幅值全合格解|种群中有相位全合格解|\n|---|---:|---:|---:|---:|\n"
        rate = lambda hit,n: f"{hit}/{n}" if n else "不适用"
        for m in metrics:
            text += f'|{m["target"]}|{rate(m["amp_ok"],m["amp_evaluable_seeds"])}|{rate(m["phase_ok"],m["phase_evaluable_seeds"])}|{rate(m["any_amp_ok"],m["amp_population_seeds"])}|{rate(m["any_phase_ok"],m["phase_population_seeds"])}|\n'
        text += "\n合格的检查标准：峰谷位置差不超过0.02个十倍频程（约4.7%的频率比）、高度差不超过3 dB、宽度比在0.5～2之间。相位转变检查位置、转弯速度与宽度。相位表使用相位变化速度的峰及其突出程度来筛选事件；它可能检出角度很小的局部起伏，不等于只评价大角度跳变，也不覆盖所有相位细节。幅值与相位的‘种群中有’可能来自不同候选，不能相加理解为存在同一条两者全合格的曲线。\n\n"
        text += "如果代表曲线不合格，但种群里已经有合格解，优先研究怎样选解以及幅相取舍；如果整个被检查的种群都没有，再把它列为当前算法和预算下的搜索困难。我们仍不能把搜索失败直接归罪于MSE。\n\n"
        circuits = sorted({m["target"].split("/")[0] for m in metrics})
        text += "## 这些电路具体是什么\n\n`+`表示串联，`//`表示并联。R是电阻、L是电感、C是电容；同一个字母后面的不同编号表示独立元件。下面是每类电路的基准参数，倍率条件在随后表中测量实际变化。\n\n"
        for ident in circuits:
            meta = target_meta[ident+"/base"]["circuit"]
            text += f'### {NAMES[ident]}\n\n`{meta["expression"]}`\n\n'
            png = ROOT.parents[0]/"mse_structure_complex_20261010_code"/f"{ident}.png"
            if png.exists():
                text += f'![{NAMES[ident]}电路图]({png.as_posix()})\n\n'
            text += "|元件|基准数值|\n|---|---:|\n"
            for name,value,unit in zip(meta["names"],meta["parameters"],meta["units"]):
                if unit=="H":
                    value,unit = value*1e3,"mH"
                elif unit=="F":
                    value,unit = value*1e9,"nF"
                elif unit=="ohm":
                    unit = "Ω"
                text += f'|{name}|{value:.6g} {unit}|\n'
            text += "\n"
        ranked = sorted([e for e in event_evidence if e["complete"] is not None and e["complete"]<e["seeds"]],key=lambda e:(e["complete"]/e["seeds"],-e["missing"],e["target"],e["event"]))
        if ranked:
            text += "## 不合格的峰谷，具体坏在哪里\n\n以下选最多8个线索，完整记录保存在 EVIDENCE.json。漏失、位置偏差、高度偏差、宽度偏差可以同时出现，所以各列不能相加当作失败seed数。\n\n|目标与事件|实际频率 / Hz|实际宽度 / Hz|漏失|位置不合格|高度不合格|宽度不合格|结构合格|该局部幅值误差占总分的均值|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
            for e in ranked[:8]:
                fraction = "不适用" if e["amp_fraction_mean"] is None else f'{100*e["amp_fraction_mean"]:.2f}%'
                text += f'|{e["target"]} / {e["event"]}|{fmt(e["frequency_hz"])}|{fmt(e["width_hz"])}|{e["missing"]}/{e["seeds"]}|{e["position_bad"]}|{e["height_bad"]}|{e["width_bad"]}|{e["complete"]}/{e["seeds"]}|{fraction}|\n'
            text += "\n局部范围是目标峰谷的两侧3 dB交点之间。‘只占总分很少’可以作为值得单独引导的线索，但不能单独证明程序不重视它；相位贡献、搜索预算和参数联动还要一起看。\n\n"
        bad_pe = sorted([e for e in phase_evidence if e["complete"]<e["seeds"]],key=lambda e:(e["complete"]/e["seeds"],-e["missing"],e["target"],e["event"]))
        if bad_pe:
            text += "## 相位转变没有做好，具体是什么意思\n\n相位曲线上有些地方转弯很快，我们用转弯速度的峰来定位它；U表示上升方向，D表示下降方向。它不是幅值的峰，也不一定是垂直跳变。下表区分‘转弯没出现’与‘转弯出现了但位置、速度或宽度不对’。各项也可能重叠。\n\n|目标与转变|目标频率 / Hz|漏失|位置不合格|转变宽度不合格|转弯速度不合格|全部合格|\n|---|---:|---:|---:|---:|---:|---:|\n"
            for e in bad_pe[:10]:
                text += f'|{e["target"]} / {e["event"]}|{fmt(e["frequency_hz"])}|{e["missing"]}/{e["seeds"]}|{e["position_bad"]}|{e["width_bad"]}|{e["slope_bad"]}|{e["complete"]}/{e["seeds"]}|\n'
            text += "\n例如首批 user_nested/base 的 U1 在约1371 Hz：2个代表解的位置不合格，但转变存在，宽度与速度都在容差内；不能写成‘相位结构丢失’。完整偏差见 EVIDENCE.json，其他电路应按自己的数据解释。\n\n"
        text += "## 改L、C之后，不能只说峰宽变了\n\n下面把每个目标的实际峰谷列出来，读表时先看同一电路的三个条件。LC倍率是参数操作，实际宽度才是曲线测出的结果。事件编号只在各自目标内有效，跨条件不能仅凭编号相同就断定是同一个物理结构。\n\n|目标|事件|频率 / Hz|宽度 / Hz|高度 / dBΩ|突出程度 / dB|\n|---|---|---:|---:|---:|---:|\n"
        for e in sorted(event_evidence,key=lambda e:(e["target"],e["event"])):
            text += f'|{e["target"]}|{e["event"]}|{fmt(e["frequency_hz"])}|{fmt(e["width_hz"])}|{fmt(e["height_db"])}|{fmt(e["prominence_db"])}|\n'
        text += "\n真实端口曲线的峰位置、高度、宽度和邻近结构可能一起改变。因此这轮能发现条件相关的困难线索，尚不能证明‘仅仅变窄导致失败’。逐目标对应记录在 base_event_correspondence.json，也只是按频率做的匹配。\n\n"
        text += "## 两项整体分数接近，结构是否仍可能不同\n\n|目标|检查seed数|有‘分数接近但形状明显不同’的seed数|\n|---|---:|---:|\n"
        for m in metrics:
            rr = [r for r in tables["similar_scores_different_shapes"] if r["target"]==m["target"] and r["generation"]=="300" and r["threshold"]=="0.05"]
            text += f'|{m["target"]}|{len(rr)}|{sum(r["has_meaningful_shape_difference"]=="True" for r in rr)}|\n'
        text += "\n两项分数都按各自尺度比较，主阈值为5%；为避免接近零时比例爆炸，分母下限为0.01 dB²与(0.5°)²换成rad²。所以上表不能一概解释为‘原始相对误差都小于5%’。形状明显不同指事件漏失状态不同、同一目标事件拟合高度相差≥6 dB、或宽度相差≥2倍。当前这张表只查幅值峰谷，不能拿它证明相位结构等价。\n\n"
        text += "## 计算成本也需要看\n\n所有seed都是200个候选、300代，共60000次参数候选评价。但复杂曲线可能需要更多频点，因此同样代数不等于同样计算成本。下面的查询数每次同时计算目标与候选；单seed耗时来自8任务并行时的运行记录，不能简单相加成整批墙钟时间。\n\n|电路|已核验seed|单seed平均耗时 / 秒|单seed平均频率查询数|\n|---|---:|---:|---:|\n"
        for ident in circuits:
            records = []
            for group in groups:
                d = ROOT/"retrieved"/group/f"batch_{group}"/ident
                records += [json.loads(p.read_text(encoding="utf-8")) for p in d.glob("*/seed_*/summary.json")]
            if records:
                c = dict(circuit=ident,seeds=len(records),seconds_mean=fmean(r["seconds"] for r in records),frequency_queries_mean=fmean(r["model_frequency_queries"] for r in records))
                costs.append(c)
                text += f'|{NAMES[ident]}|{c["seeds"]}|{c["seconds_mean"]:.1f}|{c["frequency_queries_mean"]:.0f}|\n'
        text += "\n"
        text += "## 接下来应当做什么判断\n\n1. 哪类结构反复难恢复：以逐seed漏失与形状偏差为依据，先排除只由代表选择造成的例子。\n2. 哪个条件使它更难：对照同一电路三个目标的实际结构变化；这一轮仍不足以唯一分离宽度、峰高和相邻结构的作用。\n3. 针对性指标能否改善：本轮没有加入辅助指标，暂时没有答案。指标候选仍围绕 f、Re Z、Im Z 的三维曲线；不能提前写成有效。\n4. 改善是否能推广、代价是什么：本轮扩大电路范围，但尚未验证指标，也没有新实测数据；8个seed用于探索，不宣称普遍规律。\n\n**下一步先从持续失败的局部结构中选一个最有证据的候选。** 如果加预算或换选解已能恢复，就优先解释这些因素；若仍需指标，再用相同seed、初始化和明确成本做基础目标与辅助指标的对照。\n"
    (ROOT/"FINDINGS.md").write_text(text,encoding="utf-8")
    (ROOT/"EVIDENCE.json").write_text(json.dumps(dict(verified_groups=groups,targets=metrics,events=event_evidence,phase_events=phase_evidence,costs=costs),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    faults = "# 运行与核验问题记录\n\n这份记录与结果一起保留，问题发生后不删除旧记录，也不把修复写成实验效果。\n\n"
    faults += "## 1. 首批分析停止：一个非代表候选的结构判定不稳定\n\n优化任务本身已完成48/48，没有seed运行失败。异常是 user_nested/base、seed 33003、300代、第129号候选的一个相位转变。原先4级网格的位置差为0.0200127个十倍频程，8级为0.0199385，落在0.02阈值两侧，所以检查停止。它不是主代表，也不是平衡规则的代表。\n\n保留原始代码、参数、分数和失败记录，另外加密复核到32、64级。位置差分别为0.0199747、0.0199690，两级都在同一侧；幅相分数按原容差复算通过。没有放宽阈值，没有重新优化或挑换seed。原始表仍排除这条不稳定候选，代表曲线统计不受影响；对应seed的种群比较少纳入1条候选，因此不是对200条候选全部已证明的结论。详细记录见首批 AUDIT_REFINEMENT_first.json。\n\n"
    for group in groups:
        base = ROOT/"retrieved"/group
        audit = json.loads((base/f"analysis_{group}/AUDIT.json").read_text(encoding="utf-8"))
        faults += f"## {group} 批次完整性\n\n已核验任务 {audit['completed_jobs']}/{audit['planned_jobs']}；当前主结果核验表记录异常 {len(audit['failures'])} 条。原始网格的问题与补充处理见下面说明，不能把当前通过理解为原先没有异常。\n\n"
        if group=="other" and audit["failures"]:
            supplement = json.loads((base/f"AUDIT_REFINEMENT_{group}.json").read_text(encoding="utf-8"))
            for r in supplement["records"]:
                i = r["identity"]
                faults += f"- {i['target']} / seed {i['seed']} / 第{i['generation']}代 / 候选{i['index']}：非代表候选；32、64级结构判定一致且分数按原容差通过。原表保留排除，细节见补充复核文件。\n"
            faults += "\n"
        refinement_path = base/f"AUDIT_REFINEMENT_{group}.json"
        if group=="other" and audit["passed"] and refinement_path.exists():
            refinement = json.loads(refinement_path.read_text(encoding="utf-8"))
            faults += f"末批原始网格有{len(refinement['records'])}条候选结构判定不稳定，分数复算均通过。其中包含中途代表解，不能直接按‘非代表候选剔除’处理。因此仅对这些候选加密到32、64级，其中一条仍贴近位置阈值，继续检查到128、256级。确认判定稳定并按原分数容差通过后，重新生成完整统计表；主结果使用加密后的诊断，原始整套表保存在 analysis_other_original_grid。没有换代表参数、换seed、增加优化代数或放宽合格阈值。\n\n"
            faults += "|目标|seed|代数|候选编号|加密后的分数及结构核验|\n|---|---:|---:|---:|---|\n"
            for r in refinement["records"]:
                i = r["identity"]
                faults += f"|{i['target']}|{i['seed']}|{i['generation']}|{i['index']}|通过|\n"
            faults += "\n补充记录同时保留每条候选的原始和加密诊断。诊断精度提升只影响这几条曲线的评分复算和结构判断，参数优化过程与训练分数均保持原样。\n\n"
            faults += "三级网络那条非代表候选的低频相位转变位置差，在128、256级分别为约−0.0199976、−0.0199966个十倍频程，仍非常接近−0.02的边界。这是按冻结阈值的分类，不是证明曲线完全恢复；不能用这个边界样本单独建立困难规律。逐级轨迹保留在 REFINEMENT_TRACE_other.json。\n\n"
    status = snapshot.get("pipeline") or {}
    if status.get("status")=="needs_review":
        faults += "**当前还有待处理的停止，不能称为全部完成。**\n\n````\n"+status.get("traceback","")+"\n````\n\n"
    faults += "## 这些实验仍然存在的限制\n\n- 只有8个seed、300代；这里的困难受算法和预算限定。\n- LC互换倍率会同时改变多个曲线特征，不能单独证明宽度的因果作用。\n- 各电路参数维数不同，拟合难度不能直接归因于嵌套层数。\n- 结构合格标准有容差，不能等同于完全重合；贴近阈值的样本需要加密复核。\n- 全部为正确模型生成的理论目标，真参数处幅值和相位误差可同时为零。\n- 没有进行辅助指标优化，也没有把本轮结果上传网站。\n"
    faults += "\n## 本地报告维护\n\n首批回收后的通俗报告生成曾因Windows默认GBK编码无法读取UTF-8中文目标说明而报错。已为读取显式指定UTF-8，重新生成成功；问题仅影响说明文件生成，没有影响服务器优化、原始数值或独立复核。回收程序补充了下载超时、进度状态和并发请求上限，随后继续从已校验档案生成报告。\n"
    faults += "\n## 最终回收时的连接中断\n\n末批回收期间出现一次SSH连接中断。监视程序重新连接并重试，最后正常退出，状态为complete。两个回收包均通过SHA256与ZIP完整性校验；解压后的168个任务又逐项核验预算、保存阶段和文件哈希。没有用未完成的下载生成最终结论，也没有因为连接问题重新运行优化。\n"
    faults += "\n## CPU资源使用偏低\n\n运行中确认，容器CPU配额为32核（cpu.max为3200000/100000），但优化只设置8个独立进程。8个进程各占约1核，因此实例总体CPU约25%，并非优化已经结束或程序没有推进。这属于并行调度偏保守，会延长完成时间；没有改变seed、参数、每seed60000次候选评价或结果评分。当前进程池不能直接热增并行度，且未保存完整随机状态来恢复正在计算的seed，强行重启会产生中断和重算开销。后续新批次应按容器实际配额安排更多独立seed并行，例如24～28个，再检查实际吞吐；不能用宿主机192个逻辑CPU替代容器32核配额。不同并行度下的单seed耗时也不能直接比较算法效率。\n"
    proof_path = ROOT/"retrieved/other/PARALLEL_ANALYSIS.json"
    if proof_path.exists():
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        faults += f"\n后处理分析原先也是串行，只使用约1核。此次保留原始串行诊断记录，使用{proof['workers']}个进程并行调用完全相同的冻结诊断函数；复用{proof['serial_prefix_records']}条原记录，计算{proof['new_parallel_diagnostics']}个剩余独立诊断。对{proof['fresh_serial_crosschecks']}个跨目标样本重新串行复算，结果与缓存/并行结果逐字段完全一致。首次完整复核仍保留上面说明的网格不稳定问题，随后只对这些候选加密并重新生成统计；完整结果通过核验后才结束重复的串行后处理。原始串行部分结果另存 analysis_other_serial_partial，计算依据和助手脚本哈希保存在 PARALLEL_ANALYSIS.json。参数优化没有重启、预算和seed没有改变。\n"
    if (ROOT/"retrieved/other/ARCHIVE_POLICY.md").exists():
        faults += "\n## 回收包与原始数据\n\n末批全量包约1.17 GB，其中大量空间是重复、可重建的绘图数组。另生成约98.8 MB的核心回收包，保留全部120个seed、所有保存阶段的参数与训练分数、初始种群、前沿和选解索引、完整诊断、原始/加密统计表、最终图片、冻结代码及来源证明。只省略可从参数与代码重建的三维绘图导出，以及原始网格版本重复的图片。服务器全量包仍保留，未删除。回收后逐个检查文件哈希、种群尺寸、阶段数量、初始化配对与预算；本地并不缺少任何优化seed或保存阶段。详见 ARCHIVE_POLICY.md。\n"
    (ROOT/"ISSUES.md").write_text(faults,encoding="utf-8")

if __name__=="__main__":
    interpret(json.loads((ROOT/"LAST_STATUS.json").read_text(encoding="utf-8")))

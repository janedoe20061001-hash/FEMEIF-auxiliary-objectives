from core import *
import csv
from statistics import fmean

NAMES={"case1":"Case1基准", "ladder8":"梯形网络倍率8", "mixed0p25":"混合支路倍率0.25", "control0p25":"单并联RLC窄峰对照"}

def read(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def main():
    rows=read(OUT/"analysis/REPRESENTATIVES.csv")
    summaries=json.loads((OUT/"batch/summaries.json").read_text())
    audit=json.loads((OUT/"analysis/AUDIT.json").read_text())
    integrity=json.loads((OUT/"INTEGRITY.json").read_text())
    assert integrity["checked_jobs"]==32 and integrity["evaluations"]==1920000
    table="# 第三步配对实验：三维几何指标有没有帮助？\n\n2026年10月11日。本轮结合Case1和新电路更新指标，再在本地完成32个配对优化任务。没有上传网站。\n\n"
    table+="## 先看这轮得到的答案\n\n**这个三维指标在梯形网络上有初步帮助，但还不能作为通用方案。** 重点结构合格数：梯形网络0/4→3/4，Case1仍0/4，混合支路2/4→1/4，简单RLC对照保持4/4。这里‘合格’是通过固定检查门槛，‘4’是4次新的随机起跑，不能当作已确定的长期成功率。\n\n"
    table+="Case1新增组有一次修好了尖峰，却没修好邻谷；4个配对的整体幅值、相位误差都增大。这提示我们：指标能看出局部错误，还要能提供逐步修好的信号，并防止为了局部形状损伤整条曲线。直接评分与分项数据给出了‘峰项错位后分辨度变弱’和‘谷项可能主导最大值’两条线索，尚不是失败原因的因果证明。\n\n"
    table+="58条主/补充代表的基础分数及结构判断均稳定；1条几何训练分数与加密复算不一致，保留原记录。下面有全部配对、图片与故障影响。机制解释及后续方向见[新结论通俗解读](D:/develop/FEMEIF/output/mse_structure_step3_20261011/FINDINGS.md)。\n\n"
    table+="## 先把问题说清楚\n\n前两步已经找到一些反复恢复不好的部位。现在给程序增加一个要求：在预定局部范围内，三维阻抗轨迹是否按照频率顺序伸出、转向和折回？我们希望知道它是否帮助找参数，而不只是在事后指出某条曲线不好。\n\n"
    table+="如果第一次接触这项研究，可以这样理解：阻抗曲线是在不同频率下记录电路的反应。幅值表示反应有多大，相位表示反应在时间上偏前还是偏后。MAG和PHA是两张成绩单，分别对整条曲线的幅值、相位差异取平方后求平均，分数越低越好。尖峰和谷只占很短的一段，平均成绩好时，它们仍可能画错。\n\n"
    table+="‘seed’是一组固定随机数，相当于一次可重复的随机起跑。配对意味着两种方法从相同的200组参数起跑，再各自搜索300代；不是把两条曲线的参数只改一个元件。全部元件参数都由程序同时寻找。这里的‘代表’是按预先规定的规则，从最后保留下来的200个候选中选一条，不能保证是所有参数中的理论最好答案。\n\n"
    table+="两组都保留MAG/PHA：幅值先换成dB计算平均平方误差，相位用弧度。新增组另外优化一个局部三维轨迹瓶颈分数。每个目标4个相同随机起点，种群200、300代，每次60,000次候选评价；共1,920,000次正式评价。原先的数据只用来设计这次实验，没有与新seed结果混成一个成功率。\n\n"
    table+=f"完整设置见[开跑前约定]({(OUT/'PREREGISTRATION.md').as_posix()})，通俗设计说明见[第三步更新设计]({(ROOT/'docs/三维阻抗曲线_第三步更新设计_20261011.md').as_posix()})。\n\n"
    table+=f"想先看实验对象，可以打开[四个目标的普通电路图、参数和搜索范围]({(OUT/'TARGETS.md').as_posix()})。本文后面的结果图与这些电路一一对应。\n\n"
    table+="## 新指标怎样看曲线\n\n三维轴是对数频率、原始阻抗实部、原始阻抗虚部。各局部坐标使用目标固定的频率宽度和阻抗尺度归一化，变成无量纲坐标。峰和谷各用自己的尺度，避免大峰盖住小谷。两条曲线必须按频率顺序匹配、覆盖整个窗口，只能容许0.25个目标宽度内的频率错位。\n\n"
    table+="每个窗口取最难贴合处的误差，再取重点峰谷中较大的误差，作为一个第三目标。单峰对照只有一个窗口。两个窗口之间的连接段没有直接进入几何分数，因此另查包含连接段的局部相位误差。\n\n"
    table+="这不是同频点欧姆距离的全频平均，也没有把20log|Z|当成半径。它有明确的尺度、窗口、离散采样和匹配约定，并不是不依赖坐标的物理距离。\n\n"
    table+="## 1. 按原来的幅相规则选解，结果怎样？\n\n两组都从末代保留种群按MAG＋25.12×PHA最小选一条曲线，不根据新增几何分数换代表。优化过程中MAG/PHA仍是两个独立目标；这项加权规则只用于事后统一选代表。‘结构合格’要求预定重点峰谷在同一条曲线上，位置、高度和宽度都满足原检查标准；不是仅仅找到了一个峰。\n\n"
    table+="本轮检查门槛沿用原实验：中心偏差不超过0.02个log10频率单位（约4.7%的比例偏移），高度误差不超过3 dB，3 dB宽度为目标的0.5—2倍，并先按既定事件匹配规则找到对应结构。这是一个允许偏差的筛查门槛，不等于完全重合；尤其对很窄的峰，位置门槛可能比峰本身宽得多，所以逐事件文件另保留连续偏差。P表示峰，V表示谷，后面的数字只是该目标上的编号。\n\n"
    table+="|目标|组别|重点结构全部合格|MAG均值 / dB²|PHA均值 / rad²|局部相位RMSE均值 / °|\n|---|---|---:|---:|---:|---:|\n"
    for ident in CFG["targets"]:
        for method in CFG["methods"]:
            rr=[r for r in rows if r["target"]==ident and r["method"]==method and r["rule"]=="primary"]
            assert len(rr)==4
            table+=f'|{NAMES[ident]}|{"MAG/PHA" if method=="MAG_PHA" else "＋几何指标"}|{sum(r["focus_joint_ok"]=="True" for r in rr)}/4|{fmean(float(r["mag"]) for r in rr):.6g}|{fmean(float(r["pha"]) for r in rr):.6g}|{fmean(float(r["local_phase_rmse_deg"]) for r in rr):.5g}|\n'
    table+="\n分数先按每条代表曲线独立复算，再取4个seed均值；曲线本身不平均。局部相位统计只对应各目标自己的固定范围，跨电路不当成同一个难度尺度。\n\n"
    table+="### 看每个配对，而不只看均值\n\n|目标|seed|基线结构合格|新增组结构合格|MAG变化：新增−基线 / dB²|PHA变化 / rad²|\n|---|---:|---|---|---:|---:|\n"
    for ident in CFG["targets"]:
        for seed in CFG["seeds"]:
            a,b=[next(r for r in rows if r["target"]==ident and r["seed"]==str(seed) and r["method"]==method and r["rule"]=="primary") for method in CFG["methods"]]
            table+=f'|{NAMES[ident]}|{seed}|{"是" if a["focus_joint_ok"]=="True" else "否"}|{"是" if b["focus_joint_ok"]=="True" else "否"}|{float(b["mag"])-float(a["mag"]):+.5g}|{float(b["pha"])-float(a["pha"]):+.5g}|\n'
    table+="\n正的分数变化表示退步，负的表示改善。不能用某个seed变好代表全部seed，也不能用整体分数变好代替局部结构恢复。\n\n"
    table+="### 其他部位有没有被拉坏？\n\n下面只检查主代表：‘其他事件’是重点峰谷之外、能定义3 dB宽度的目标幅值事件；‘多余事件’按原1 dB突出程度规则匹配后统计。事件数量是筛查，不能当成严重程度。\n\n|目标|组别|其他目标幅值事件不合格数均值|多余幅值事件数均值|全部可评价幅值事件都合格的seed|\n|---|---|---:|---:|---:|\n"
    for ident in CFG["targets"]:
        for method in CFG["methods"]:
            dd=[json.loads((OUT/f"analysis/details/{ident}_{method}_{seed}_primary.json").read_text(encoding="utf-8")) for seed in CFG["seeds"]]
            incomplete=[sum(e["event_id"] not in CFG["focus_events"][ident] and e["width_evaluable"] and not e["complete"] for e in d["diagnostic"]["events"]) for d in dd]
            table+=f'|{NAMES[ident]}|{"MAG/PHA" if method=="MAG_PHA" else "＋几何指标"}|{fmean(incomplete):.4g}|{fmean(len(d["diagnostic"]["extra_events"]) for d in dd):.4g}|{sum(d["summary"]["all_amp_ok"] for d in dd)}/4|\n'
    table+="\n‘全部合格’只表示预定目标事件通过门槛，并不保证没有多余起伏、相位全部正确，或每一点完全重合。\n\n"
    table+="## 2. 允许事先规定的小幅取舍后，是否出现好结构？\n\n另一条选解规则允许MAG、PHA各不超过配对基线代表的1.10倍，再加固定小误差下限0.01 dB²及(0.5°)²对应rad²。在符合范围的曲线中选几何分数最小者。两组都使用这条规则。这里只是一个事先限定的取舍机会，不代表工程上已认可这些退步。\n\n"
    table+="|目标|组别|训练范围内有候选的seed|独立复算仍在范围内|结构合格且独立复算仍在范围内|\n|---|---|---:|---:|---:|\n"
    for ident in CFG["targets"]:
        for method in CFG["methods"]:
            rr=[r for r in rows if r["target"]==ident and r["method"]==method and r["rule"]=="limited_tradeoff"]
            valid=[r for r in rr if r["secondary_reference_eligible"]=="True"]
            table+=f'|{NAMES[ident]}|{"MAG/PHA" if method=="MAG_PHA" else "＋几何指标"}|{len(rr)}/4|{len(valid)}/4|{sum(r["focus_joint_ok"]=="True" for r in valid)}/4|\n'
    table+="\n没有候选不是丢掉这个seed后再算成功率，而是记为没有满足既定要求的解。此表和主结果分别解读：只在这里改善，说明改善需要改变取舍。\n\n"
    table+="## 3. 少量实际曲线\n\n按seed编号固定展示第一个41000和最后一个41003，使用相同主选解规则，所有曲线分别画出。图是理解例子，完整判断仍看4个配对与逐事件数据。\n\n"
    for ident in CFG["targets"]:
        table+=f'### {NAMES[ident]}\n\n![{NAMES[ident]}局部幅相曲线]({(OUT/f"figures/{ident}_local.png").as_posix()})\n\n'
        if ident in ("case1","ladder8"):
            table+=f'![{NAMES[ident]}三维轨迹]({(OUT/f"figures/{ident}_3d.png").as_posix()})\n\n三维图只画seed 41000的两组代表与目标；实部、虚部为原始Ω，透视和轴尺度影响视觉距离，不能直接从图片量误差。\n\n'
    if (OUT/"figures/case1_partial_peak_example.png").exists():
        pick=json.loads((OUT/"PLOT_SELECTION.json").read_text())["additional_example"]
        table+=f'### 单独解释一个“峰恢复、组合仍失败”的例子\n\n以下seed {pick["seed"]}按明确的事后诊断规则选出：新增组主代表的P3合格，但峰谷组合仍不合格；若有多个，取编号最小者。这是解释局部改善的例子，不是随机图，也不替代前面的4个配对统计。\n\n![局部峰恢复的代价]({(OUT/"figures/case1_partial_peak_example.png").as_posix()})\n\n峰已经较好恢复，谷和整条曲线仍要一起评价。对应全局幅相分数见逐seed表，不能只凭这张局部好图宣布成功。\n\n'
    table+="## 4. 计算开销\n\n|目标|组别|单seed平均耗时 / 秒|单seed基础候选频点查询数均值|新增几何候选频点查询数均值|\n|---|---|---:|---:|---:|\n"
    for ident in CFG["targets"]:
        for method in CFG["methods"]:
            rr=[r for r in summaries if r["target"]==ident and r["method"]==method]
            table+=f'|{NAMES[ident]}|{"MAG/PHA" if method=="MAG_PHA" else "＋几何指标"}|{fmean(r["seconds"] for r in rr):.1f}|{fmean(r["base_frequency_queries"] for r in rr):.0f}|{fmean(r["geometry_frequency_queries"] for r in rr):.0f}|\n'
    table+="\n两组候选评价次数相同，但计算成本不同。基础自适应频点数还会因搜索到了不同参数而改变；几何项额外做轨迹匹配。查询数是候选频点评价口径，不是浮点运算数；耗时来自4任务并行的本地记录，受其他任务负载影响，不能用它证明服务器或优化器的纯算法速度。\n\n"
    table+="## 5. 独立检查与结论范围\n\n"
    table+=f'32个任务、每任务60,000次评价、15个保存阶段、配对初始化和文件哈希已核验。独立检查了{audit["curves"]}条主/补充代表记录（两种规则可能选中同一候选），基础分数或结构判定稳定性异常{audit["structure_and_base_score_failures"]}条，几何采样检查异常{audit["geometry_sampling_failures"]}条。异常原始记录全部保留，不能把“程序运行成功”当成“指标已经验证有效”。\n\n'
    table+="训练几何项使用129点网格，复核增加257、513、1025点及候选模态锚点，保持尺度与容差固定。详细核验见AUDIT.json。基础分数与结构检查稳定，指计算判断可重复，不表示曲线一定拟合合格。\n\n"
    table+="异常是梯形网络新增组seed 41003的主代表：几何分数由训练的0.473403变为加密的0.445386，粗网格偏高约6.29%，超过原容差；加密结果已稳定，独立幅相与结构判断仍稳定。因此保留结构合格计数，但不能声称全部训练几何分数都可靠。若暂不使用这个配对作敏感性说明，剩余3个配对仍为基线0/3、新增2/3；这不替代正式4个配对的统计。\n\n"
    table+="只有4个seed，而且所有电路都参与指标开发。本轮只能判断这个版本在这些条件下是否提供初步帮助，不能证明所有三维指标有效或无效，也不能宣布MSE理论上不可能拟合。不同电路边界和维数不同，没有把困难唯一归因于宽度或嵌套层数。\n\n"
    table+="目标数由2个变成3个，本身也会改变候选保留的方式。本轮没有无关第三目标的对照，尚不能把改善唯一归因于顺序匹配这一项几何信息。\n\n"
    table+=f'## 数据与设计入口\n\n- [更新设计]({(ROOT/"docs/三维阻抗曲线_第三步更新设计_20261011.md").as_posix()})\n- [冻结实验约定]({(OUT/"PREREGISTRATION.md").as_posix()})\n- [逐seed代表与独立分数]({(OUT/"analysis/REPRESENTATIVES.csv").as_posix()})\n- [逐事件的连续偏差与合格判断]({(OUT/"analysis/EVENTS.csv").as_posix()})\n- [候选数量和几何分数]({(OUT/"analysis/RETAINED_SCORES.csv").as_posix()})\n- [核验记录]({(OUT/"analysis/AUDIT.json").as_posix()})\n- [运行问题与影响]({(OUT/"ISSUES.md").as_posix()})\n- [直接评分诊断]({(OUT/"DIRECT_SCORE_PROBES.json").as_posix()})\n'
    table+=f'- [新结论与下一步方向]({(OUT/"FINDINGS.md").as_posix()})\n- [目标电路结构和参数]({(OUT/"TARGETS.md").as_posix()})\n'
    (OUT/"REPORT.md").write_text(table,encoding="utf-8")
    print("Evidence tables and readable report generated; human interpretation still required",flush=True)

if __name__=="__main__":
    main()

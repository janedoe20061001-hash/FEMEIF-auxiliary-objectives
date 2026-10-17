"""Formal diagnosis, provisional process specification, and immutable provenance."""
from diagnose import *
import csv,re
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'output/valley_process_20261017'
def link(name):return '/D:/develop/FEMEIF/output/valley_process_20261017/'+name
def table(headers,rows):return '|'+ '|'.join(headers)+'|\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('|'+ '|'.join(str(v) for v in r)+'|' for r in rows)+'\n'
def main():
    data=json.loads((OUT/'DIAGNOSTICS.json').read_text());s=json.loads((OUT/'SUMMARY.json').read_text());records=data['records'];targets=[r for r in records if r['source'] is None];cohort=list(csv.DictReader((OUT/'COHORT_TABLE.csv').open(encoding='utf-8-sig')));complete=[r for r in cohort if r['old_complete']=='True'];example=next(r for r in records if r['id']=='held_003_better');ref=next(t for t in targets if t['target']==example['target']);rows=[]
    for t in targets:
        v=next(v for v in t['levels'][-1]['core']['radial_extrema'] if v['kind']=='valley')
        rows.append([t['target'],f"{v['frequency_hz']:.2f}",f"{v['real_ohm']:.3f}",f"{v['imag_ohm']:.3f}",f"{v['phase_deg']:.2f}",f"{v['R_dR']:.3f}",f"{v['X_dX']:.3f}"])
    porttable=table(['目标','精化谷频率/Hz','ReZ/Ω','ImZ/Ω','相位/°',"RR'/Ω²每decade","XX'/Ω²每decade"],rows)
    oldtable=table(['旧完整候选','原配对关注位置','1dB宽比','3dB宽比','运动跨度比','弧长比','核心区径向极值顺序'],[[r['id'],r['original_pair_region'],f"{float(r['width1_ratio']):.4f}" if r['width1_ratio'] else '—',f"{float(r['width3_ratio']):.4f}" if r['width3_ratio'] else '—',f"{float(r['core_arc80_ratio']):.4f}",f"{float(r['core_length_ratio']):.4f}",r['core_radial_sequence'] or '无'] for r in complete])
    sequences={key:sum(r['core_radial_sequence']==key for r in complete) for key in ['', 'peak valley','valley']}
    assert sequences=={'':4,'peak valley':3,'valley':3}
    report=r'''# 混合网络谷结构：应恢复怎样的复轨迹过程

日期：2026年10月17日。范围：本地诊断现有理论电路及已保存候选；未运行新优化，未设计新的优化辅助目标。

## 摘要与直接判断

**这类谷需要恢复的是：在正确的频率区间，复阻抗沿目标所示路径逐步接近谷底状态，再按目标过程离开；谷底一个点及一个3 dB宽度不足以代表完成恢复。**具体应同时检查绝对Re/Im状态、进入/离开的方向与频率持续范围、附加往返过程，以及核心区外副作用。

本轮固定检查两个混合网络条件的全部32条旧候选及两个目标。34条响应的128/256细分、弧长独立Gauss积分、径向极值根和节点电路抽样全部通过。10条候选被旧规则判为V3完整；其中在固定目标核心区内，4条没有径向谷转换，3条是额外峰后接谷，仅3条与目标同为单谷顺序。这三个子类不是新的合格率，更不能据此宣布仅3条真正恢复；它们仍需状态和持续过程检查。

代表样本held_003正侧的3 dB宽度是目标的67.3%，但1 dB宽度只有0.897%，中间80%复轨迹弧长对应的频率跨度只有目标的1.91%。3 dB右侧距离约为左侧525倍。**宽度数字达标可以由窄谷与缓慢背景共同产生，无法保证宽谷过程恢复。**

另一个关键事实：两个目标的V3谷中心电抗分别约+93.0 Ω和+44.5 Ω，并非电抗过零或纯阻性点。因此，不能统一把这类谷的恢复设为ImZ=0，也不能要求它必须形成圆环。本文形成的是待冻结的过程检查框架，不追改旧标签或之前指标失败结论。

## 1. 分析对象和固定条件

执行前已保存[诊断方案](/D:/develop/FEMEIF/docs/混合网络谷结构完整过程_诊断方案_20261017.md)。样本来自上一轮冻结24对库中两个混合网络条件的16对、32条候选。原先配对关注P2、V1、V3或间隙的候选，本轮均检查同一个目标V3；没有按本轮诊断结果再挑样本。两个目标另作基准。

这些候选已经参与过开发，且并非32个独立随机样本。这里只研究已见缺陷，不作总体发生率、显著性或独立推广结论。源码、样本参数、旧标签和来源哈希先于本次诊断输出冻结。

固定两种区间：原目标V3窗口；以及目标左/右3 dB交点之间的核心区。候选在相同绝对频率上分析，不移动频率或按候选宽度对齐。核心区边界沿用旧目标记录；另精化目标驻点用于解释，二者没有混成一套新标签。

旧“完整”的门槛是位置偏差≤0.02 decade、高度偏差≤3 dB、3 dB宽度比在0.5至2之间。它并未要求候选谷中心落在目标3 dB核心区内，也未约束整段Re/Im运动。这能解释部分核心区无谷的样本，无需假设旧事件检测发生数值故障。

## 2. 先明确目标自身的过程

设x=log10 f、Z=R+jX、v=dZ/dx，定义：

\[
v_r=\frac{\operatorname{Re}(\overline Z v)}{|Z|},\qquad
v_t=\frac{\operatorname{Im}(\overline Z v)}{|Z|}.
\]

vr表示沿复平面原点径向的变化速度：负值时|Z|减小，正值时增大。vt表示横向推进。这里的“速度”指随对数频率变化，不是空间中的物体运动或电路时域暂态。

幅值谷的驻点满足：

\[
RR'+XX'=0,\qquad v_r:\text{负}\rightarrow\text{正}.
\]

**这个条件不等于X=0。**本例两个目标在谷中心的数值为：

{{PORTTABLE}}

两个目标在核心区都只有一次负到正的径向转换。谷中心处RR'为负、XX'为正，二者平衡；复阻抗仍处于正电抗区域。该描述来自端口复函数，尚未把现象归因到唯一内部支路或唯一储能模态。

![两个目标的过程]({{BASE}}figures/target_processes.png)

作为代数参照，串联RLC的Z=R+j(ωL−1/(ωC))也有幅值谷，但其Re/Im路径是一条直线，谷底X=0。这说明谷可以对应不同的复轨迹，不能把有圆环、无圆环或纯阻性作为统一判据。

径向/切向量分别等价于d|Z|/dx和|Z|dargZ/dx。因此该分解是解释工具，本身不是新的数学创新或替代幅相拟合的独立信息来源。后续方法若要有特点，需要证明对频率顺序、状态与联合路径的处理提供实际附加作用。

## 3. 一个3 dB宽度为什么会误导

held_003正侧旧记录：位置偏差0.00235 decade、高度偏差+1.58 dB、3 dB宽度比0.673，因而被旧规则判为完整。连续曲线精化得到：

|诊断|目标|该候选|
|---|---|---|
|1 dB谷宽/decade|0.0227146|0.000203662|
|3 dB谷宽/decade|0.0600644|0.0404207|
|3 dB右/左距离比|3.43|524.57|
|固定核心区复弧长/Ω|130.66|269.98|
|端点位移/弧长|0.808|0.096|
|中间80%弧长所占频率跨度/目标|1|0.0191|

它不是将目标宽谷略微压窄，而是在背景上出现很窄的峰—谷及复平面往返。1 dB交点都靠近局部谷底；3 dB右交点则延伸到后面的缓慢背景，使总宽度看起来仍接近目标。这里背景的作用由同一连续曲线交点直接体现，不是推测采样漏峰。

![代表样本：点与过程不同]({{BASE}}figures/representative_process.png)

图中累计弧长比例按候选自身总弧长显示，用于观察变化集中在哪里；总弧长另列，不能只看归一化比例而忽略运动量。中间80%指10%到90%的弧长分位范围，不是最短80%区间，也不是电路能量宽度。

上述结果不能证明旧配对负侧更好：held_003负侧的谷偏到核心区之外，固定核心区主要是背景，同样没有恢复目标过程。它说明正侧符合旧局部标签，也仍存在明显三维过程缺陷；不能用标签局限倒过来宣布JV已经成功。

## 4. 全部32条候选的检查

在32条候选中，旧V3匹配15条、完整10条。下表完整保留这10条，不只展示极端反例。原配对的正/负侧标签只针对原配对关注结构；例如P2配对的负侧也可能满足V3旧规则。

{{OLDTABLE}}

四条旧完整候选在固定目标核心区内没有谷转换，三条出现peak→valley，三条只出现valley。前一类不代表整个频带没有谷；后一类也不等于其状态、宽度或复轨迹已经正确。统计基于无噪声模型解析导数的全部稳定根，没有用新的显著性阈值删去小起伏。

![全体候选]({{BASE}}figures/all_candidates.png)

10条旧完整候选的核心区弧长比范围为0.0608至2.066，运动跨度比为0.0191至1.189。跨度接近1也不能单独代表恢复：一条几乎只剩背景的曲线，可能在整个核心区缓慢变化，比例跨度很宽但总量很小。因此必须联合检查**量、位置、持续范围、方向和绝对状态**，不能只换一个“新宽度”代替旧宽度。

![五个既有失败对的宽度对照]({{BASE}}figures/width_comparison.png)

## 5. 完整过程的可检查定义：待冻结版本

建议将“完整”拆成以下互补检查；本轮只报告连续量，不从这32条样本反向挑选新合格线。

|检查层|具体检查什么|避免哪种误判|
|谷处状态|目标左侧、谷中心、右侧三个固定频率的ReZ、ImZ；保留平均背景偏置|形状相似但整段处于错误阻抗位置|
|频率位置及顺序|目标区域内径向负→正转换是否存在；额外正→负及负→正如何排列|谷偏到窗口外，或先出现多余峰再形成窄谷|
|过程持续范围|在固定区间比较累计弧长分布及10/50/90%频率位置，同时保留总弧长|用一瞬间的窄往返冒充持续的宽谷；或只剩背景却分位跨度很宽|
|联合方向|左侧进入、中心经过、右侧离开时的复切向方向与分量变化|幅值三点接近，但Re/Im沿不同路线变化|
|往返与外部副作用|弧长、端点位移/弧长、径向极值顺序及核心区外结构|局部补谷后多出环路、峰或谷|

其中三个固定频率的状态和切向夹角已逐候选记录，原窗口与核心区的弧长、位移效率及根序列也均保留。方向在零速度处无定义，后续协议需处理这类退化点；本次三个固定锚点没有依赖强行赋值获得通过结论。

这一定义以**目标实际复轨迹**为准：目标是直线就不能强迫拟合形成环，目标不是纯阻性就不能强迫电抗过零；频率坐标必须保留，不能通过对齐后说位置已经恢复。它是评价对象的定义，不是已经验证有效的辅助目标。

## 6. 下一步的明确顺序

1. 将旧完整标签在后续报告中称为“位置/高度/3 dB事件达标”，保留原字段和值；另列过程诊断。原报告不追改。
2. 为状态、频率顺序、总量与分位分布分别形成目标参照，保留各项连续误差；涉及多项取舍的样本明确标为取舍，不强行生成一个总排名。
3. 用独立于当前32条候选的物理扰动对检验诊断：位置移动、阻抗偏置、频率压缩、额外窄往返分别处理；涉及真实电路时承认这些因素不能全部独立控制。先判断诊断是否真的区分所声明缺陷，再设新的容差或辅助目标。
4. 只有评价定义冻结后，才讨论轻量辅助指标及配对优化。新指标必须同时面对“窄往返”和“仅剩背景”两类反例，不能靠修复其中一类宣布成功。

以上第3、4项是后续安排，本轮没有新增扰动实验、优化或服务器任务。当前证据不足以确定各过程量如何加权，更不足以宣布某个新指标可用。

## 7. 复核与局限

34条目标/候选全部通过128/256折线弧长和分位稳定检查；解析导数的32阶加密、64阶更密Gauss弧长独立复核通过；核心区及原窗口的径向根数、类型和位置稳定。最大驻点归一化径向点积约7.61×10^-12。每条响应在17个分布位置及三个固定锚点的独立节点解抽样通过，均无需高精度回退。

旧宽度解释另用全频带模态锚点寻找最近绝对高度交点，再以连续电路式精化；该诊断不是重跑原SciPy事件匹配。它不能把不显著的极值强行当作旧规则匹配事件；只对源记录确实匹配的谷补充交点解释。核心区外谷仍明确区分。

这32条来自旧搜索历史、两个目标均参与开发；没有新拓扑、实测或噪声验证。弧长及导数会受实测噪声影响；核心区的目标3 dB边界也只是当前既有物理尺度，不能自动变成普适窗口。端口诊断没有证明某个内部谐振支路的因果，也没有证明MSE理论不能拟合。

|持续追踪的问题|本轮进展|
|---|---|
|哪类结构困难？|在背景上用窄峰—谷/往返替代目标宽谷，或谷偏出目标核心区但事件仍达标。|
|什么条件更难？|幅值阈值交点受到背景影响；绝对位置容差大于局部核心尺度；归一化形状可能掩盖总量不足。|
|针对性指标能改善吗？|本轮只确定评价对象，未提出或验证新辅助目标，JV旧失败保持不变。|
|能推广、代价是什么？|仍是两个已用目标；本地可复算，无需服务器，尚无实测噪声和推广结论。|

## 8. 文件入口

- [过程检查框架](/D:/develop/FEMEIF/docs/谷结构完整过程_检查框架_20261017.md)、[全部候选表]({{BASE}}ALL_TABLES.md)。
- [固定样本]({{BASE}}COHORT.json)、[连续诊断]({{BASE}}DIAGNOSTICS.json)、[简表CSV]({{BASE}}COHORT_TABLE.csv)、[数值响应档案]({{BASE}}PROFILES.npz)。
- [物理参照]({{BASE}}PHYSICAL_REFERENCE.json)、[问题及影响]({{BASE}}ISSUES.md)、[来源哈希]({{BASE}}PROVENANCE.json)、[最终状态]({{BASE}}REVIEW_STATUS.json)。

本轮未修改旧标签或旧报告，未合成新辅助分数，未运行优化，未连接或操作服务器，未发布网站或创建监控自动化。
'''
    report=report.replace('{{BASE}}',link('')).replace('{{PORTTABLE}}',porttable).replace('{{OLDTABLE}}',oldtable)
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')
    framework=r'''# 谷结构完整过程的检查框架（待冻结）

本框架来自两个混合网络目标及32条旧候选的开发诊断，不是新的合格标准或优化指标。完整证据见[诊断报告](/D:/develop/FEMEIF/output/valley_process_20261017/REPORT.md)。

## 当前可以明确的评价对象

需要恢复的是：**在正确频率区间，从正确复阻抗状态进入谷底，再按目标的方向和持续范围离开，同时不引入目标不支持的附加过程。**

1. **状态**：在固定目标频率比较ReZ、ImZ，保留背景偏置，不能按候选自己的状态平移或缩放。
2. **频率和顺序**：用复轨迹径向速度的负→正转换识别谷，同时保留附加峰、谷顺序。不强制ImZ=0或出现圆环。
3. **运动量与持续范围**：总弧长与累计弧长的10/50/90%位置并列报告。只看归一化形状，会让总量很小的背景看起来覆盖了整个区间。
4. **方向与途径**：比较进入、经过、离开时的复切向量，目标直线、弯曲或往返分别按其自身参照处理。
5. **副作用**：同时检查核心区外，局部改善不能抵消新增结构而不报告。

## 当前不宜继续使用的简写

旧“完整”字段应在新展示中注明“位置/高度/3 dB事件达标”。它仍是有用的点和宽度检查，但不代表整段过程恢复。旧数据值和既有结论全部保留。

不要将“3 dB宽度相近”“弧长分布跨度相近”“有谷”“没有环”“电抗过零”任意一项直接等同完整恢复。径向/切向分解与幅相导数存在代数等价关系，本身也不是创新结论。

## 下一协议需要预先固定什么

- 目标窗口、频率坐标、阻抗尺度以及边界不确定性。
- 状态容差、位置容差如何结合局部尺度，而不是沿用一个对所有谷相同的绝对频率容差。
- 导数零点、极小速度、实测噪声下起伏显著性如何处理。
- 运动量、频率分布、额外过程发生冲突时，哪些是必要条件，哪些只能报告取舍。
- 独立物理扰动与新参数样本，先冻结再检查，不能从当前32条旧样本挑出最有利阈值。

本轮没有确定数值容差、加权公式或新的合格率。先将这些诊断转换成可信的比较依据，再设计轻量辅助目标；避免把评价标签的盲区继续带入指标筛选。
'''
    (ROOT/'docs/谷结构完整过程_检查框架_20261017.md').write_text(framework,encoding='utf-8')
    issues='''# 问题及影响

|问题|处理|结论影响|
|---|---|---|
|首次脚本导入复用了旧模块的路径变量，冻结源哈希时找不到本轮脚本|在本轮导入后明确设置自己的路径；冻结和计算均在修正后执行|评分、旧结果、旧源未被修改；没有为了通过数值门槛重算不同公式。|
|旧“完整”标签允许固定核心区无谷、附加峰谷或运动极窄|保留旧标签，额外列过程诊断，明确不生成新通过率|评价范围不足，不等于旧代码数值故障，也不能用来挽救JV。|
|3 dB交点可能延伸到背景|补充连续1/3 dB左右交点及复轨迹弧长分位|支持背景影响的具体解释，不将幅值宽度改为唯一新指标。|
|部分旧匹配谷位于当前核心区之外|全带寻找源事件对应的谷，仅用于宽度解释；核心区过程保持固定频率|明确区分“整个频带有谷”和“目标区间过程正确”。|
|归一化弧长分布会隐藏总量不足|同时保留弧长、端点位移及复状态|不单用分位跨度定义恢复；未调新阈值。|

34条数值检查均通过，节点抽样全部双精度通过，无数值失败被删除。分析只使用本地现有理论参数，未连接服务器、未追加搜索预算、未修改旧标签或报告、未上传网站、未创建监控。
'''
    (OUT/'ISSUES.md').write_text(issues,encoding='utf-8')
    alltable='# 全部32条候选及连续过程量\n\n原正/负侧仅针对原配对关注结构，不自动代表本轮V3的优劣。所有量为开发诊断，没有新通过门槛。\n\n'+table(['候选','原关注位置','V3旧匹配','V3旧完整','旧位置偏差','旧高度偏差','旧宽度比','1dB宽度比','3dB宽度比','运动跨度比','弧长比','位移效率','径向顺序'],[[r[k] if r[k] else '—' for k in ['id','original_pair_region','old_matched','old_complete','old_shift_decades','old_height_delta_db','old_width_ratio','width1_ratio','width3_ratio','core_arc80_ratio','core_length_ratio','core_endpoint_efficiency','core_radial_sequence']] for r in cohort])
    (OUT/'ALL_TABLES.md').write_text(alltable,encoding='utf-8')
    assert len(records)==34 and data['numerical_passed']==34
    frozen=json.loads((OUT/'COHORT.json').read_text());assert frozen['source_sha256']==sha(HERE/'diagnose.py') and frozen['bank_sha256']==sha(OLD)
    # Check the complete prior result ledger, so the old scientific conclusions remain unchanged.
    prior=ROOT/'output/variation_transport_20261017';p=json.loads((prior/'PROVENANCE.json').read_text());assert all(sha(ROOT/k)==h for k,h in p['outputs'].items());assert all(sha(ROOT/k)==h for k,h in p['analysis_sources'].items())
    for path in re.findall(r'\]\((/D:/[^)]+)\)',report):assert Path(path[1:]).exists() or path.endswith(('REVIEW_STATUS.json','PROVENANCE.json')),path
    sources={q.relative_to(ROOT).as_posix():sha(q) for q in HERE.glob('*.py')};sources['docs/混合网络谷结构完整过程_诊断方案_20261017.md']=sha(ROOT/'docs/混合网络谷结构完整过程_诊断方案_20261017.md');sources['docs/谷结构完整过程_检查框架_20261017.md']=sha(ROOT/'docs/谷结构完整过程_检查框架_20261017.md')
    outputs={q.relative_to(ROOT).as_posix():sha(q) for q in OUT.rglob('*') if q.is_file() and q.name not in ['PROVENANCE.json','REVIEW_STATUS.json','LAST_STATUS.json']}
    dump(OUT/'PROVENANCE.json',dict(sources=sources,outputs=outputs,prior_provenance_sha256=sha(prior/'PROVENANCE.json'),prior_result_files_unchanged=True,cohort_sha256=sha(OUT/'COHORT.json')))
    state=dict(status='complete_local_diagnosis',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cohort_candidates=32,target_references=2,numerical_passed=34,old_complete_count=10,old_complete_core_sequence=sequences,old_labels_unchanged=True,prior_result_ledger_unchanged=True,new_auxiliary_metric=False,new_optimization=False,server_connected=False,website_published=False,automation_created=False,manually_reviewed_figures=sorted(p.name for p in (OUT/'figures').glob('*.png')),report_sha256=sha(OUT/'REPORT.md'),provenance_sha256=sha(OUT/'PROVENANCE.json'))
    dump(OUT/'REVIEW_STATUS.json',state);dump(OUT/'LAST_STATUS.json',state);print(json.dumps(state),flush=True)
if __name__=='__main__':main()

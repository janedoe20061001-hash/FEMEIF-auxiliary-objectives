# 这批复杂电路的结果，应该怎样理解

已完成优化 168/168 个 seed；以下解释只使用已回收且通过原始或补充核验的批次：first, other。全部优化、核验和回收已完成。

建议先读 [研究结果：通俗解读](研究结果_通俗解读.md)，那里用少量局部图解释主要发现；本文保留完整电路、参数和逐条件统计。

## 先把本轮任务说清楚

我们现在没有让程序学习新指标，而是只给它两项要求：整条曲线的幅值平均平方误差（MAG）和相位平均平方误差（PHA）尽量小。然后另外检查：峰谷有没有、位置是否对、宽度是否对，以及相位有没有在正确位置转弯。

每个目标用8次独立的随机起点（seed）拟合。一个seed里有200条末代候选；主表按事先固定的 MAG＋25.12×PHA 选一条。这里的25.12只负责挑展示和统计用的代表，并没有把双目标优化改成单目标。图中是几条真实结果各画各的，曲线不平均。

## 目前最直接的结果

已核验 21 个目标条件，其中 11 个条件的代表曲线在全部有效seed中都恢复了可评价的幅值峰谷；10 个条件有幅值结构不合格；15 个条件有相位转变不合格。这是在当前阈值下的计数，幅值合格不等于相位合格，也不等于每个点都完全重合。

|目标|代表曲线幅值全合格|代表曲线相位全合格|种群中有幅值全合格解|种群中有相位全合格解|
|---|---:|---:|---:|---:|
|control_rlc/base|8/8|8/8|8/8|8/8|
|control_rlc/lc_reciprocal_0p25|8/8|8/8|8/8|8/8|
|control_rlc/lc_reciprocal_8p0|8/8|8/8|8/8|8/8|
|dual_nested/base|8/8|6/8|8/8|7/8|
|dual_nested/lc_reciprocal_0p25|8/8|6/8|8/8|7/8|
|dual_nested/lc_reciprocal_8p0|8/8|8/8|8/8|8/8|
|loaded_tank/base|8/8|4/8|8/8|8/8|
|loaded_tank/lc_reciprocal_0p25|6/8|2/8|7/8|8/8|
|loaded_tank/lc_reciprocal_8p0|6/8|5/8|6/8|7/8|
|mixed_resonators/base|4/8|1/8|5/8|2/8|
|mixed_resonators/lc_reciprocal_0p25|4/8|0/8|5/8|1/8|
|mixed_resonators/lc_reciprocal_8p0|6/8|4/8|6/8|4/8|
|rlc_ladder/base|1/8|1/8|2/8|1/8|
|rlc_ladder/lc_reciprocal_0p25|8/8|4/8|8/8|5/8|
|rlc_ladder/lc_reciprocal_8p0|1/8|0/8|1/8|1/8|
|three_level/base|4/8|0/8|6/8|0/8|
|three_level/lc_reciprocal_0p25|7/8|0/8|7/8|0/8|
|three_level/lc_reciprocal_8p0|7/8|6/8|8/8|8/8|
|user_nested/base|8/8|6/8|8/8|8/8|
|user_nested/lc_reciprocal_0p25|8/8|8/8|8/8|8/8|
|user_nested/lc_reciprocal_8p0|8/8|8/8|8/8|8/8|

合格的检查标准：峰谷位置差不超过0.02个十倍频程（约4.7%的频率比）、高度差不超过3 dB、宽度比在0.5～2之间。相位转变检查位置、转弯速度与宽度。相位表使用相位变化速度的峰及其突出程度来筛选事件；它可能检出角度很小的局部起伏，不等于只评价大角度跳变，也不覆盖所有相位细节。幅值与相位的‘种群中有’可能来自不同候选，不能相加理解为存在同一条两者全合格的曲线。

如果代表曲线不合格，但种群里已经有合格解，优先研究怎样选解以及幅相取舍；如果整个被检查的种群都没有，再把它列为当前算法和预算下的搜索困难。我们仍不能把搜索失败直接归罪于MSE。

## 这些电路具体是什么

`+`表示串联，`//`表示并联。R是电阻、L是电感、C是电容；同一个字母后面的不同编号表示独立元件。下面是每类电路的基准参数，倍率条件在随后表中测量实际变化。

### 单并联 RLC 对照

`(R1 // L1 // C1)`

![单并联 RLC 对照电路图](../../archive/LOCAL_ONLY.md#ref-b66779de945f)

|元件|基准数值|
|---|---:|
|R1|3000 Ω|
|L1|1.59155 mH|
|C1|159.155 nF|

### 双嵌套支路

`(Rs + ((R1 + (L1 // (R2 + C1))) // (R3 + (C2 // (R4 + L2)))))`

![双嵌套支路电路图](../../archive/LOCAL_ONLY.md#ref-f83c7dca7f26)

|元件|基准数值|
|---|---:|
|Rs|5 Ω|
|R1|2 Ω|
|L1|1.59155 mH|
|R2|3 Ω|
|C1|159.155 nF|
|R3|7 Ω|
|C2|72.3432 nF|
|R4|4 Ω|
|L2|0.723432 mH|

### 带负载的谐振网络

`(Rs + ((R1 + L1) // (C1 + (R2 // L2 // C2))))`

![带负载的谐振网络电路图](../../archive/LOCAL_ONLY.md#ref-62415d1fa4fd)

|元件|基准数值|
|---|---:|
|Rs|5 Ω|
|R1|2 Ω|
|L1|1.59155 mH|
|C1|159.155 nF|
|R2|1500 Ω|
|L2|0.723432 mH|
|C2|72.3432 nF|

### 混合谐振支路

`(Rs + ((R1 + L1 + C1) // (L2 + (R2 // C2 // (R3 + L3 + C3)))))`

![混合谐振支路电路图](../../archive/LOCAL_ONLY.md#ref-b224df6665bf)

|元件|基准数值|
|---|---:|
|Rs|5 Ω|
|R1|2 Ω|
|L1|1.59155 mH|
|C1|159.155 nF|
|L2|0.723432 mH|
|R2|1500 Ω|
|C2|72.3432 nF|
|R3|7 Ω|
|L3|0.397887 mH|
|C3|39.7887 nF|

### 两级 RLC 梯形网络

`(Rs + (C1 // (R1 + L1 + (C2 // (R2 + L2)))))`

![两级 RLC 梯形网络电路图](../../archive/LOCAL_ONLY.md#ref-047ec23cd2b4)

|元件|基准数值|
|---|---:|
|Rs|5 Ω|
|C1|159.155 nF|
|R1|2 Ω|
|L1|1.59155 mH|
|C2|72.3432 nF|
|R2|3 Ω|
|L2|0.723432 mH|

### 三级 RLC 网络

`(Rs + (C1 // (R1 + L1 + (C2 // (R2 + L2 + (R3 // L3 // C3))))))`

![三级 RLC 网络电路图](../../archive/LOCAL_ONLY.md#ref-34786bd18666)

|元件|基准数值|
|---|---:|
|Rs|5 Ω|
|C1|159.155 nF|
|R1|2 Ω|
|L1|1.59155 mH|
|C2|72.3432 nF|
|R2|3 Ω|
|L2|0.723432 mH|
|R3|2000 Ω|
|L3|0.397887 mH|
|C3|39.7887 nF|

### 所提出的嵌套电路

`(R1 + (L1 // (R2 + ((R3 + C1) // L2))))`

![所提出的嵌套电路电路图](../../archive/LOCAL_ONLY.md#ref-4e27d5e11224)

|元件|基准数值|
|---|---:|
|R1|2 Ω|
|L1|1.59155 mH|
|R2|3 Ω|
|R3|7 Ω|
|C1|159.155 nF|
|L2|0.723432 mH|

## 不合格的峰谷，具体坏在哪里

以下选最多8个线索，完整记录保存在 EVIDENCE.json。漏失、位置偏差、高度偏差、宽度偏差可以同时出现，所以各列不能相加当作失败seed数。

|目标与事件|实际频率 / Hz|实际宽度 / Hz|漏失|位置不合格|高度不合格|宽度不合格|结构合格|该局部幅值误差占总分的均值|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|rlc_ladder/lc_reciprocal_8p0 / P2|4.821e+04|179.5|4/8|0|3|0|1/8|17.21%|
|rlc_ladder/lc_reciprocal_8p0 / V1|4.732e+04|682.3|4/8|0|3|2|1/8|14.49%|
|rlc_ladder/base / P2|2.724e+04|585.7|0/8|0|7|0|1/8|24.92%|
|mixed_resonators/base / P2|4.941e+04|6764|2/8|0|0|2|4/8|20.66%|
|mixed_resonators/base / V3|5.268e+04|7574|2/8|0|0|2|4/8|23.17%|
|mixed_resonators/lc_reciprocal_0p25 / P1|1.07e+04|550.1|1/8|0|2|3|4/8|18.69%|
|mixed_resonators/lc_reciprocal_0p25 / V1|9956|706.8|1/8|0|3|0|4/8|15.77%|
|three_level/base / P2|2.318e+04|458.4|0/8|0|4|0|4/8|12.46%|

局部范围是目标峰谷的两侧3 dB交点之间。‘只占总分很少’可以作为值得单独引导的线索，但不能单独证明程序不重视它；相位贡献、搜索预算和参数联动还要一起看。

## 相位转变没有做好，具体是什么意思

相位曲线上有些地方转弯很快，我们用转弯速度的峰来定位它；U表示上升方向，D表示下降方向。它不是幅值的峰，也不一定是垂直跳变。下表区分‘转弯没出现’与‘转弯出现了但位置、速度或宽度不对’。各项也可能重叠。

|目标与转变|目标频率 / Hz|漏失|位置不合格|转变宽度不合格|转弯速度不合格|全部合格|
|---|---:|---:|---:|---:|---:|---:|
|three_level/base / U3|5.113e+04|6/8|1|2|2|0/8|
|three_level/lc_reciprocal_0p25 / D3|4.384e+04|5/8|3|1|1|0/8|
|three_level/lc_reciprocal_0p25 / U3|4.349e+04|5/8|3|1|1|0/8|
|mixed_resonators/lc_reciprocal_0p25 / D3|6.928e+04|3/8|2|3|5|0/8|
|rlc_ladder/lc_reciprocal_8p0 / U3|4.997e+04|4/8|0|3|0|1/8|
|mixed_resonators/lc_reciprocal_0p25 / U3|7.618e+04|3/8|4|4|4|1/8|
|rlc_ladder/lc_reciprocal_8p0 / U2|4.741e+04|3/8|0|4|4|1/8|
|rlc_ladder/base / D2|2.719e+04|0/8|0|7|4|1/8|
|rlc_ladder/base / U2|2.652e+04|0/8|0|7|6|1/8|
|rlc_ladder/lc_reciprocal_8p0 / D2|4.82e+04|4/8|0|2|0|2/8|

例如首批 user_nested/base 的 U1 在约1371 Hz：2个代表解的位置不合格，但转变存在，宽度与速度都在容差内；不能写成‘相位结构丢失’。完整偏差见 EVIDENCE.json，其他电路应按自己的数据解释。

## 改L、C之后，不能只说峰宽变了

下面把每个目标的实际峰谷列出来，读表时先看同一电路的三个条件。LC倍率是参数操作，实际宽度才是曲线测出的结果。事件编号只在各自目标内有效，跨条件不能仅凭编号相同就断定是同一个物理结构。

|目标|事件|频率 / Hz|宽度 / Hz|高度 / dBΩ|突出程度 / dB|
|---|---|---:|---:|---:|---:|
|control_rlc/base|P1|1e+04|332.5|69.54|69.54|
|control_rlc/lc_reciprocal_0p25|P1|1e+04|83.14|69.54|81.58|
|control_rlc/lc_reciprocal_8p0|P1|1e+04|2660|69.54|51.48|
|dual_nested/base|P1|1.482e+04|1463|53.63|35.61|
|dual_nested/lc_reciprocal_0p25|P1|1.856e+04|4142|39.87|21.88|
|dual_nested/lc_reciprocal_8p0|P1|1.098e+04|395.4|67.59|48.33|
|loaded_tank/base|P1|8094|198.9|66.24|49.25|
|loaded_tank/base|P2|2.718e+04|1464|65.61|46.15|
|loaded_tank/base|V1|1.231e+04|971.8|19.47|46.15|
|loaded_tank/lc_reciprocal_0p25|P1|9371|180.2|72.39|55.4|
|loaded_tank/lc_reciprocal_0p25|P2|2.348e+04|385.3|66.35|49.28|
|loaded_tank/lc_reciprocal_0p25|V1|1.766e+04|786.3|17.06|49.28|
|loaded_tank/lc_reciprocal_8p0|P1|4508|496.6|41.3|13.03|
|loaded_tank/lc_reciprocal_8p0|P2|4.815e+04|1.146e+04|64.19|38.46|
|loaded_tank/lc_reciprocal_8p0|V1|5270|1114|28.27|13.03|
|mixed_resonators/base|P1|1.275e+04|699.5|47.33|27.3|
|mixed_resonators/base|P2|4.941e+04|6764|46.49|6.179|
|mixed_resonators/base|V1|9986|692.7|16.89|46.39|
|mixed_resonators/base|V2|1.71e+04|2049|20.03|27.3|
|mixed_resonators/base|V3|5.268e+04|7574|40.31|6.179|
|mixed_resonators/lc_reciprocal_0p25|P1|1.07e+04|550.1|28.9|8.843|
|mixed_resonators/lc_reciprocal_0p25|P2|6.888e+04|无法定义|47.43|1.2|
|mixed_resonators/lc_reciprocal_0p25|V1|9956|706.8|16.74|46.26|
|mixed_resonators/lc_reciprocal_0p25|V2|1.21e+04|2095|20.05|8.843|
|mixed_resonators/lc_reciprocal_0p25|V3|7.637e+04|无法定义|46.23|1.2|
|mixed_resonators/lc_reciprocal_8p0|P1|1.445e+04|1024|51.54|30.67|
|mixed_resonators/lc_reciprocal_8p0|P2|4.139e+04|714.3|46.96|11.69|
|mixed_resonators/lc_reciprocal_8p0|V1|9991|693.8|16.89|46.45|
|mixed_resonators/lc_reciprocal_8p0|V2|2.113e+04|2378|20.87|30.67|
|mixed_resonators/lc_reciprocal_8p0|V3|4.2e+04|1105|35.27|11.69|
|rlc_ladder/base|P1|8095|373.1|68.11|48.02|
|rlc_ladder/base|P2|2.724e+04|585.7|41.81|16.39|
|rlc_ladder/base|V1|2.639e+04|999.2|25.42|16.39|
|rlc_ladder/lc_reciprocal_0p25|P1|9376|552.9|64.94|44.88|
|rlc_ladder/lc_reciprocal_8p0|P1|4564|105.9|79.19|58.34|
|rlc_ladder/lc_reciprocal_8p0|P2|4.821e+04|179.5|46.11|29.08|
|rlc_ladder/lc_reciprocal_8p0|V1|4.732e+04|682.3|17.03|29.08|
|three_level/base|P1|7325|322.8|68.96|48.83|
|three_level/base|P2|2.318e+04|458.4|47.38|23.61|
|three_level/base|V1|2.205e+04|772.9|23.78|23.61|
|three_level/lc_reciprocal_0p25|P1|7883|356.6|68.39|48.29|
|three_level/lc_reciprocal_0p25|P2|2.559e+04|439|44.8|20.54|
|three_level/lc_reciprocal_0p25|V1|2.471e+04|754.6|24.26|20.54|
|three_level/lc_reciprocal_8p0|P1|4791|287.5|68.67|48.18|
|three_level/lc_reciprocal_8p0|P2|1.936e+04|798.6|49.54|24.84|
|three_level/lc_reciprocal_8p0|V1|1.706e+04|1383|24.7|24.84|
|user_nested/base|P1|1.789e+04|2330|52.8|31.19|
|user_nested/lc_reciprocal_0p25|P1|1.591e+04|7484|32.93|11.34|
|user_nested/lc_reciprocal_8p0|P1|3.194e+04|1125|77.03|53.84|

真实端口曲线的峰位置、高度、宽度和邻近结构可能一起改变。因此这轮能发现条件相关的困难线索，尚不能证明‘仅仅变窄导致失败’。逐目标对应记录在 base_event_correspondence.json，也只是按频率做的匹配。

## 两项整体分数接近，结构是否仍可能不同

|目标|检查seed数|有‘分数接近但形状明显不同’的seed数|
|---|---:|---:|
|control_rlc/base|8|0|
|control_rlc/lc_reciprocal_0p25|8|0|
|control_rlc/lc_reciprocal_8p0|8|0|
|dual_nested/base|8|0|
|dual_nested/lc_reciprocal_0p25|8|0|
|dual_nested/lc_reciprocal_8p0|8|0|
|loaded_tank/base|8|0|
|loaded_tank/lc_reciprocal_0p25|8|0|
|loaded_tank/lc_reciprocal_8p0|8|0|
|mixed_resonators/base|8|2|
|mixed_resonators/lc_reciprocal_0p25|8|6|
|mixed_resonators/lc_reciprocal_8p0|8|1|
|rlc_ladder/base|8|0|
|rlc_ladder/lc_reciprocal_0p25|8|0|
|rlc_ladder/lc_reciprocal_8p0|8|2|
|three_level/base|8|0|
|three_level/lc_reciprocal_0p25|8|1|
|three_level/lc_reciprocal_8p0|8|0|
|user_nested/base|8|0|
|user_nested/lc_reciprocal_0p25|8|0|
|user_nested/lc_reciprocal_8p0|8|0|

两项分数都按各自尺度比较，主阈值为5%；为避免接近零时比例爆炸，分母下限为0.01 dB²与(0.5°)²换成rad²。所以上表不能一概解释为‘原始相对误差都小于5%’。形状明显不同指事件漏失状态不同、同一目标事件拟合高度相差≥6 dB、或宽度相差≥2倍。当前这张表只查幅值峰谷，不能拿它证明相位结构等价。

## 计算成本也需要看

所有seed都是200个候选、300代，共60000次参数候选评价。但复杂曲线可能需要更多频点，因此同样代数不等于同样计算成本。下面的查询数每次同时计算目标与候选；单seed耗时来自8任务并行时的运行记录，不能简单相加成整批墙钟时间。

|电路|已核验seed|单seed平均耗时 / 秒|单seed平均频率查询数|
|---|---:|---:|---:|
|单并联 RLC 对照|24|83.7|55813544|
|双嵌套支路|24|160.0|110850758|
|带负载的谐振网络|24|150.4|97189157|
|混合谐振支路|24|222.6|112136618|
|两级 RLC 梯形网络|24|180.7|101653627|
|三级 RLC 网络|24|239.0|125723022|
|所提出的嵌套电路|24|141.2|85981157|

## 接下来应当做什么判断

1. 哪类结构反复难恢复：以逐seed漏失与形状偏差为依据，先排除只由代表选择造成的例子。
2. 哪个条件使它更难：对照同一电路三个目标的实际结构变化；这一轮仍不足以唯一分离宽度、峰高和相邻结构的作用。
3. 针对性指标能否改善：本轮没有加入辅助指标，暂时没有答案。指标候选仍围绕 f、Re Z、Im Z 的三维曲线；不能提前写成有效。
4. 改善是否能推广、代价是什么：本轮扩大电路范围，但尚未验证指标，也没有新实测数据；8个seed用于探索，不宣称普遍规律。

**下一步先从持续失败的局部结构中选一个最有证据的候选。** 如果加预算或换选解已能恢复，就优先解释这些因素；若仍需指标，再用相同seed、初始化和明确成本做基础目标与辅助指标的对照。

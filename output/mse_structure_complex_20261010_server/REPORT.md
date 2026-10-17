# 嵌套电路第二步实验：进展与结果

2026年10月10日启动。当前服务器阶段：`complete`；已完成 168/168 个 seed 任务。

这次还是让程序同时减小幅值和相位的整体误差，观察哪些峰、谷和相位转变仍恢复不好。没有加入辅助指标，没有更新网站。

先运行你提出的电路与单并联 RLC 对照，再运行另外五类嵌套电路。每类 3 个目标、每目标 8 个 seed、每 seed 300 代、种群 200；全部预算为 10,080,000 次参数候选评价。

生成目标时，指定 L 乘 0.25/1/8、C 除以同一倍率。拟合时允许所有元件参数改变。嵌套结构的实际峰宽、位置与相邻结构可能一起改变，倍率本身不当作峰宽。

first 批次有 1 条非代表候选在原检查网格下结构判定不稳定；加密至32、64级后结构判定一致，分数按原容差复算通过。原始 AUDIT.json 与补充复核记录均保留。下面的代表曲线统计未受影响，种群比较仍保守排除原来不稳定的候选。

## 先看整体分数与结构是否一起变好

MAG 是 dB 幅值平均平方误差，PHA 是弧度相位平均平方误差，两项越小越好。每个 seed 按固定规则选一条代表曲线，再对这些曲线的分数取平均；曲线本身不平均。

表中‘幅值结构全部合格’只评价能定义 3 dB 宽度的目标峰谷：频率位置差≤0.02个十倍频程、高度差≤3 dB、宽度比在0.5～2之间。相位转变另外检查。没有可算宽度的目标事件时标为不适用。

|电路与条件|有效seed/计划|MAG均值 / dB²|PHA均值 / rad²|幅值结构全部合格的比例|
|---|---:|---:|---:|---:|
|单并联 RLC 对照 / base|8/8|6.1371e-06|2.5675e-08|1|
|单并联 RLC 对照 / lc_reciprocal_0p25|8/8|0.00019418|1.3239e-07|1|
|单并联 RLC 对照 / lc_reciprocal_8p0|8/8|1.4297e-06|3.6883e-10|1|
|你提出的嵌套电路 / base|8/8|0.0046171|0.00022345|1|
|你提出的嵌套电路 / lc_reciprocal_0p25|8/8|2.1091e-05|1.7596e-07|1|
|你提出的嵌套电路 / lc_reciprocal_8p0|8/8|0.00079971|1.8107e-05|1|
|双嵌套支路 / base|8/8|0.0095364|0.00010844|1|
|双嵌套支路 / lc_reciprocal_0p25|8/8|0.0090616|0.00010598|1|
|双嵌套支路 / lc_reciprocal_8p0|8/8|0.0052267|0.00011069|1|
|带负载的谐振网络 / base|8/8|0.11843|0.00046408|1|
|带负载的谐振网络 / lc_reciprocal_0p25|8/8|0.69867|0.005387|0.75|
|带负载的谐振网络 / lc_reciprocal_8p0|8/8|0.68555|0.0086106|0.75|
|混合谐振支路 / base|8/8|0.2884|0.0032054|0.5|
|混合谐振支路 / lc_reciprocal_0p25|8/8|0.30865|0.0037619|0.5|
|混合谐振支路 / lc_reciprocal_8p0|8/8|0.11775|0.00091627|0.75|
|两级 RLC 梯形网络 / base|8/8|0.32547|0.0041539|0.125|
|两级 RLC 梯形网络 / lc_reciprocal_0p25|8/8|0.021746|0.00025835|1|
|两级 RLC 梯形网络 / lc_reciprocal_8p0|8/8|0.56553|0.0071483|0.125|
|三级 RLC 网络 / base|8/8|0.40984|0.0046154|0.5|
|三级 RLC 网络 / lc_reciprocal_0p25|8/8|0.25233|0.0027936|0.875|
|三级 RLC 网络 / lc_reciprocal_8p0|8/8|0.14214|0.0014372|0.875|

‘有效 seed’同时通过分数复算和结构判断稳定性检查。最终只按 MAG+25.12×PHA 选代表的表列在这里；另一种平衡选解规则同样保存在 summary.csv，避免把选解偏好误认为优化器没有找到结构。

## 具体是哪些峰谷没有恢复

下面每个目标挑出合格率最低的两个可计算宽度的事件，作为后续查曲线的入口。这是从结果中筛出的线索，不代替完整逐 seed 数据，也不是预先证明某种结构最难。峰谷编号只在该目标内使用。

|目标|事件|目标频率 / Hz|目标实际3 dB宽度 / Hz|漏失次数 / 有效seed|合格次数 / 有效seed|
|---|---|---:|---:|---:|---:|
|control_rlc/base|P1|10000|332.54|0/8|8/8|
|control_rlc/lc_reciprocal_0p25|P1|10000|83.136|0/8|8/8|
|control_rlc/lc_reciprocal_8p0|P1|10000|2660.3|0/8|8/8|
|dual_nested/base|P1|14825|1463.4|0/8|8/8|
|dual_nested/lc_reciprocal_0p25|P1|18565|4142.3|0/8|8/8|
|dual_nested/lc_reciprocal_8p0|P1|10984|395.39|0/8|8/8|
|loaded_tank/base|P1|8094.4|198.94|0/8|8/8|
|loaded_tank/base|P2|27181|1463.8|0/8|8/8|
|loaded_tank/lc_reciprocal_0p25|P2|23481|385.32|0/8|6/8|
|loaded_tank/lc_reciprocal_0p25|V1|17656|786.29|0/8|7/8|
|loaded_tank/lc_reciprocal_8p0|P1|4508.1|496.64|2/8|6/8|
|loaded_tank/lc_reciprocal_8p0|V1|5270.5|1113.7|2/8|6/8|
|mixed_resonators/base|P2|49405|6764.3|2/8|4/8|
|mixed_resonators/base|V3|52684|7573.6|2/8|4/8|
|mixed_resonators/lc_reciprocal_0p25|P1|10705|550.14|1/8|4/8|
|mixed_resonators/lc_reciprocal_0p25|V1|9956.4|706.83|1/8|4/8|
|mixed_resonators/lc_reciprocal_8p0|P2|41387|714.3|2/8|6/8|
|mixed_resonators/lc_reciprocal_8p0|V3|42005|1105.2|2/8|6/8|
|rlc_ladder/base|P2|27244|585.67|0/8|1/8|
|rlc_ladder/base|V1|26391|999.17|0/8|6/8|
|rlc_ladder/lc_reciprocal_0p25|P1|9376.1|552.85|0/8|8/8|
|rlc_ladder/lc_reciprocal_8p0|P2|48208|179.53|4/8|1/8|
|rlc_ladder/lc_reciprocal_8p0|V1|47323|682.29|4/8|1/8|
|three_level/base|P2|23179|458.36|0/8|4/8|
|three_level/base|V1|22047|772.9|0/8|5/8|
|three_level/lc_reciprocal_0p25|P2|25590|439.04|0/8|7/8|
|three_level/lc_reciprocal_0p25|V1|24714|754.64|0/8|7/8|
|three_level/lc_reciprocal_8p0|P1|4790.8|287.48|0/8|7/8|
|three_level/lc_reciprocal_8p0|P2|19359|798.57|0/8|7/8|
|user_nested/base|P1|17893|2330|0/8|8/8|
|user_nested/lc_reciprocal_0p25|P1|15913|7483.5|0/8|8/8|
|user_nested/lc_reciprocal_8p0|P1|31936|1125.1|0/8|8/8|

这是目标实际峰宽，不是给 L、C 设定的倍率；每一行都来自对应目标曲线的两侧 3 dB 交点。无法定义宽度的浅峰谷仍在 events.csv 检查匹配和位置，没有填成0。

## 是代表解没恢复，还是保留的整个种群都没恢复

|目标|末代检查的seed|种群中存在幅值结构全合格的seed|两项分数差≤5%却有明显形状差异的seed|
|---|---:|---:|---:|
|control_rlc/base|8|8/8|0/8|
|control_rlc/lc_reciprocal_0p25|8|8/8|0/8|
|control_rlc/lc_reciprocal_8p0|8|8/8|0/8|
|dual_nested/base|8|8/8|0/8|
|dual_nested/lc_reciprocal_0p25|8|8/8|0/8|
|dual_nested/lc_reciprocal_8p0|8|8/8|0/8|
|loaded_tank/base|8|8/8|0/8|
|loaded_tank/lc_reciprocal_0p25|8|7/8|0/8|
|loaded_tank/lc_reciprocal_8p0|8|6/8|0/8|
|mixed_resonators/base|8|5/8|2/8|
|mixed_resonators/lc_reciprocal_0p25|8|5/8|6/8|
|mixed_resonators/lc_reciprocal_8p0|8|6/8|1/8|
|rlc_ladder/base|8|2/8|0/8|
|rlc_ladder/lc_reciprocal_0p25|8|8/8|0/8|
|rlc_ladder/lc_reciprocal_8p0|8|1/8|2/8|
|three_level/base|8|6/8|0/8|
|three_level/lc_reciprocal_0p25|8|7/8|1/8|
|three_level/lc_reciprocal_8p0|8|8/8|0/8|
|user_nested/base|8|8/8|0/8|
|user_nested/lc_reciprocal_0p25|8|8/8|0/8|
|user_nested/lc_reciprocal_8p0|8|8/8|0/8|

‘形状差异’指漏失状态不同、同一目标事件拟合高度差至少6 dB、或拟合宽度相差至少2倍。只检查末代保留的200条候选，不代表所有曾经产生或可能产生的曲线。统计单位仍是 seed。

## 中途有没有进步

|目标|代数|全局MAG均值|全局PHA均值|未合格幅值事件数均值|漏失相位转变数均值|
|---|---:|---:|---:|---:|---:|
|control_rlc/base|20|0.060263|0.00032917|0.125|0|
|control_rlc/base|100|0.004147|5.7782e-07|0|0|
|control_rlc/base|300|6.1371e-06|2.5675e-08|0|0|
|control_rlc/lc_reciprocal_0p25|20|0.10359|7.9869e-05|0.25|0|
|control_rlc/lc_reciprocal_0p25|100|0.01017|4.9366e-06|0|0|
|control_rlc/lc_reciprocal_0p25|300|0.00019418|1.3239e-07|0|0|
|control_rlc/lc_reciprocal_8p0|20|0.013411|3.643e-05|0|0|
|control_rlc/lc_reciprocal_8p0|100|0.00010189|1.1392e-06|0|0|
|control_rlc/lc_reciprocal_8p0|300|1.4297e-06|3.6883e-10|0|0|
|dual_nested/base|20|0.57506|0.0051137|0.25|0.375|
|dual_nested/base|100|0.04207|0.0005184|0|0.125|
|dual_nested/base|300|0.0095364|0.00010844|0|0|
|dual_nested/lc_reciprocal_0p25|20|0.60817|0.0052551|0.25|0.125|
|dual_nested/lc_reciprocal_0p25|100|0.024573|0.00027023|0|0|
|dual_nested/lc_reciprocal_0p25|300|0.0090616|0.00010598|0|0|
|dual_nested/lc_reciprocal_8p0|20|0.64696|0.0057627|0.375|0|
|dual_nested/lc_reciprocal_8p0|100|0.030278|0.00031644|0|0|
|dual_nested/lc_reciprocal_8p0|300|0.0052267|0.00011069|0|0|
|loaded_tank/base|20|5.1405|0.042001|2.125|0|
|loaded_tank/base|100|0.35417|0.0021791|0|0|
|loaded_tank/base|300|0.11843|0.00046408|0|0|
|loaded_tank/lc_reciprocal_0p25|20|7.0883|0.078688|2.625|0.125|
|loaded_tank/lc_reciprocal_0p25|100|1.0903|0.0097397|0.5|0|
|loaded_tank/lc_reciprocal_0p25|300|0.69867|0.005387|0.375|0|
|loaded_tank/lc_reciprocal_8p0|20|2.642|0.031816|2|1.5|
|loaded_tank/lc_reciprocal_8p0|100|0.76037|0.0094077|0.625|0.5|
|loaded_tank/lc_reciprocal_8p0|300|0.68555|0.0086106|0.5|0.5|
|mixed_resonators/base|20|4.193|0.04354|4.25|2|
|mixed_resonators/base|100|0.70593|0.0066631|2.125|0.875|
|mixed_resonators/base|300|0.2884|0.0032054|1|0.5|
|mixed_resonators/lc_reciprocal_0p25|20|1.0516|0.012103|2.75|2.875|
|mixed_resonators/lc_reciprocal_0p25|100|0.43277|0.0052116|1.5|1.375|
|mixed_resonators/lc_reciprocal_0p25|300|0.30865|0.0037619|1.375|0.875|
|mixed_resonators/lc_reciprocal_8p0|20|3.7539|0.033045|4.125|1.75|
|mixed_resonators/lc_reciprocal_8p0|100|0.39512|0.0024746|1.625|1|
|mixed_resonators/lc_reciprocal_8p0|300|0.11775|0.00091627|0.5|0.5|
|rlc_ladder/base|20|0.9535|0.0097623|2.25|1.375|
|rlc_ladder/base|100|0.44534|0.0051862|1.625|0.125|
|rlc_ladder/base|300|0.32547|0.0041539|1.125|0|
|rlc_ladder/lc_reciprocal_0p25|20|0.57232|0.0035032|0.25|2|
|rlc_ladder/lc_reciprocal_0p25|100|0.037846|0.00041836|0|1.25|
|rlc_ladder/lc_reciprocal_0p25|300|0.021746|0.00025835|0|0.75|
|rlc_ladder/lc_reciprocal_8p0|20|1.0651|0.011808|2.5|2.875|
|rlc_ladder/lc_reciprocal_8p0|100|0.62469|0.0074474|1.75|2|
|rlc_ladder/lc_reciprocal_8p0|300|0.56553|0.0071483|1.75|1.5|
|three_level/base|20|2.0231|0.022563|2.125|1.375|
|three_level/base|100|0.66768|0.0079348|1.25|0.875|
|three_level/base|300|0.40984|0.0046154|0.875|0.75|
|three_level/lc_reciprocal_0p25|20|1.6633|0.016791|2.375|2.375|
|three_level/lc_reciprocal_0p25|100|0.47088|0.0056632|1.25|1.875|
|three_level/lc_reciprocal_0p25|300|0.25233|0.0027936|0.25|1.25|
|three_level/lc_reciprocal_8p0|20|3.7839|0.039246|2.5|0.75|
|three_level/lc_reciprocal_8p0|100|0.42756|0.0046997|0.5|0|
|three_level/lc_reciprocal_8p0|300|0.14214|0.0014372|0.25|0|
|user_nested/base|20|0.41249|0.0046641|0.125|0|
|user_nested/base|100|0.018034|0.00064861|0|0|
|user_nested/base|300|0.0046171|0.00022345|0|0|
|user_nested/lc_reciprocal_0p25|20|0.39324|0.0069807|0.25|0.125|
|user_nested/lc_reciprocal_0p25|100|0.015463|0.000181|0|0|
|user_nested/lc_reciprocal_0p25|300|2.1091e-05|1.7596e-07|0|0|
|user_nested/lc_reciprocal_8p0|20|0.12567|0.0011379|0|0.125|
|user_nested/lc_reciprocal_8p0|100|0.0017949|3.8696e-05|0|0|
|user_nested/lc_reciprocal_8p0|300|0.00079971|1.8107e-05|0|0|

这张表平均的是数值统计，不是把不同 seed 的阻抗曲线拼成一条。局部误差对总分的贡献还在 local.csv；没有明显事件的区域可看 bands.csv。

## 只看少量实际曲线

以下图各叠加固定三个 seed 的代表解与目标，选择基准目标300代展示；其他阶段和条件保留在文件夹里。

### 你提出的嵌套电路

![你提出的嵌套电路](../../archive/LOCAL_ONLY.md#ref-5800071a3f81)

### 两级 RLC 梯形网络

![两级 RLC 梯形网络](../../archive/LOCAL_ONLY.md#ref-209ca8408990)

### 三级 RLC 网络

![三级 RLC 网络](../../archive/LOCAL_ONLY.md#ref-0f4c71579704)

## 这一轮能回答到哪里

先看困难是否在不同电路中反复出现，再检查目标实际位置、宽度、突出程度和相邻结构怎样变化。单靠三档倍率不能把原因唯一归到峰宽；不同拓扑元件数不同，也不能直接归到‘复杂度’。8个seed是探索规模，不据此宣布普遍规律。

理论模型的真参数仍在搜索范围内，存在零误差解。当前结果描述规定算法和300代预算下的恢复情况；不能写成‘MSE理论上永远无法优化这些结构’。之后才根据反复出现的缺陷，选择一个围绕三维曲线的辅助指标。

始终保留四个问题：哪类结构反复难恢复；哪个条件使它更难；针对性指标能否改善；改善是否能推广、代价是什么。

先读 [研究结果：通俗解读](研究结果_通俗解读.md)；完整电路、参数和统计见 [FINDINGS.md](FINDINGS.md)；运行问题与影响见 [ISSUES.md](ISSUES.md)。

# 联合面积实验：自动统计初稿

已回收并逐文件核对 192/192 个任务；数值及结构复核接受 191 个。

这份表不是最终结论。尚需人工检查局部连续误差、全种群机会、三维轨迹和反例。失败样本不能悄悄排除；通过数值检查也不代表结构拟合成功。

|条件|方法|已审计|通过复核|幅值MSE中位数|相位MSE中位数|关注结构完整数中位数|CPU核时|
|---|---|---:|---:|---:|---:|---:|---:|
|mixed_resonators/base|MAG_PHA|8|7|0.29394|0.0020636|2|0.707|
|mixed_resonators/base|MAG_PHA_ISLANDS|8|8|0.51252|0.0032947|2|0.633|
|mixed_resonators/base|G0|8|8|0.26009|0.0031579|1|1.651|
|mixed_resonators/base|J0|8|8|0.28788|0.0030723|0|2.977|
|mixed_resonators/base|D|8|8|0.29636|0.0021699|2|23.349|
|mixed_resonators/base|H|8|8|0.34629|0.0038815|0|23.391|
|mixed_resonators/base|A|8|8|0.41479|0.002894|0.5|4.304|
|mixed_resonators/base|ARX|8|8|0.32702|0.0033954|1.5|4.320|
|mixed_resonators/lc_reciprocal_8p0|MAG_PHA|8|8|0.22511|0.0019634|0|0.721|
|mixed_resonators/lc_reciprocal_8p0|MAG_PHA_ISLANDS|8|8|0.18969|0.0018713|1|0.723|
|mixed_resonators/lc_reciprocal_8p0|G0|8|8|0.15716|0.0015194|0.5|1.687|
|mixed_resonators/lc_reciprocal_8p0|J0|8|8|0.15535|0.0017428|0|3.004|
|mixed_resonators/lc_reciprocal_8p0|D|8|8|0.16139|0.0020249|0|24.461|
|mixed_resonators/lc_reciprocal_8p0|H|8|8|0.12668|0.0012828|0|24.383|
|mixed_resonators/lc_reciprocal_8p0|A|8|8|0.098556|0.00088276|1|4.346|
|mixed_resonators/lc_reciprocal_8p0|ARX|8|8|0.15587|0.00077117|1.5|4.325|
|case1|MAG_PHA|8|8|14.491|0.11245|2.5|0.136|
|case1|MAG_PHA_ISLANDS|8|8|18.065|0.11473|2.5|0.129|
|case1|G0|8|8|20.193|0.15504|3|1.435|
|case1|J0|8|8|34.932|0.28397|0|7.232|
|case1|D|8|8|17.533|0.16599|3|66.113|
|case1|H|8|8|15.571|0.16217|2.5|66.004|
|case1|A|8|8|28.263|0.28901|1|12.522|
|case1|ARX|8|8|22.713|0.18337|1|12.195|

方案见 experiments/complex_area_20261014/PROTOCOL.md；原始逐seed、事件连续量及配对差在 analysis/。所有条件都参与过指标开发，8个seed不足以宣称独立推广。

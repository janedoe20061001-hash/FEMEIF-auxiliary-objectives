# Case2：拟合参考解周围的参数扰动与指标敏感性

完成 1765 条谱。参考点固定为100个正式baseline选解中WSMSE最低者（seed 7），它是拟合参考解，不是真实元件参数。
扰动包含15个单参数正反方向、R/L/C/全参数整体缩放正反方向，以及R/L/C/全参数各64个固定随机方向；档位为0.1%、0.3%、1%、3%、10%、30%。
所有误差都相对原始201点实测目标计算；参数不裁剪到优化边界。该实验描述局部响应，不能证明辅助目标加入进化后有效。

## 1%全参数随机方向的秩相关

|辅助指标|幅值MSE|相位MSE|
|---|---:|---:|
|pdf_turning_event_global_count_5|nan|nan|
|pdf_turning_event_coverage_5_of_6|nan|nan|
|pdf_reactance_crossing_direction_consistency|nan|nan|
|pdf_reactance_crossing_global_count_10|nan|nan|
|pdf_turning_event_global_count_4|nan|nan|
|pdf_paper_ptc|0.042|0.038|
|pdf_reactance_crossing_global_count_8|nan|nan|
|pdf_characteristic_frequency_shift_error_decades|0.708|0.477|
|pdf_transition_speed_relative_error|0.855|0.245|
|pdf_constrained_dtw_window_4|0.554|0.477|
|pdf_soft_dtw_divergence_gamma_0.10|0.352|0.043|
|pdf_hausdorff95_error|-0.340|-0.040|
|pdf_sobolev_h1_error|0.640|0.643|
|pdf_translation_aligned_spatial_shape_error|0.545|0.488|
|pdf_turning_event_weak_location|-0.063|-0.053|
|pdf_turning_event_existence_count_guard|nan|nan|
|pdf_normal_shape_error_relative|0.380|0.080|
|pdf_turning_event_coverage_4_of_6|nan|nan|
|paper_SS|0.033|0.020|
|paper_PTC|0.001|0.123|
|paper_DTW|0.389|-0.532|
|paper_LFP|nan|nan|
|paper_LMEP|-0.016|-0.665|

完整逐样本结果见 all_runs.csv；summary.csv、monotonicity.csv 和 correlations_1pct_all.csv 分别保存分组分位数、轨迹单调性及固定档位相关。

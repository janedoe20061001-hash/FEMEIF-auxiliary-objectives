# 可复算数据索引

全部CSV使用UTF-8 BOM。频率单位Hz；x与shift_decades为log10频率；κ为dφ/dln f，φ在该导数中使用弧度；real_db为20log10(R_e/R_e目标)。相位极值角度及偏差为度。

- [通俗报告](REPORT.md)
- [电路、目标参数与原普通电路图](TARGETS.md)
- [原诊断运行前约定](../../experiments/physical_state_diagnosis_20261011/PROTOCOL.md)
- [第一轮完成后的补充约定](../../experiments/physical_state_diagnosis_20261011/SUPPLEMENT_PROTOCOL.md)
- [252条代表的过零诊断](REPRESENTATIVES.csv)
- [1092组目标过零对应记录](CROSSINGS.csv)
- [三轮分组汇总](GROUPS.csv)
- [105条原参数干预](PROBES.csv)
- [原干预的逐事件诊断](PROBE_CROSSINGS.csv)
- [35条整体尺度控制](SCALE_CONTROLS.csv)
- [14项独立非整倍数尺度复核](NONBINARY_SCALE_CHECKS.json)
- [252条代表的相位峰谷诊断](PHASE_REPRESENTATIVES.csv)
- [逐相位峰谷对应记录](PHASE_EXTREMA.csv)
- [最新轮幅值通过、相位未通过的37条](AMP_PASS_PHASE_FAIL.csv)
- [1086个相位极值的独立节点法核验记录](INDEPENDENT_PHASE_CHECKS.json)
- [输入哈希与原始种群核验](INPUT_MANIFEST.json)
- [补充状态、来源与检查结果](SUPPLEMENT_STATUS.json)
- [最终数值和输入核验](VERIFICATION.json)
- [图的示例选择规则](PLOT_SELECTION.json)
- [故障和研究边界](ISSUES.md)

details/{目标}.json保存：原始目标电路、参数、过零参考、每条代表的参数、4/8级结果、独立节点法检查、缺失与额外事件，以及各干预的完整记录。phase_details/{目标}.json保存≥5°相位极值的双级提取、对应及额外事件。文件中的index、event_index只在对应目标内有效。

“matched”是在固定目标邻域内找到同类事件；“missing”不是在全部频带不存在。过零方向、相位峰谷类型与相位正负都有明确匹配要求。额外事件单列。

所有新诊断标记均不替代原幅值/相位事件成功标记。几种检查门槛不同，不能以报警条数直接比较准确率。方法间共享seed与初始化，不把252条当作独立同分布样本。

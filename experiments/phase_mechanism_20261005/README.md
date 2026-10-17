# 相位改善能力诊断（2026-10-05）

[中文报告](../../output/相位改善能力探索_Case1机制与指标设计_20261005.md)记录公式盲区、实际误差分解、与其他组合的对比、最终种群可达性、相关性及面向指标设计的假设。

本轮在 `/root/autodl-tmp/phase_mechanism_20261005` 完成，只读使用前轮 1,200 次优化的完整最终种群和初始种群，以及旧 Case1 800 次优化的选中参数。**新增优化为 0。** 原有效组合和正式成绩保持不变。

发现：速度损失下降并不稳定伴随 Case1 相位下降；本批相位净恶化主要来自两组都符号正确的频点上的角度偏差。固定种群提高相位选解权重仍未消除差距。带方向幅相割线可作为有限候选继续研究，尚未证明优化增益；事件位置敏感性仅是另一项结构线索。

## 文件与复现

- `ANALYSIS_PLAN.md`：首轮计算前确定的诊断问题。
- `FOLLOWUP_NOTE.md`：看到首轮结果后追加检查的缘由。
- `analyze_phase.py --source ../crosscase_consolidation_20261005/case1 --out case1`：全种群、共同初始种群及选解诊断；Case2 同理。
- `summarize_phase.py`：在两个 Case 诊断 CSV 完成后生成 SUMMARY.json。
- `followup_phase.py --source ../crosscase_consolidation_20261005/case1 --out case1`：四状态误差分解与带方向割线检查；Case2 同理。
- `historical_phase.py`：从本目录的 historical_case1_selected.csv 重建旧八组结果；输入由原 case1_dual_validation/server_run/selected_individuals.csv 复制。
- `server_run/results/`、`server_run/followup/`、`server_run/historical/`：三批服务器结果的解压副本；同名 `.tar.gz` 下载包保留。
- `scripts/report_phase_mechanism_20261005.py`（项目根目录下）：校验下载包哈希，生成报告、四张图和报告完整性记录。

分析脚本沿用前轮冻结环境，运行时需原始实验目录及其 case1/case2 代码、完整种群。历史曲线重建重新核对了幅值/相位 MSE。新诊断量不含新优化成绩。

解释边界：事后相关性、偏相关、误差贡献和种群可达性不能单独证明搜索因果；错误符号子集与幅值预算可行子集均明确报告条件与样本数。候选的定义审查通过不等于获得相位改善能力。

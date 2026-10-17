# 加密采样受控实验

承接2026-10-07曲线结构诊断，只验证采样可靠性与有限预算优化效果。新增真实目标加密和旧点插值加密两个条件，保留WSMSE、原Ω距离、幅相双MSE。旧结果不重选，不发布网站。

方案在 `PROTOCOL.md`；10个电路、280个配对seed块，共1680次新运行、74,880,000次评价，旧粗采样复用840次。Case2未加入，因为没有测量频点之间的目标真值。

## 输入、原始结果和来源

- `frozen_scalar/`：原单目标实验的网络、边界、目标与算法源码。
- `grids/*.npz`：冻结加密频点、单元积分权重、解析真值目标、旧点Bode插值目标、原频点与单元总权重。
- `CONFIG.json`：固定电路、seed、种群、300代、α及预算。
- `PREPARATION.json`：10个目标网格独立检查误差、点数、所有优化输入/源码SHA256。
- `server_run/retrieved/`：新运行的初始/最终种群、原点/训练指标、选解、前沿、逐代历史和运行环境。
- `server_run/results.tar.gz` 与 `ARCHIVES.json`：完整下载归档、服务器SHA256和本地核验。

新频点来自目标解析信息，不使用拟合结果。每个原log f单元总权重固定1/N，单元内用梯形积分，避免加密窄峰时按新增点数重复加权。插值组和真值组的频点/权重完全一样。

真值参数只用于生成合成目标、准备采样频点和离线诊断。优化器仍在原边界内用原随机初始种群搜索；没有真值初始化、缩边界或按参数误差选解。此实验属于可取得目标细节的理想采样验证，不是测量仪器自动寻峰的验证。

## 本地复算

在项目根目录PowerShell：

```powershell
$env:PYTHONPATH='D:/develop/FEMEIF/tmp/curve_refinement_deps'
$env:PYTHONIOENCODING='utf-8'
$samplingPython='C:/Users/Admin（无密码）/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $samplingPython experiments/sampling_refinement_20261007/analyze.py --runs experiments/sampling_refinement_20261007/server_run/retrieved/runs --output output/sampling_refinement_20261007 --workers 8
& $samplingPython experiments/sampling_refinement_20261007/report.py
```

此命令只重算已下载的结果，不再次优化。完整报告为 [REPORT.md](../../output/sampling_refinement_20261007/REPORT.md)，完整附表为 [DETAILS.md](../../output/sampling_refinement_20261007/DETAILS.md)。

分析会核对249,600个末代个体、全部1680次新运行，与旧840次运行的初始化和选解；同时验证训练目标、原点评分、非支配前沿、末代选解、逐代历史、预算、参数边界及来源。独立电路公式采用Case1有理RLC表达式、串并联积/和表达式与桥式节点方程。

依赖原 `output/optimized_curve_structure_20261007/selected/` 的旧选解、对应原始服务器结果、`experiments/optimized_curve_structure_20261007/{diagnostics,refined_grid}.py` 及 `experiments/seed_replication_20261007/analyze.py`。分析源码SHA256保存在报告目录 `ANALYSIS_SOURCES.json`。

## 统计数据

`selected/*.npz` 的主要参数/曲线维度为seed×采样条件×优化目标×参数/频点，采样顺序 original/dense_interpolated/dense_true，目标顺序 wsmse/mean_distance/dual_mse。`original_metrics`始终是原频点等权四指标，`training_metrics`是各自训练条件四指标，二者不要混用。`balanced_log_parameters`为同一双目标运行的均衡前沿选解，作为敏感性分析。

`selected_metrics.csv`原点和训练误差；`original_group_means.csv`原点均值；`per_seed_diagnostics.csv`共同解析网格诊断；`band_diagnostics.csv`八频带；`grid_refinement.csv`网格稳定性；`structure_means.csv`解析结构均值；`paired_comparisons.csv`逐电路差值；`aggregate_comparisons.csv`电路等权汇总；`run_timings.csv`每运行耗时。

主要比较是dense_true减original、dense_true减dense_interpolated。Bootstrap10,000次，按同电路seed块抽样，不将频点当独立样本，不让Case1的100seed在跨电路汇总时占更高权重。匹配特征的频移/宽度遇到漏失时保留缺失值，不伪造零误差。均衡选解不算另一批独立实验。

解析诊断在每seed共同网格包含目标和12个主要/均衡选解的模态，1/2档检查不稳定时继续2/4、4/8档。最终仍不稳定的行完整保留、标记，不删seed或改变优化目标。

同种群/代数表示相同候选评价次数，新增频点会增加每次评价运算量；不是相同墙钟时间或相同浮点运算预算。结果只比较这些固定电路和300代搜索，不证明全局最优或普遍指标优劣。

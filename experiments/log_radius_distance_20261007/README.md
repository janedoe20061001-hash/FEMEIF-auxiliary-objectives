# 老师式对数半径距离实验

本地报告：[REPORT.md](../../archive/LOCAL_ONLY.md#ref-76d42f649312)。本目录用于复现本轮，未发布网站。

优化前方案在 `PROTOCOL.md`，输入及优化源码由 `PREPARATION.json` 的SHA256冻结。`frozen_scalar/` 为原单目标实验输入副本。本轮只新增老师式距离目标，Case1/Case2各100个seed、300代；原两组结果来自上一轮seed_replication实验，保留原选解。

服务器原始结果在 `server_run/retrieved/`，完整归档为 `server_run/results.tar.gz`，校验记录为 `server_run/ARCHIVES.json`。归档包含200个新运行的初始种群、最终种群、选中个体、每代最优历史、摘要、运行环境和冻结输入。服务器任务已结束。

独立分析依赖上一轮 `experiments/seed_replication_20261007/analyze.py` 的替代电路公式和原对照结果，以及 `experiments/optimized_curve_structure_20261007/{diagnostics,refined_grid}.py`。分析源码哈希记在报告目录的 `ANALYSIS_SOURCES.json`。数据不依赖网站可用性。

在项目根目录 PowerShell 运行（当前本机已有所需NumPy/SciPy/Matplotlib）：

```powershell
$env:PYTHONPATH='D:/develop/FEMEIF/tmp/curve_refinement_deps'
$env:PYTHONIOENCODING='utf-8'
$experimentPython='C:/Users/Admin（无密码）/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $experimentPython experiments/log_radius_distance_20261007/analyze.py --runs experiments/log_radius_distance_20261007/server_run/retrieved/runs --output output/log_radius_distance_20261007 --workers 8
& $experimentPython experiments/log_radius_distance_20261007/report.py
```

这两步只分析下载结果和生成两张典型图，不再次优化。`analyze.py` 会独立重算35,000个最终个体，核对200次新运行及两组原初始化/选解，计算100个配对seed的区间、600条曲线诊断和300条Case1解析加密诊断。

`selected/case1.npz`、`selected/case2.npz` 包含原频点、目标、seeds、组名、选解log10参数、原复阻抗曲线及五项指标；其维度为seed×组×参数/频点/指标。组序为WSMSE、原Ω距离、老师式对数半径距离；五项指标序为幅值MSE(dB²)、相位MSE(rad²)、WSMSE、原距离(Ω)、老师式距离。

`selected_metrics.csv`逐seed结果；`group_means.csv`均值；`SUMMARY.json`配对bootstrap及结构统计；`per_seed_structure.csv`标记原/加密网格；`band_diagnostics.csv`保留原频点频带误差；`grid_refinement.csv`两档网格变化；`special_regions.csv`保留近1Ω与对数幅值异号区域诊断。无失败seed替换、剔除或按新目标重选旧组个体。

目标1Ω相位退化的数学检查在 `server_run/retrieved/smoke/CHECKS.json`。两个Case、1Ω参考和固定300代的结果只支持本实验条件下的比较，不能证明目标的普适性或全局优化能力。

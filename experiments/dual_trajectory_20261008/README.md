# 复算入口

交付目录：`D:/develop/FEMEIF/output/dual_trajectory_20261008`。

1. `CONFIG.json`、`PROTOCOL.md` 固定目标、随机展示seed和选解规则。
2. `frozen_dual/` 是旧双目标算法与目标网络的副本；`SOURCES.json` 记录服务器执行输入哈希。
3. `run.py` 重放两个Case全部100个seed，在每20代保存环境选择后的种群与前沿。历史指标每代保存；没有保存每代全部子代。
4. 原始阶段数据在 `server_run/retrieved/runs/{case}/seed_{seed}/generation_{generation:03}.npz`，可换选解规则重新画图。
5. 在项目根目录运行 `python experiments/dual_trajectory_20261008/background.py`，核验全部阶段与旧末代种群，输出100seeds统计。
6. 运行 `python experiments/dual_trajectory_20261008/analyze.py`，独立核验随机展示seed，输出Markdown和静态科学图。
7. 运行 `findings.py`、`zoom.py`、`model_limits.py`（同目录），生成后台结论、局部放大图与Case2模型可表示性核查。
8. 两份报告均完成后运行 `finalize.py`，将末代三个seed的参数整理进Case页面，交叉引用采样结果，检查链接/数据行数，并生成最终交付哈希清单。

需要 numpy、scipy、matplotlib；服务器优化还需要冻结版本pymoo。本地可使用 `tmp/curve_refinement_deps` 作为PYTHONPATH。不运行发布命令。后台默认6进程，服务器本轮通过 `STRUCTURE_ANALYSIS_WORKERS=20` 加速离线诊断；不改变优化配置。

缓存位于输出目录 `background_chunks/`，仅用于这个固定结果归档的断点续算。如果更换原始归档、脚本或诊断阈值，应使用新输出目录或移走旧缓存；不可跨实验复用。

Case1的连续诊断与Case2的测点诊断分别标明，数值不等价。双目标代表解可能随代数换个体，某一指标不保证逐代单调。没有生成平均阻抗曲线。

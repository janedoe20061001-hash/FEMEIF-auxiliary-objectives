# Case I：中文 PDF 的 18 项单辅助目标

当前协议：2026-09-14 已获授权启动正式实验。实际运行状态见 server_run/progress.json 快照及服务器实时记录。

## 两个筛选阶段

本协议仍使用 NSGA-II。辅助组优化 (幅值 MSE，相位 MSE，一个辅助损失)，基线优化前两项。

|阶段|纯双主目标基线|单辅助组|
|---|---|---|
|交配时选择亲本|按两个目标的 rank / crowding|按三个目标的 rank / crowding|
|父代与子代合并后选择下一代|按两个目标排序和截断|按三个目标排序和截断|
|最终选取报告个体|固定 WSMSE 选一个|相同固定 WSMSE 选一个|
|补充前沿统计|只按两个主目标|只按两个主目标|

因此本代码**没有**把每一代的环境选择改成仅按主目标，也没有只用辅助目标参与交配。
config.json 中 selection_objectives=all、report_front=wsmse_selected 明确锁定这两个阶段。运行中记录双主目标前沿是只读统计，不将该前沿回填为下一代种群。

辅助目标的作用是给搜索增加保留候选解的理由。例如，一个个体当前两项 MSE 较差，但目标频段的转折结构较接近，辅助维度可能让它进入下一代并参与繁殖。随后使用相同交叉/变异算子，也会因亲本不同而产生不同后代。最终只按主目标评估，不会抹去之前的搜索差异。

这只说明机制，不保证被保留的个体确实更易调好。“增加多样性”应具体指辅助特征上的不同候选得到保留机会，不应被写成多样性越多越好，或所有几何特征都有严格电路机理。

如果未来每代的亲本选择和环境选择都只使用两个主目标，且算子、初始种群、随机数状态相同，辅助损失只计算而不被任何选择规则使用，那么它不会对算法产生作用。若仅交配使用辅助、环境选择仅用主目标，仍可能与基线不同，但属于另一套算法设计，本协议没有采用。

## 预定实验

- 两个主目标：幅值 dB MSE、相位 rad MSE；原有定义不变，不合成为 WSMSE。
- 18 个指标各自单独作为第三目标，加纯双主目标基线，共 **19 组**。
- 延续上一批设置：100 seeds（0—99）、种群 200、300 代，共 **1,900 次预定运行**。
- 第 1 代是初始种群评估，随后 299 轮繁殖；每次 60,000 次候选评估。
- 同 seed 的各组共享初始种群和算法 seed。log10 参数编码、范围、SBX / PM、rank+crowding 锦标赛均与上一批一致。
- 不运行这些 18 项的两两组合或三项组合。

主要结果：最终种群中选取 WSMSE 最小的一个个体，WSMSE = 幅值 MSE + 25.12 × 相位 MSE。选取该同一个体的两项 MSE 后，跨 100 个 seed 等权平均，与同 seed 基线比较。权重 25.12 在正式运行前固定，分数相同取最终种群最早行。仅选择最后一代，不选择历史最优。主目标前沿均值、完整目标前沿均值均作为补充保留，不混用。

WSMSE 仅用于只读统计，不改变两个主目标的优化；测试已验证改变最终选解权重不会改变进化得到的种群。最终选解的 LaTeX 代码：

~~~latex
i_s^*=\arg\min_i\left(L_{A,s,i}+25.12 L_{\phi,s,i}\right),\qquad
\overline{L}_A=\frac1{100}\sum_{s=1}^{100}L_{A,s,i_s^*},\qquad
\overline{L}_{\phi}=\frac1{100}\sum_{s=1}^{100}L_{\phi,s,i_s^*}.
~~~

前沿均值不是 Pareto 支配关系的替代，也不是“最优幅值与最优相位来自同一个解”的保证；这里只把它作为与现有统计一致的描述性筛选口径，不自动宣称显著改善。

## 指标清单

|PDF 编号|程序名|指标|
|---|---|---|
|I-01|turning_event_global_count_5|转折全局计数 5|
|I-02|turning_event_coverage_5_of_6|转折覆盖 5-of-6|
|I-03|reactance_crossing_direction_consistency|电抗过零方向|
|I-04|reactance_crossing_global_count_10|过零计数 10|
|I-05|turning_event_global_count_4|转折全局计数 4|
|I-06|paper_ptc|中文 PDF 的最近匹配、均值 PTC|
|I-07|reactance_crossing_global_count_8|过零计数 8|
|II-01|characteristic_frequency_shift_error_decades|局部特征频移|
|II-02|transition_speed_relative_error|转变速度相对误差|
|II-03|constrained_dtw_window_4|三维路径窗口 DTW|
|II-04|soft_dtw_divergence_gamma_0.10|Soft-DTW 散度|
|II-05|hausdorff95_error|95% Hausdorff|
|II-06|sobolev_h1_error|Sobolev H1|
|II-07|translation_aligned_spatial_shape_error|去平移空间形状|
|II-08|turning_event_weak_location|转折弱位置|
|II-09|turning_event_existence_count_guard|存在性与过量事件|
|II-10|normal_shape_error_relative|去局部频移后的相对形变|
|II-11|turning_event_coverage_4_of_6|转折覆盖 4-of-6|

## 公式对齐

公式来源为根目录《双主目标实验_第一第二层级指标公式详解.pdf》，实现按第 2—10 节及附录 B：

- 路径使用 (u,R/sR,X/sX)，所有尺度固定由目标计算。转角的 95% 分位门槛、候选侧 0.2 倍最弱目标事件阈值、稳定降序和最小索引间隔 4 均保留。
- 覆盖、过零方向和中文 PTC 的区域两端都闭合，保留共享边界可能重复覆盖的原定义。过零位置为索引对的 u 中点，保留 signbit 对负零的行为。
- 中文 PTC 使用紧邻峰谷高度差 0.2 dB、最近同类匹配和目标峰谷平均；与英文论文的全匹配累计 PTC 区分。
- 复导数按非均匀坐标 np.gradient(..., edge_order=2)，投影使用 Re(residual × conjugate(target_gradient))。
- DTW 使用三维欧氏距离、窗口 4；先最小累计距离，再除以选定路径步数；同分前驱依次优先对角、上方、左方。
- Soft-DTW 使用三维平方欧氏距离、窗口 10、gamma=0.10；包含目标自项、候选自项修正，除以 n、截断后开方。
- Hausdorff95 使用双向最近邻的线性插值 95% 分位数再取最大。Sobolev 分母保留频率坐标导数 1 的贡献。
- 去平移按两坐标中位数，不替换成均值或几何中位数。弱位置和过量事件系数均为 0.10。
- 只跳过数学上无关的缓存计算，没有以“轻量化”变体替换指标。
- 原公共事件初始化要求目标有转折事件；对应事件类不满足前提时明确报错，不擅自返回零。无目标峰谷时中文 PTC 返回零；无目标过零时方向匹配返回零，计数公式则拒绝零分母。

## 频率网格与可比性

默认沿用上一批论文五项实验的 **180 点对数均匀网格，100 Hz—1 MHz**，保持新旧对照的频率与主目标一致。支持通过 frequency.file 导入一列无表头、严格递增的 Hz 数值文件。

中文 PDF 原来的 401 点频率文件和原始实现文件不在当前工作区。因此这里是按公式重建并在当前协议下测试，不能称为 401 点原实验的精确复现，也不能直接把 PDF 中的旧结果当成本批基线。

180 点目标同样检测到 6 个转折、11 个过零、11 个幅值极值，但归一化实部尺度约 76.94 Ω，与 PDF 的约 728.68 Ω 不同。**事件数量相同不代表采样后的目标谱和尺度相同。** 特别是高 Q 窄峰可能被稀疏采样低估。实际尺度和事件位置会写入 prepared_plan/manifest.json。

不要在得到结果后单独为某个指标更换网格或常数。若更换频率文件，应作为新的统一协议并重跑对应基线和各对照，不能沿用旧网格下的误差值。

## 文件与用法

本目录独立于已完成的论文五项目录；旧实验的代码、指纹和结果不做覆盖。SOURCE.json 记录本次实验引擎的来源。

- objectives.py：18 项指标和电路响应。
- experiment.py：plan / smoke / run；不带合法子命令不会运行优化。
- summarize.py：先输出 wsmse_selected，再输出 primary_front / full_front 的配对统计。
- launch_batch.py：未来 Linux 服务器并行调度，默认 28 workers，实际应按服务器资源调整；启动参数见运行记录。
- test_experiment.py：解析例子、边界、递推枚举、小规模进化流程检查。
- requirements-lock.txt：与上一批相同的依赖锁定。

当前工作区可复用上一批的隔离 Python；在本目录执行：

~~~powershell
& ..\case1_dual_paper\.venv\Scripts\python.exe experiment.py plan --output prepared_plan
& ..\case1_dual_paper\.venv\Scripts\python.exe -m unittest -v test_experiment
~~~

服务器准备好后，可在本目录创建隔离环境并安装 requirements-lock.txt。以下为服务器执行说明：

~~~bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m unittest -v test_experiment
nohup .venv/bin/python launch_batch.py --workers 28 --output runs > batch.log 2>&1 < /dev/null &
.venv/bin/python summarize.py runs
~~~

最后一行汇总应在批次完成后执行；协调器完成时也会自动生成 paired_summary.csv。
单次结果包含完整种群、参数、全部优化目标、最终两种前沿索引与逐代统计。--resume 只跳过已完整完成且指纹相同的整次运行，不从半途代数恢复。

正式批次结束后 report_results.py 会审计选中个体、生成报告和 selected_individuals.csv，并打包 raw_results.tar.gz；完成后应将原始归档回传本地。

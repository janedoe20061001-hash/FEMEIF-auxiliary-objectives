# Case I：转折覆盖、峰谷一致性、局部频移的三指标集合

当前状态：代码准备和旧结果分析已完成；尚未启动任何新正式实验。

## 固定候选和下一批实验

- A / I-02：目标转折区域覆盖5-of-6。
- B / I-06：中文 PDF 峰谷一致性（最近匹配、均值）。
- C / II-01：局部特征频移误差。

三结合与整个集合验证放在同一批实验：baseline、A、B、C、A+B、A+C、B+C、A+B+C，共八组。全部使用新的 seed 200–299，每组100次、200个体、300代，共800次运行、4800万次目标评估。0–99用于候选筛选，100–199已有实验结果被查看，因此本次不将它们称为新的独立seed。

本次不额外重跑旧seed下的三结合；首次正式检验三结合就在上述新批次中完成。旧三个单指标和配对只用于已有数据的探索性分析，不混入新批次统计。

算法、指标定义、初始化、交叉、变异、亲本及环境选择与此前保持一致。辅助目标各占一个维度参与进化。每个seed从最后一代完整种群中，按固定权重25.12的WSMSE选一个个体，两项误差来自同一个体，然后跨seed求均值。所有组使用相同规则，完整种群继续保存。

仍采用Case I的180点对数均匀频率网格，不表示已验证case2，也不声称精确复现中文PDF的401点采样。

## 已有结果的稳定性分析

analyze_existing.py 仅读取此前九指标配对的完整归档，提取baseline、A、B、C、A+B、A+C、B+C的700次历史运行。

读取时检查归档SHA256、源指纹、配置和指标代码，重新从每次完整种群选择WSMSE最优个体，核对两项误差，并验证同seed各组初始种群一致。不会调用优化器，也不改变旧结果。

三个配对各与两个组成单指标比较，共六次配对比较、12项误差。对seed成对bootstrap 20000次，固定分析随机种子20260916，给出均值差、单项95%区间、12项Bonferroni调整区间，以及去掉改善最大的5% seed后的敏感性诊断。删减只作诊断，不改变正式均值；两项指标可能删去不同seed。

候选由同一批旧结果筛出，所以这些区间属于选择后的探索性分析，调整区间也不能消除选择偏差。所有配对均保留，区间跨零解释为证据不足，不直接断言无效。

## 新实验报告

- RESULTS.md：八组的均值、相对baseline变化、两项同降seed数。
- STABILITY.md：七个非baseline组与baseline的配对置信区间及敏感性诊断。14项误差作为一个比较集合。
- CONSTITUENTS.md：三个配对相对其单指标，以及三结合相对三个单指标和三个配对，共12次比较、24项误差，单独给出调整区间。
- selected_individuals.csv：每个seed实际选中的参数和误差。
- paired_summary.csv：同时保留最终WSMSE选解、仅主目标前沿和完整目标前沿三种口径。
- raw_results.tar.gz及SHA256：完整种群和报告归档。

研究主问题仍是配对与三结合能否同时改善两个主目标；是否进一步优于组成项单独报告，不用更强条件偷偷替换原问题。bootstrap百分位区间覆盖率是近似的，多个比较集合也不等于对所有结论做了统一全局错误率控制。

analysis_config.json在正式实验前固定并纳入指纹。不会根据本次新seed结果修改候选或权重后仍将其描述为原方案的独立验证。

## 本地检查与分析

在本目录运行：

~~~powershell
& '..\case1_dual_paper\.venv\Scripts\python.exe' -m unittest -v test_validation
& '..\case1_dual_paper\.venv\Scripts\python.exe' experiment.py plan --output prepared_plan
& '..\case1_dual_paper\.venv\Scripts\python.exe' analyze_existing.py --archive ../case1_dual_nine_pairs/server_run/raw_results.tar.gz --output existing_analysis
~~~

软件测试仅在临时目录运行24次小规模检查（8个体、3代），不作为实验结果。已有数据报告见existing_analysis/CONSTITUENTS.md，来源审计见existing_analysis/source_audit.json。

## 服务器就绪后执行（本次未执行）

上传本目录顶层.py、.json、requirements*.txt及run_server.sh。不需要上传旧归档或existing_analysis。

~~~bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m unittest -v test_validation
nohup bash run_server.sh > batch.log 2>&1 < /dev/null &
~~~

run_server.sh默认28个worker，按服务器CPU配额调整。启动器使用独占锁；拒绝覆盖未完成目录或混用不同配置和代码指纹。优化完成后才执行完整结果审计、稳定性报告和归档。

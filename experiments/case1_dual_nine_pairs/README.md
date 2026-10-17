# Case I：九个候选指标的两两组合

状态：仅准备代码和历史对照，等待服务器后运行；没有启动新增正式实验。

## 候选集合

这里的“有效”仅指此前 seed 0–99 上，两项主目标的均值均低于 baseline，不代表全部指标已经通过稳定性或显著性验证。

|编号|指标|
|---|---|
|I-01|全局转折数量：至少5个|
|I-02|目标转折区域覆盖：5-of-6|
|I-03|电抗过零方向一致性|
|I-04|全局电抗过零数量：至少10次|
|I-05|全局转折数量：至少4个|
|I-06|中文 PDF 峰谷一致性（最近匹配、均值）|
|II-01|局部特征频移误差|
|II-02|转变速度相对误差|
|II-05|95% Hausdorff 距离|

检查全部 36 个无序配对，不按结果挑选其中某些配对。已有 I-03+II-01、I-03+II-02、II-01+II-02 三组，直接复用；新增其余 33 组。

## 固定预算与结果口径

每组 seed 0–99、200 个体、300 代，新增 3300 次运行、1.98 亿次目标评估。最终报告包含 baseline、九个单指标、36 个配对，共 46 组。历史对照共 1300 次：baseline 和九个单指标 1000 次，加三个配对 300 次。

NSGA-II、指标计算、初始化、交叉、变异和筛选规则保持不变。每个配对有四个独立目标：幅值 MSE、相位 MSE、辅助目标1、辅助目标2。辅助目标参与交配和环境选择，不合并成加权辅助损失。

最终每个 seed 从最后一代种群中选固定 WSMSE 最小的一个个体，权重仍为 25.12；同一个体的两项误差分别跨 seed 求均值。完整种群、参数及两种前沿补充统计继续保存。

报告分别比较配对与 baseline、配对与两个组成单指标。所有组都输出，不只列改善的组合。均值下降仅作探索性筛选，不宣称显著改善或没有相互干扰。

本轮继续使用已有的 180 点对数均匀频率网格，不声称精确复现中文 PDF 原 401 点采样。

## 与独立验证的关系

本目录用于扩大候选组合的探索范围，使用旧 seed 与已有单指标结果配对。case1_dual_validation 中 seed 100–199 的八组固定验证代码保留不变，也没有启动。不能把本轮筛选结果当成独立验证结果；如果以后根据本轮改变验证候选，需要明确另行制定验证方案。

## 代码与数据

- config.json 只列 33 个待运行配对。
- pair_plan.json 列出九个候选、全部36个配对、复用的三个配对及完整报告组别。
- reference/singles 与 reference/pairs 分别保留来自两批实验的原始 manifest、完成记录和完整种群，不改写历史 fingerprint。
- prepare_reference.py 校验两批原始归档，只提取所需的1300次历史运行；reference.tar.gz 可直接上传，reference.sha256 用于校验。
- sources.py 在统计前核对每个来源的配置、频率、指标与选解代码哈希、依赖版本。
- 频率网格配置要求完全一致；Windows/Linux 生成的频点只允许不超过4倍双精度机器精度的末位舍入差，启动预检记录实际差值。这不允许改动采样点、范围或密度。
- report_results.py 审计所有4600个新旧完成记录与实际选中个体，检查配对初始种群，生成完整比较报告及包含历史对照的原始归档。

## 本地检查（无正式实验）

~~~powershell
& '..\case1_dual_paper\.venv\Scripts\python.exe' -m unittest -v test_nine_pairs
& '..\case1_dual_paper\.venv\Scripts\python.exe' experiment.py plan --output prepared_plan
~~~

测试在临时目录中运行46组、每组一个seed、8个体、3代，仅检查从计算到报告的流程；不作为科学结果。

## 服务器就绪后

上传顶层 .py、.json、requirements*.txt、run_server.sh、reference.tar.gz、reference.sha256。建立 .venv 并按 requirements-lock.txt 安装，然后校验和展开历史对照：

~~~bash
sha256sum -c reference.sha256
tar -xzf reference.tar.gz
nohup bash run_server.sh > batch.log 2>&1 < /dev/null &
~~~

run_server.sh 默认28个worker，部署时按 CPU 配额调整。新输出写入 runs，历史输出保持在 reference。启动器拒绝覆盖未完成目录或混用不同指纹。

# Case I：固定三指标集合的独立 seed 验证

状态：仅准备代码和分析已有数据，尚未启动新一轮正式优化。软件检查仅使用临时目录中的小种群短运行。

## 固定实验

A = I-03 电抗过零方向一致性；B = II-01 局部特征频移误差；C = II-02 转变速度相对误差。

八组为 baseline、A、B、C、A+B、A+C、B+C、A+B+C。全部使用新 seed 100–199，包括重新运行 baseline 和三个单指标，共 800 次运行。每次 200 个体、300 代，总计 4800 万次目标评估。旧 seed 0–99 仅供探索性分析，不混入新批次，也不复用旧 baseline 作新 seed 对照。

NSGA-II 的目标定义、参数边界、初始化、交叉、变异、亲本选择、环境选择、频率网格与上批完全一致。全部辅助目标分别占用一个目标维度，参与进化选择。

最终每个 seed 从最后一代种群中按固定 WSMSE 权重 25.12 选一个个体，两项误差取自同一个体，跨 seed 等权求均值。权重只影响结果选取，不影响进化。保留全目标前沿和仅主目标前沿作为补充统计。

继续使用 180 点对数均匀频率网格。中文 PDF 原始 401 点坐标缺失，本验证不声称复现该采样；也不代表已经验证 case2 或真实电路。

## 分析预设

analysis_config.json 在正式实验前固定并纳入运行指纹。

1. 针对相同 seed 计算候选组减 baseline 的两个误差差值，负数表示改善。主报告保留完整 100 seed 的均值，不删除运行。
2. 对整个配对 seed 重采样 20000 次，同时保留两项误差的对应关系，固定分析 RNG seed 为 20260915。报告配对均值差的百分位 bootstrap 95% 区间。
3. 对七个非 baseline 组的两项误差共 14 项，另报告 Bonferroni 调整区间（每项约 99.643%，名义整体覆盖率 95%）。bootstrap 是近似方法，尤其极端尾部分位点存在 Monte Carlo 误差。区间跨零不等于证明无效。
4. 固定敏感性诊断：差值两端各截尾 5%、去掉改善最大的 5%、逐个删除 seed 的均值差范围、最大 5% seed 占全部正收益的比例。删减各误差分别进行，不能将两项删减结果理解为同一子样本。所有这些是诊断，不能替代主报告或用于挑 seed。
5. 组合相对其组成项的比较保存在 paired_stability.csv，仅作探索性诊断。针对 baseline 的 14 项整体判据不适用于这些额外比较。

研究主问题是四种组合是否都改善两项主目标。报告分别区分：两项均值均下降、两项单项区间均低于零、两项调整区间均低于零。新 seed 结果若不足，不通过追加 seed、改变选解权重或排除不利组来改变预设结论。

旧数据用于挑选了当前三指标，其区间是选择后的探索性诊断，不能作为独立确认。新一轮仍仅检验同一个 case1 上对随机种子的稳定性，不代表对不同电路的泛化。

## 本地分析与检查

在本目录下使用已有 Python 环境运行：

~~~powershell
& '..\case1_dual_paper\.venv\Scripts\python.exe' -m unittest -v test_validation
& '..\case1_dual_paper\.venv\Scripts\python.exe' experiment.py plan --output prepared_plan
& '..\case1_dual_paper\.venv\Scripts\python.exe' stability.py --archive ../case1_dual_pdf18_combinations/server_run/raw_results.tar.gz --output discovery_analysis
~~~

第三条命令仅读取已完成归档并校验 SHA256，不调用优化器。输出在 discovery_analysis/STABILITY.md、paired_stability.csv、stability.json；source.json 保存源归档校验值与两批源 manifest。

## 服务器就绪后执行（本次未执行）

上传本目录顶层的 .py、.json、requirements*.txt 和 run_server.sh。无需上传旧 reference 或 discovery_analysis。

~~~bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
nohup bash run_server.sh > batch.log 2>&1 < /dev/null &
~~~

run_server.sh 默认使用 28 个 worker；部署时根据可用 CPU 配额调整该脚本。线程库被限制为每 worker 一个线程。启动器保留独占锁，拒绝不同指纹混跑，拒绝覆盖未完成目录。只跳过有完整完成记录且来源相同的运行。

完成后审计全部 800 次运行、配对初始化、每个实际选中个体，生成 RESULTS.md、STABILITY.md、三种统计口径、选中参数表和完整原始归档及 SHA256。如果运行失败，脚本退出并保留错误，不把部分结果称为完成。

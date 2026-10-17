# 本地原因追查：方法、文件与限制

## 1. 来源与输出

冻结代码和输入：`D:/develop/FEMEIF/output/complex_joint_20261012/retrieved/experiments/complex_joint_20261012`。

原任务结果：`D:/develop/FEMEIF/output/complex_joint_20261012/retrieved/output/complex_joint_20261012/jobs`。

本次所有新文件仅写入 `D:/develop/FEMEIF/output/complex_joint_causes_20261013`。没有运行优化器、网络连接、采样仪器或上传。

|文件|用途|
|---|---|
|[MODE_CARDS.csv](MODE_CARDS.csv)|32个候选：位置、衰减、留数、端口强弱诊断及曲线重建误差|
|[MODES.csv](MODES.csv)|所有有限端口极点、留数、背景|
|[modal_decompositions.npz](modal_decompositions.npz)|目标与候选的复数极点、零点、留数和背景数组|
|[MODAL_FACTORIAL.csv](MODAL_FACTORIAL.csv)|32×8种诊断替换，保留增加误差的情况|
|[MODAL_SWAP.csv](MODAL_SWAP.csv)|整体换高频成分、整体换其余成分的诊断|
|[EVENT_WEIGHT.csv](EVENT_WEIGHT.csv)|原复核3dB核心窗口的频率权重和误差贡献|
|[A_COMPONENT_SCORES.csv](A_COMPONENT_SCORES.csv)|160个候选的A分量、J0、D/H、连续局部误差及加密一致性|
|[A_COMPONENT_WINDOWS.csv](A_COMPONENT_WINDOWS.csv)|逐窗口A分量及原目标归一化分母|
|[A_COMPONENT_RANKS.csv](A_COMPONENT_RANKS.csv)|分数与局部事件失败数、连续误差的Spearman相关|
|[MSE_MATCHED_RANKING.csv](MSE_MATCHED_RANKING.csv)|MSE相近候选的排序检查，10%与25%都保留|
|[SCORE_DOMINANCE.csv](SCORE_DOMINANCE.csv)|哪个窗口主导总分、P2/V3占比|
|[SCORE_RELIABILITY.csv](SCORE_RELIABILITY.csv)|16/32级与原锚点值的核对；原特征门槛不变|
|[CASE1_GRID_CHECK.csv](CASE1_GRID_CHECK.csv)|56个候选的原标记、网格诊断，4个分层样本补充深度计算|
|[CASE1_INITIAL_DEEP_D.csv](CASE1_INITIAL_DEEP_D.csv)|初轮8个D样本1/8/16/32级及32级锚点检查|
|[CASE1_BRANCH_GRID.csv](CASE1_BRANCH_GRID.csv)|各候选六支路的中心、Q、近似宽度与网格间隔|
|[CASE1_WINDOW_PEAKS.csv](CASE1_WINDOW_PEAKS.csv)|各窗口粗网格最大值与支路中心直接响应的差|
|[POPULATION_LOCAL.csv](POPULATION_LOCAL.csv)|32个保存种群的局部改善机会；不改实际选解|
|[AREA_IDENTITY_CHECK.csv](AREA_IDENTITY_CHECK.csv)|频率–Re面积的线性积分恒等式数值检查|
|[PROVENANCE.json](PROVENANCE.json)|冻结输入/代码与216份保存种群、复核曲线的哈希核验|

`FIXED_CANDIDATE_SCORES.csv`、`SCORE_CORRELATIONS.csv`、`STABLE_SCORE_CORRELATIONS.csv`、`WINDOW_SCORE.csv`保存较低层级的交叉检查；正文A分量与排序表使用32级计算。未删除初轮结果。

## 2. 动态分解方法

构造端口电流激励的MNA描述系统：

\[
(G+uE)x=bI,\quad Z(u)=b^T(G+uE)^{-1}b,\quad u=s/(2\pi\,10^4).
\]

电阻进入G，电容和电感进入E；电感电流为额外变量。矩阵行列对称均衡后，用齐次广义特征值区分有限与无限特征值。有限模态留数使用左右特征向量：

\[
r_k=\frac{(b^Tv_k)(w_k^Hb)}{w_k^HEv_k},\qquad
Z(u)=d+eu+\sum_k\frac{r_k}{u-p_k}.
\]

原电路递归响应用于重建核对；另以MNA端口响应和冻结有理多项式根进行交叉检查。没有用展开多项式的求根作为主要提取方式。所有有限根数和根位置匹配检查通过。

34个分解中，32候选的重建最大相对误差约1.76×10⁻¹⁰，MNA响应核对最大约2.39×10⁻¹¹；目标也通过同样检查。重建、节点与特征方程残差门槛均为10⁻⁸。微小背景虚部保留为数值结果，未赋予物理意义。

以目标成分频率和振荡/实模态类型做一一匹配。这两个条件的已检查解均有两个振荡对和一个实极点。若未来模态数改变或根重合，不应强行使用本匹配规则。

诊断高频成分是目标中频率最高的振荡对。候选和目标的其他成分、背景不同，所以模态替换的局部效果可以受到相消影响。替换响应不保证拓扑可实现或无源。

## 3. 误差、评分和比较规则

局部连续误差为：

\[
E_{loc}=\frac{1}{x_b-x_a}\int_{x_a}^{x_b}\frac{|Z(x)-Z^*(x)|^2}{S_{loc}^2}\,dx,
\quad x=\log_{10}f,
\]

局部范围预先按已有研究频段固定：原始条件45–58kHz，另一条件38.5–45kHz，S取该范围目标最大幅值。种群诊断使用另一份预设略宽的log频率窗口，两个表不混算绝对误差。

理论辅助指标使用原目标窗口、原尺度、原归一化分母和原物理半步。32级表示每窗口4097点，不是新的测量点。特征一致性用原阈值：绝对0.01＋相对1%。不修改历史通过标签。

J0是同窗口、同Ω尺度的复数点误差平均。它不是逐项匹配A分母的完整权重控制，也不是已经运行的辅助优化组。

A的三个分量按冻结代码计算，包含各窗口起点平移和前缀面积。对频率–Re分量，用梯形积分核对了恒等式：

\[
A_{tR}(t)=\frac12t[R(t)-R(0)]-\int_0^t[R(v)-R(0)]\,dv.
\]

因此频率–Re/Im分量可视为各自的线性积分过程，而Re–Im分量包含跨分量关系。数学区别不自动证明搜索收益，也不代表这些特征是新数学工具。

评分排序使用各条件全部80个既有最终候选。相近MSE候选对要求两种MSE各自最大/最小≤1.10或1.25，且P2/V3失败数不同。候选对重叠，不使用独立样本p值，不作为推广证据。

Case1每支路中心与Q按其已知并联RLC计算；近似对数3dB宽度为1/(Q ln10)，只用于分辨率诊断。支路中心响应不等于全曲线峰顶。16/32级及锚点补充样本包含失败与历史通过样本，深度检查不是全56样本。

## 4. 运行问题与处理

1. 初版描述系统分解将一个数值上极大的特征值误当作有限值，导致目标重建误差超过门槛，计算立即停止。改用齐次广义特征值识别无限值，并核对有限根数、根位置、节点响应和重建。原冻结科学代码未改动，未放宽门槛。
2. 初次尝试对全部56样本做32级高密度锚点检查，完成前8个D后发现本地成本较高，主动停止该诊断进程，保留已经完成的数据。改为全56个候选做粗网格/支路检查，4个分层样本补充深度计算。没有停止或重启历史优化任务，也没有把未计算项填成通过。
3. 第一次读取文件时，旧分析脚本和任务摘要的路径猜测未命中；找到实际文件后继续。没有生成替代源数据。

## 5. 复算顺序

使用本地科学Python环境，依次运行同目录下：

1. `investigate.py`：32候选动态分解和替换。
2. `scoring_check.py --case1`：56样本与4个深度样本；初轮8个D深度数据单独保留。
3. `followup_checks.py`：辅助评分可靠性、窗口与Case1粗网格。
4. `area_ablation.py`：32级分量与相近MSE排序。
5. `figures_and_provenance.py`：图与来源核验。

若需复算初轮固定候选与种群表，用`scoring_check.py`不带参数运行。它会再次计算Case1部分；这是本地诊断成本，不会运行优化器。

正文为人工分析，不由这些脚本自动覆盖。后续方案须另开版本，保留原代码、预算和来源。

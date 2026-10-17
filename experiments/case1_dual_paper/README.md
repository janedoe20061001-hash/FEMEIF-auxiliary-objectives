# Case I：双主目标 + 论文五个辅助目标

状态：代码准备与小规模正确性检查；未启动正式的 100 seeds × 200 个体 × 300 代实验。

本实验将论文 SS、PTC、DTW、LFP、LMEP 接入 NSGA-II，始终保留幅值 MSE 和相位 MSE 两个独立主目标。这里不加入新提出的谐振位置、抵消位置或带宽指标，不使用轻量化 DT、交叉硬保护等框架改动。

## 实验矩阵

|项目|配置|依据|
|---|---|---|
|独立运行次数|100 seeds，编号 0—99|英文论文第 IV 节规定 100 次；具体 seed 编号由本次定义|
|种群 / 每代子代数|200 / 200|Table V 的种群；采用等量子代|
|进化代数|300|Table IV / Fig.11；本代码按 pymoo 约定初始评估为第 1 代，随后 299 轮繁殖|
|每组每 seed 评估次数|60,000|200 × 300，无提前收敛终止|
|算法|NSGA-II|沿用现有双主目标框架；英文论文后续组合消融使用 GrEA，本次不切换|
|基线|幅值 MSE + 相位 MSE|没有辅助项|
|辅助分组|5 单项 + 10 两项 + 10 三项 + 5 四项 + 1 全部|覆盖组合筛选需要|
|总量|32 组 × 100 seeds = 3,200 次运行|仅预配置，未执行|

同一 seed 的全部组共享同一个初始种群和算法随机种子；初始种群单独保存并记录哈希。种群从 Table II 范围内按 log10 均匀初始化，不注入目标参数，不围绕真值初始化。种群演化后不保证各组继续使用相同的交配对象。

参数按 R1,L1,C1,…,R6,L6,C6 排列，6 个并联 RLC 块串联。参数上下界：
R ∈ [1,10^5] Ω，L ∈ [10^-7,10^-2] H，C ∈ [10^-9,10^-4] F；真值来自 Table III。

## 目标和实现约定

主目标严格为：

- LA = mean[(20log10|Zfit| − 20log10|Ztarget|)^2]，单位 dB²。
- Lphase = mean[(angle Zfit − angle Ztarget)^2]，单位 rad²；直接相位差，不额外 unwrap。
- 两者始终占目标向量前两列，不使用 α=25.12 合成 WSMSE。

辅助项在默认 auxiliary_transform = loss 下均越小越好：

|辅助项|实现|
|---|---|
|SS|Table I、式 (3)—(6)；相邻斜率编码绝对差和 / N|
|PTC|目标峰谷中点分区；区内所有同类峰谷累加 exp(γ × log10 频率偏差 / 总频宽)−1；缺失罚 γ|
|DTW|60 点，绝对 dB 差、无窗口约束、无路径长度平均的累计 DTW|
|LFP|从低频开始连续通过的区段数 c；损失 1−c/区段总数|
|LMEP|幅值最大绝对 dB 差|

超参数照 Table V：THk=8、β=0.4、μ=0.5、γ=0.41、THlf=1.6、DTW 点数 60，交叉和变异分布指数均为 1。

论文没有完整交代以下细节，当前选择明确写入配置，不能声称与作者代码完全相同：

1. **频率坐标**：默认重新构造 100 Hz—1 MHz 的 180 点对数均匀网格。论文只给点数，没有原始频率表；现有中文实验的 401 点原始坐标也不在工作区。正式跨实验比较应统一频率文件，不能把旧 401 点或稠密扰动实验的 MSE 与本实验数值直接比较。尤其 PTC 在稀疏网格上可能看不到小峰移。
2. **自定义频率**：frequency.file 可设为相对 config.json 的路径或绝对路径；格式为一列 Hz 数值的 CSV，无表头，严格递增。设置后以文件为准，不以 points 截断。DTW 在该文件的频率范围内重算 60 个对数均匀点的同一电路响应；未使用作者未知的降采样坐标。
3. **通道**：五项辅助按幅值 dB 计算；原文未逐项完整标出通道/单位，本次与先前五项指标扰动对比保持一致。
4. **峰谷与 SS 端点**：峰谷由相邻样本确定，平顶取右端；不加 0.2 dB 等原文未给定的显著性门槛，不做插值。PTC 不采用“只取最近匹配后再平均”的变体。SS 只计算可得的 N−2 项，分母仍为 N；斜率等于门槛归平缓区。
5. **LFP 门槛**：各段幅值 MSE 严格小于 1.6 才通过；等于则失败。目标 180 点网格有 11 个峰谷区段，自定义网格按实际目标区段数计算。
6. **交叉、变异**：SBX 个体对交叉概率取 0.9；交叉时的参数概率使用固定依赖版 pymoo 的 SBX 默认值 0.5。论文 λ=0.1 未明确作用层次，本次取每个参数变异概率 0.1、个体进入变异算子的概率 1.0。均可配置；log10 编码是本次实现选择。
7. **选择**：二元锦标赛按非支配等级、拥挤距离比较；环境选择为标准 NSGA-II。关闭重复消除以保持每代恰好 200 次评估，不做额外修复、目标动态调权或早停。
8. **倒数转换**：英文论文多数辅助项最大化 1/error；本次默认直接最小化 error，以延续双主目标实验的误差形式。配置也支持 negative_reciprocal，对应最小化 −1/(error+epsilon)，LFP 为 −c。两种形式保持单项优劣排序，但会改变拥挤距离，**不能视为相同实验**。epsilon 默认 1e−12，避免完全吻合时除零；该稳定化约定不等于原作者实现。

## 代码与环境

- objectives.py：电路响应、两个主目标、五项辅助；仅 NumPy。
- experiment.py：实验矩阵、NSGA-II、按 seed 保存、断点续跑。
- summarize.py：与同 seed 基线配对汇总。
- test_experiment.py：数值核验和小规模流水线检查。
- config.json：正式参数配置。
- requirements-lock.txt：本次验证环境的完整依赖版本；Python 3.12。

已在当前目录 .venv 建立隔离环境。在其他环境可执行：

~~~powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
~~~

算法 API 参考：[pymoo NSGA-II 官方文档](https://pymoo.org/algorithms/moo/nsga2.html)、
[Mutation 官方文档](https://pymoo.org/operators/mutation.html)。依赖固定为 pymoo 0.6.1.6，配置同时依据安装版本源代码核对。

## 使用

以下命令均在本目录运行。**只查看计划，不进行任何种群进化：**

~~~powershell
& .\.venv\Scripts\python.exe experiment.py plan --output prepared_plan
~~~

数值检查只执行极小规模的测试，不生成正式结果：

~~~powershell
& .\.venv\Scripts\python.exe -m unittest -v test_experiment
~~~

可选冒烟检查使用 1 seed、12 个体、3 代，输出隔离在 smoke 下：

~~~powershell
& .\.venv\Scripts\python.exe experiment.py smoke
~~~

**以下才是未来启动正式实验的命令，本次未执行：**

~~~powershell
& .\.venv\Scripts\python.exe experiment.py run
~~~

可以只运行配置内的某组、某 seed：

~~~powershell
& .\.venv\Scripts\python.exe experiment.py run --group "SS+PTC" --seed 0
~~~

--resume 仅跳过完成且配置指纹一致的整次运行，不从中间代恢复。存在未完成目录时停止，避免静默覆盖；需要保留该目录用于诊断，并用新的 --output 目录重跑。配置、代码、频率或关键运行时版本变化会拒绝混合结果。

~~~powershell
& .\.venv\Scripts\python.exe experiment.py run --resume
& .\.venv\Scripts\python.exe summarize.py runs
~~~

plan 展示配置的完整矩阵，--group/--seed 运行筛选仅用于 run/smoke。

## 结果口径

每次运行保存初始参数、逐代 history.csv、最终完整种群与两个前沿的索引，以及最后写入的完成标记 summary.json。

主要统计延续中文实验稿式 (1.2)—(1.3)：先对**完整目标空间的最终非支配前沿**分别计算幅值 MSE 和相位 MSE 的均值，再对 seed 等权平均，计算相对基线的变化。由于辅助目标可能保留主目标较差的个体，同时另存**仅由两个主目标重新筛选的前沿**统计，二者分别标记，不能混用。

summarize.py 只对同一 seed 的组与基线配对，标注已有/预期 seed 数以及完整性。报告两项均值是否同时下降，以及多少个配对 seed 的两项都下降。这里是描述性结果，不自动宣称统计显著性，也不把分别取出的两个单项最优值当作同一个解。

不把全部 seed 的所有解混在一起平均；不使用 WSMSE 选一个代表解。若后续需要显著性检验、多重比较或组合筛选规则，应在正式统计前固定规则。

## 来源

- 根目录英文 PDF：第 3—5 页辅助目标定义；第 5 页 Table II 参数范围；第 6 页 Table III/IV/V 和第 IV 节实验次数。
- 根目录中文 PDF：第 3 页既有双主目标实验的跨 seed 汇总口径；两个主目标定义。
- 既有扰动对比数据仅用于核验辅助函数数值，不作为算法初始种群或正式拟合结果。

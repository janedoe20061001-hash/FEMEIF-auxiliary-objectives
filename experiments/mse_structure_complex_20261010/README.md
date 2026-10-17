# 第二步扩展：嵌套串并联电路

代码准备版本，2026-10-10。**尚未运行正式优化**。无需原工程、原 Case1 代码或 GPU；本目录可独立复制到 Linux 服务器。

## 服务器环境

用 Python 3.12 建立独立环境，CPU 即可。锁定依赖与上一轮环境相同。首次建议 2 个工作进程，再根据服务器资源调整；每个进程的数值库线程限制为 1。

```bash
cd mse_structure_complex_20261010
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python test_validation.py --report VALIDATION.json
```

测试只检查数值计算，并用静态假种群检查保存/分析流程；不进行进化搜索。正式运行必须先通过测试。

## 先查看计划，不开始优化

```bash
.venv/bin/python run.py plan --circuits user_nested
.venv/bin/python run.py plan
.venv/bin/python run.py plan --profile extended
```

无参数运行 `run.py` 也只显示计划。默认 core：7 类 × 3 个目标 × 8 seeds × 300 代 × 200 个体，10,080,000 次评价。用户电路单独运行为 1,440,000 次评价。extended 是预备扩展，共 23,520,000 次评价；不建议尚未看探索结果就全部铺开。

## 第一批建议：用户电路

先生成解析目标和核验记录，再正式运行。生成目标不是优化实验。

```bash
.venv/bin/python prepare.py --circuits user_nested --out inputs_user
.venv/bin/python run.py run --circuits user_nested --prepared inputs_user --validation VALIDATION.json --out batch_user --workers 2
.venv/bin/python analyze.py --batch batch_user --prepared inputs_user --out analysis_user --population all --plots
```

Windows 对应解释器路径为 `.venv/Scripts/python.exe`。本地提供的 VALIDATION.json 说明本地代码已核验；服务器应在自己的环境中重新生成。

## 多类电路及补充条件

```bash
.venv/bin/python prepare.py --out inputs_core
.venv/bin/python run.py run --prepared inputs_core --validation VALIDATION.json --out batch_core --workers 2
.venv/bin/python analyze.py --batch batch_core --prepared inputs_core --out analysis_core --population all --plots
```

`--circuits` 可以一次给多个 ID，例如 `control_rlc user_nested rlc_ladder`。这些 ID 在 `circuits.py`，参数也在其中。补充阻尼和背景参数对照时，在 prepare 和 run 中同时加 `--profile extended`。core 与 extended 各用新目录，不混合重复的 base/seed 运行。

如需新种子或更多代数，复制 config.json 为新配置文件，修改后对 test_validation.py、prepare.py、run.py 都加 `--config 新配置文件路径`，并使用新核验记录、新输入目录和新输出目录。不能拿原配置的核验记录批准修改后的运行。

## 数据和检查口径

- 目标结构：JSON 显式 S/P 树，S 为串联、P 为并联。不会猜测运算符优先级。电阻 Ω、电感 H、电容 F，各物理元件名字独立。
- 优化目标：MAG = dB 幅值 MSE；PHA = 弧度相位 MSE。两项都独立交给 NSGA-II，没有辅助指标。
- 数值训练：目标和每条候选的极点/零点作为加密入口，再按幅相变化自适应细分；保留原 180 个对数频率单元各 1/180 的权重。纯数值加密不增加窄峰的评分权重。此策略只适用于有解析模型的理论电路。
- 每个拓扑的边界是基准参数上下 1.5 个 log10 数量级。同一拓扑所有条件共用边界，真参数在范围内；初始化均匀随机，未把真参数放入初始种群。不同拓扑的维数、边界和响应不同，不视为严格配对。
- 相同拓扑、seed 在各条件下的初始种群逐元素一致。NSGA-II：种群/子代各 200、300 代；SBX prob=.9/eta=1，PM prob=1/prob_var=.1/eta=1，rank/crowding 锦标赛，不去重，与上一轮方法保持一致。
- 每 20 代及最后一代保存整个保留种群、前沿、两条代表解。primary 在前沿取 MAG+25.12×PHA 最小，balanced 取 max(MAG,25.12×PHA) 最小；权重只用于展示/选解，不是单目标训练。
- 幅值事件：突出度至少 1 dB；突出度不到 3 dB 时不强算 3 dB 宽度。匹配相同类型、频差≤.25 decade；合格要求位置差≤.02 decade、高度差≤3 dB、宽度比在 .5–2。没有事件不是成功，有事件但不能定义宽度也不强行计为失败。
- 相位事件：解析相位斜率的正/负极值，斜率至少 30°/decade、突出度至少 15°/decade。对应频率差≤.25 decade；完整恢复门槛位置差≤.02 decade、斜率和半高宽比各 .5–2。只是探索性诊断定义，尚未进行噪声稳健性验证。
- 独立检查：非自适应加密参考网格逐级 1/2/4/8 加密，检验目标分数、事件匹配与合格分类是否稳定；评分差容差为 1% 加绝对下限 [0.01 dB², (0.5°)²]。超过限制会标记，未通过样本不悄悄计入正常均值。
- 默认分析每个保存阶段的两条代表解，以及末代整个种群。`--all-stages` 进一步检查所有保存阶段整个种群；它仍然不包括被淘汰的全部历史子代。
- `population.csv` 看检查范围内是否存在结构合格的候选；`similar_scores_different_shapes.csv` 按 2%/5%/10% 总分接近门槛，检查漏失状态不同、高度差≥6 dB、宽度比≥2 的候选对。统计单位是 seed。
- 后台比较所有 seed；图只叠加预先固定的 33000/33003/33006，各自代表曲线与目标，不平均曲线。3D 数据同时保存 f、Re Z、Im Z，二维图采用 log f/dB/角度。诊断分频段和局部窗口误差另外保留。

`AUDIT.json` 先看完整性和数值问题，`REPORT.md` 是初步结果表；CSV 和 details.jsonl 保留细节，可复算。报告生成器不自动宣布“MSE无法优化”或“峰越窄越难”，这些解释需要结合目标之间的实际结构变化。

## 维护与中断

新目录禁止覆盖。config、拓扑、源码和目标元数据均做 SHA-256 校验；改代码/配置后先重新测试，再重新生成 inputs。每个已完成 seed 的初始化、历史和所有阶段文件也做哈希检查；仅有“完成”标记不算完整。正式运行记录依赖、平台、参数、初始化、预算、时间和失败原因。分析代码也在本次哈希中，方便锁定本轮口径。

同一机器、相同版本可用原 run 命令加 `--resume`：它只跳过已完整完成的 seed。如果有中途失败/中断的 seed，其目录保留并明确报错，应在新输出目录重跑需要的任务。**没有保存 RNG 状态，不声称支持从某一代恢复。** 跨平台不声称逐位相同；应在服务器重新核验并准备输入。

远程断开后可用服务器现有的 tmux/screen 会话维持进程。耗时需要正式运行时测量，目前没有速度结果。

## 文件职责

|文件|作用|
|---|---|
|circuits.py|7 类拓扑、参数、阻抗、导数、节点方程、极点/零点|
|schematic.py|标准 R/L/C 符号、串并联分支布局、图中接线与模型核对、SVG/PNG|
|config.json|条件、种子、算法预算、阈值|
|sampling.py|训练自适应积分与独立参考积分|
|diagnostics.py|实际峰谷/相位事件、匹配、局部贡献、分频段检查|
|prepare.py|冻结目标、结构图、三维曲线、目标事件表|
|run.py|默认只计划；显式 run 启动优化及记录阶段数据|
|analyze.py|代表解和种群复核、配对数据、总分近似而结构不同、少量曲线图|
|test_validation.py|不进行搜索的数值/模拟流程检查|
|deliver.py|生成通俗说明、输入清单和可携带代码 ZIP|

所有模型均为自设计理想无源集中参数 RLC 网络，未采用外部实测数据；仍有正确模型内的零误差解。Case1 历史实验数据不修改，Case2 实测模型限制不混入本轮。

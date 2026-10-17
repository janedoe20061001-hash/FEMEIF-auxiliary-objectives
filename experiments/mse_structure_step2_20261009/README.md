# 第二步本地探索：原始实验目录

正式结果入口：`D:/develop/FEMEIF/output/mse_structure_step2_20261009/REPORT.md`。

本目录保存固定协议、数值源、目标输入、24次原始运行，以及复算和作图程序。网站没有更新。

## 固定对照

- Case1六个并联RLC单元串联；18个拟合参数全部未知。
- 目标第2单元：R2不变，L2乘s，C2除以s；六档目标核验，正式运行s=0.25、1、8。
- 三档共用seed 31000—31007，对应初始200×18参数矩阵完全相同。
- NSGA-II双目标MAG/PHA，200种群/200子代，300代，SBX概率0.9/eta1，PM概率1/逐变量0.1/eta1，不去重，rank/crowding tournament。
- 自适应积分评分沿用原180个对数频率单元，每单元质量1/180；Windows纯Python实现。
- 主代表规则MAG+25.12PHA最小；另一规则max(MAG,25.12PHA)最小。25.12仅用于选解。
- 每20代完整种群和两条代表均保存；主结果是同曲线P3/V2同时满足原结构要求。

## 文件关系

- `CONFIG.json` / `output/.../PROTOCOL.json`：新结果前冻结的主协议。
- `frozen/`、`analysis_toolkit/`、`adaptive.py`、`structure.py`：从第一步复制的数值源；未改动旧实验。
- `targets/`：六档独立目标响应、结构目录、评分基础网格。
- `batch/manifest.json`：环境版本、协议和输入散列；`batch/status.json`、`batch/summaries.json`：状态和成本。
- `batch/s*/seed_*/initial_log_parameters.npy`：精确初始种群。
- `generation_020.npz`至`generation_300.npz`：完整参数、训练目标、前沿及代表索引。
- `trajectory.npz` / `history.json`：每20代两种代表及每代成绩。
- `smoke/` / `timing/`：独立seed的环境与耗时预检，不计入正式统计。

补充评分探针是探索性诊断，在首批运行期间新增，另有`SCORING_PROBE_PROTOCOL.json`，不当作提前设定的主检验，也不把42个确定参数候选视为42次随机优化。

## 复算

交付目录的REPRODUCE.md说明复算顺序和依赖版本。运行程序保护已有运行目录，拒绝覆盖原始结果。独立复现要使用另外目录并更改common.py中的输出路径。

独立环境`.venv/`及Python缓存不纳入研究数据版本记录。没有连接服务器，也不使用Linux native.so。

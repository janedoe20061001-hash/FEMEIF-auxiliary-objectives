# 本地复现

在D:/develop/FEMEIF下使用独立Python环境。运行所用版本保存在experiments/mse_structure_step2_20261009/batch/manifest.json；依赖采用experiments/primary_loss_comparison_20261007/frozen/case1/requirements-lock.txt。

## 从已有原始结果重新生成分析

1. 使用该实验目录.venv/Scripts/python.exe运行offline.py（需要第一步本地原始结果和分析CSV）。
2. 运行analyze.py --workers 2；已完成的chunks会复用。若要完整重算，先把自己的旧分析输出备份到另外目录，不覆盖提供的数据。
3. 运行scoring_probes.py生成42条确定参数评分诊断；运行full_population.py补查末代全部4800个候选结构。
4. 依次运行summarize.py、plots.py、audit.py、report.py；最后运行deliver.py，核验完整种群分类与前沿/代表一致，并更新交付检查与本地文档入口。

## 重新优化

setup_inputs.py复制第一步数值源和固定规则；prepare.py生成六档目标；preflight.py核验；run.py smoke --workers 2与run.py timing --workers 2检查环境；run.py batch --workers 4运行正式24组。

run.py不会覆盖已经存在的smoke/timing/batch目录。独立复现实验应复制源码/目标到另一个实验目录并调整输出路径，保留已交付原始数据。不要删除本次结果来腾出复现目录。

源码和输入散列见SOURCE_MANIFEST.json；旧输入散列见OFFLINE_INPUT_HASHES.json；环境和运行摘要分别保存在batch/manifest.json、batch/status.json、batch/summaries.json。

本地全部使用Python评分；不加载第一步服务器Linux native.so，不连接服务器。

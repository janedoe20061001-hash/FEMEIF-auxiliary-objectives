# 物理状态诊断与候选假设

结果：[通俗报告](../../output/physical_state_diagnosis_20261011/REPORT.md)。本地运行，无优化器、服务器连接或网站操作。

从项目根目录依次运行：

```text
experiments/mse_structure_step2_20261009/.venv/Scripts/python.exe experiments/physical_state_diagnosis_20261011/run.py
experiments/mse_structure_step2_20261009/.venv/Scripts/python.exe experiments/physical_state_diagnosis_20261011/supplement.py
experiments/mse_structure_step2_20261009/.venv/Scripts/python.exe experiments/physical_state_diagnosis_20261011/plot.py
experiments/mse_structure_step2_20261009/.venv/Scripts/python.exe experiments/physical_state_diagnosis_20261011/catalog.py
experiments/mse_structure_step2_20261009/.venv/Scripts/python.exe experiments/physical_state_diagnosis_20261011/verify.py
```

PROTOCOL.md为原诊断约定，SUPPLEMENT_PROTOCOL.md明确是见到第一轮结果后的补充。run.py核验并使用原膝点种群的同一索引，不调用任何拟合优化。

两次补充诊断故障的源码和记录保留在output/physical_state_diagnosis_20261011/failures；完成版没有放宽门槛或删除失败样本。目标参数与电路图索引、通俗解释为人工整理，重跑脚本不覆盖这些说明。

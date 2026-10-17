# Case2：补齐 Case1 组合筛选并做独立 seed 验证

本目录包含两个预先固定、相互隔离的正式批次。

1. `config_pair_screen.json`：严格沿用 Case1 九候选标签，运行 baseline、9 个单项和全部 36 个两两组合；seed 0–99，共 4600 次。Case2 上已知 I-01=I-05，但为了回答“Case1 原九候选的完整组合能否迁移”，保留两个标签；报告不得把重复标签当成独立证据。
2. `config_validation.json`：对此前两套三指标集合使用全新的 seed 100–199，重新运行 baseline、5 个单项、6 个两两组合和 2 个三项组合；共 1400 次。这个批次不复用 seed 0–99 结果。

两批均使用 Case2 的 201 个原始测量点、15 元件 Fig.15 电路、种群 150、300 代、变异率 0.08 和 WSMSE 报告权重 17.38。同 seed 的全部组共享初始种群。WSMSE 只选择最终报告个体，不参与 NSGA-II 进化。

`report_results.py` 会重新计算每个最终种群的两个主目标、核对选解、检查同 seed 初始种群，生成配对 bootstrap、全组 Bonferroni 调整区间和组合相对组成项的区间。

正式运行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
screen -dmS case2-extended bash -lc 'cd /path/to/this/directory && bash run_all.sh > server.log 2>&1'
```

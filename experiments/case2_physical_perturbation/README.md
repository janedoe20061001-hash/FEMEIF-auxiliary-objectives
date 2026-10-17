# Case2 参数扰动与指标敏感性

Case2 没有真实元件参数，因此不能把 Case1 的真值扰动原样搬过来。本实验预先规定：从正式 baseline 的100个 WSMSE 代表解中选 WSMSE 最低者作为“拟合参考解”，再围绕它做乘性参数扰动。结论仅描述该拟合邻域，不把参考参数称为真值。

运行脚本同时评价18个中文PDF指标、英文论文五指标及两个主目标，共1765条谱。它应在 `case2_extended_pairs_validation` 的配对筛选报告完成后执行：

```bash
python run_study.py --baseline-selected ../case2_extended_pairs_validation/pair_screen_runs/selected_individuals.csv
```

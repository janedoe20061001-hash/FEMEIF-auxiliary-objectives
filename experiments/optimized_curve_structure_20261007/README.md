# 从优化代表解诊断阻抗曲线结构

本轮增加与既有单目标结果受控配对的双MSE进化组；后台分析全部seed的代表曲线与多参数有限变化的非加和剩余。PROTOCOL.md为运行前方案，DIAGNOSTIC_CLARIFICATIONS.md为分析前实现细化，NUMERICAL_GRID_NOTE.md记录网格检查暴露的问题和离线修复。

## 输入与运行

冻结输入来自 ../primary_loss_comparison_20261007：circuits.json、model.py、frozen目录、全部正式末代种群与已核验的两组代表解。新组不使用任何辅助目标，种群/300代/边界/算子/2300系列seeds完全一致。

服务器计算目录为 `/root/autodl-tmp/optimized_curve_structure_20261007`，沿用既有环境。新增1160次运行，总45060000次评价；全部成功。双目标最后保留非支配前沿，在其中按固定WSMSE规则选一个主要代表，并保存minmax等其他固定选解供敏感性检查。

```sh
python run_dual.py smoke --output smoke
python run_dual.py batch --workers 28 --output runs
```

初始种群与原WSMSE组逐元素相同；全部150200个双目标末代个体重算目标，并独立核对非支配集合、选解、参数边界、300代及评价次数。服务器回传包 `server_run/dual_results.tar.gz` 与远端SHA-256一致，清单见 server_run/ARCHIVES.json。

## 后处理

在工作区根目录使用包含numpy、scipy、matplotlib、markdown的Python环境。分析需要此前单目标的 output/selected 和 server_run/retrieved/runs，不能只用本轮优化包复现整个诊断。

```sh
python experiments/optimized_curve_structure_20261007/run_analysis.py --dual-runs experiments/optimized_curve_structure_20261007/server_run/retrieved/runs --output output/optimized_curve_structure_20261007_first_grid --workers 8
python experiments/optimized_curve_structure_20261007/refine_analysis.py --input output/optimized_curve_structure_20261007_first_grid --output output/optimized_curve_structure_20261007 --workers 8
python experiments/optimized_curve_structure_20261007/report.py
```

这些脚本拒绝覆盖已有分析目录。首次分析曾使用正式输出名，结束后完整移到first_grid，再进行数值网格修复；保留初次结果及检查记录。修复没有重新运行优化或改变原频点上的指标和选解。Case2无测点间真值，仅在全部201个测量点分析，不删除107°点。

主要结果目录包含：

- per_seed_diagnostics.csv：3480个三组主要代表解 + 1160个双目标minmax代表解；后者不是新增独立运行。
- band_diagnostics.csv：每个代表解八个等宽log频带。
- grid_refinement.csv：解析目标的两档加密差异和稳定标记，保留不稳定项。
- feature_matches.csv：按类型、方向、0.25 decades门限进行一对一匹配的记录。
- dual_selection_sensitivity.csv：同一前沿内五种固定选解的原目标误差。
- selected/*.npz：原频点上的参数、曲线、四项主误差和seed。密集曲线从这些选解与冻结网络重建，不批量保存图片。
- REPORT.md / index.html、少量figures和example页面：主要发现、典型曲线、结构及参数。

诊断项不直接作为新优化目标。峰谷门限1dB/5°、频移门限0.25decades，宽度用半prominence；幅值0.5/2dB敏感性已保留。明显误差样本用幅值RMSE>1dB或相位RMSE>5°界定，只用于避免把数值近零解当成强交互例子。极坐标距离贡献、峰谷匹配和非加和剩余均为描述性后处理。

## 查看和维护

通过工作区统一研究首页查看“优化曲线的结构诊断”。公开站点保持既有身份和访问范围；版本化离线包、启动、检查、发布和回退详见 ../../研究网站维护说明.md。发布流程不会上传服务器连接资料或全部原始运行包。

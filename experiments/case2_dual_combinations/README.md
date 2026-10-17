# Case2：迁移Case1的两套三指标集合

状态：800次新增运行已完成，0失败。报告、1400条选解记录和完整原始种群压缩包均已下载校验。原始包SHA256为`b3ce1a63b58d6e99b73b403bb7a3a41ca53d44b62ecaf4be67b3435e740d751a`。加权选解结果见server_run/RESULTS.md；不使用17.38的双主目标前沿结果见server_run/PARETO_WEIGHT_FREE.md。

集合A：I-03过零方向、II-01局部频移、II-02转变速度。
集合B：I-02转折覆盖、I-06中文PTC、II-01局部频移。

各集合新增3个两两组合和1个三结合，共8组×100 seeds（0–99）=800次正式运行。复用已完成的Case2 baseline及5个单指标，共600次；最终报告包含14组1400条结果。复用记录保持原始指纹，独立放在reference目录。

预算按论文Case2：150个体、300代，新增36000000次评估。全部201个原始测量点、15元件电路、NSGA-II、变异率0.08、WSMSE报告权重17.38及辅助公式均保持与Case2单指标实验一致。各组同seed的初始种群相同。WSMSE仅用于最终选解，进化使用两个主目标和该组各辅助目标。

判断分三层：

1. 八个组合相对Case2 baseline，是否同时降低幅值和相位MSE。配对bootstrap和16项比较调整区间见STABILITY.md。
2. 组合是否进一步优于各组成项。包含24对比较、48项误差区间，单独报告于CONSTITUENTS.md，不与baseline判断混淆。
3. 与Case1已有批次逐项对照，见CROSS_CASE.md。保留不同seed批次中成功与未成功的结果，比较各Case相对自身baseline的变化，不直接比较绝对误差。该范围仅覆盖两套集合，不能推断Case1所有结论已经迁移，也不能把差别单独归因于噪声。

预先固定的Case1对照摘要、哈希和来源指纹保存在case1_reference及SOURCE.json。集合A对照seed0–99和100–199；集合B对照seed0–99的三个两两组合及seed200–299的完整八组。

运行顺序（Linux，使用已安装锁定依赖的Python）：

```bash
python test_combinations.py
python experiment.py plan --output prepared
python prepare_reference.py --source /path/to/completed_case2_singles/runs
python launch_batch.py --run --workers 28 --output runs
python report_results.py runs
```

也可用已提供的run_server.sh --run衔接批次和报告；PYTHON环境变量可指定解释器。协调器在复用校验未通过时不会调度运行。代码/数据/预算/seed/运行库不一致或结果缺失会报错，避免混用。报告再次核验最终种群主目标与选中个体，打包runs和reference并生成SHA256。

软件检查使用8个体、3代、2个seed，产生28条临时运行记录；不计入正式统计。

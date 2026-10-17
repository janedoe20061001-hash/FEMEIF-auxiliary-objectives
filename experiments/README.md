# Experiment index

## Scalar primary-loss comparison — 2026-10-07

[中文报告](../archive/LOCAL_ONLY.md#ref-69c504fd40ba)、[逐电路图册](../archive/LOCAL_ONLY.md#ref-723b3419e0f3)与[复现入口](primary_loss_comparison_20261007/README.md)：WSMSE 与原始 Re/Im 平均欧姆距离分别作唯一进化目标，每 seed 按本组目标取末代最小值解，再跨 seeds 平均，交叉比较两项 MSE。两 Case 各 100 seeds，16 自设计拓扑×3 参数目标各 20 seeds，2,320 次、90,120,000 次评价全部完成；300,400 个末代个体重算通过。距离组在两 Case 降低欧姆距离，但两项 MSE 均增大；48 自设计电路中距离/WSMSE 两项均值同降分别为 18/27 个（含近零数值差异），结果不支持两种损失等价或一种普遍更优。

## Signed-phase optimization validation — 2026-10-07

[中文报告](../output/indicator_phase_validation_20261007/REPORT.md)与[复现入口](indicator_phase_validation_20261007/README.md)：用户提供新服务器并明确要求在预筛未过之后继续独立优化验证。双主目标、I-03＋II-02、I-03＋D_phi 三组；两 Case 各 20 个配对新种子（2200–2219），各 300 代，120 次、630 万次评价全部完成，21,000 个末代个体重算核验通过。候选相对原组合平均 WSMSE 在两 Case 分别增加 49.7%/32.9%，相位误差没有可靠改善，不支持直接替换 II-02。保留主值相位、全部测量频点与固定 WSMSE 选解。

## Indicator design probe — 2026-10-06

[预筛报告](../output/indicator_design_probe_20261006/REPORT.md)、[追加敏感性检查](../output/indicator_design_probe_20261006/SENSITIVITY.md)与[复现入口](indicator_design_probe_20261006/README.md)：重建前轮1,200个完整最终种群、200个共同初始种群，共245,000个个体。以两项主误差分别接近的15,078对曲线检查带方向相位割线D_phi能否识别更差的局部角度误差。Case1最终与Case2初始出现退步，候选未通过计算前固定门槛；新增优化为0。排序统计以种子为单位，敏感性分析标明事后追加，结论限于预定局部质量判据。旧服务器端点未确认SSH可用，预筛在本地完成。

## Parameter-to-curve population study — 2026-10-06

[中文总体报告](../archive/LOCAL_ONLY.md#ref-c6a9c5f55924)、[分条件规律与限制](../archive/LOCAL_ONLY.md#ref-4332505ee24e)、[图册](../archive/LOCAL_ONLY.md#ref-1b3be2327d48)与[复现入口](parameter_curve_population_20261006/README.md)：扩展至16种自设计拓扑、2,304个基准电路和207,360条扰动谱，两个独立抽样批次，12,828项数值检查通过。以基准电路为统计单位，按拓扑、元件连接位置与参数构造区间分析；解析恒等关系与抽样现象分别报告。全部本地计算，新增优化为0；统计结果覆盖本轮理想无源RLC参数空间，不能外推为任意电路的普适证明。

## Parameter-to-curve mechanisms — 2026-10-06

[中文机制报告](../archive/LOCAL_ONLY.md#ref-a590a410dcf7)、[逐参数速查](../archive/LOCAL_ONLY.md#ref-b4bd757b08bd)、[图册](../archive/LOCAL_ONLY.md#ref-fc778ab89dd6)与[复现入口](parameter_curve_mechanism_20261006/README.md)：本地分析5种自设计电路、Case1真值和Case2的3组拟合参考。覆盖单参数、整体缩放、整体频移、LC补偿及局部强弱方向，共1,932条有限扰动谱；保留模型变化与实测误差的区别，审查I-03、II-01、II-02与累计弧长。新增优化为0，未连接服务器。结论用于下一步指标机制审查，不能直接替换正式优化收益结论。

## Phase mechanism exploration — 2026-10-05

[相位改善能力报告](../output/相位改善能力探索_Case1机制与指标设计_20261005.md)与[复现入口](phase_mechanism_20261005/README.md)：用户要求以失败分析指导指标设计，本轮复核既有 1,200 次完整种群与旧 Case1 800 个选中解，新增优化为 0。给出同号角度误差、过零位置盲区及选解敏感性证据；带方向幅相变化仅列为待验证替换方向，已有有效组合继续保留。当前无新测量，外部验证暂缓。

## Latest cross-case consolidation — 2026-10-05

[统一研究报告](../output/近期研究统一总结_指标演变与验证_20261005.md)汇总六项数量先验的删选、七轮几何指标演变及改造前后效果，并新增服务器 1,200 次正式优化、40 项指标的同样本相关性和逐频点分析。新批次使用 seeds 2000–2099；累计弧长与 I-03+II-02 在 Case2 保持较大均值改善，但直接比较未分出双项主误差的胜者；T-01 暂不进入核心目标。

复现与原始结果入口：[本轮实验目录](crosscase_consolidation_20261005/README.md)。下方 Case2 推荐保留为前序实验结论，其模拟噪声/采样比例是已测试条件，不是普适使用阈值；新批次的统计结果以统一报告为准。

## Recommended entry points

- `CASE2_REMAINING_EXPERIMENTS_SUMMARY.md`：Case1 到 Case2 的全部迁移实验与最终结论。
- `case2_champion_validation/FINAL_METHOD.md`：论文可用的 motivation、idea、指标分层和最终方法。
- `case2_champion_validation/runs/RESULTS.md`：seeds 200–399 冠军复赛均值结果。
- `case2_champion_validation/runs/STABILITY.md`：配对 bootstrap 与多重比较校正。
- `case2_champion_validation/runs/CONSTITUENTS.md`：组合相对组成项的直接比较。
- `case2_champion_validation/runs/PARAMETER_STABILITY.md`：参数离散度和边界命中分析。
- `case2_robustness_validation/runs/RESULTS.md`：三档噪声、稀疏频点和连续缺失频段鲁棒性。
- `case2_robustness_all_finalists/runs/RESULTS.md`：十个入围组合在全部鲁棒性条件下的排名。
- `case2_robustness_fallback_validation/runs/RESULTS.md`：高噪声与连续缺段条件型候选的独立复验。
- `ARTIFACTS.sha256`：六个正式原始结果包的 SHA-256 清单。
- `case2_final_report_bundle.tar.gz`：不含大体积种群文件的论文、报告、配置和复现实验代码轻量包。

## Final recommendation

默认使用幅值 MSE、相位 MSE、I-03 和 II-02 进行四目标 NSGA-II；将 II-01、P-03 和 P-05 保留为独立诊断指标。已知高噪声约为 0.5 dB/2° 时，可切换到经独立验证的 P-03+P-05。使用前应确认频点覆盖完整且不少于约原采样的一半；关键频段连续缺失时退回基线或补测。不要重新加入 II-05 或 P-01，也不要把所有诊断指标同时扩展为 Pareto 目标。

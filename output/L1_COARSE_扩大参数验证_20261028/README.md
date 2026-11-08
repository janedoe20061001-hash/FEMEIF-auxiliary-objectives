> 2026-11-08 精选归档：以下为原日期研究记录；“active／下一步／完成”等是当时状态。现行结论见[研究现状](../../docs/研究现状_20261108.md)。此目录是核心阅读与复核材料，原运行包未全量附带；JSON中的历史路径/证书不是本次重跑证明。

# 扩大参数范围验证补充

见[完整报告](REPORT.md)和FINAL_ACCEPTANCE.json。原L1_COARSE实现未改，输入JSON不含真元件。运行 `python verify_replay.py` 复放24个预定代表。旧范围的验收证据不被本补充覆盖。

依赖NumPy与SciPy。候选数组每行为log10(R,L,C)，极点频率单位Hz；方法选择L1_COARSE。公式与原适用前提见[MATH_original_scope.md](MATH_original_scope.md)。9710个原始文件完整恢复，1783项冻结源码/输入未变；预算、选解和严格判决的本地重核全部通过。

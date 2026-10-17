# 第三步唯一有效的后处理入口：MAG/PHA膝点

结果：[通俗解释](../../output/mse_structure_step3_knee_20261011/FINDINGS.md)。定义见PROTOCOL.md，选法在selector.py。原优化代码保持冻结；其历史加权代表字段已停用。

从项目根目录使用`experiments/mse_structure_step2_20261009/.venv/Scripts/python.exe`运行：

```text
experiments/mse_structure_step3_knee_20261011/reselect.py pilot --workers 12
experiments/mse_structure_step3_knee_20261011/reselect.py revision --workers 12
experiments/mse_structure_step3_knee_20261011/reselect.py family --workers 12
```

这三个入口只重新选解、复核，不启动优化。每轮完整种群在新版output/source_populations目录。`--reuse`用于输入核验已完成且所有原文件哈希仍一致的后处理续跑。

三轮分别运行plot.py、extras.py，然后运行summarize.py重建数据汇总和REPORT.md。FINDINGS.md和ISSUES.md是人工解释，不能靠重跑覆盖。verify.py检查交付。cleanup.ps1只在新版验收通过后执行用户授权的旧结果清理，不由科学复算自动触发。

单条独立评分不一致会留在AUDIT.json和detail的原标记中；增加参考分辨率不将其改成通过。

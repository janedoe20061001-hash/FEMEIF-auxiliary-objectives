# 老师疑点验证，2026-10-11

服务器正式矩阵800次1000代，135,000,000评价。目标、算法和膝点规则见PROTOCOL.md；复现异常后的流程修改明确留存，不调整算法或数据。

1. run.py：服务器正式优化，full两Case三方法、Case2频段两方法，各100seed。
2. replay_probe.py：两个历史差异seed的原/新回调300代诊断，先后运行两批，共8次。
3. recover_two.py：保留首次失败，重新跑至1000代，300代必须与首次尝试逐位一致。
4. diagnose.py：服务器350,000条个体/阶段记录独立核验，16次有界最小二乘与曲线诊断。
5. finish_server.py：服务器来源与文件哈希、压缩归档。
6. retrieve.py：本地回收包SHA256与4,854个文件哈希核验。完整数据在server_run/retrieved/teacher_doubts_validation_20261011。
7. analyze.py：本地绘图与100seed配对bootstrap。没有本地正式优化。
8. VERDICT.md：基于结果的逐疑点判断；report.py生成报告、HTML和交付压缩包。

本地分析环境：experiments/mse_structure_step2_20261009/.venv/Scripts/python.exe；HTML生成使用tmp/curve_refinement_deps中的Markdown。服务器科学环境、冻结源与CSV哈希在SERVER_PROVENANCE.json。先前严格核验失败不是优化数值崩溃；398/400历史重放匹配，两个差异seed未剔除。

图和精修以实际阶段、真实膝点、额外预算明确标记。原96频点、旧程序和7月结果缺失，频段实验不冒充原论文完全复现。

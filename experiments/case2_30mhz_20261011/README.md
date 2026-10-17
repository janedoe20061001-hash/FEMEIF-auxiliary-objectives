# Case2拟合上限30MHz的服务器重跑

运行前设置见PROTOCOL.md。core.py、run.py沿用上轮冻结模型及算子，仅新增upto30m目标子集和独立输出目录。batch.py在服务器启动3方法各100seed、1000代优化。finish_server.py归档；retrieve.py核验归档SHA256与1,804个文件哈希。analyze.py独立重算120,000条个体/阶段记录，统一在183点评价旧全频训练代表，并绘制300/1000代目标叠图。report.py生成报告和交付压缩包。

完整数据在server_run/retrieved/case2_30mhz_20261011。没有本地正式优化，没有添加精修或辅助目标。两阶段预算分开报告；最高原测点29MHz，不补造30MHz测点。

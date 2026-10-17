# 新种子复现实验

本实验只改变两个Case的优化seed：2026100700–2026100799。400次正式运行全部完成，21,000,000次评价，服务器正式运行耗时78.13秒，失败0次。另有4次12个体/3代冒烟运行，不进入统计。

主报告：`../../output/seed_replication_20261007/REPORT.md`。新批原始结果独立保存，不覆盖旧批。没有生成或发布网站版本。

运行顺序：

1. `prepare.py` 从原单目标实验冻结优化输入和源码，保存来源SHA-256及运行前方案，生成上传包。
2. 在服务器既有虚拟环境运行 `run_replication.py smoke --output <独立目录>`，然后 `run_replication.py batch --workers 28 --output <独立目录>`。
3. 本次完整原始归档为 `server_run/results.tar.gz`，SHA-256为 `d95951bee3600cd4043def97dde2893be94a07724b0c76f2464a3c220538ee05`。`extract.py`核对哈希及源文件后解包；只在首次解包运行。
4. `analyze.py --runs server_run/retrieved/runs --output ../../output/seed_replication_20261007` 独立重算70,000个末代个体，并按预登记方法分析新旧seed区块。
5. `report.py` 生成本地Markdown、检查链接和表格，并更新输出SHA-256清单。

Windows本次使用Codex自带Python，并设置 `PYTHONPATH=D:/develop/FEMEIF/tmp/curve_refinement_deps` 和 `PYTHONIOENCODING=utf-8`。服务器Python/NumPy/SciPy/pymoo/moocore版本见原始包内 `runs/manifest.json`，资源检查见 `server_environment.json`。

原第一次服务器目录因Windows清单路径分隔符不兼容，在正式优化前中止；修复后实际目录为 `/root/autodl-tmp/seed_replication_20261007_v2`。它不构成第二批统计结果，没有更换正式种子或实验条件。

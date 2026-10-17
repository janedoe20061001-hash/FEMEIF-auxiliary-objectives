# 两个标量主目标：WSMSE 与平均欧姆距离

本轮是单目标直接对照。用户确认 D=mean|Z-Z*|，不归一化。设计见 PROTOCOL.md。

Case1/Case2 各 100 个配对新 seeds 2300–2399；16 种自设计拓扑×3 组独立参数=48 个电路，各 20 个配对 seeds 2300–2319。两组，无辅助项，300 代，总 2,320 次、90,120,000 次评价。

各组在每 seed 的末代种群按自身目标取最小值解，再跨 seeds 等权平均。该同一个解另计算两项主 MSE 和另一种综合指标。不能比较 WSMSE 数值与 Ω 数值谁小来判断优劣。

## 复现

依赖沿用上一轮锁定环境；服务器目录 `/root/autodl-tmp/primary_loss_comparison_20261007`，Python 为 `/root/autodl-tmp/indicator_phase_validation_20261007/.venv/bin/python`。所有工作进程的 BLAS/OpenMP 线程数为 1，28 个进程。

```sh
python run.py smoke --output smoke
python run.py batch --workers 28 --output runs
python analyze.py --runs runs --output analysis
```

`prepare.py` 依赖本地既有参数研究和冻结电路源文件，用于首次构建目标。复现优化无需重新生成目标，直接使用包内 circuits.json 与 frozen 目录；prepare 的公式检查结果已保存在 PREPARATION_CHECKS.json。程序拒绝覆盖正式任务目录，失败保留为 FAILED.json。

中文报告及逐电路页面：`../../output/primary_loss_comparison_20261007/REPORT.md`。服务器回传包、哈希清单和原始目录保存在本目录 server_run。`report.py` 只读取统计与预先生成的电路数据，画电路结构、列参数并生成报告，不改优化结果。

论文对照：英文论文 PDF 第 2 页式(2)与第 6 页表 V；本轮 Case2 使用全部 201 个已有测量点，非论文表内 96 点。自设计 alpha 为目标 dB 幅值跨度与 rad 相位跨度之比的平方。真值中心±2 decades 的合成边界及 240 点采样限制必须连同结果解释。

全部优化和末代核验通过，原始包为 server_run/results.tar.gz，下载哈希与服务器一致。第一遍分析仅在末尾哈希记录遇到路径类型错误，修正后完整重跑；两遍统计 SUMMARY 字节一致，记录见 EXECUTION_NOTE.md。此修复未改优化代码、目标、选解或种子。

图册文件为 output/primary_loss_comparison_20261007/index.html，离线可直接查看。原临时预览进程已停止；现在使用工作区根目录的“打开研究网站.cmd”启动可维护的本机预览，http://127.0.0.1:8787/ 自动跳转到研究首页下的目标对照页。外部读者使用托管公开网址或版本化离线包，详见 ../../研究网站维护说明.md。原浏览器检查确认 Case1、Case2 和桥式自设计电路、50 个选择项可用；此历史检查不代表预览进程始终在线。verify_artifacts.py 核对全部电路图元件名、102 张 PNG、报告链接和原始包 SHA。

# 2026-10-05 跨 Case 统一实验与研究汇总

状态：正式优化、全量最终种群复核、相关性分析、结果下载均已完成。共 1,200 次优化、6,300 万次目标评价；服务器目录 `/root/autodl-tmp/crosscase_consolidation_20261005`。

- [完整中文总结](../../output/近期研究统一总结_指标演变与验证_20261005.md)：删选依据、历史路线、父子直接比较、相关性、Case 差异、负结果与下一步判断。
- [启动前协议](PROTOCOL.md)：六组方案、种子 2000–2099、60 项统一 Holm 校正和不改主目标的约束。
- `server_run/review_results/`：已解压的冻结代码、配置、预检、选中解曲线、统计和审计。这里是实际服务器版本的复现来源。
- `server_run/formal_results.tar.gz`：完整原始包，含 1,200 个初始/最终种群及逐代汇总；并非保存了全部中间个体的 6,300 万次评价明细。
- `server_run/review_results.tar.gz`：轻量代码与分析包。
- `server_run/ARCHIVES.json`、`server_run/download_audit.json`：服务器清单、下载 SHA256 与文件计数核验。

六组为双主目标基线、I-03+II-01、I-03+II-02、R-05-C、T-01、G-01-LP。Case1 每次种群 200、300 代；Case2 种群 150、300 代。两项主误差均取自按原固定 WSMSE 选出的同一个体。

本轮决定：保留局部组合与原累计弧长作为工作对照；T-01 没有足够增量收益，不加入核心目标。历史各轮的显著性保留其原检验口径，新轮使用全部直接比较的 60 项校正，不能混成一个跨轮排名。

## 复现入口

在 Linux 独立目录解压完整原始包后，用 Python 3.12 创建环境并按包内根目录的 `requirements-lock.txt` 安装固定依赖。保留原包用于核验；重算统计可运行 `run_analysis.py`（会重写该副本的分析产物，需要 GCC 编译方向指标的 C 后端）。软件检查和正式运行入口分别为 `preflight_case.py`、`run_formal.py --run --workers 28`；后者读取 `PREPARED.json` 核验源文件并冻结 `FROZEN.json`，已有运行会按原批处理器的完成状态跳过，因此原包上重启不等于产生新独立实验。新实验应另开目录和协议。

报告重建：在项目根目录使用含 NumPy/Matplotlib/SciPy 的环境，依次运行 `scripts/collect_history_20261005.py` 和 `scripts/build_consolidated_report_20261005.py`。前者读取七轮旧 JSON；后者读取已核验的新分析、生成 CSV 汇总和五张科研图，并写中文 Markdown。报告生成无需再连接服务器或优化。

本轮没有创建继续运行的自动任务，所有实验及分析已结束。

# 相位斜率指标优化验证

运行前设计见 [PROTOCOL.md](PROTOCOL.md)。新增种子 2200–2219，两 Case、三组，合计 120 次、6,300,000 次评价。预筛未过，按用户 2026-10-07 的明确指令运行；旧预筛不变。

服务器目录：`/root/autodl-tmp/indicator_phase_validation_20261007`。独立 venv 与冻结依赖，28 个单线程工作进程。SSH 密码仅通过私密提示输入，不保存到实验文件。

## 复现

在全新目录解包 bundle.tar.gz 后安装 frozen/case1/requirements-lock.txt。运行前核验：

```sh
python run.py smoke --case case1 --output smoke/case1
python run.py smoke --case case2 --output smoke/case2
python run.py batch --workers 28 --output runs
python analyze.py --runs runs --output analysis
```

程序拒绝覆盖已有运行目录。小种群检查不计入正式统计。正式结果保留初始种群、末代种群、每代历史、选解和源代码哈希。分析程序独立重算候选割线公式以及原指标与主误差，核对全部 21,000 个末代个体、配对初始化及预算。

原始运行与分析归档放在本目录 server_run；中文报告在 ../../output/indicator_phase_validation_20261007/REPORT.md。

全部 120 次正式运行成功，末代 21,000 个体独立重算通过。服务器原始结果包 SHA-256 记录在 server_run/ARCHIVES.json，本地回传后检查字节哈希与冻结代码一致。finalize.py 为报告补充结果解读，服务器原始分析报告保留在归档内。

结果不支持用 D_phi 替换 II-02：候选相对原组合的平均 WSMSE 在 Case1/Case2 分别增加 49.7%/32.9%；相位均值增加但置信区间跨零。候选保持诊断用途，未继续改公式或追加调参。

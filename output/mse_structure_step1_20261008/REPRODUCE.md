# 维护和复算说明

主报告：[REPORT.md](REPORT.md)。所有材料只保存在本地，本轮没有发布网站。

## 已保存的材料

实验目录：[experiments/mse_structure_step1_20261008](../../experiments/mse_structure_step1_20261008)。

- `server_run/training_results.tar.gz`：200次正式运行，初始化、每20代两种代表、300/1000代全种群与前沿、训练历史、配置、编译内核与源码校验。
- `server_run/analysis_results.tar.gz`：全部阶段/前沿的逐事件和逐频带CSV、网格稳定核验、分析源码散列及环境记录。
- `server_run/quadrature_results.tar.gz`：4万个自适应保留个体的独立积分核验、每seed结果、核验脚本与运行记录。
- `server_run/frozen_inputs.tar.gz`：与正式训练、分类校验值对应的完整输入和代码，包含100份旧300代对照、目标网格、编译内核和复算脚本。无需依赖更早实验的可变源文件。

下载包已经在本地验证SHA-256。主报告引用的CSV、精确参数、JSON和图片都有[MANIFEST.json](MANIFEST.json)校验记录。文件名日期保持启动日20261008；报告整理完成于次日，不因此重命名原始运行目录。

不在这些文件中保存SSH密码。服务器上的中断性能试运行保留作审计，不与正式200次比较结果混用。

## 不重跑优化，怎样重新生成统计？

在项目根目录使用具备numpy、scipy、matplotlib的Python环境：

```text
python experiments/mse_structure_step1_20261008/summarize.py
python experiments/mse_structure_step1_20261008/detail_stats.py
python experiments/mse_structure_step1_20261008/plot_results.py
python experiments/mse_structure_step1_20261008/package_results.py
```

这些脚本读取本地已解包的`server_run/retrieved`，生成汇总和固定seed图片，保留代表规则。第一条会打印较长JSON，可以重定向到日志；不应把日志当作额外独立证据。Markdown解释文字经过人工核对，不会被脚本自动改写。

如果换一台机器，需要按脚本中的`HERE`目录结构放置本地解包结果；Markdown中的绝对文件链接与图片路径需要相应替换。数值结果和归档本身不依赖这些显示路径。

## 完整重跑的入口

把`frozen_inputs.tar.gz`解压到新的空目录，使用记录中的Linux数值环境。`native.so`是Linux二进制，Windows不会直接运行它。重编译后的二进制SHA可能依编译环境变化；改变编译器或数值库时应重做预检并冻结新的运行记录。

训练入口`run.py batch --output runs --workers 24`要求`SOURCES.json`每个输入散列匹配、预检通过，并要求输出目录尚不存在。先用`run.py smoke --output smoke_new --workers 2`做小规模运行，验证环境，不覆盖旧数据。

分类入口`classify_v2.py --runs runs --output analysis --workers 24`核对独立的`ANALYSIS_SOURCES.json`。积分入口`check_quadrature.py --runs runs --output quadrature --workers 8`；`check_quadrature_v2.py`仅增加已完成seed续算与原子写入能力，数值计算函数一致。换源码或输入时不能续用旧的分析分块或积分记录，应另开目录。

注意：本地重新执行`preflight.py`会更新`PREFLIGHT.json`中的耗时等记录，随后必须核对结果并重新冻结其SHA；不能为了绕过源文件检查直接删掉核验。`SOURCES.json`包括编译内核和BUILD记录；Python评分与C评分容差内一致，不保证自适应优化逐代路径完全相同。

## 后续实验的约束

保留本轮结果，下一轮另开目录、冻结新方案和评分口径。第一步不再无边界扩大采样或预算；下一步应有控制地研究困难条件。任何辅助指标的公式和权重，在使用其验证结果调整后，都不能继续声称该验证集未参与开发。

四个主线问题始终是：哪类结构反复困难；哪个条件加重困难；指标能否改善；改善能否推广且代价如何。没有明确发布要求，不更新网站。

# WSMSE主目标基础检查与双目标膝点对照

2026-10-11。按用户要求在服务器运行Case1/Case2，各100个seed，300代。双目标主结果来自已完成的相同seed历史末代完整种群，按膝点重新选解；另重放6次双目标确认一致。未更改目标、边界、采样和算子，未发布网站。

实验配置与膝点规则见 `PROTOCOL.md`。优化入口 `check.py --workers 28`。服务器工作区为 `/root/autodl-tmp/wsmse_teacher_check_20261011_workspace`；使用已有 `/root/autodl-tmp/femeif_nested_20261010_01/.venv/bin/python`。输入及源文件哈希在 `INPUT_MANIFEST.json`，服务器结果包与下载核验在 `server_run/`。密码只经隐藏标准输入进入SSH进程内存，不保存。

本地分析使用 `experiments/mse_structure_step2_20261009/.venv/Scripts/python.exe`，科学依赖与服务器版本一致。报告HTML的Markdown解析模块来自已有 `tmp/curve_refinement_deps`，不参与优化或误差计算。

结果位于 `output/wsmse_teacher_check_20261011/REPORT.md` 与 `index.html`。保存两Case最佳曲线叠图、复阻抗图、固定3seed配对图、膝点前沿图、600行逐seed交叉指标、曲线/参数CSV及逐seed膝点选择记录。`SUMMARY.json` 包含按seed配对的bootstrap区间；主图最佳seed使用统一WSMSE排名，但双目标候选必须先经过每seed膝点选解。

复核/重建报告：

```powershell
& 'experiments/mse_structure_step2_20261009/.venv/Scripts/python.exe' experiments/wsmse_teacher_check_20261011/analysis.py
& 'experiments/mse_structure_step2_20261009/.venv/Scripts/python.exe' experiments/wsmse_teacher_check_20261011/finalize.py
```

首次服务器预检因包中漏带根目录CSV而中止，无优化任务执行；补传同哈希原文件后预检及206次运行成功。最初临时本地依赖目录尚在安装时导入失败；正式优化始终在服务器完成，分析改用工作区已有完整虚拟环境。临时依赖安装不影响原实验环境或科学结果。下载哈希核验脚本曾多录入一个字符，重新与服务器SHA256核对修正后，全包及1006个文件哈希通过。原始服务器首次与重试日志均留存。

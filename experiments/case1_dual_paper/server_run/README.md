# 服务器正式实验

已于 2026-09-13 启动正式实验，非测试。当前目录中的 progress.json 是拉取时的快照，不会自动刷新。

- SSH：root@connect.westb.seetacloud.com，端口 45370。
- 远程目录：/root/autodl-tmp/femeif_case1_dual_paper_20260913
- 配置：32 组，100 seeds，200 个体，300 代，28 个工作进程。
- 后台协调进程启动 PID：1436；以远程 runs/progress.json 中的实时状态为准。
- 代码与配置的本地/服务器哈希已核对；实际环境指纹保存在 manifest.json。
- 服务器端 9 项检查通过，1 项本地历史数据对照因未上传该数据而跳过；并行短流程 32/32 完成。

远程查看状态：

~~~bash
cd /root/autodl-tmp/femeif_case1_dual_paper_20260913
cat runs/progress.json
tail -20 batch.log
~~~

正式任务由 nohup 在后台运行，断开 SSH 不会中断。
实验结束后会自动生成：

- runs/paired_summary.csv：与基线按 seed 配对的双主目标统计。
- runs/RESULTS.md：分组结果，以及三项组合和所有两项子组合的筛选。
- runs/comparison.png：两项主目标相对基线的变化图。
- runs/分组/seed_编号/：该次实验的初始种群、逐代历史、最终参数与目标、完成标记。

服务器后台仅执行计算和结果生成；本地结果回传需要再次通过 SSH/SFTP 拉取。
不要把 server_smoke 目录下的结果当作正式结果。

重启协调程序时，已完整完成且指纹一致的运行会被跳过；发现半途留下的目录会停止并要求先检查。
不要同时启动重复批次；协调器持有文件锁。

本目录不保存 SSH 密码。

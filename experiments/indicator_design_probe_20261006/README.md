# 带方向相位指标预筛（2026-10-06）

计算前协议见 `PROTOCOL.md`。主候选D_phi采用原频率点、主值相位、带方向相邻割线；D_LP只作诊断。通过主误差接近的个体对，比较候选和旧速度II-02对局部角度误差的排序。质量判据由原频带8个等log区间的最大相位RMSE定义，不含导数。

输入只读使用 `crosscase_consolidation_20261005/server_run/formal_results.tar.gz` 和轻量包的选中解曲线。完整包用流式读入，不在旧目录解压或修改。旧selected_curves.npz中名为parameters的字段实际上存log10参数；本脚本重建并核对保存曲线和原主误差。

运行：使用项目附带Python运行 `probe.py`，随后运行 `verify_probe.py`。SciPy通过已有 `tmp/curve_refinement_deps` 加载；筛查不依赖优化器。结果位于 `output/indicator_design_probe_20261006`。

报告解释预设门槛是否通过。排序正确率按种子汇总，不能把数万个配对当成数万个独立实验。主候选、配对容差、分窗方式与门槛不会按结果调整。通过筛查不保证新优化收益，未通过也不能排除所有带方向指标。

服务器只检查项目现存SSH端点；TCP可达不等于服务器开机或SSH可用。密码不写入文件。

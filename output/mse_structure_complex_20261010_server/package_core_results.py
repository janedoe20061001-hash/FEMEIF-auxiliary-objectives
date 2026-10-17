"""Package all raw optimization data without duplicate/reconstructible plot arrays."""
import hashlib
import json
from pathlib import Path
import time
import zipfile
ROOT = Path(__file__).resolve().parent
state_path = ROOT/"PIPELINE_STATUS.json"
state = json.loads(state_path.read_text())
assert state["status"]=="complete" and state["other_supplemental_audit_passed"]
assert json.loads((ROOT/"analysis_other/AUDIT.json").read_text())["passed"]
assert json.loads((ROOT/"AUDIT_REFINEMENT_other.json").read_text())["final_audit_passed"]
full = dict(path=state["archive_other"],sha256=state["archive_other_sha256"])
policy = """# 回收包包含什么

本包保留全部120个seed的原始优化数据：初始种群、每20代保存的完整种群参数及训练分数、选解索引、前沿、运行历史、每seed预算和文件哈希。

保留全部解析目标、目标曲线、结构参数、冻结计算代码、逐候选独立诊断、逐seed统计、原始失败表、加密后的表、补充复核和并行处理证明；最终幅值/相位图片全部保留。

仅省略分析目录里可从参数与冻结代码重建的 stage_*_3d_data.npz 绘图导出，以及原始网格版本重复的图片。全量原包和这些派生导出仍在服务器保留，未删除。f、Re Z、Im Z 可用保存的参数调用 circuits.Circuit.response，或重新调用冻结的 analyze.plot_stage 生成。

这些目标来自理论电路，绘图数组是计算的派生结果，不是测量原始数据。本次打包没有丢弃任何优化seed或保存阶段，也没有改变任何统计值。
"""
(ROOT/"ARCHIVE_POLICY.md").write_text(policy,encoding="utf-8")
(ROOT/"FULL_ARCHIVE_OTHER.json").write_text(json.dumps(full,indent=2))
names = ["batch_other","analysis_other","analysis_other_original_grid","analysis_other_serial_partial","inputs_core","mse_structure_complex_20261010","SERVER_ENVIRONMENT.json","VALIDATION.json","PIPELINE_STATUS.json","PARALLEL_ANALYSIS.json","AUDIT_REFINEMENT_other.json","REFINEMENT_TRACE_other.json","parallel_analysis.py","refine_parallel_analysis.py","trace_unstable.py","inspect_audit.py","resume_pipeline.py","package_core_results.py","ARCHIVE_POLICY.md","FULL_ARCHIVE_OTHER.json","parallel_analysis.log","refine_parallel_analysis.log","refine_parallel_analysis_retry.log","trace_unstable.log"]
dest = ROOT/"results_other_core.zip"
omitted = []
with zipfile.ZipFile(dest,"x",compression=zipfile.ZIP_DEFLATED) as z:
    for name in names:
        p = ROOT/name
        files = sorted(p.rglob("*")) if p.is_dir() else [p]
        for file in files:
            if not file.is_file() or "__pycache__" in file.parts:
                continue
            rel = file.relative_to(ROOT)
            if name.startswith("analysis_") and (file.name.endswith("_3d_data.npz") or name=="analysis_other_original_grid" and file.suffix==".png"):
                omitted.append(rel.as_posix())
                continue
            z.write(file,rel.as_posix())
digest = hashlib.sha256(dest.read_bytes()).hexdigest()
(ROOT/"results_other_core.sha256").write_text(digest+"\n")
state.update(archive_other=str(dest),archive_other_sha256=digest,full_archive_other=full,core_archive_omits_only_derived_plots=True,updated_unix=time.time())
state_path.write_text(json.dumps(state,indent=2))
print(json.dumps(dict(bytes=dest.stat().st_size,sha256=digest,omitted_reconstructible_files=len(omitted))))

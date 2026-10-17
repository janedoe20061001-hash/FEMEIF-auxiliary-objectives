"""Freeze second-round sources while retaining all preceding raw results."""
from pathlib import Path
import hashlib,json,shutil,tarfile,sys
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'experiments/curve3d_logpolar_20261003'
PREVIOUS=ROOT/'experiments/curve3d_objectives_20261003'
shutil.copy2(PREVIOUS/'global_metrics.py',DEST/'curve_math.py')
sys.path.insert(0,str(DEST))
from global_metrics import AUX
for case in ['case1','case2']:
    source=PREVIOUS/case;target=DEST/case
    if (target/'runs').exists():raise RuntimeError('Never rebuild a running protocol')
    target.mkdir(parents=True,exist_ok=True)
    for p in source.iterdir():
        if p.is_file() and p.suffix in ['.py','.json','.txt']:shutil.copy2(p,target/p.name)
    for name in ['global_metrics.py','curve_math.py']:shutil.copy2(DEST/name,target/name)
    if case=='case2':shutil.copytree(source/'data',target/'data',dirs_exist_ok=True)
    cfg=json.loads((source/'config.json').read_text(encoding='utf-8'))
    cfg.update(groups=[[]]+[[name] for name in AUX],protocol='curve3d_logpolar_20261003_v1')
    (target/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    text=(source/'experiment.py').read_text(encoding='utf-8').replace('"global_metrics.py","legacy_objectives.py",','"global_metrics.py","curve_math.py","legacy_objectives.py",')
    (target/'experiment.py').write_text(text,encoding='utf-8')
    provenance={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}
    (target/'ROUND_SOURCE.json').write_text(json.dumps(dict(source=str(source.relative_to(ROOT)),sha256=provenance),indent=2),encoding='utf-8')
shutil.copy2(PREVIOUS/'audit_case.py',DEST/'audit_case.py')
runner=(PREVIOUS/'run_formal.py').read_text(encoding='utf-8').replace('completed_runs=2200','completed_runs=1400').replace('evaluations=115500000','evaluations=73500000')
(DEST/'run_formal.py').write_text(runner,encoding='utf-8')
report=(PREVIOUS/'finish_report.py').read_text(encoding='utf-8')
report=report.replace('holm_adjusted_p_40_primary_comparisons','holm_adjusted_p_24_primary_comparisons')
report=report.replace('三维曲线指标：Case I / Case II 正式实验结果','三维对数幅值—相位坐标：第二轮正式实验结果')
report=report.replace('各 11 组','各 7 组').replace('2,200 次运行、115,500,000','1,400 次运行、73,500,000')
report=report.replace('10 辅助指标 × 2 主目标共 40','6 辅助指标 × 2 主目标共 24')
report=report.replace('五个几何方向各比较两种坐标。','五个既有几何方向改用对数幅值—相位坐标，另加全频率对方向差。')
report=report.replace('两种坐标在实验前冻结，不能把事后选出的最好坐标视为独立验证。','本轮在看到第一轮负结果后提出，定义在本轮启动前冻结，但不属于独立数据验证。')
(DEST/'finish_report.py').write_text(report,encoding='utf-8')
archive=ROOT/'tmp/curve3d_logpolar_20261003_code.tar.gz'
with tarfile.open(archive,'w:gz') as tf:
    for p in sorted(DEST.rglob('*')):
        if p.is_file() and not any(v in p.parts for v in ['__pycache__','runs','server_run','prepared']):tf.add(p,arcname=str(p.relative_to(DEST)))
print(archive);print(hashlib.sha256(archive.read_bytes()).hexdigest())

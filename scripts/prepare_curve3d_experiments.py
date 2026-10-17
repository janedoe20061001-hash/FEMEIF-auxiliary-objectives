"""Build a frozen full-budget 3D protocol without overwriting existing runs."""
from pathlib import Path
import hashlib,json,shutil,tarfile,sys
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'experiments/curve3d_objectives_20261003'
PREVIOUS=ROOT/'experiments/global_objectives_20261003'
sys.path.insert(0,str(DEST))
from global_metrics import AUX
shared=['experiment.py','launch_batch.py','summarize.py','final_selection.py',
        'report_results.py','requirements.txt','requirements-lock.txt']
for case in ['case1','case2']:
    source=ROOT/f'experiments/{case}_dual_pdf18';target=DEST/case
    if (target/'runs').exists():raise RuntimeError('Never rebuild a running protocol')
    target.mkdir(parents=True,exist_ok=True)
    files=shared+(['case2.py','stability.py','analysis_config.json'] if case=='case2' else [])
    for name in files:shutil.copy2(source/name,target/name)
    shutil.copy2(source/'objectives.py',target/'legacy_objectives.py')
    shutil.copy2(DEST/'global_metrics.py',target/'global_metrics.py')
    if case=='case2':
        (target/'data').mkdir(exist_ok=True)
        shutil.copy2(source/'data/targetImpedance_case2.csv',target/'data/targetImpedance_case2.csv')
    cfg=json.loads((source/'config.json').read_text(encoding='utf-8'))
    cfg.update(features={},groups=[[]]+[[name] for name in AUX],protocol='curve3d_objectives_20261003_v1')
    (target/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    text=(source/'experiment.py').read_text(encoding='utf-8')
    text=text.replace('"experiment.py","objectives.py",','"experiment.py","objectives.py","global_metrics.py","legacy_objectives.py",')
    text=text.replace('Feature constants must match the PDF definitions','Frozen 3D definitions have no tunable feature constants')
    (target/'experiment.py').write_text(text,encoding='utf-8')
    shutil.copy2(PREVIOUS/case/'objectives.py',target/'objectives.py')
    provenance={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in files+['objectives.py','config.json']}
    (target/'SOURCE.json').write_text(json.dumps({'source':str(source.relative_to(ROOT)),'sha256':provenance},indent=2),encoding='utf-8')
shutil.copy2(PREVIOUS/'audit_case.py',DEST/'audit_case.py')
runner=(PREVIOUS/'run_formal.py').read_text(encoding='utf-8').replace('completed_runs=800','completed_runs=2200').replace('evaluations=42000000','evaluations=115500000')
(DEST/'run_formal.py').write_text(runner,encoding='utf-8')
report=(PREVIOUS/'finish_report.py').read_text(encoding='utf-8')
report=report.replace('holm_adjusted_p_12_primary_comparisons','holm_adjusted_p_40_primary_comparisons')
report=report.replace('新全局指标：Case I / Case II 正式实验结果','三维曲线指标：Case I / Case II 正式实验结果')
report=report.replace('各 4 组','各 11 组').replace('800 次运行、42,000,000','2,200 次运行、115,500,000')
report=report.replace('3 辅助指标 × 2 主目标共 12','10 辅助指标 × 2 主目标共 40')
report=report.replace('G-01 与 G-02 来自同一累积残差思路；G-05 是简单相对复误差对照。单辅助结果不能推断组合效果。','五个几何方向各比较两种坐标。所有组保留双主目标并仅添加一个辅助目标；单辅助结果不能推断组合效果。')
report=report.replace('未进行事后阈值、积分窗口或权重调优。','未进行事后阈值、积分窗口或权重调优；两种坐标在实验前冻结，不能把事后选出的最好坐标视为独立验证。')
(DEST/'finish_report.py').write_text(report,encoding='utf-8')
archive=ROOT/'tmp/curve3d_objectives_20261003_code.tar.gz'
with tarfile.open(archive,'w:gz') as tf:
    for p in sorted(DEST.rglob('*')):
        if p.is_file() and not any(v in p.parts for v in ['__pycache__','runs','server_run','prepared']):tf.add(p,arcname=str(p.relative_to(DEST)))
print(archive);print(hashlib.sha256(archive.read_bytes()).hexdigest())

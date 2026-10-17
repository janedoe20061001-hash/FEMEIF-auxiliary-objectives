"""Prepare files only. Scientific preflight and all optimization run on server."""
from pathlib import Path
import shutil,json,tarfile,hashlib

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'experiments/curve3d_interval_20261003'
DST=ROOT/'experiments/crosscase_consolidation_20261005'
if (DST/'PREPARED.json').exists():
    raise RuntimeError('Already prepared')
groups=[[],['reactance_crossing_direction_consistency','characteristic_frequency_shift_error_decades'],
        ['reactance_crossing_direction_consistency','transition_speed_relative_error'],
        ['cumulative_length_logpolar'],['all_interval_mean_residual_logpolar'],['bidirectional_residual_logpolar']]
for case in ['case1','case2']:
    target=DST/case;target.mkdir()
    for p in (SRC/case).iterdir():
        if p.is_file() and p.suffix in ['.py','.txt']:
            shutil.copy2(p,target/p.name)
    if case=='case2':
        shutil.copytree(SRC/case/'data',target/'data')
        shutil.copy2(SRC/case/'analysis_config.json',target/'analysis_config.json')
    shutil.copy2(SRC/'global_metrics.py',target/'length_metrics.py')
    for name in ['global_metrics.py','trend_kernel.py']:
        shutil.copy2(DST/name,target/name)
    cfg=json.loads((SRC/case/'config.json').read_text())
    cfg.update(seeds=dict(start=2000,count=100),groups=groups,protocol='crosscase_consolidation_20261005_v1')
    (target/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    # Include the added scientific dependencies in each run's source fingerprint.
    p=target/'experiment.py';s=p.read_text(encoding='utf-8')
    s=s.replace('"experiment.py","objectives.py","global_metrics.py",',
                '"experiment.py","objectives.py","global_metrics.py","length_metrics.py","trend_kernel.py","trend_kernel.npy",')
    p.write_text(s,encoding='utf-8')
files={p.relative_to(DST).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
       for p in sorted(DST.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
(DST/'PREPARED.json').write_text(json.dumps(files,indent=2))
archive=ROOT/'tmp/crosscase_20261005_code.tar.gz'
with tarfile.open(archive,'w:gz') as tf:
    for p in DST.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:
            tf.add(p,arcname=p.relative_to(DST).as_posix())
print(json.dumps(dict(archive=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))

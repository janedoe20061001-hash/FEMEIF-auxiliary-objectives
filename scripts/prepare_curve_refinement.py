"""Prepare an immutable new protocol; never touch earlier result files."""
from pathlib import Path
import hashlib,json,shutil,tarfile,sys
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'experiments/curve3d_logpolar_native_20261003'
DEST=ROOT/'experiments/curve3d_refinement_20261003'
if (DEST/'FROZEN.json').exists():raise RuntimeError('Protocol already frozen')
shutil.copy2(SOURCE/'curve_math.py',DEST/'curve_math.py')
sys.path.insert(0,str(DEST))
from global_metrics import AUX
for case in ['case1','case2']:
    src=SOURCE/case;dst=DEST/case;dst.mkdir(exist_ok=True)
    for p in src.iterdir():
        if p.is_file() and p.suffix in ['.py','.json','.txt'] and p.name not in ['native_backend.py','build_native.json']:
            shutil.copy2(p,dst/p.name)
    for name in ['global_metrics.py','curve_math.py']:shutil.copy2(DEST/name,dst/name)
    if case=='case2':shutil.copytree(src/'data',dst/'data')
    cfg=json.loads((src/'config.json').read_text(encoding='utf-8'))
    cfg.update(seeds=dict(start=100,count=100),groups=[[]]+[[name] for name in AUX],protocol='curve3d_refinement_20261003_v1')
    (dst/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    engine=(dst/'experiment.py').read_text(encoding='utf-8')
    engine=engine.replace(',"native_backend.py","orientation_native.c","orientation_native.so","build_native.json"','')
    (dst/'experiment.py').write_text(engine,encoding='utf-8')
    (dst/'ROUND_SOURCE.json').write_text(json.dumps(dict(source=str(src.relative_to(ROOT)),
        sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in src.iterdir() if p.is_file()}),indent=2),encoding='utf-8')
audit=(SOURCE/'audit_case.py').read_text(encoding='utf-8')
audit=audit.replace('for seed in range(100):','for seed in range(100,200):').replace('for s in range(100)','for s in range(100,200)')
(DEST/'audit_case.py').write_text(audit,encoding='utf-8')
shutil.copy2(SOURCE/'run_formal.py',DEST/'run_formal.py')
files={p.relative_to(DEST).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DEST.rglob('*'))
       if p.is_file() and '__pycache__' not in p.parts}
(DEST/'FROZEN.json').write_text(json.dumps(dict(protocol='curve3d_refinement_20261003_v1',files_sha256=files),indent=2),encoding='utf-8')
archive=ROOT/'tmp/curve3d_refinement_20261003_code.tar.gz'
with tarfile.open(archive,'w:gz') as tf:
    for p in sorted(DEST.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:tf.add(p,arcname=str(p.relative_to(DEST)))
print(archive);print(hashlib.sha256(archive.read_bytes()).hexdigest())

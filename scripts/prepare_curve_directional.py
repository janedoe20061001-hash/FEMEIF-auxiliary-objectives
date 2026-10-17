"""Prepare the fourth version without rewriting any earlier protocol."""
from pathlib import Path
import hashlib,json,shutil,tarfile,sys
root=Path(__file__).resolve().parents[1];src=root/'experiments/curve3d_refinement_20261003';dst=root/'experiments/curve3d_directional_20261003'
if (dst/'FROZEN.json').exists():raise RuntimeError('Already frozen')
shutil.copy2(src/'curve_math.py',dst/'curve_math.py')
sys.path.insert(0,str(dst));from global_metrics import GROUPS
for case in ['case1','case2']:
    a,b=src/case,dst/case;b.mkdir(exist_ok=True)
    for p in a.iterdir():
        if p.is_file() and p.suffix in ['.py','.json','.txt']:shutil.copy2(p,b/p.name)
    if case=='case2':shutil.copytree(a/'data',b/'data')
    shutil.copy2(dst/'global_metrics.py',b/'global_metrics.py')
    cfg=json.loads((a/'config.json').read_text(encoding='utf-8'))
    cfg.update(seeds=dict(start=200,count=100),groups=[list(g) for g in GROUPS],protocol='curve3d_directional_20261003_v1')
    (b/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    engine=(b/'experiment.py').read_text(encoding='utf-8')
    engine=engine.replace('if not isinstance(g,list) or len(g)>1 or any(a not in AUX for a in g):',
        'if not isinstance(g,list) or tuple(g) not in GROUPS:')
    engine=engine.replace('from objectives import AUX,','from global_metrics import GROUPS\nfrom objectives import AUX,')
    engine=engine.replace('This protocol allows baseline or exactly one auxiliary','Only the frozen baseline, single auxiliaries and one declared pair are allowed')
    engine=engine.replace('explicit singleton lists','explicit frozen group lists')
    (b/'experiment.py').write_text(engine,encoding='utf-8')
    (b/'ROUND_SOURCE.json').write_text(json.dumps(dict(source=str(a.relative_to(root)),
        sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.iterdir() if p.is_file()}),indent=2),encoding='utf-8')
audit=(src/'audit_case.py').read_text(encoding='utf-8').replace('range(100,200)','range(200,300)')
(dst/'audit_case.py').write_text(audit,encoding='utf-8')
shutil.copy2(src/'run_formal.py',dst/'run_formal.py')
files={p.relative_to(dst).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(dst.rglob('*'))
       if p.is_file() and '__pycache__' not in p.parts}
(dst/'FROZEN.json').write_text(json.dumps(dict(protocol='curve3d_directional_20261003_v1',files_sha256=files),indent=2),encoding='utf-8')
archive=root/'tmp/curve3d_directional_20261003_code.tar.gz'
with tarfile.open(archive,'w:gz') as tf:
    for p in sorted(dst.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:tf.add(p,arcname=p.relative_to(dst).as_posix())
print(archive);print(hashlib.sha256(archive.read_bytes()).hexdigest())

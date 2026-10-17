from pathlib import Path
import json,hashlib,shutil,tarfile,py_compile
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];SOURCE=ROOT/'experiments/primary_loss_comparison_20261007';DEST=HERE/'frozen_scalar';DEST.mkdir(exist_ok=True)
for name in ['run.py','model.py','circuits.json','prepare.py','PROTOCOL.md','PREPARATION_CHECKS.json']:shutil.copy2(SOURCE/name,DEST/name)
shutil.copytree(SOURCE/'frozen',DEST/'frozen',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();files=[p for p in DEST.rglob('*') if p.is_file() and '__pycache__' not in str(p)]
for p in files:assert sha(p)==sha(SOURCE/p.relative_to(DEST))
manifest=json.loads((ROOT/'experiments/seed_replication_20261007/server_run/retrieved/runs/manifest.json').read_text())
for name,h in manifest['source_sha256'].items():
    if name.startswith('frozen_scalar/'):assert sha(HERE/name)==h
py_compile.compile(str(HERE/'run_logradius.py'),doraise=True)
prep=dict(passed=True,baseline_inputs_hash_match=True,reference_ohm=1.,new_runs=200,planned_evaluations=10500000,sha256={p.relative_to(HERE).as_posix():sha(p) for p in [HERE/'PROTOCOL.md',HERE/'run_logradius.py',*files]})
(HERE/'PREPARATION.json').write_text(json.dumps(prep,indent=2),encoding='utf-8');bundle=HERE/'optimization_bundle.tar.gz'
with tarfile.open(bundle,'w:gz') as t:
    for p in [HERE/'PROTOCOL.md',HERE/'run_logradius.py',HERE/'PREPARATION.json',*files]:t.add(p,arcname=p.relative_to(HERE).as_posix())
print(json.dumps({'prepared':True,'bundle_bytes':bundle.stat().st_size,'sha256':sha(bundle)}))

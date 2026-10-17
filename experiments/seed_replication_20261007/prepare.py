from pathlib import Path
import json,hashlib,shutil,tarfile,py_compile
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];SOURCE=ROOT/'experiments/primary_loss_comparison_20261007';DEST=HERE/'frozen_scalar'
DEST.mkdir(exist_ok=True)
for name in ['run.py','model.py','circuits.json','prepare.py','PROTOCOL.md','PREPARATION_CHECKS.json']:shutil.copy2(SOURCE/name,DEST/name)
shutil.copytree(SOURCE/'frozen',DEST/'frozen',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files=[p for p in DEST.rglob('*') if p.is_file() and '__pycache__' not in str(p)]
for p in files:assert sha(p)==sha(SOURCE/p.relative_to(DEST))
old_manifest=json.loads((SOURCE/'server_run/retrieved/runs/manifest.json').read_text())
for name,h in old_manifest['source_sha256'].items():
    if (DEST/name).exists():assert sha(DEST/name)==h
import csv
old={int(r['seed']) for r in csv.DictReader((ROOT/'output/primary_loss_comparison_20261007/selected_metrics.csv').open(encoding='utf-8-sig'))}
new=set(range(2026100700,2026100800));assert not new&old
py_compile.compile(str(HERE/'run_replication.py'),doraise=True)
manifest={'passed':True,'frozen_files':len(files),'frozen_original_hashes_matched':True,'seeds_disjoint_from_original':True,'new_seed_start':min(new),'new_seed_end':max(new),'planned_runs':400,'planned_evaluations':21000000,'sha256':{p.relative_to(HERE).as_posix():sha(p) for p in [HERE/'PROTOCOL.md',HERE/'run_replication.py',*files]}}
(HERE/'PREPARATION.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
bundle=HERE/'replication_bundle.tar.gz'
with tarfile.open(bundle,'w:gz') as t:
    for p in [HERE/'PROTOCOL.md',HERE/'run_replication.py',HERE/'PREPARATION.json',*files]:t.add(p,arcname=p.relative_to(HERE).as_posix())
print(json.dumps({'prepared':True,'bundle_bytes':bundle.stat().st_size,'bundle_sha256':sha(bundle),'frozen_files':len(files)}))

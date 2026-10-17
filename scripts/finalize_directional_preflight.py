"""Preserve the draft and freeze a reporting-typo correction before any run."""
from pathlib import Path
import hashlib,json,tarfile
root=Path(__file__).resolve().parents[1];dst=root/'experiments/curve3d_directional_20261003'
assert not (dst/'case1/runs').exists() and not (dst/'case2/runs').exists()
assert not (dst/'PRELAUNCH_REVISION.json').exists()
old=(dst/'FROZEN.json').read_bytes();old_manifest=json.loads(old)
draft=root/'tmp/curve3d_directional_20261003_code.tar.gz'
draft_sha=hashlib.sha256(draft.read_bytes()).hexdigest()
backup=dst/'preflight_draft';backup.mkdir()
(backup/'FROZEN.json').write_bytes(old)
with tarfile.open(draft) as tf:
    (backup/'finish_report.py').write_bytes(tf.extractfile('finish_report.py').read())
assert hashlib.sha256((backup/'finish_report.py').read_bytes()).hexdigest()==old_manifest['files_sha256']['finish_report.py']
for name,sha in old_manifest['files_sha256'].items():
    if name!='finish_report.py':assert hashlib.sha256((dst/name).read_bytes()).hexdigest()==sha
(dst/'PRELAUNCH_REVISION.json').write_text(json.dumps(dict(reason='Report prose 1,440 corrected to actual 1,400; no formulas/configuration changes',
    no_optimization_started=True,draft_archive_sha256=draft_sha,draft_archive=str(draft)),indent=2),encoding='utf-8')
files={p.relative_to(dst).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(dst.rglob('*'))
       if p.is_file() and '__pycache__' not in p.parts and p.name not in ['FROZEN.json','software_checks.json']}
(dst/'FROZEN.json').write_text(json.dumps(dict(protocol='curve3d_directional_20261003_v1',files_sha256=files),indent=2),encoding='utf-8')
archive=root/'tmp/curve3d_directional_20261003_final_code.tar.gz'
with tarfile.open(archive,'w:gz') as tf:
    for p in sorted(dst.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:tf.add(p,arcname=p.relative_to(dst).as_posix())
print(archive);print(hashlib.sha256(archive.read_bytes()).hexdigest())

"""Package completed, audited experiment; exclude environments and archives."""
from pathlib import Path
import json,tarfile,hashlib
root=Path(__file__).resolve().parent
assert json.loads((root/'analysis/COMPLETE.json').read_text())['status']=='passed'
files=[p for p in root.rglob('*') if p.is_file() and '.venv' not in p.parts and '__pycache__' not in p.parts
       and not p.name.endswith(('.tar.gz','.lock','.tmp')) and p.name!='ARCHIVES.json']
records=[]
for name,chosen in [('review_results.tar.gz',[p for p in files if 'runs' not in p.relative_to(root).parts]),('formal_results.tar.gz',files)]:
    dest=root/name
    with tarfile.open(dest,'w:gz',compresslevel=3) as tf:
        for p in sorted(chosen):tf.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
    records.append(dict(file=name,bytes=dest.stat().st_size,files=len(chosen),sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
(root/'ARCHIVES.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records),flush=True)

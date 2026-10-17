/root/autodl-tmp/femeif_complex_joint_20261012_01/.venv/bin/python - <<'PY'
from pathlib import Path
import json,time,psutil
r=Path('/root/autodl-tmp/femeif_complex_joint_20261012_01')
out=r/'output/complex_joint_20261012'
pid=int((r/'pipeline.pid').read_text())
p=psutil.Process(pid)
children=p.children(recursive=True)
progress=[]
for f in (out/'jobs').rglob('progress.json'):
 v=json.loads(f.read_text());progress.append(dict(job=str(f.parent.relative_to(out/'jobs')),generation=v['generation'],evaluations=v['evaluations']))
records=[]
for c in children:
 try:
  ct=c.cpu_times();records.append(dict(pid=c.pid,status=c.status(),cpu_seconds=ct.user+ct.system,rss_bytes=c.memory_info().rss))
 except psutil.NoSuchProcess:pass
print(json.dumps(dict(time_unix=time.time(),pid=pid,parent_status=p.status(),children=records,
 progress=progress,status=json.loads((out/'STATUS.json').read_text()),
 pipeline_failure=(out/'PIPELINE_FAILED.json').exists(),optimization_failures=len(list((out/'jobs').rglob('FAILED.json')))),indent=2))
PY

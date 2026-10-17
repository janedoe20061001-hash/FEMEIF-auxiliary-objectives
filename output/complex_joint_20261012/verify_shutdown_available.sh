/root/autodl-tmp/femeif_complex_joint_20261012_01/.venv/bin/python - <<'PY'
from pathlib import Path
import os,json,hashlib
p=Path('/usr/bin/shutdown')
print(json.dumps(dict(shutdown_path=str(p),exists=p.exists(),executable=os.access(p,os.X_OK),resolved_path=str(p.resolve()),
 sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None,
 server_task_directory_exists=Path('/root/autodl-tmp/femeif_complex_joint_20261012_01/pipeline.pid').exists())))
PY

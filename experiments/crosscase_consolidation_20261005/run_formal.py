"""Preflight, freeze, then complete the preregistered paired experiment."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib,subprocess,time,traceback,argparse

root=Path(__file__).resolve().parent
def state(status,**kw):
    p=root/'status.tmp';p.write_text(json.dumps(dict(status=status,pid=os.getpid(),time=time.time(),**kw),indent=2));p.replace(root/'status.json')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');ap.add_argument('--workers',type=int,default=28);a=ap.parse_args()
    if not a.run:ap.error('Explicit --run required')
    import fcntl
    lock=(root/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        prepared=json.loads((root/'PREPARED.json').read_text())
        for name,digest in prepared.items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
        for case in ['case1','case2']:
            state('preflight',case=case)
            with (root/f'{case}_preflight.log').open('w') as log:
                subprocess.run([sys.executable,str(root/'preflight_case.py'),str(root/case)],stdout=log,stderr=subprocess.STDOUT,check=True)
        files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file() and '.venv' not in p.parts and '__pycache__' not in p.parts
            and p.suffix in ['.py','.json','.npy','.md','.txt'] and p.name not in ['status.json','FROZEN.json']}
        (root/'FROZEN.json').write_text(json.dumps(dict(time=time.time(),files_sha256=files),indent=2))
        for case in ['case1','case2']:
            state('running',case=case,workers=a.workers)
            cmd=[sys.executable,'launch_batch.py','--workers',str(a.workers)]
            if case=='case2':cmd.append('--run')
            with (root/f'{case}.log').open('w') as log:
                subprocess.run(cmd,cwd=root/case,stdout=log,stderr=subprocess.STDOUT,check=True)
        state('optimization_complete',completed_runs=1200,expected_evaluations=63000000)
    except Exception:
        state('failed',error=traceback.format_exc());raise
if __name__=='__main__':main()

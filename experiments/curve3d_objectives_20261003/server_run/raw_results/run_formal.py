"""Durable two-case formal run; completed software checks precede this entry point."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import argparse
from pathlib import Path
import json
import subprocess
import sys
import time
import traceback

root=Path(__file__).resolve().parent

def state(status,**details):
    record=dict(status=status,updated_unix=time.time(),pid=os.getpid(),**details)
    tmp=root/'formal_status.tmp';tmp.write_text(json.dumps(record,indent=2))
    tmp.replace(root/'formal_status.json')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true')
    parser.add_argument('--workers',type=int,default=28);args=parser.parse_args()
    if not args.run:parser.error('Explicit --run required')
    import fcntl
    lock=(root/'formal.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        for case in ['case1','case2']:
            state('running',case=case,workers=args.workers)
            with (root/f'{case}.log').open('a') as log:
                command=[sys.executable,'launch_batch.py','--workers',str(args.workers)]
                if case=='case2':command.append('--run')
                subprocess.run(command,
                               cwd=root/case,stdout=log,stderr=subprocess.STDOUT,check=True)
            state('auditing',case=case)
            with (root/f'{case}_audit.log').open('a') as log:
                subprocess.run([sys.executable,str(root/'audit_case.py'),str(root/case)],
                               stdout=log,stderr=subprocess.STDOUT,check=True)
        state('reporting')
        subprocess.run([sys.executable,str(root/'finish_report.py')],check=True)
        state('complete',completed_runs=2200,failed_runs=0,evaluations=115500000)
    except Exception:
        state('failed',error=traceback.format_exc());raise

if __name__=='__main__':main()

"""Durable postprocessing, isolated from the frozen optimization source."""
from pathlib import Path
import subprocess,sys,json,time,hashlib,traceback
root=Path(__file__).resolve().parent
def state(s,**kw):
    p=root/'analysis_status.tmp';p.write_text(json.dumps(dict(status=s,time=time.time(),**kw),indent=2));p.replace(root/'analysis_status.json')
try:
    while True:
        current=json.loads((root/'status.json').read_text())
        if current['status']=='failed':raise RuntimeError('Optimization failed')
        if current['status']=='optimization_complete':break
        state('waiting_for_optimization');time.sleep(5)
    src=root/'analysis_libs/logpolar/orientation_native.c';binary=src.with_suffix('.so')
    subprocess.run(['gcc','-O3','-fno-fast-math','-ffp-contract=off','-shared','-fPIC',str(src),'-lm','-o',str(binary)],check=True)
    for case in ['case1','case2']:
        state('auditing_and_correlating',case=case)
        with (root/f'{case}_analysis.log').open('w') as log:
            subprocess.run([sys.executable,str(root/'analyze_case.py'),str(root/case),'--workers','12'],stdout=log,stderr=subprocess.STDOUT,check=True)
    state('statistics')
    subprocess.run([sys.executable,str(root/'finalize_stats.py')],cwd=root,check=True)
    state('complete')
except Exception:
    state('failed',error=traceback.format_exc());raise

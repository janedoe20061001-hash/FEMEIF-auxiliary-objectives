"""Stop only this task's verified process tree; retain every experiment file."""
import os,signal,subprocess,time,json
from pathlib import Path
root=Path('/root/autodl-tmp/femeif_curve3d_logpolar_20261003').resolve()
pid=1951
assert Path(os.readlink(f'/proc/{pid}/cwd')).resolve()==root
command=Path(f'/proc/{pid}/cmdline').read_bytes()
assert b'run_formal.py' in command and b'--run' in command
pairs=[tuple(map(int,line.split())) for line in subprocess.check_output(['ps','-eo','pid=,ppid='],text=True).splitlines()]
owned={pid};changed=True
while changed:
    before=len(owned);owned.update(p for p,parent in pairs if parent in owned);changed=len(owned)>before
verified={}
for p in sorted(owned):
    try:
        assert Path(os.readlink(f'/proc/{p}/cwd')).resolve() in [root,root/'case1',root/'case2']
        cmd=Path(f'/proc/{p}/cmdline').read_bytes();assert b'python' in cmd
        verified[p]=Path(f'/proc/{p}/stat').read_text().split()[21]
    except FileNotFoundError:pass
progress=json.loads((root/'case1/runs/progress.json').read_text())
record=dict(status='superseded_for_numerical_efficiency',reason='same formula, bitwise-equivalent native arithmetic; restart full protocol separately',
            stopped_unix=time.time(),completed_runs_at_interruption=progress['completed_runs'],
            stopped_process_ids=list(verified),all_files_retained=True,partial_results_not_used_for_metric_selection=True)
(root/'INTERRUPTED.json').write_text(json.dumps(record,indent=2))
for p in verified:
    try:os.kill(p,signal.SIGTERM)
    except ProcessLookupError:pass
time.sleep(2)
for p,start in verified.items():
    try:
        state=Path(f'/proc/{p}/stat').read_text().split()
        if state[2]!='Z' and state[21]==start:
            assert Path(os.readlink(f'/proc/{p}/cwd')).resolve() in [root,root/'case1',root/'case2']
            os.kill(p,signal.SIGKILL)
    except FileNotFoundError:pass
print(json.dumps(record))

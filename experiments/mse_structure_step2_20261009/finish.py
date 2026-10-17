from common import *
import time,subprocess
while True:
    state=HERE/'batch/status.json';val=OUT/'analysis/VALIDATION.json'
    if state.exists() and json.loads(state.read_text())['status']=='failed':raise RuntimeError('Training failed')
    if val.exists() and state.exists() and json.loads(state.read_text())['status']=='complete':break
    time.sleep(5)
with (OUT/'FINALIZE_LOG.txt').open('w',encoding='utf8') as log:
    for script in ['summarize.py','plots.py','audit.py','report.py']:
        print(f'Finalizing {script}',flush=True);p=subprocess.run([sys.executable,str(HERE/script)],stdout=log,stderr=log)
        if p.returncode:raise RuntimeError(f'{script} failed; see FINALIZE_LOG.txt')
    # Ensure local Markdown links/images resolve before opening the report.
    import re
    report=OUT/'REPORT.md';text=report.read_text(encoding='utf8');bad=[]
    for path in re.findall(r'\]\(([^)]+)\)',text):
        if path.startswith(('D:/','C:/')) and not Path(path).exists():bad.append(path)
    assert not bad,bad
    dump(OUT/'DELIVERY_CHECK.json',dict(passed=True,report_bytes=report.stat().st_size,local_links_checked=True,missing_links=bad,report_sha256=hashlib.sha256(report.read_bytes()).hexdigest()))
print('All training, analysis and report finished',flush=True)

from core import *
import time,subprocess,shutil,tarfile,platform,traceback
deadline=time.time()+7200
try:
    while True:
        status=json.loads((OUT/'STATUS.json').read_text())
        if status['status']=='failed':raise RuntimeError('Optimization batch failed; preserve all jobs')
        if status['status']=='complete':break
        if time.time()>deadline:raise TimeoutError('Batch completion deadline')
        time.sleep(5)
    log=OUT/'diagnosis_server.log'
    with log.open('w') as f:
        subprocess.run([sys.executable,str(HERE/'diagnose.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    shutil.copy2('/root/autodl-tmp/teacher_doubts_validation_20261011.log',OUT/'optimization_server.log')
    source={str(p.relative_to(ROOT)):sha(p) for p in list(HERE.glob('*.py'))+list(HERE.glob('*.md'))+list(BASE.rglob('*.py'))+[BASE/'circuits.json',ROOT/'targetImpedance_case2.csv']}
    dump(OUT/'SERVER_PROVENANCE.json',dict(python=sys.executable,python_version=platform.python_version(),
                                         system=platform.platform(),cpu_affinity=sorted(os.sched_getaffinity(0)),
                                         workspace=str(ROOT),source_sha256=source,completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    hashes={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file()}
    dump(OUT/'SERVER_FILES.json',hashes)
    archive=Path('/root/autodl-tmp/teacher_doubts_validation_20261011_results.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:tar.add(OUT,arcname=OUT.name)
    record=dict(status='complete',archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,files=len(hashes))
    dump('/root/autodl-tmp/teacher_doubts_archive.json',record);print(json.dumps(record),flush=True)
except Exception:
    dump('/root/autodl-tmp/teacher_doubts_archive.json',dict(status='failed',traceback=traceback.format_exc()))
    raise

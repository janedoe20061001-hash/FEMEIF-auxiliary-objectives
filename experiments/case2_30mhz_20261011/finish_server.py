from core import *
import time,tarfile,shutil
deadline=time.time()+3600
while True:
    status=json.loads((OUT/'STATUS.json').read_text())
    if status['status']=='failed':raise RuntimeError('Batch failed, preserve every seed')
    if status['status']=='complete':break
    if time.time()>deadline:raise TimeoutError('Batch deadline')
    time.sleep(5)
assert status['completed']==300 and not status['failed']
shutil.copy2('/root/autodl-tmp/case2_30mhz_20261011.log',OUT/'server.log')
dump(OUT/'SERVER_FILES.json',{p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file()})
archive=Path('/root/autodl-tmp/case2_30mhz_20261011_results.tar.gz')
with tarfile.open(archive,'w:gz') as tar:tar.add(OUT,arcname=OUT.name)
dump('/root/autodl-tmp/case2_30mhz_archive.json',dict(status='complete',sha256=sha(archive),bytes=archive.stat().st_size,archive=str(archive)))

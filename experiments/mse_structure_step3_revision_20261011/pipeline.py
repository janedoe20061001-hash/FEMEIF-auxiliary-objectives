"""Server pipeline, resource measurement, and checksummed result archive."""
from core import *
import subprocess,time,zipfile,threading,csv

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    stop=threading.Event()
    samples=[]
    quota=Path('/sys/fs/cgroup/cpu.max').read_text().split()
    allocated=min(len(os.sched_getaffinity(0)),float(quota[0])/float(quota[1]) if quota[0]!='max' else len(os.sched_getaffinity(0)))
    memory_limit=Path('/sys/fs/cgroup/memory.max').read_text().strip()
    dump(OUT/'SERVER_ENVIRONMENT.json',dict(affinity=len(os.sched_getaffinity(0)),allocated_cpus=allocated,cpu_max=quota,memory_max=memory_limit,python=sys.version))
    def usage():
        fields=Path('/sys/fs/cgroup/cpu.stat').read_text().split()
        return int(dict(zip(fields[::2],fields[1::2]))['usage_usec'])
    def sample():
        last=time.monotonic(); count=usage()
        with (OUT/'CPU_SAMPLES.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['unix','elapsed','allocated_cpus','cpu_percent','memory_bytes','stage'])
            writer.writeheader()
            while not stop.wait(2):
                now=time.monotonic(); current=usage()
                row=dict(unix=time.time(),elapsed=now-started,allocated_cpus=allocated,cpu_percent=(current-count)/(now-last)/1e6/allocated*100,memory_bytes=int(Path('/sys/fs/cgroup/memory.current').read_text()),stage=state['stage'])
                writer.writerow(row);f.flush();samples.append(row)
                dump(OUT/'RESOURCE_STATUS.json',row)
                last,count=now,current
    started=time.monotonic()
    state=dict(status='running',stage='preflight',pid=os.getpid(),started_unix=time.time(),workers=min(CFG['workers'],int(allocated)))
    dump(OUT/'PIPELINE_STATUS.json',state)
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    try:
        for stage,argv in [('preflight',[HERE/'run.py','preflight']),('smoke',[HERE/'run.py','smoke']),('optimization',[HERE/'run.py','batch']),('independent_audit',[HERE/'analyze.py'])]:
            state.update(stage=stage,updated_unix=time.time());dump(OUT/'PIPELINE_STATUS.json',state)
            with (OUT/f'{stage}.log').open('w') as log:
                child=subprocess.Popen([sys.executable,*map(str,argv)],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                state['child_pid']=child.pid;dump(OUT/'PIPELINE_STATUS.json',state)
                if child.wait()!=0: raise RuntimeError(f'{stage} failed; original log preserved')
        state.update(stage='packaging',child_pid=None,updated_unix=time.time());dump(OUT/'PIPELINE_STATUS.json',state)
        stop.set();thread.join()
        useful=[r['cpu_percent'] for r in samples if r['stage']=='optimization']
        dump(OUT/'CPU_SUMMARY.json',dict(allocated_cpus=allocated,samples=len(samples),optimization_samples=len(useful),optimization_mean_percent=float(np.mean(useful)) if useful else None,optimization_p90_percent=float(np.percentile(useful,90)) if useful else None,peak_memory_bytes=max(r['memory_bytes'] for r in samples) if samples else None,note='Real useful computation, not an artificial utilization target; tails included'))
        archive=ROOT/'results_revision.zip'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
            for p in OUT.rglob('*'):
                if p.is_file() and not p.name.endswith('.tmp'):z.write(p,p.relative_to(ROOT).as_posix())
            for p in HERE.glob('*'):
                if p.is_file():z.write(p,p.relative_to(ROOT).as_posix())
        h=hashlib.sha256()
        with archive.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        state.update(status='complete',stage='complete',archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=h.hexdigest(),finished_unix=time.time(),elapsed_seconds=time.monotonic()-started)
        dump(OUT/'PIPELINE_STATUS.json',state)
    except Exception as exc:
        state.update(status='failed',error=str(exc),updated_unix=time.time());dump(OUT/'PIPELINE_STATUS.json',state)
        raise
    finally:
        stop.set();thread.join()

if __name__=='__main__':main()

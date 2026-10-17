"""Quota-sized queue, preflight, all paired jobs, independent audit, archived recovery."""
from core import *
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from run import one as optimize,path_for,verify
from audit import one as audit_one
import argparse,time,traceback,zipfile,hashlib,platform,fcntl
from importlib.metadata import version

def usage():
    p=Path('/sys/fs/cgroup/cpuacct/cpuacct.usage')
    if p.exists():return int(p.read_text())/1e9
    p=Path('/sys/fs/cgroup/cpu.stat')
    if p.exists():return float(dict(x.split() for x in p.read_text().splitlines())['usage_usec'])/1e6
    return None

def queue(jobs,worker,label,workers,costs=None):
    oldcpu=usage();oldtime=time.monotonic();done=failed=0;total=len(jobs)
    ordered=sorted(jobs,key=lambda j:-(costs or {}).get(j[0],{}).get(j[1],0))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(worker,j):j for j in ordered}
        while pending:
            completed,_=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
            for future in completed:
                job=pending.pop(future)
                try:r=future.result()
                except Exception:r=dict(status='worker_crashed',target=job[0],method=job[1],seed=job[2],traceback=traceback.format_exc())
                ok=r['status'] in ('complete','audited');done+=int(ok);failed+=int(not ok)
                if not ok:dump(OUT/'pipeline_failures'/f'{label}_{job[0].replace("/","_")}_{job[1]}_{job[2]}.json',r)
                print(json.dumps(dict(phase=label,target=job[0],method=job[1],seed=job[2],status=r['status'],done=done,failed=failed,total=total)),flush=True)
            now=time.monotonic();cpu=usage();percent=None if oldcpu is None or cpu is None else 100*(cpu-oldcpu)/(now-oldtime)/workers
            status=dict(phase=label,completed_phase=done,failed_phase=failed,total_phase=total,pending=len(pending),workers=workers,
                allocated_cpu_usage_percent=percent,updated_unix=time.time())
            dump(OUT/'STATUS.json',status)
            with (OUT/'CPU_USAGE.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(status)+'\n')
            oldcpu=cpu;oldtime=now
    return failed

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dump(OUT/'PROCESS.json',dict(pid=os.getpid(),start_unix=time.time(),workers=allocated_cpus()))
    from preflight import main as preflight
    validation=OUT/'PREFLIGHT.json'
    if not validation.exists():preflight()
    checked=json.loads(validation.read_text());assert checked['passed'] and checked['hashes']==hashes(),'Stale preflight'
    from development_report import main as development
    development()
    manifest=dict(protocol=CFG,hashes=hashes(),versions=checked['versions'],python=sys.version,platform=platform.platform(),workers=allocated_cpus(),native_sha256={p.relative_to(HERE).as_posix():sha(p) for p in [FROZEN/'native.so',HERE/'area_native.so'] if p.exists()})
    mf=OUT/'MANIFEST.json'
    if mf.exists():assert json.loads(mf.read_text())==manifest,'Resume code/inputs/runtime changed'
    else:dump(mf,manifest)
    smoke=OUT/'smoke'
    jobs=[(ident,method,20261015,str(smoke),True) for ident in target_ids() for method in CFG['methods'] if not path_for(smoke,ident,method,20261015).exists()]
    if jobs and queue(jobs,optimize,'smoke',allocated_cpus()):raise RuntimeError('Smoke failure; no formal jobs launched')
    costs={}
    for ident in target_ids():
        costs[ident]={}
        for method in CFG['methods']:
            d=path_for(smoke,ident,method,20261015)
            rec=json.loads((d/'summary.json').read_text());assert rec['status']=='complete' and rec['evaluations']==24
            costs[ident][method]=rec['seconds']*60000/24
    dump(OUT/'SMOKE_COST_ESTIMATE.json',dict(rough_seconds_per_job=costs,warning='24 candidates only; initialization and convergence alter actual costs',rough_core_hours=sum(sum(v.values()) for v in costs.values())*len(CFG['seeds'])/3600))
    formal=[(ident,method,seed,str(OUT),False) for ident in target_ids() for method in CFG['methods'] for seed in CFG['seeds']]
    assert len(formal)==CFG['planned_jobs']
    pending=[]
    for j in formal:
        d=path_for(OUT,*j[:3])
        if (d/'summary.json').exists():verify(d)
        elif d.exists():raise RuntimeError(f'Incomplete job retained, manual repair required: {d}')
        else:pending.append(j)
    failure=queue(pending,optimize,'optimization',allocated_cpus(),costs) if pending else 0
    valid=[j for j in formal if (path_for(OUT,*j[:3])/'summary.json').exists()]
    audits=[j for j in valid if not (path_for(OUT,*j[:3])/'audit.json').exists()]
    failure+=queue(audits,audit_one,'audit',allocated_cpus()) if audits else 0
    completed=len(valid);audited=sum((path_for(OUT,*j[:3])/'audit.json').exists() for j in valid)
    accepted=sum(json.loads((path_for(OUT,*j[:3])/'audit.json').read_text())['accepted'] for j in valid if (path_for(OUT,*j[:3])/'audit.json').exists())
    dump(OUT/'STATUS.json',dict(phase='finished' if failure==0 else 'failed',completed=completed,audited=audited,accepted=accepted,
        failed=failure,total=len(formal),updated_unix=time.time(),workers=allocated_cpus()))
    # Recovery includes all original failures, inputs/source provenance and populations.
    archive=ROOT/'results_arx_light_20261015.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.name!='pipeline.lock':z.write(p,p.relative_to(ROOT).as_posix())
        for p in sorted(HERE.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(ROOT).as_posix())
    # Root execution logs are retained separately on the server and retrieved with the archive.
    dump(ROOT/'results_arx_light_20261015.json',dict(file=archive.name,sha256=sha(archive),bytes=archive.stat().st_size,
        status=json.loads((OUT/'STATUS.json').read_text())))
    if failure:raise SystemExit(1)

if __name__=='__main__':
    try:main()
    except Exception:
        dump(OUT/'PIPELINE_FAILED.json',dict(traceback=traceback.format_exc(),updated_unix=time.time()))
        raise

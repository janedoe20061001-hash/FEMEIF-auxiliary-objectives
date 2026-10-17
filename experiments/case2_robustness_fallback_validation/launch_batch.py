"""Durable coordinator for fallback validation."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[name]="1"
import argparse,json,time,traceback,fcntl
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import experiment as e
def atomic(path,data):
    temp=path.with_suffix('.tmp');e.BASE.dump(temp,data);temp.replace(path)
def job(cfg,f,z,c,g,s,p,fp):return e.run_one(cfg,f,z,c,g,s,p,fp)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');ap.add_argument('--workers',type=int,default=28);ap.add_argument('--output',type=Path,default=e.HERE/'runs');args=ap.parse_args()
    if not args.run:ap.error('--run required')
    cfg,f,z=e.config(e.HERE/'config.json');root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    mf=e.manifest(cfg,f,z);mp=root/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())['fingerprint']!=mf['fingerprint']:raise ValueError('fingerprint differs')
    if not mp.exists():e.BASE.dump(mp,mf)
    pending=[];complete=[];failed=[]
    for c in cfg['conditions']:
      for s in range(cfg['seeds']['start'],cfg['seeds']['start']+cfg['seeds']['count']):
       for g in e.GROUPS:
        p=root/c['name']/e.group_name(g)/f'seed_{s:03d}';mark=p/'summary.json'
        if mark.exists():complete.append(json.loads(mark.read_text()))
        elif p.exists():raise ValueError(f'incomplete {p}')
        else:pending.append((c,g,s,p))
    total=len(complete)+len(pending);start=time.time()
    def progress(state):atomic(root/'progress.json',{'state':state,'total_runs':total,'completed_runs':len(complete),'failed_runs':len(failed),'elapsed_seconds':time.time()-start,'failures':failed,'recent_completions':[{'condition':r['condition'],'group':r['group'],'seed':r['seed']} for r in complete[-5:]]})
    progress('running')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
      futures={pool.submit(job,cfg,f,z,c,g,s,p,mf['fingerprint']):(c,g,s) for c,g,s,p in pending}
      for future in as_completed(futures):
       c,g,s=futures[future]
       try:r=future.result();complete.append(r);print(f"{len(complete)}/{total} {c['name']} {e.group_name(g)} {s}",flush=True)
       except Exception:failed.append({'condition':c['name'],'group':e.group_name(g),'seed':s,'error':traceback.format_exc()})
       progress('running')
    progress('failed' if failed else 'complete');print(f'Batch finished: {len(complete)} complete, {len(failed)} failed',flush=True)
    if failed:raise SystemExit(1)
if __name__=='__main__':main()


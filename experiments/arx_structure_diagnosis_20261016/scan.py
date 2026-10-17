"""Read-only full-band amplitude-event scan of all saved Case1 populations."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib,time,argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
CODE=ROOT/'experiments/arx_light_20261015';sys.path.insert(0,str(CODE))
from core import target,CFG,diag,anchors,sha,dump,knee
OUT=ROOT/'output/arx_structure_diagnosis_20261016'
OLD=ROOT/'output/arx_light_20261015'
DATA=(OLD/'retrieved/output/arx_light_20261015') if (OLD/'retrieved').exists() else OLD
METHODS=CFG['methods'];SEEDS=CFG['seeds'];STAGES=[0]+list(range(20,301,20))

def inventory():
    OUT.mkdir(parents=True,exist_ok=True);lookup=np.empty((6,12,16,200),np.int32)
    manifest=json.loads((DATA/'MANIFEST.json').read_text())
    for name,digest in manifest['hashes'].items():assert sha(ROOT/name)==digest,('source_or_input',name)
    primary=np.full((6,12,16,200,2),np.nan);aux=np.full((6,12,16,200),np.nan);selected=np.full((6,12,16),-1,np.int16)
    ids={};values=[];source_hashes={};data=target('case1')
    for mi,m in enumerate(METHODS):
        for si,s in enumerate(SEEDS):
            d=DATA/'jobs/case1'/m/f'seed_{s}';summary=json.loads((d/'summary.json').read_text())
            for gi,g in enumerate(STAGES):
                file=d/('initial.npy' if g==0 else f'generation_{g:04}.npz')
                digest=sha(file);assert digest==summary['files_sha256'][file.name]
                source_hashes[str(file.relative_to(DATA))]=digest
                if g==0:X=np.load(file)
                else:
                    with np.load(file) as a:X=a['X'];F=a['F'];index=int(a['selected_index'])
                    assert knee(F[:,:2])[0]==index
                    primary[mi,si,gi]=F[:,:2];selected[mi,si,gi]=index
                    if m!='MAG_PHA':aux[mi,si,gi]=F[:,2]
                assert X.shape==(200,18) and np.isfinite(X).all()
                assert np.all(X>=data['lower']-1e-12) and np.all(X<=data['upper']+1e-12)
                for ci,x in enumerate(X):
                    key=np.asarray(x,dtype='<f8').tobytes()
                    if key not in ids:ids[key]=len(values);values.append(x.copy())
                    lookup[mi,si,gi,ci]=ids[key]
    unique=np.array(values)
    np.savez_compressed(OUT/'INVENTORY.npz',X=unique,lookup=lookup,primary=primary,auxiliary=aux,selected=selected)
    dump(OUT/'INVENTORY.json',dict(methods=METHODS,seeds=SEEDS,stages=STAGES,records=int(lookup.size),unique_candidates=len(values),final_records=14400,
        source_hashes=source_hashes,inventory_sha256=sha(OUT/'INVENTORY.npz'),source_manifest_sha256=sha(DATA/'MANIFEST.json'),
        protocol_sha256=sha(HERE/'PROTOCOL.md'),scan_source_sha256=sha(HERE/'scan.py')))
    print(json.dumps(dict(phase='inventory',records=int(lookup.size),unique_candidates=len(values))),flush=True)

_worker=None
def initialize():
    global _worker
    data=target('case1');c=data['circuit'];q=data['quadrature'];cfg=q.config
    with np.load(OUT/'INVENTORY.npz') as a:X=a['X']
    with np.load(OUT/'INVENTORY.npz') as a:
        force=set()
        for mi in range(6):
            for si in range(12):
                for gi in range(1,16):force.add(int(a['lookup'][mi,si,gi,int(a['selected'][mi,si,gi])]))
        final=np.unique(a['lookup'][:,:,-1].ravel());force.update(final[np.linspace(0,len(final)-1,32).round().astype(int)].tolist())
    grids={level:q.reference_grid(None,level) for level in (1,2,4,8)}
    _worker=dict(data=data,c=c,q=q,cfg=cfg,X=X,grids=grids,force=force)

def extract(x,z):
    w=_worker;fit=diag.events(x,z,w['cfg']);rows,extras=diag.match(w['data']['record']['events'],fit,w['cfg'])
    match=np.array([r['matched'] for r in rows]);complete=np.array([r['complete'] for r in rows]);numbers=np.array([[np.nan if r[k] is None else r[k] for k in ('shift_decades','height_delta_db','width_ratio')] for r in rows])
    return match,complete,numbers,len(extras),len(fit)

def stability(a,b):
    same=(a[0]==b[0])&(a[1]==b[1]);va,vb=a[2],b[2]
    same &= np.isclose(va[:,0],vb[:,0],atol=1e-5,rtol=0,equal_nan=True)
    same &= np.isclose(va[:,1],vb[:,1],atol=.02,rtol=0,equal_nan=True)
    same &= np.isclose(va[:,2],vb[:,2],atol=1e-8,rtol=.01,equal_nan=True)
    return same

def evaluate(uid):
    w=_worker;p=10**w['X'][uid]
    ti=next(j for j,e in enumerate(w['data']['record']['events']) if e['event_id']=='P3');target_event=w['data']['record']['events'][ti]
    pos=w['cfg']['events']['position_decades'];lower=10**(target_event['x']-pos);upper=10**(target_event['x']+pos)
    v=p.reshape(6,3);r,l,c=v[:,0],v[:,1],v[:,2]
    fc=1/(2*np.pi*np.sqrt(l*c));ww=2*np.pi*np.clip(fc,lower,upper);b=ww*c-1/(ww*l)
    bound=float(np.sum(1/np.sqrt((1/r)**2+b*b)))*(1+1e-8)
    minimum=10**((target_event['height_db']-w['cfg']['events']['height_db'])/20)
    certified=bound<minimum*(1-1e-8)
    if certified and uid not in w['force']:
        stable=np.zeros(11,bool);stable[ti]=True
        return dict(uid=uid,matched=np.zeros(11,bool),complete=np.zeros(11,bool),values=np.full((11,3),np.nan),stable=stable,extra=-1,nfit=-1,level=0,evaluated_full=False,amplitude_upper_bound=bound,certified_height_incomplete=True)
    modes=w['c'].modes(p)
    adapter=SimpleNamespace(modes=lambda parameters:modes)
    previous=None;stable=np.zeros(11,bool)
    for level in (1,2,4,8):
        xx=np.unique(np.r_[w['grids'][level],anchors(adapter,p,2.,6.,24*level)])
        z,_=w['c'].response(10**xx,p);now=extract(xx,z)
        if previous is not None:
            stable=stability(previous,now)
            if stable.all() and previous[3:]==now[3:]:break
        previous=now
    assert not (certified and now[1][ti]),('analytic_certificate_conflict',uid,bound)
    return dict(uid=uid,matched=now[0],complete=now[1],values=now[2],stable=stable,extra=now[3],nfit=now[4],level=level,evaluated_full=True,amplitude_upper_bound=bound,certified_height_incomplete=certified)

def batch(item):
    label,ids=item;path=OUT/'batches'/f'{label}.npz'
    if path.exists():return dict(batch=label,candidates=len(ids),cached=True)
    st=time.time();rows=[]
    for i in ids:
        try:rows.append(evaluate(int(i)))
        except Exception as exc:
            dump(OUT/'failures'/f'candidate_{i}.json',dict(uid=int(i),error=repr(exc)))
            raise
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**{k:np.array([r[k] for r in rows]) for k in rows[0]})
    return dict(batch=label,candidates=len(ids),cached=False,seconds=time.time()-st)

def run(which,workers):
    with np.load(OUT/'INVENTORY.npz') as a:
        final=np.unique(a['lookup'][:,:,-1].ravel());allids=np.arange(len(a['X']))
    if which=='pilot':ids=final[np.linspace(0,len(final)-1,32).round().astype(int)]
    elif which=='final':ids=final
    else:ids=np.setdiff1d(allids,final)
    jobs=[(f'{which}_{j:05}',part) for j,part in enumerate(np.array_split(ids,max(1,int(np.ceil(len(ids)/32)))))]
    count=0;st=time.time();results=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize) as pool:
        futures=[pool.submit(batch,j) for j in jobs]
        for f in as_completed(futures):
            result=f.result();results.append(result);count+=result['candidates']
            status=dict(phase=which,completed=count,total=len(ids),workers=workers,elapsed_seconds=time.time()-st,updated_unix=time.time())
            dump(OUT/'STATUS.json',status)
            if len(results)%8==0 or count==len(ids):print(json.dumps(status),flush=True)
    dump(OUT/f'{which.upper()}_EXECUTION.json',dict(phase=which,candidates=len(ids),workers=workers,seconds=time.time()-st,batches=results))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['inventory','pilot','final','history']);parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    if args.mode=='inventory':inventory()
    else:run(args.mode,args.workers)

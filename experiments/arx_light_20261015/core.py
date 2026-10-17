"""Frozen models, target adapters, and common scientific provenance."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
import sys,json,hashlib,importlib.util
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'output/arx_light_20261015'
INPUTS=HERE/'inputs'
FROZEN=HERE/'frozen'
sys.path.insert(0,str(FROZEN))
from circuits import Circuit, S, P
from sampling import Quadrature, anchors
import diagnostics as diag
from case1_adaptive import Adaptive
from selector import knee
CFG=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
BASECFG=json.loads((FROZEN/'nested_config.json').read_text(encoding='utf-8'))

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()),encoding='utf-8')
    import time
    for attempt in range(20):
        try:temp.replace(path);break
        except PermissionError:
            if attempt==19:raise
            time.sleep(.1)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def hashes():
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.c'))+[HERE/'protocol.json',HERE/'PROTOCOL.md']
    files+=list(FROZEN.glob('*'))+list(INPUTS.rglob('*'))
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))
            if p.is_file() and p.suffix not in ('.so','.dll','.pyc')}

def case1_modes(parameters):
    v=np.asarray(parameters).reshape(6,3)
    fc=1/(2*np.pi*np.sqrt(v[:,1]*v[:,2]))
    width=np.clip(6/(v[:,0]*np.sqrt(v[:,2]/v[:,1]))/np.log(10),1e-12,.5)
    return list(zip(fc,width))

class Case1Quadrature(Quadrature):
    def __init__(self,circuit,record):
        self.c=circuit;self.config=BASECFG;self.tolerance_factor=1
        with np.load(FROZEN/'case1_grid.npz') as a:
            self.adaptive=Adaptive(record,a['frequency'])
        self.edges=self.adaptive.edges
    def score(self,x):return self.adaptive.score(x)
    @property
    def queries(self):return self.adaptive.queries
    def reference_grid(self,p=None,refinement=1):
        lo,hi=self.edges[[0,-1]]
        pp=[self.c.p] if p is None else [self.c.p,np.asarray(p)]
        return np.unique(np.r_[np.linspace(lo,hi,2048*refinement+1),self.edges,
             *[anchors(self.c,v,lo,hi,24*refinement) for v in pp]])

class MeasuredQuadrature:
    def __init__(self,circuit,x,z):
        self.c=circuit;self.x=x;self.z=z;self.queries=0;self.config=BASECFG
    def score(self,logp):
        zz,_=self.c.response(10**self.x,10**np.atleast_2d(logp))
        self.queries+=zz.size
        return np.column_stack([np.mean((20*np.log10(abs(zz)/abs(self.z)))**2,axis=1),
                                np.mean((np.angle(zz)-np.angle(self.z))**2,axis=1)])
    def reference(self,logp,refinement=1):
        zz,_=self.c.response(10**self.x,10**np.asarray(logp))
        loss=np.array([(20*np.log10(abs(zz)/abs(self.z)))**2,(np.angle(zz)-np.angle(self.z))**2])
        return loss.mean(axis=1),self.x,zz,self.z
    def reference_grid(self,p=None,refinement=1):return self.x

def target(ident):
    rec=json.loads((INPUTS/ident/'target.json').read_text(encoding='utf-8'))
    cr=rec['circuit'];values=dict(zip(cr['names'],cr['parameters']))
    ll=next(n for n in cr['names'] if n.startswith('L'))
    cc=next(n for n in cr['names'] if n.startswith('C'))
    rr=next(n for n in cr['names'] if n.startswith('R'))
    c=Circuit(ident,cr['title'],cr['tree'],values,(ll,cc),rr,rr)
    if rec['kind']=='case1':
        c.modes=lambda p=None:case1_modes(c.p if p is None else p)+Circuit.modes(c,c.p if p is None else p)
        record=json.loads((FROZEN/'case1_record.json').read_text(encoding='utf-8'))
        q=Case1Quadrature(c,record)
    elif rec['kind']=='measured':
        with np.load(INPUTS/ident/'target_curve.npz') as a:x=a['x'];z=a['z']
        q=MeasuredQuadrature(c,x,z)
    else:q=Quadrature(c,BASECFG)
    return dict(id=ident,record=rec,circuit=c,quadrature=q,kind=rec['kind'],
                lower=np.array(rec['lower_log10']),upper=np.array(rec['upper_log10']))

def target_ids():return json.loads((INPUTS/'manifest.json').read_text(encoding='utf-8'))['targets']

def allocated_cpus():
    affinity=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1
    v2=Path('/sys/fs/cgroup/cpu.max')
    if v2.exists():
        q,p=v2.read_text().split();quota=affinity if q=='max' else float(q)/float(p)
    else:
        q=Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us');p=Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
        quota=affinity if not q.exists() or float(q.read_text())<0 else float(q.read_text())/float(p.read_text())
    return max(1,min(CFG['workers'],int(quota),affinity))

def checked_nodal(c,f,p):
    """Retain double failure; resolve ill-conditioned port-current cancellation at 70 digits."""
    recursive,_=c.response(f,p)
    try:
        ordinary=c.nodal_response(f,p)
        passed=np.all(abs(recursive-ordinary)<=CFG['audit']['nodal_atol']+CFG['audit']['nodal_rtol']*abs(recursive))
        error=float(np.max(abs(recursive-ordinary)/np.maximum(abs(recursive),1e-30)))
    except np.linalg.LinAlgError:
        passed=False;error=None
    if passed:return dict(mode='double',double_max_relative_error=error)
    import mpmath as mp
    with mp.workdps(70):
        edges=c.netlist();internal=sorted({k for _,a,b in edges for k in (a,b) if k>=2});loc={n:i for i,n in enumerate(internal)}
        parameters={n:mp.mpf(float(v)) for n,v in zip(c.names,p)};answer=[]
        for freq in f:
            w=2*mp.pi*mp.mpf(float(freq));y={}
            for n,a,b in edges:
                v=parameters[n];z=v if n[0]=='R' else (mp.j*w*v if n[0]=='L' else 1/(mp.j*w*v));y[n]=1/z
            mat=mp.matrix(len(internal));rhs=mp.matrix(len(internal),1)
            for n,a,b in edges:
                for node,other in ((a,b),(b,a)):
                    if node in loc:
                        i=loc[node];mat[i,i]+=y[n]
                        if other in loc:mat[i,loc[other]]-=y[n]
                        elif other==1:rhs[i]+=y[n]
            v=mp.lu_solve(mat,rhs) if internal else []
            volt={0:mp.mpf(0),1:mp.mpf(1),**dict(zip(internal,v))}
            current=sum(y[n]*(1-volt[b if a==1 else a]) for n,a,b in edges if a==1 or b==1)
            answer.append(complex(1/current))
    np.testing.assert_allclose(recursive,np.array(answer),rtol=CFG['audit']['nodal_rtol'],atol=CFG['audit']['nodal_atol'])
    return dict(mode='70_digit',double_max_relative_error=error,high_precision_max_relative_error=float(np.max(abs(recursive-answer)/np.maximum(abs(recursive),1e-30))))

import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BASE=ROOT/'experiments/dual_trajectory_20261008/frozen_dual'
sys.path.insert(0,str(BASE))
from model import Model
OUT=ROOT/'output/teacher_doubts_validation_20261011'
METRICS=['magnitude_mse_db2','phase_mse_rad2','wsmse','mean_distance_ohm']
SHOW=[2307,2312,2321]

def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()),encoding='utf8');q.replace(p)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def model(case,condition='full'):
    m=Model(case)
    if condition=='paper_band':
        mask=(m.f>=1e4)&(m.f<=3e7);m.f=m.f[mask];m.target=m.target[mask]
        m.amp=20*np.log10(abs(m.target));m.phase=np.angle(m.target)
        assert case=='case2' and len(m.f)==108
    return m

def metrics(z,target,alpha):
    z=np.atleast_2d(z);a=np.mean((20*np.log10(abs(z)/abs(target)))**2,axis=1)
    p=np.mean((np.arctan2(z.imag,z.real)-np.arctan2(target.imag,target.real))**2,axis=1)
    return np.column_stack([a,p,a+alpha*p,np.mean(abs(z-target),axis=1)])

def front(v):
    dominates=np.all(v[:,None,:]<=v[None,:,:],axis=2)&np.any(v[:,None,:]<v[None,:,:],axis=2)
    return np.flatnonzero(~dominates.any(axis=0))

def knee(v):
    f=front(v);_,first=np.unique(v[f],axis=0,return_index=True);indices=f[first];values=v[indices]
    low=values.min(axis=0);high=values.max(axis=0);span=high-low
    floor=1e-14*np.maximum(1,np.maximum(abs(low),abs(high)));uv=(values-low)/np.maximum(span,floor)
    ideal=np.sum(uv**2,axis=1);distance=(1-uv.sum(axis=1))/np.sqrt(2)
    reason='convex_knee'
    if len(values)==1:reason='unique_front_point'
    elif np.any(span<=floor):reason='degenerate_span'
    elif len(values)<3:reason='insufficient_front_points'
    elif distance.max()<=1e-12:reason='no_convex_knee'
    eligible=np.flatnonzero(distance>=distance.max()-1e-12) if reason=='convex_knee' else np.arange(len(values))
    selected=int(indices[eligible[np.lexsort((indices[eligible],ideal[eligible]))[0]]])
    return selected,reason

def nodal(m,parameters,f=None):
    """Build an independent admittance graph, drive 1A, solve input voltage."""
    frequency=m.f if f is None else np.asarray(f);parameters=np.atleast_2d(parameters)
    edges=[];next_node=2
    def newnode():
        nonlocal next_node
        result=next_node;next_node+=1;return result
    def visit(tree,a,b):
        if isinstance(tree,str):edges.append((tree,a,b));return
        if tree[0]=='P':
            for child in tree[1:]:visit(child,a,b)
        elif tree[0]=='S':
            nodes=[a]+[newnode() for _ in tree[1:-1]]+[b]
            for child,left,right in zip(tree[1:],nodes[:-1],nodes[1:]):visit(child,left,right)
        else:raise ValueError(tree[0])
    visit(m.record['tree'],1,0);names=m.record['parameter_names'];w=2*np.pi*frequency
    answers=[];conditions=[]
    for p in parameters:
        values=dict(zip(names,p));matrix=np.zeros((len(w),next_node-1,next_node-1),complex)
        for name,a,b in edges:
            value=values[name];y=np.ones(len(w))/value if name[0]=='R' else (1/(1j*w*value) if name[0]=='L' else 1j*w*value)
            if a:matrix[:,a-1,a-1]+=y
            if b:matrix[:,b-1,b-1]+=y
            if a and b:matrix[:,a-1,b-1]-=y;matrix[:,b-1,a-1]-=y
        current=np.zeros((len(w),next_node-1,1),complex);current[:,0,0]=1
        # Broad log bounds can make this graph ill conditioned. Build stamps
        # again from component values at 60 digits, never from rounded matrices.
        import mpmath as mp
        voltage=[]
        with mp.workdps(60):
            for frequency_value in frequency:
                mat=mp.matrix(next_node-1);rhs=mp.matrix(next_node-1,1);rhs[0]=1
                omega=2*mp.mpf(str(np.pi))*mp.mpf(str(float(frequency_value)))
                for name,a,b in edges:
                    value=mp.mpf(str(float(values[name])))
                    y=1/value if name[0]=='R' else (1/(mp.j*omega*value) if name[0]=='L' else mp.j*omega*value)
                    if a:mat[a-1,a-1]+=y
                    if b:mat[b-1,b-1]+=y
                    if a and b:mat[a-1,b-1]-=y;mat[b-1,a-1]-=y
                voltage.append(complex(mp.lu_solve(mat,rhs)[0]))
        answers.append(voltage);conditions.append(float(np.max(np.linalg.cond(matrix))))
    return np.asarray(answers),conditions

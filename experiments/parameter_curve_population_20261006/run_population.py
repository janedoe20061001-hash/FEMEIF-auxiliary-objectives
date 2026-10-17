"""Large local passive-circuit ensemble. Streaming, deterministic, no optimization."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numpy.polynomial import polynomial as poly
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[2]
OUT_DEFAULT=ROOT/'output/parameter_curve_population_20261006'
BAND=(100.,1e8)
WREF=2*np.pi*1e5
LEVELS=np.array([.1,.3,1.,3.,10.,30.])
Q_RANGES=((.5,2.),(2.,20.),(20.,200.))
GAP_RANGES=((1.05,1.4),(1.5,4.),(6.,30.))
BLUE,RED,GREEN='#155aa8','#c44536','#168778'
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')


def S(*children): return ['S',*children]
def P(*children): return ['P',*children]


FAMILIES=[
    ('rc_series','RC 串联',S('R1','C1')),
    ('rc_parallel','RC 并联',P('R1','C1')),
    ('rl_series','RL 串联',S('R1','L1')),
    ('rl_parallel','RL 并联',P('R1','L1')),
    ('rlc_series','单串联 RLC',S('R1','L1','C1')),
    ('rlc_parallel','单并联 RLC',P('R1','L1','C1')),
    ('two_parallel_units','两个并联 RLC 单元串联',S(P('R1','L1','C1'),P('R2','L2','C2'))),
    ('three_parallel_units','三个并联 RLC 单元串联',S(P('R1','L1','C1'),P('R2','L2','C2'),P('R3','L3','C3'))),
    ('two_series_units','两个串联 RLC 支路并联',P(S('R1','L1','C1'),S('R2','L2','C2'))),
    ('three_series_units','三个串联 RLC 支路并联',P(S('R1','L1','C1'),S('R2','L2','C2'),S('R3','L3','C3'))),
    ('rc_two_process','串联电阻与两个 RC 过程',S('Rs',P('R1','C1'),P('R2','C2'))),
    ('rl_two_process','串联电阻与两个 RL 过程',S('Rs',P('R1','L1'),P('R2','L2'))),
    ('rc_ladder','RC 梯形网络',S('Rs',P('C1',S('R1',P('C2','R2'))))),
    ('rlc_ladder','RLC 梯形网络',S('R1',P('C1',S('L1',P('C2',S('R2','L2')))))),
    ('rlc_background','两个谐振单元与串联 RLC 背景',S('Rs','Ls','Cs',P('R1','L1','C1'),P('R2','L2','C2'))),
    ('rlc_bridge','RLC 桥式网络',['B','R1','L1','C1','R2','R3']),
]
RULES={
    'R_amp_up':'R增大，各采样频点幅值均不降低',
    'R_phase_to_zero':'R增大，各采样频点相位绝对值均不增大',
    'C_phase_down':'C增大，各采样频点相位均不升高',
    'L_phase_up':'L增大，各采样频点相位均不降低',
    'peak_count_stable':'所有单参数正反扰动后，频带内峰数保持',
    'rms_monotone':'所有单参数正反轨迹的幅相RMS随扰动档位单调不减',
}


def clean(v):
    if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [clean(x) for x in v]
    if isinstance(v,np.ndarray): return clean(v.tolist())
    if isinstance(v,(np.integer,np.bool_)): return v.item()
    if isinstance(v,(float,np.floating)): return float(v) if np.isfinite(v) else None
    return v


def dump(path,value): path.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def csvout(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(clean(rows))


def topology(node):
    if isinstance(node,str): return node
    if node[0]=='B': return '桥：端口→u:R1；u→地:L1；端口→v:C1；v→地:R2；u↔v:R3'
    return '('+(' + ' if node[0]=='S' else ' || ').join(topology(c) for c in node[1:])+')'


def leaves(node):
    if isinstance(node,str): return [node]
    return [n for c in node[1:] for n in leaves(c)]


def rat_normalize(num,den):
    while len(num)>1 and num[-1]==0: num=num[:-1]
    while len(den)>1 and den[-1]==0: den=den[:-1]
    # Remove exact common origin factors introduced by algebraic operations.
    while len(num)>1 and len(den)>1 and num[0]==0 and den[0]==0:
        num=num[1:]; den=den[1:]
    scale=max(np.max(abs(num)),np.max(abs(den)),1e-300)
    return num/scale,den/scale


def rat_add(a,b):
    return rat_normalize(poly.polyadd(poly.polymul(a[0],b[1]),poly.polymul(b[0],a[1])),poly.polymul(a[1],b[1]))


@dataclass
class Network:
    ident: str
    family: str
    title: str
    tree: list
    names: list
    p: np.ndarray
    split: str
    damping: int
    spacing: int
    design: dict

    def response(self,f,p=None):
        pars=np.asarray(self.p if p is None else p,float)
        single=pars.ndim==1
        pars=np.atleast_2d(pars); f=np.asarray(f,float); w=2*np.pi*(f[None,:] if f.ndim==1 else f)
        lookup={n:pars[:,j,None] for j,n in enumerate(self.names)}
        def evaluate(node):
            if isinstance(node,str):
                value=lookup[node]
                if node.startswith('R'): return np.broadcast_to(value,np.broadcast_shapes(value.shape,w.shape)),np.zeros(np.broadcast_shapes(value.shape,w.shape),complex)
                z=1j*w*value if node.startswith('L') else 1/(1j*w*value)
                return z,z if node.startswith('L') else -z
            if node[0]=='B':
                ys=[]; dys=[]
                for item in node[1:]:
                    z,dz=evaluate(item); ys.append(1/z); dys.append(-dz/z**2)
                def terms(indices):
                    vals=[]; deriv=[]
                    for indices_one in indices:
                        vals.append(np.prod(np.stack([ys[k] for k in indices_one]),axis=0))
                        deriv.append(sum(dys[k]*np.prod(np.stack([ys[j] for j in indices_one if j!=k]),axis=0) for k in indices_one))
                    return sum(vals),sum(deriv)
                det,ddet=terms([(0,2),(0,3),(1,2),(1,3),(4,0),(4,1),(4,2),(4,3)])
                num,dnum=terms([(0,1,2),(0,1,3),(0,2,3),(1,2,3),(4,0,1),(4,0,3),(4,1,2),(4,2,3)])
                return det/num,(ddet*num-det*dnum)/num**2
            parts=[evaluate(c) for c in node[1:]]
            if node[0]=='S': return sum(z for z,dz in parts),sum(dz for z,dz in parts)
            y=sum(1/z for z,dz in parts); dy=-sum(dz/z**2 for z,dz in parts)
            return 1/y,-dy/y**2
        z,dz=evaluate(self.tree)
        if not np.isfinite(z).all() or np.any(z.real<=0): raise FloatingPointError(self.ident)
        if single and f.ndim==1: return z[0],dz[0]
        return z,dz

    def rational(self,p=None):
        values=dict(zip(self.names,self.p if p is None else p))
        def build(node):
            if isinstance(node,str):
                v=values[node]
                if node.startswith('R'): return np.array([v]),np.array([1.])
                if node.startswith('L'): return np.array([0.,v*WREF]),np.array([1.])
                return np.array([1.]),np.array([0.,v*WREF])
            if node[0]=='B':
                a=1/values['R1']; b=1/(values['L1']*WREF); c=values['C1']*WREF; d=1/values['R2']; e=1/values['R3']
                return rat_normalize(np.array([b*(d+e),a*d+b*c+e*(a+d),c*(a+e)]),
                    np.array([a*b*(d+e),b*c*(a+d+e)+e*a*d,c*d*(a+e)]))
            parts=[build(c) for c in node[1:]]
            if node[0]=='P': parts=[(den,num) for num,den in parts]
            result=parts[0]
            for part in parts[1:]: result=rat_add(result,part)
            if node[0]=='P': result=(result[1],result[0])
            return rat_normalize(*result)
        return build(self.tree)

    def modes(self,p=None):
        num,den=self.rational(p); result=[]
        for coef in (num,den):
            for root in np.roots(coef[::-1]):
                if abs(root)<1e-12 or root.imag < -1e-9*max(abs(root),1): continue
                center=abs(root)*WREF/(2*np.pi)
                width=.5 if abs(root.imag)<1e-8*abs(root) else min(.5,max(6*abs(root.real)/abs(root.imag),1e-6))
                if BAND[0]<=center<=BAND[1]: result.append((center,width))
        return result

    def record(self):
        return dict(id=self.ident,family=self.family,title=self.title,tree=self.tree,names=self.names,parameters=self.p,
            split=self.split,damping_bin=self.damping,spacing_bin=self.spacing,design=self.design)


def lhs(rng,n,d):
    return np.column_stack([(rng.permutation(n)+rng.random(n))/n for _ in range(d)])


def sample_network(family,title,tree,split,di,si,draw,u):
    qlo,qhi=Q_RANGES[di]; slo,shi=GAP_RANGES[si]
    f1=10**(3+2*u[0]); gap=np.exp(np.log(slo)+u[1]*np.log(shi/slo))
    fc=f1*gap**np.arange(3); z0=10**(1+2*u[2]); zc=z0*10**(1.2*(u[3:6]-.5))
    qs=np.exp(np.log(qlo)+u[6:9]*np.log(qhi/qlo)); w=2*np.pi*fc
    values={}
    for k in range(3):
        r=zc[k]
        if family in ('rlc_series','two_series_units','three_series_units','rlc_ladder','rlc_bridge'): r/=qs[k]
        elif family in ('rlc_parallel','two_parallel_units','three_parallel_units','rlc_background'): r*=qs[k]
        values.update({f'R{k+1}':r,f'L{k+1}':zc[k]/w[k],f'C{k+1}':1/(zc[k]*w[k])})
    values['Rs']=z0*10**(2*u[9]-1)
    values['Ls']=z0*10**(2*u[10]-1)/w[0]
    values['Cs']=1/(z0*w[0]*10**(2*u[11]-1))
    if family=='rlc_bridge': values['R3']=z0*10**(2*u[12]-1)
    names=leaves(tree); assert len(names)==len(set(names))
    return Network(f'{family}_{split}_d{di+1}s{si+1}_{draw:02d}',family,title,tree,names,
        np.array([values[n] for n in names]),split,di,si,
        dict(frequency_anchors_hz=fc,spacing_ratio=gap,nominal_unit_Q=qs,impedance_scales_ohm=zc))


def ensemble(pilot=False):
    models=[]
    for fi,(family,title,tree) in enumerate(FAMILIES):
        if pilot:
            rng=np.random.default_rng(20261006+1000*fi)
            models.append(sample_network(family,title,tree,'A',2,0,0,lhs(rng,1,16)[0])); continue
        for bi,split in enumerate(('A','B')):
            for di in range(3):
                for si in range(3):
                    rng=np.random.default_rng(20261006+fi*1000+bi*100+di*10+si)
                    for draw,u in enumerate(lhs(rng,8,16)):
                        models.append(sample_network(family,title,tree,split,di,si,draw,u))
    return models


def plan(model):
    n=len(model.p); directions=[('single',name,np.eye(n)[j]) for j,name in enumerate(model.names)]
    directions.extend([('structured','impedance_scale',np.array([-1 if n.startswith('C') else 1 for n in model.names])),
                       ('structured','frequency_shift',np.array([0 if n.startswith('R') else 1 for n in model.names]))])
    ls=[j for j,n in enumerate(model.names) if n.startswith('L')]; cs=[j for j,n in enumerate(model.names) if n.startswith('C')]
    if ls and cs:
        v=np.zeros(n); v[ls[0]]=1; v[cs[0]]=-1; directions.append(('structured','first_LC_reciprocal',v))
    meta=[]; params=[]; deltas=[]
    for kind,name,v in directions:
        for level in LEVELS:
            for sign in (-1,1):
                delta=sign*np.log1p(level/100)*v
                meta.append(dict(kind=kind,direction=name,level_pct=float(level),sign=sign))
                params.append(model.p*np.exp(delta)); deltas.append(delta)
    return meta,np.array(params),np.array(deltas)


def grids(model,params,refinement=1):
    """Equal global grid plus exact rational pole/zero neighborhoods per spectrum."""
    base=np.log(np.geomspace(*BAND,1024*refinement+1)); ref_modes=model.modes()
    all_modes=[ref_modes+model.modes(p) for p in params]; max_modes=max([len(m) for m in all_modes]+[0])
    result=np.full((len(params),len(base)+max_modes*(96*refinement+1)),np.log(BAND[0]))
    result[:,:len(base)]=base
    for i,modes in enumerate(all_modes):
        for k,(fc,width) in enumerate(modes):
            result[i,len(base)+k*(96*refinement+1):len(base)+(k+1)*(96*refinement+1)]=np.linspace(max(np.log(BAND[0]),np.log(fc)-width),min(np.log(BAND[1]),np.log(fc)+width),96*refinement+1)
    result.sort(axis=1)
    return np.exp(result)


def weights(f):
    x=np.log(f); d=np.diff(x,axis=-1); w=np.zeros_like(x)
    w[...,:-1]+=d/2; w[...,1:]+=d/2
    return w/np.log(BAND[1]/BAND[0])


def statistics(model,params,f):
    z,dz=model.response(f,params); z0,dz0=model.response(f,model.p); wt=weights(f)
    da=20*np.log10(abs(z)/abs(z0)); dp=np.rad2deg(np.angle(z)-np.angle(z0))
    dabs=np.rad2deg(abs(np.angle(z))-abs(np.angle(z0)))
    deriv=(z.conj()*dz).real; deriv0=(z0.conj()*dz0).real
    count=np.sum((deriv[:,:-1]>0)&(deriv[:,1:]<0),axis=1)
    count0=np.sum((deriv0[:,:-1]>0)&(deriv0[:,1:]<0),axis=1)
    return dict(magnitude_rmse_db=np.sqrt(np.sum(wt*da**2,axis=1)),phase_rmse_deg=np.sqrt(np.sum(wt*dp**2,axis=1)),
        relative_complex_rmse=np.sqrt(np.sum(wt*abs((z-z0)/z0)**2,axis=1)),
        magnitude_delta_min_db=da.min(axis=1),magnitude_delta_max_db=da.max(axis=1),
        phase_delta_min_deg=dp.min(axis=1),phase_delta_max_deg=dp.max(axis=1),
        phase_abs_delta_max_deg=dabs.max(axis=1),
        magnitude_up_frequency_fraction=np.sum(wt*(da>1e-6),axis=1),magnitude_down_frequency_fraction=np.sum(wt*(da< -1e-6),axis=1),
        phase_up_frequency_fraction=np.sum(wt*(dp>1e-6),axis=1),phase_down_frequency_fraction=np.sum(wt*(dp< -1e-6),axis=1),
        magnitude_min_frequency_hz=f[np.arange(len(f)),np.argmin(da,axis=1)],
        phase_min_frequency_hz=f[np.arange(len(f)),np.argmin(dp,axis=1)],phase_max_frequency_hz=f[np.arange(len(f)),np.argmax(dp,axis=1)],
        port_peak_count=count,reference_port_peak_count=count0,
        grid_points=np.full(len(params),f.shape[1])),(z,z0)


def audit(model,params,deltas):
    f=np.geomspace(*BAND,513); x=np.log10(f); u=(x-x[0])/(x[-1]-x[0]); z0=model.response(f)[0]; z=model.response(f,params)[0]
    a0=20*np.log10(abs(z0)); phi0=np.angle(z0); a=20*np.log10(abs(z)); phi=np.angle(z)
    g=np.gradient(z0,x,edge_order=2); speed=abs(g); positive=speed[speed>0]
    epsilon=.05*max(float(np.median(positive)) if len(positive) else float(np.median(abs(z0))),1e-12)
    den=speed**2+epsilon**2; weight=speed**2/den
    shift=((z-z0)*g.conj()).real/den
    speed_loss=np.mean(abs(abs(np.gradient(z,x,axis=1,edge_order=2))-speed)/(speed+epsilon),axis=1)
    target_ids=np.flatnonzero(np.signbit(z0.imag[:-1])!=np.signbit(z0.imag[1:]))
    mids=(u[:-1]+u[1:])/2; events=mids[target_ids]; types=np.where(z0.imag[target_ids]<z0.imag[target_ids+1],1,-1)
    candidate_cross=np.signbit(z.imag[:,:-1])!=np.signbit(z.imag[:,1:]); candidate_types=np.where(z.imag[:,:-1]<z.imag[:,1:],1,-1)
    i03=np.full(len(params),np.nan)
    if len(events):
        edges=np.r_[0,(events[:-1]+events[1:])/2,1]
        i03=sum(~np.any(candidate_cross&(candidate_types==typ)&(mids>=lo)&(mids<=hi),axis=1) for typ,lo,hi in zip(types,edges[:-1],edges[1:]))/len(events)
    ranges=np.array([np.ptp(a0),np.ptp(phi0)]); cumulative=np.full(len(params),np.nan)
    if np.all(ranges>1e-12):
        path0=np.c_[u,(a0-a0.min())/ranges[0],(phi0-phi0.min())/ranges[1]]
        path=np.stack([np.broadcast_to(u,a.shape),(a-a0.min())/ranges[0],(phi-phi0.min())/ranges[1]],axis=-1)
        l0=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path0,axis=0),axis=1))]
        lc=np.c_[np.zeros(len(params)),np.cumsum(np.linalg.norm(np.diff(path,axis=1),axis=2),axis=1)]
        e=(lc-l0)/l0[-1]; cumulative=np.sqrt(np.sum(np.diff(u)/3*(e[:,:-1]**2+e[:,:-1]*e[:,1:]+e[:,1:]**2),axis=1))
    h=1e-5; jp=model.p*np.exp(h*np.eye(len(model.p))); jm=model.p*np.exp(-h*np.eye(len(model.p)))
    jac=(model.response(f,jp)[0]-model.response(f,jm)[0])/(2*h)
    residual=z-z0; first=deltas@jac
    actual_norm=np.sqrt(np.mean(abs(residual/z0)**2,axis=1)); linear=np.sqrt(np.mean(abs((residual-first)/z0)**2,axis=1))/np.maximum(actual_norm,1e-30)
    linear[actual_norm<1e-12]=np.nan
    return dict(sampled_magnitude_mse_db2=np.mean((a-a0)**2,axis=1),sampled_phase_mse_rad2=np.mean((phi-phi0)**2,axis=1),
        I03=i03,II01=np.sum(weight*abs(shift),axis=1)/max(weight.sum(),1e-12),II02=speed_loss,
        cumulative_length=cumulative,fixed_grid_linearization_relative_error=linear)


def model_checks(model):
    f=np.geomspace(*BAND,31); z,dz=model.response(f); num,den=model.rational()
    zr=poly.polyval(1j*2*np.pi*f/WREF,num)/poly.polyval(1j*2*np.pi*f/WREF,den)
    error=np.max(abs((zr-z)/z)); assert error<1e-7,(model.ident,error)
    a=1.3; scale=np.array([-1 if n.startswith('C') else 1 for n in model.names]); shift=np.array([0 if n.startswith('R') else 1 for n in model.names])
    e1=np.max(abs((model.response(f,model.p*a**scale)[0]-a*z)/(a*z)))
    e2=np.max(abs((model.response(f,model.p*a**shift)[0]-model.response(f*a)[0])/model.response(f*a)[0]))
    assert max(e1,e2)<1e-10,(model.ident,e1,e2)
    fd=(model.response(f*np.exp(1e-6))[0]-model.response(f*np.exp(-1e-6))[0])/2e-6
    e3=np.linalg.norm(fd-dz)/max(np.linalg.norm(dz),1e-30); assert e3<1e-4,(model.ident,e3)
    result=[dict(circuit_id=model.ident,check=k,value=v,limit=limit,passed=True) for k,v,limit in [
        ('rational_vs_direct',error,1e-7),('impedance_scale_identity',e1,1e-10),('frequency_shift_identity',e2,1e-10),('analytic_frequency_derivative',e3,1e-4)]]
    if model.family=='rlc_bridge':
        p=dict(zip(model.names,model.p)); w=2*np.pi*f
        y1=np.full(len(f),1/p['R1'],complex); y2=1/(1j*w*p['L1']); y3=1j*w*p['C1']; y4=np.full(len(f),1/p['R2'],complex); y5=np.full(len(f),1/p['R3'],complex)
        matrix=np.empty((len(f),2,2),complex); matrix[:,0,0]=y1+y2+y5; matrix[:,1,1]=y3+y4+y5; matrix[:,0,1]=matrix[:,1,0]=-y5
        internal=np.linalg.solve(matrix,np.c_[y1,y3][...,None])[...,0]
        zn=1/(y1*(1-internal[:,0])+y3*(1-internal[:,1]))
        e=np.max(abs((zn-z)/z)); assert e<1e-7,(model.ident,e)
        result.append(dict(circuit_id=model.ident,check='bridge_independent_nodal_solution',value=e,limit=1e-7,passed=True))
    return result


def rules_for_circuit(model,rows):
    result=[]
    for level in LEVELS:
        subset=[r for r in rows if r['kind']=='single' and r['level_pct']==level]
        positive=[r for r in subset if r['sign']==1]
        for rule,letter,key,test in [
            ('R_amp_up','R','magnitude_delta_min_db',lambda x:x>=-1e-6),
            ('R_phase_to_zero','R','phase_abs_delta_max_deg',lambda x:x<=1e-6),
            ('C_phase_down','C','phase_delta_max_deg',lambda x:x<=1e-6),
            ('L_phase_up','L','phase_delta_min_deg',lambda x:x>=-1e-6)]:
            rr=[r for r in positive if r['direction'].startswith(letter)]
            if rr: result.append(dict(circuit_id=model.ident,family=model.family,split=model.split,damping_bin=model.damping+1,spacing_bin=model.spacing+1,
                rule=rule,level_pct=float(level),supported=all(test(r[key]) for r in rr),parameter_trials=len(rr)))
        result.append(dict(circuit_id=model.ident,family=model.family,split=model.split,damping_bin=model.damping+1,spacing_bin=model.spacing+1,
            rule='peak_count_stable',level_pct=float(level),supported=all(r['port_peak_count']==r['reference_port_peak_count'] for r in subset),parameter_trials=len(subset)))
    stable=True
    for name in model.names:
        for sign in (-1,1):
            rr=sorted([r for r in rows if r['kind']=='single' and r['direction']==name and r['sign']==sign],key=lambda r:r['level_pct'])
            for field in ('magnitude_rmse_db','phase_rmse_deg'):
                values=np.array([r[field] for r in rr]); stable &= bool(np.all(np.diff(values)>=-1e-6))
    result.append(dict(circuit_id=model.ident,family=model.family,split=model.split,damping_bin=model.damping+1,spacing_bin=model.spacing+1,
        rule='rms_monotone',level_pct=30.,supported=stable,parameter_trials=len(model.names)*2))
    return result


def aggregate(records):
    groups={}
    for r in records:
        key=(r['family'],r['split'],r['rule'],r['level_pct'])
        groups.setdefault(key,[]).append(r)
    result=[]
    for (family,split,rule,level),rr in groups.items():
        n=len(rr); good=sum(r['supported'] for r in rr)
        result.append(dict(family=family,split=split,rule=rule,level_pct=level,circuits=n,supported_circuits=good,support_fraction=good/n))
    return result


def candidate_examples(model,rows,current):
    for r in rows:
        if r['kind']!='single' or r['sign']!=1 or r['level_pct']!=10: continue
        typ=None; strength=0
        if r['direction'].startswith('R') and r['magnitude_delta_min_db']<-.05:
            typ='R_amplitude_counterexample'; strength=-r['magnitude_delta_min_db']
        elif r['direction'].startswith('C') and r['phase_delta_max_deg']>.5:
            typ='C_phase_counterexample'; strength=r['phase_delta_max_deg']
        elif r['direction'].startswith('L') and r['phase_delta_min_deg']<-.5:
            typ='L_phase_counterexample'; strength=-r['phase_delta_min_deg']
        if typ:
            key=(model.family,typ)
            if key not in current or strength>current[key]['strength']:
                current[key]=dict(model=model,row=r,strength=strength,type=typ)


def parameter_aggregates(model,rows,groups):
    port={n:('port_series' if model.tree[0]=='S' else 'port_parallel') for n in model.tree[1:] if isinstance(n,str) and model.tree[0] in ('S','P')}
    for r in rows:
        if r['kind']!='single' or r['sign']!=1: continue
        name=r['direction']; letter=name[0]
        tests=[]
        if letter=='R': tests=[('R_amp_up',r['magnitude_delta_min_db']>=-1e-6),('R_phase_to_zero',r['phase_abs_delta_max_deg']<=1e-6)]
        elif letter=='C': tests=[('C_phase_down',r['phase_delta_max_deg']<=1e-6)]
        elif letter=='L': tests=[('L_phase_up',r['phase_delta_min_deg']>=-1e-6)]
        for rule,supported in tests:
            key=(model.family,model.split,name,port.get(name,'internal'),rule,r['level_pct'])
            counts=groups.setdefault(key,[0,0]); counts[0]+=1; counts[1]+=int(supported)


def style():
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,
        'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'#fafcfe','text.color':'#213a55'})


def savefig(fig,path): fig.savefig(path,dpi=150,bbox_inches='tight'); plt.close(fig)


def parameter_plot(model,name,path):
    idx=model.names.index(name); v=np.eye(len(model.p))[idx]
    pars=np.array([model.p*np.exp(s*np.log1p(l/100)*v) for l in (1.,10.,30.) for s in (-1,1)])
    f=grids(model,pars,2); z,z0=statistics(model,pars,f)[1]
    fig,axes=plt.subplots(2,2,figsize=(12.5,8)); colors=['#a3c4e5','#edb5af','#5991c8','#d4776d',BLUE,RED]
    for i,(l,s) in enumerate([(l,s) for l in (1.,10.,30.) for s in (-1,1)]):
        ls=':' if l==1 else '--' if l==10 else '-'; label=f'{name} ×{np.exp(s*np.log1p(l/100)):.4g}'
        axes[0,0].semilogx(f[i],20*np.log10(abs(z[i])),color=colors[i],ls=ls,label=label)
        axes[0,1].semilogx(f[i],np.rad2deg(np.angle(z[i])),color=colors[i],ls=ls)
        axes[1,0].semilogx(f[i],20*np.log10(abs(z[i])/abs(z0[i])),color=colors[i],ls=ls)
        axes[1,1].semilogx(f[i],np.rad2deg(np.angle(z[i])-np.angle(z0[i])),color=colors[i],ls=ls)
    axes[0,0].semilogx(f[0],20*np.log10(abs(z0[0])),color='#172b40',lw=1.5,label='参考')
    axes[0,1].semilogx(f[0],np.rad2deg(np.angle(z0[0])),color='#172b40',lw=1.5)
    labels=['20log10|Z| / dBΩ','相位 / °','幅值变化 / dB','相位变化 / °']
    for ax,label in zip(axes.flat,labels): ax.set_xlabel('频率 / Hz'); ax.set_ylabel(label); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8); axes[1,0].axhline(0,color='#172b40',lw=.7); axes[1,1].axhline(0,color='#172b40',lw=.7)
    fig.suptitle(model.title+'｜'+name+'｜'+model.ident,fontsize=12,y=1.01); fig.tight_layout(); savefig(fig,path)


def overview_plot(models,path):
    fig,axes=plt.subplots(1,2,figsize=(13,4.8)); f=np.geomspace(*BAND,2501)
    colors=plt.cm.viridis(np.linspace(.05,.95,len(models)))
    for m,c in zip(models,colors):
        # Only a baseline coverage illustration; sharp peaks use model-guided grid.
        ff=grids(m,np.array([m.p]))[0]; z=m.response(ff)[0]
        label=f'构造档 d{m.damping+1}/间距 s{m.spacing+1}'
        axes[0].semilogx(ff,20*np.log10(abs(z)),color=c,lw=1,label=label)
        axes[1].semilogx(ff,np.rad2deg(np.angle(z)),color=c,lw=1)
    axes[0].set_ylabel('幅值 / dBΩ'); axes[1].set_ylabel('相位 / °')
    for ax in axes: ax.set_xlabel('频率 / Hz'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7,ncol=2); fig.suptitle(models[0].title+'｜九个构造区域各一例',y=1.01); fig.tight_layout(); savefig(fig,path)


def summary_plots(summary,refs,out):
    family_names=[t for f,t,tree in FAMILIES]; family_codes=[f for f,t,tree in FAMILIES]
    fig,axes=plt.subplots(1,2,figsize=(14,7),sharey=True)
    for ax,split in zip(axes,('A','B')):
        matrix=np.full((16,4),np.nan)
        for i,f in enumerate(family_codes):
            for j,rule in enumerate(('R_amp_up','R_phase_to_zero','C_phase_down','L_phase_up')):
                rr=[r for r in summary if r['family']==f and r['split']==split and r['rule']==rule and r['level_pct']==10]
                if rr: matrix[i,j]=rr[0]['support_fraction']
        im=ax.imshow(matrix,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
        ax.set_xticks(range(4),['R：幅值不降','R：|相位|不增','C：相位不升','L：相位不降'],rotation=35,ha='right',fontsize=9)
        ax.set_yticks(range(16),family_names); ax.set_title(f'独立批次 {split}：参数正向×1.1')
        for i in range(16):
            for j in range(4):
                if np.isfinite(matrix[i,j]): ax.text(j,i,f'{100*matrix[i,j]:.0f}%',ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>.7 else '#17324d')
    fig.suptitle('经验命题必须按连接位置和拓扑判断（空格=无该类元件）',y=.995)
    fig.subplots_adjust(left=.23,bottom=.22,right=.88,wspace=.15,top=.92)
    fig.colorbar(im,cax=fig.add_axes([.915,.29,.018,.55]),label='满足命题的基准电路比例')
    savefig(fig,out/'figures/rule_support.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for split,c in [('A',BLUE),('B',RED)]:
        rr=[r for r in refs if r['split']==split]
        axes[0].scatter([r['design']['frequency_anchors_hz'][0] for r in rr],[r['design']['nominal_unit_Q'][0] for r in rr],s=5,alpha=.2,color=c,label=split)
        axes[1].scatter([r['design']['spacing_ratio'] for r in rr],[r['design']['impedance_scales_ohm'][0] for r in rr],s=5,alpha=.2,color=c,label=split)
    for ax in axes: ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(alpha=.2); ax.legend()
    axes[0].set_xlabel('第一频率锚点 / Hz'); axes[0].set_ylabel('元件构造的名义 Q 档')
    axes[1].set_xlabel('相邻锚点频率比例'); axes[1].set_ylabel('阻抗构造尺度 / Ω')
    fig.suptitle('参数覆盖：耦合网络构造 Q 不等于端口真实 Q',y=1.01); fig.tight_layout(); savefig(fig,out/'figures/parameter_coverage.png')


def make_report(summary,rule_records,parameter_summary,examples,checks,refs,total,elapsed):
    def table(headers,rows):
        return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(str(v).replace('|',r'\|') for v in r)+' |\n' for r in rows)
    def rate(f,rule,split):
        r=next((r for r in summary if r['family']==f and r['rule']==rule and r['level_pct']==10 and r['split']==split),None)
        return '—' if r is None else f"{r['supported_circuits']}/{r['circuits']} ({100*r['support_fraction']:.1f}%)"
    def parameter_rate(f,name,rule,split):
        r=next((r for r in parameter_summary if r['family']==f and r['parameter']==name and r['rule']==rule and r['level_pct']==10 and r['split']==split),None)
        return '—' if r is None else str(r['supported_circuits'])+'/'+str(r['circuits'])
    port_rows=[]
    for family,name,role in sorted({(r['family'],r['parameter'],r['port_role']) for r in parameter_summary if r['port_role'] in ('port_series','port_parallel') and r['parameter'].startswith('R')}):
        port_rows.append([next(t for f,t,tr in FAMILIES if f==family)+' / '+name,'端口串联' if role=='port_series' else '端口并联']+[' / '.join(parameter_rate(family,name,rule,split) for split in ('A','B')) for rule in ('R_amp_up','R_phase_to_zero')])
    lines=['# 大样本电路参数—阻抗变化研究','',
        f'2026-10-06。新增16种连接模板、{len(refs):,}个基准电路、{total:,}条有限扰动谱，全部本地计算；两批独立固定种子样本，未运行优化器。每条谱的频点不是独立电路。','',
        '## 本轮回答什么','',
        '第一轮的5个自设计例子适合解释机制，不能代表广泛参数区域。本轮对每种模板加入144组参数，以两批独立样本检查经验方向是否稳定；同时主动寻找反例，避免把某个元件在一个电路中的作用写成普遍规律。',
        '两个精确缩放关系适用于本轮所有理想RLC网络，并有解析依据：R、L同乘a且C除以a，使Z乘a；R不变而L、C同乘a，使Z_new(f)=Z_old(af)。大量样本用于检查实现和覆盖，普遍性来自元件公式与网络串并联/节点方程的齐次关系。',
        '', '## 模板与参数覆盖','',
        table(['模板','连接关系','元件数','基准样本'],[[t,topology(tree),len(leaves(tree)),sum(r['family']==f for r in refs)] for f,t,tree in FAMILIES]),'',
        '每批72组/模板，按3个名义阻尼档×3个频率间距档×8组Latin hypercube样本构造。第一频率锚点1–100kHz；相邻比例1.05–1.4、1.5–4、6–30；Q构造档0.5–2、2–20、20–200。独立RLC单元的Q由公式定义；梯形/桥式等耦合网络的这些档只是元件构造控制，不能称为端口Q。RC/RL没有谐振Q，其144组样本不作Q解释。','',
        '![参数覆盖](figures/parameter_coverage.png)','',
        '模板不同不保证端口函数互不等价。本轮研究的是连接位置和参数区域下的响应，不把可交换支路标签、频点或扰动档位重复算成新增电路。范围仅为理想正值被动集总RLC，仍不能覆盖任意电路。','',
        '## 哪些直觉能稳定成立，哪些有反例','',
        '下表参数正向×1.1。每个基准电路只有在其所有该类元件都满足命题时才记支持；按电路计数，分批报告。条件为所计算的加密频点、1e-6 dB/°容差内不违反指定方向，不是未观测连续频率上的严格证明。', '',
        table(['模板','R幅值不降 A / B','R相位向0 A / B','C相位不升 A / B','L相位不降 A / B'],[[t]+[rate(f,k,'A')+' / '+rate(f,k,'B') for k in ('R_amp_up','R_phase_to_zero','C_phase_down','L_phase_up')] for f,t,tree in FAMILIES]),'',
        '![分拓扑命题支持率](figures/rule_support.png)','',
        '必须把串联电阻与并联电阻区分开。例如孤立串联RLC增大R会使相位绝对值减小；孤立并联RLC增大R在非过零位置反而使相位绝对值增大。这是公式可解释的条件规律，不应将两个方向平均成一个“电阻效应”。',
        '多单元或耦合网络中，一个参数可以在不同频段引起相反变化。完整逐扰动数据同时保存上升/下降频段占比、幅相RMS、峰数及反例频率，不只保存一个总误差。','',
        '## 可解析解释的端口元件规律','',
        '全电路串联的电阻增加ΔR>0时，Z_new=Z+ΔR；被动网络ReZ≥0，因此|Z_new|²−|Z|²=2ΔR ReZ+ΔR²>0，且相位绝对值减小。这个结论依赖“串联在整个端口”，不适用于任意内部电阻。',
        '在整个端口串联的L增大，或串联C增大，都使电抗上升，所以相位上升；在整个端口并联的C增大，或并联L增大，都使导纳虚部上升，所以阻抗相位下降。连接位置是方向规律的一部分。',
        '并联在整个端口的电阻增大使电导下降，因此幅值上升、相位绝对值通常增大。简单串并联例子是这些一般端口关系的特例。',
        '按具体参数及连接位置的支持比例保存在 parameter_rule_summary.csv；这样不会把串联背景Rs的稳定作用掩盖在“所有电阻是否同时满足”的严格汇总中。','',
        table(['模板/元件','位置','R幅值不降 A / B','R相位向0 A / B'],port_rows),'',
        '## 反例：先排除数值假象，再解释机制','',
        '反例按10档正向下最大的方向违反量确定性选出，用加密网格复算确认。它们用于反驳无条件命题，不代表该现象常见；常见程度应看上面的全样本比例。','',
        table(['模板','参数','反例类型','违反幅度','基准编号'],[[e['model'].title,e['row']['direction'],e['type'],f"{e['strength']:.5g}"+(' dB' if e['type']=='R_amplitude_counterexample' else '°'),e['model'].ident] for e in examples]),'',
    ]
    for i,e in enumerate(examples): lines.extend([f"![反例{i+1}](figures/counterexample_{i+1:02d}.png)",''])
    lines.extend(['## 扰动大小与峰数','',
        '单参数与结构组合均取0.1/0.3/1/3/10/30档正向和倒数负向。每个参数相对参考的RMS是否随档位增加而单调，另按电路聚合保存在 circuit_rules.csv；不预设“大参数偏差必然对应更大的所有曲线特征误差”。',
        '峰数只是观察频带内的局部幅值极大值数量。改变参数后峰数变化可能来自峰合并、背景掩盖或跨出频带，不能直接叫作内部模式消失；本轮不在大样本中用固定窗强行匹配峰宽。','',
        '## 旧指标的机制审查','',
        'I-03、II-01、II-02和累计弧长在统一513点等log频率网格上计算，另存 sampled_* 列。这些数值不能和Case1/Case2正式180/201点成绩直接混比。无目标过零时I-03为未定义，而非零损失。',
        '纯整体阻抗缩放保持相位和所有特征频率不变，I-03及累计弧长不响应，而II-01和II-02通常响应。II-01的切向投影也接收幅值变化，因此不能把它当成真实频移的直接量尺。这个结论与第一轮分析一致，但并不自动决定该指标对优化是否有用。','',
        '## 数值验证与研究边界','',
        f'网络有理式与直接复阻抗计算、两种缩放恒等式及解析频率导数均逐基准检查；桥式另与独立节点方程核对。共{len(checks):,}条检查，失败数{sum(not r["passed"] for r in checks)}。正式计算耗时{elapsed/60:.2f}分钟。',
        '极点/零点引导每条谱独立加密，积分按等log频率权重，重复补齐网格点权重为零。每个构造区域的预定样本做网格加密对照，反例另加密；不宣称已对全部参数空间建立误差界。参数正反变化为乘性倒数，30档负向约-23.08%，并非-30%。',
        '两批覆盖同一设计范围，支持率是该抽样设计下的描述统计，不是任意真实电路的发生概率，也没有用曲线数量夸大独立样本量。没有噪声或新实测数据，未证明指标的优化收益。','',
        '## 如何继续使用','',
        '优先保留“带条件的规律”：元件位于哪里、频段在哪里、阻尼与谐振重叠程度怎样。将高支持率命题与反例一起用于指标审查；若要推广到文献电路或真实设备，再补充结构不同的模型和测量数据。',
        '所有基准参数保存在 data/reference_circuits.json；全量扰动在 data/perturbations.csv；按电路聚合的结果在 data/circuit_rules.csv，按模板/批次汇总在 data/rule_summary.csv。脚本可按任意基准编号和参数重新画图，不需要重跑整批。',''])
    return '\n'.join(lines)


def make_html(out,report,examples):
    spec=importlib.util.spec_from_file_location('population_gallery_site',Path(__file__).with_name('gallery_site.py'))
    site=importlib.util.module_from_spec(spec); spec.loader.exec_module(site)
    site.render(out,FAMILIES,report)



def load_record(record):
    return Network(record['id'],record['family'],record['title'],record['tree'],record['names'],np.asarray(record['parameters']),record['split'],record['damping_bin'],record['spacing_bin'],record['design'])


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--pilot',action='store_true'); ap.add_argument('--output',type=Path,default=OUT_DEFAULT)
    ap.add_argument('--plot-id'); ap.add_argument('--parameter'); args=ap.parse_args(); out=args.output.resolve()
    if args.pilot and args.output==OUT_DEFAULT: out=ROOT/'output/parameter_curve_population_20261006_pilot'
    (out/'data').mkdir(parents=True,exist_ok=True); (out/'figures').mkdir(exist_ok=True); style()
    if args.plot_id:
        records=json.loads((out/'data/reference_circuits.json').read_text(encoding='utf-8'))
        record=next(r for r in records if r['id']==args.plot_id); model=load_record(record)
        if args.parameter not in model.names: raise ValueError('Choose one of '+','.join(model.names))
        file=out/'figures'/f'custom_{args.plot_id}_{args.parameter}.png'; parameter_plot(model,args.parameter,file); print(file); return
    models=ensemble(args.pilot); start=time.time(); refs=[m.record() for m in models]
    dump(out/'data/reference_circuits.json',refs)
    dump(out/'data/PROTOCOL.json',dict(local_only=True,optimizer_runs=0,baseline_circuits=len(models),topology_templates=16,
        sampling='two independent seeds; 9 construction strata x 8 LHS draws per split and template',pilot=args.pilot,
        seed_formula='20261006+family_index*1000+split_index*100+damping_index*10+spacing_index',frequency_band_hz=BAND,
        levels_pct=LEVELS,negative_factor='1/(1+level/100)',source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),Path(__file__).with_name('README.md')]}))
    all_rules=[]; all_checks=[]; candidates={}; parameter_groups={}; scale_audit=[]; total=0; writer=None
    with (out/'data/perturbations.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        for index,model in enumerate(models):
            all_checks.extend(model_checks(model)); meta,params,deltas=plan(model)
            results=[]
            for offset in range(0,len(params),48):
                pp=params[offset:offset+48]; ff=grids(model,pp); stat,_=statistics(model,pp,ff)
                for k in range(len(pp)): results.append({name:value[k] for name,value in stat.items()})
            aux=audit(model,params,deltas); rows=[]
            for j,(m,r) in enumerate(zip(meta,results)):
                row=dict(circuit_id=model.ident,family=model.family,split=model.split,damping_bin=model.damping+1,spacing_bin=model.spacing+1,**m,**r)
                row.update({name:v[j] for name,v in aux.items()}); rows.append(row)
            # Predetermined coverage checks: first draw of every construction cell, both independent splits.
            if args.pilot or model.ident.endswith('_00'):
                selected=[j for j,m in enumerate(meta) if m['level_pct'] in (.1,30.) and m['sign']==1 and m['direction'] in (model.names[0],'frequency_shift')]
                ff=grids(model,params[selected],2); fine,_=statistics(model,params[selected],ff)
                for k,j in enumerate(selected):
                    for field in ('magnitude_rmse_db','phase_rmse_deg','relative_complex_rmse'):
                        err=abs(fine[field][k]-rows[j][field])/max(fine[field][k],1e-12)
                        all_checks.append(dict(circuit_id=model.ident,check=f'grid_{meta[j]["direction"]}_{meta[j]["level_pct"]}_{field}',value=err,limit=.01,passed=bool(err<.01)))
                        assert err<.01,(model.ident,meta[j],field,err)
            if writer is None: writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader()
            writer.writerows(clean(rows)); total+=len(rows); all_rules.extend(rules_for_circuit(model,rows)); candidate_examples(model,rows,candidates)
            parameter_aggregates(model,rows,parameter_groups)
            for r in rows:
                if r['direction']=='impedance_scale' and r['level_pct']==30 and r['sign']==1:
                    scale_audit.append({k:r[k] for k in ('circuit_id','family','split','sampled_magnitude_mse_db2','sampled_phase_mse_rad2','I03','II01','II02','cumulative_length')})
            if (index+1)%24==0 or args.pilot:
                stream.flush(); print(json.dumps(dict(completed_circuits=index+1,total_circuits=len(models),spectra=total,seconds=round(time.time()-start,1)),ensure_ascii=False),flush=True)
    csvout(out/'data/circuit_rules.csv',all_rules); summary=aggregate(all_rules); csvout(out/'data/rule_summary.csv',summary)
    parameter_summary=[dict(family=key[0],split=key[1],parameter=key[2],port_role=key[3],rule=key[4],level_pct=key[5],circuits=counts[0],supported_circuits=counts[1],support_fraction=counts[1]/counts[0]) for key,counts in parameter_groups.items()]
    csvout(out/'data/parameter_rule_summary.csv',parameter_summary)
    csvout(out/'data/scaling_metric_audit.csv',scale_audit)
    # At most one magnitude-decrease example per template, plus strongest global C/L direction reversals.
    examples=[e for (f,t),e in candidates.items() if t=='R_amplitude_counterexample']
    for typ in ('C_phase_counterexample','L_phase_counterexample'):
        rr=[e for (f,t),e in candidates.items() if t==typ]
        if rr: examples.append(max(rr,key=lambda e:e['strength']))
    examples=sorted(examples,key=lambda e:(e['model'].family,e['type']))
    for i,e in enumerate(examples):
        model=e['model']; row=e['row']; j=model.names.index(row['direction']); pp=np.array([model.p*np.exp(np.log1p(.1)*np.eye(len(model.p))[j])])
        ff=grids(model,pp,4); stat,_=statistics(model,pp,ff)
        field='magnitude_delta_min_db' if e['type']=='R_amplitude_counterexample' else 'phase_delta_max_deg' if e['type']=='C_phase_counterexample' else 'phase_delta_min_deg'
        value=float(stat[field][0]); passed=value<-.05 if field=='magnitude_delta_min_db' else value>.5 if field=='phase_delta_max_deg' else value<-.5
        assert passed,(model.ident,row['direction'],field,value)
        all_checks.append(dict(circuit_id=model.ident,check='counterexample_dense_sign_confirmation',value=0.,limit=1e-12,passed=True))
        parameter_plot(model,row['direction'],out/'figures'/f'counterexample_{i+1:02d}.png')
    csvout(out/'data/numerical_checks.csv',all_checks)
    dump(out/'data/counterexamples.json',[dict(reference=e['model'].record(),direction=e['row']['direction'],kind=e['type'],strength=e['strength'],original_row=e['row']) for e in examples])
    summary_plots(summary,clean(refs),out)
    for family,title,tree in FAMILIES:
        selected=[m for m in models if m.family==family and m.split=='A' and m.ident.endswith('_00')]
        overview_plot(selected,out/'figures'/f'{family}_overview.png')
        model=next((m for m in selected if m.damping==1 and m.spacing==1),selected[0])
        for letter in ('R','L','C'):
            names=[n for n in model.names if n.startswith(letter)]
            if names: parameter_plot(model,names[0],out/'figures'/f'{family}_parameter_{names[0]}.png')
    elapsed=time.time()-start; report=make_report(summary,all_rules,parameter_summary,examples,all_checks,refs,total,elapsed)
    (out/'REPORT.md').write_text(report,encoding='utf-8'); make_html(out,report,examples)
    files={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and p.name!='INTEGRITY.json'}
    dump(out/'INTEGRITY.json',dict(completed=True,local_only=True,optimizer_runs=0,baseline_circuits=len(models),topology_templates=16,perturbed_spectra=total,
        numerical_checks=len(all_checks),all_checks_passed=all(r['passed'] for r in all_checks),counterexamples=len(examples),elapsed_seconds=elapsed,
        figures=len(list((out/'figures').glob('*.png'))),artifacts_sha256=files))
    print(json.dumps(dict(completed=True,baseline_circuits=len(models),spectra=total,checks=len(all_checks),seconds=round(elapsed,1),output=str(out)),ensure_ascii=False),flush=True)


if __name__=='__main__': main()

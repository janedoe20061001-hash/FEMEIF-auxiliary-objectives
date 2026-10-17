"""Read-only investigation of recovered solutions; never runs an optimizer.
Descriptor MNA modal decomposition is independent of expanded polynomial roots.
Diagnostic hybrid modal responses are not asserted to be realizable RLC circuits.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys,json,csv,hashlib
import numpy as np
from scipy import linalg
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
CODE=ROOT/'output/complex_joint_20261012/retrieved/experiments/complex_joint_20261012'
DATA=ROOT/'output/complex_joint_20261012/retrieved/output/complex_joint_20261012/jobs'
sys.path.insert(0,str(CODE))
from core import target,CFG
from features import Features,areas,average
WREF=2*np.pi*1e4

def read(p): return json.loads(p.read_text(encoding='utf8'))
def save(name,rows):
    if not rows:return
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def load(ident,method,seed,gen=300):
    d=DATA/ident/method/f'seed_{seed}'
    with np.load(d/f'generation_{gen:04d}.npz') as a:
        ix=int(a['selected_index']);p=10**a['X'][ix];F=a['F'][ix]
    return p,F,read(d/'audit.json')

def descriptor(c,p):
    edges=c.netlist();nodes=sorted({n for _,a,b in edges for n in (a,b) if n!=0})
    loc={n:i for i,n in enumerate(nodes)};inds=[n for n,a,b in edges if n.startswith('L')]
    m=len(nodes)+len(inds);G=np.zeros((m,m));E=np.zeros((m,m));vals=dict(zip(c.names,p))
    for name,a,b in edges:
        v=np.zeros(len(nodes))
        if a:v[loc[a]]+=1
        if b:v[loc[b]]-=1
        if name.startswith('R'): G[:len(nodes),:len(nodes)]+=np.outer(v,v)/vals[name]
        elif name.startswith('C'): E[:len(nodes),:len(nodes)]+=np.outer(v,v)*vals[name]*WREF
        else:
            j=len(nodes)+inds.index(name);G[:len(nodes),j]+=v;G[j,:len(nodes)]-=v;E[j,j]=vals[name]*WREF
    b=np.zeros(m);b[loc[1]]=1
    # Symmetric row/column equilibration preserves transfer function.
    scale=1/np.sqrt(np.maximum(np.max(abs(G)+abs(E),axis=1),1e-30))
    G=scale[:,None]*G*scale[None,:];E=scale[:,None]*E*scale[None,:];b=b*scale
    eig,vl,vr=linalg.eig(-G,E,left=True,right=True,homogeneous_eigvals=True)
    finite=abs(eig[1])>1e-12*(abs(eig[0])+abs(eig[1]))
    poles=eig[0,finite]/eig[1,finite];vl=vl[:,finite];vr=vr[:,finite]
    residues=np.array([(b@vr[:,j])*(vl[:,j].conj()@b)/(vl[:,j].conj()@E@vr[:,j]) for j in range(len(poles))])
    ga=np.zeros((m+1,m+1));ea=np.zeros_like(ga);ga[:m,:m]=G;ga[:m,m]=-b;ga[m,:m]=b;ea[:m,:m]=E
    ze=linalg.eigvals(-ga,ea,homogeneous_eigvals=True)
    zfinite=abs(ze[1])>1e-12*(abs(ze[0])+abs(ze[1]));zeros=ze[0,zfinite]/ze[1,zfinite]
    num,den=c.rational(p)
    for vv,coef in ((poles,den),(zeros,num)):
        roots=np.roots(coef[::-1]);assert len(roots)==len(vv),(len(roots),len(vv))
        assert max(min(abs(roots-v))/max(abs(v),1e-10) for v in vv)<1e-6
    f=np.geomspace(10,1e7,257);u=1j*2*np.pi*f/WREF
    zz,_=c.response(f,p)
    part=np.sum(residues[None,:]/(u[:,None]-poles[None,:]),axis=1)
    # All passive RLC driving-point impedances have at most a first-order polynomial background.
    bg=np.linalg.lstsq(np.column_stack([np.ones(len(u)),u]),zz-part,rcond=None)[0]
    reconstruction=part+bg[0]+bg[1]*u
    error=float(np.max(abs(reconstruction-zz)/np.maximum(abs(zz),1e-12)))
    node=np.array([b@linalg.solve(G+v*E,b) for v in u[::16]])
    nodeerr=float(np.max(abs(node-zz[::16])/np.maximum(abs(zz[::16]),1e-12)))
    residual=max(float(np.linalg.norm((G+po*E)@vr[:,j]) / max((np.linalg.norm(G)+abs(po)*np.linalg.norm(E))*np.linalg.norm(vr[:,j]),1e-30)) for j,po in enumerate(poles))
    assert error<1e-8 and nodeerr<1e-8 and residual<1e-8,(c.ident,error,nodeerr,residual)
    return dict(poles=poles,res=residues,bg=bg,zeros=zeros,error=error,node_error=nodeerr,eig_residual=residual)

def response(mod,f):
    u=1j*2*np.pi*np.asarray(f)/WREF
    return np.sum(mod['res'][None,:]/(u[:,None]-mod['poles'][None,:]),axis=1)+mod['bg'][0]+mod['bg'][1]*u
def groups(mod):
    pp=mod['poles'];used=set();out=[]
    for j,p in enumerate(pp):
        if j in used:continue
        if abs(p.imag)>1e-8*max(abs(p),1e-12):
            k=int(np.argmin(abs(pp-p.conjugate())));ix=[j,k];used.update(ix)
            freq=abs(p.imag)*1e4
        else:ix=[j];used.add(j);freq=abs(p.real)*1e4
        out.append(dict(ix=ix,freq=float(freq),alpha=float(-p.real*WREF),osc=len(ix)==2))
    return sorted(out,key=lambda g:g['freq'])
def local_score(z,t,x):
    return float(np.trapezoid(abs(z-t)**2,x=x)/(x[-1]-x[0])/(np.max(abs(t))**2))

def mixed():
    modalrows=[];cards=[];hybrids=[];weights=[];sensitivity=[];factorial=[]
    allmods={}
    for ident,new,band in [('mixed_resonators/base','D',(45000,58000)),('mixed_resonators/lc_reciprocal_8p0','A',(38500,45000))]:
        d=target(ident);c=d['circuit'];tm=descriptor(c,c.p);tg=groups(tm)
        print('target',ident,[(g['freq'],g['alpha'],g['ix']) for g in tg],flush=True)
        # High oscillatory port mode is explicitly identified by target frequency, before seed results.
        hi=max((g for g in tg if g['osc']),key=lambda g:g['freq']);thi=hi['ix']
        x=np.linspace(np.log10(band[0]),np.log10(band[1]),4097);f=10**x;t,_=c.response(f)
        modes=dict(target=tm)
        for method in ['MAG_PHA',new]:
            for seed in CFG['seeds']:
                p,F,a=load(ident,method,seed);cm=descriptor(c,p);gg=groups(cm)
                cost=np.array([[abs(np.log10(max(g['freq'],1e-20)/max(q['freq'],1e-20)))+(0 if g['osc']==q['osc'] else 10) for q in gg] for g in tg])
                ri,cj=linear_sum_assignment(cost);matching=dict(zip(ri,cj));gi=matching[tg.index(hi)];cg=gg[gi]
                assert cg['osc'] and len(tg)==len(gg)
                ci=sorted(cg['ix'],key=lambda j:cm['poles'][j].imag);ti=sorted(thi,key=lambda j:tm['poles'][j].imag)
                z,_=c.response(f,p);modes[f'{method}_{seed}']=cm
                ar=[e for e in a['amplitude_events'] if e['event_id'] in ('P2','V3')]
                target_res=tm['res'][ti[-1]]*WREF;candidate_res=cm['res'][ci[-1]]*WREF
                current=local_score(z,t,x)
                cards.append(dict(target=ident,method=method,seed=seed,accepted=a['accepted'],mag=a['objective'][0],pha=a['objective'][1],local_complex_error=current,
                    pair_complete=sum(e['complete'] for e in ar),mode_frequency_hz=cg['freq'],target_mode_frequency_hz=hi['freq'],frequency_error_hz=cg['freq']-hi['freq'],
                    decay_per_s=cg['alpha'],target_decay_per_s=hi['alpha'],decay_ratio=cg['alpha']/hi['alpha'],
                    residue_abs_ratio=abs(candidate_res)/abs(target_res),residue_angle_delta_deg=float(np.angle(candidate_res/target_res)*180/np.pi),
                    visibility_ratio=(abs(candidate_res)/cg['alpha'])/(abs(target_res)/hi['alpha']),
                    nearest_zero_distance_over_decay=float(min(abs(cm['zeros']-cm['poles'][ci[-1]]))/abs(cm['poles'][ci[-1]].real)),
                    residue_complex_relative_error=abs(candidate_res-target_res)/abs(target_res),reconstruction_error=cm['error'],node_error=cm['node_error']))
                for j,po in enumerate(cm['poles']):
                    modalrows.append(dict(target=ident,method=method,seed=seed,mode_index=j,pole_real_per_s=po.real*WREF,pole_imag_per_s=po.imag*WREF,
                        residue_real=cm['res'][j].real*WREF,residue_imag=cm['res'][j].imag*WREF,bg_constant_real=cm['bg'][0].real,bg_inductance_real=cm['bg'][1].real/WREF,
                        reconstruction_error=cm['error'],node_error=cm['node_error'],eigen_residual=cm['eig_residual']))
                # Factorial 2^3 swap: high-mode frequency, decay and residue. Background/other modes remain fixed.
                for mask in range(8):
                    hh={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in cm.items()}
                    for ii,jj in zip(ci,ti):
                        po=hh['poles'][ii]
                        hh['poles'][ii]=(tm['poles'][jj].real if mask&2 else po.real)+1j*(tm['poles'][jj].imag if mask&1 else po.imag)
                        if mask&4:hh['res'][ii]=tm['res'][jj]
                    score=local_score(response(hh,f),t,x)
                    factorial.append(dict(target=ident,method=method,seed=seed,swap_frequency=bool(mask&1),swap_decay=bool(mask&2),swap_residue=bool(mask&4),local_error=score,
                        reduction_fraction=1-score/max(current,1e-30),note='Diagnostic response; not optimized or guaranteed realizable circuit'))
                # Replace only high mode / everything except high mode using target decomposition.
                hc=np.sum(cm['res'][None,ci]/(1j*2*np.pi*f[:,None]/WREF-cm['poles'][None,ci]),axis=1)
                ht=np.sum(tm['res'][None,ti]/(1j*2*np.pi*f[:,None]/WREF-tm['poles'][None,ti]),axis=1)
                for name,hz in [('target_high_mode',z-hc+ht),('target_rest',t-ht+hc)]:
                    er=local_score(hz,t,x);hybrids.append(dict(target=ident,method=method,seed=seed,swap=name,local_error=er,reduction_fraction=1-er/max(current,1e-30)))
                for e in a['local']:
                    if e['event_id'] in ('P2','V3'):weights.append(dict(target=ident,method=method,seed=seed,**e))
        allmods[ident]=modes
    save('MODE_CARDS.csv',cards);save('MODES.csv',modalrows);save('MODAL_SWAP.csv',hybrids);save('MODAL_FACTORIAL.csv',factorial);save('EVENT_WEIGHT.csv',weights)
    np.savez_compressed(OUT/'modal_decompositions.npz',**{ident.replace('/','_')+'__'+key+'__'+field:value[field] for ident,mm in allmods.items() for key,value in mm.items() for field in ('poles','res','bg','zeros')})
    return cards

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    mixed()

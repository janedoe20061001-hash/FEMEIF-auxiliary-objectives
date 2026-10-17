"""Exploratory follow-up prompted by same-sign residuals; fixed saved curves only."""
from pathlib import Path
import argparse,sys,os,json,csv
import numpy as np
from scipy.stats import rankdata
ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
src=Path(a.source).resolve();out=Path(a.out).resolve();sys.path.insert(0,str(src));os.chdir(src)
import experiment as e
cfg,f=e.config(src/'config.json')
if src.name=='case1':
    from legacy_objectives import impedance,TARGET
    target=impedance(f,TARGET)[0]
else:
    from case2 import load_target,impedance
    _,target=load_target(cfg['target_data'])
d=np.load(src/'analysis/selected_curves.npz');z=d['curves'];groups=d['groups'];seeds=d['seeds']
phi=np.angle(z);pt=np.angle(target);err=(phi-pt)**2;sign=np.signbit(z.imag)!=np.signbit(target.imag)
x=np.log10(f);u=(x-x[0])/(x[-1]-x[0]);h=np.diff(u)
amp=20*np.log10(abs(z));at=20*np.log10(abs(target));st=np.array([np.ptp(at),np.ptp(pt)])
y=np.stack([amp,phi],axis=-1)/st;yt=np.stack([at,pt],axis=-1)/st
dy=np.diff(y,axis=-2)/h[:,None];dyt=np.diff(yt,axis=-2)/h[:,None]
vector=np.sqrt(np.sum(h*np.sum((dy-dyt)**2,axis=-1),axis=-1)/np.sum(h*np.sum(dyt*dyt,axis=-1)))
slope=np.sqrt(np.sum(h*(dy[...,1]-dyt[...,1])**2,axis=-1)/np.sum(h*dyt[...,1]**2))
primary=err.mean(axis=-1);mag=((amp-at)**2).mean(axis=-1)
def rho(a,b):return float(np.corrcoef(rankdata(a),rankdata(b))[0,1]) if np.ptp(a)>0 and np.ptp(b)>0 else None
def partial(a,b,c):
    a,b,c=map(rankdata,[a,b,c]);X=np.c_[np.ones(len(c)),c]
    aa=a-X@np.linalg.lstsq(X,a,rcond=None)[0];bb=b-X@np.linalg.lstsq(X,b,rcond=None)[0]
    return float(np.corrcoef(aa,bb)[0,1])
rows=[];aggregate=[];byfreq=[]
for k,g in enumerate(groups):
    masks={'both_correct':(~sign[:,0])&(~sign[:,k]),'both_wrong':sign[:,0]&sign[:,k],
           'repaired':sign[:,0]&(~sign[:,k]),'introduced':(~sign[:,0])&sign[:,k]}
    terms={name:np.mean((err[:,k]-err[:,0])*mask,axis=1) for name,mask in masks.items()}
    np.testing.assert_allclose(sum(terms.values()),primary[:,k]-primary[:,0],atol=1e-15)
    same=float(np.mean(err[:,k]*(~sign[:,k])));wrong=float(np.mean(err[:,k]*sign[:,k]))
    for j,seed in enumerate(seeds):
        rows.append(dict(case=src.name,group=str(g),seed=int(seed),phase=float(primary[j,k]),
            LP_signed_vector=float(vector[j,k]),phase_secant=float(slope[j,k]),
            **{name:float(v[j]) for name,v in terms.items()}))
    aggregate.append(dict(group=str(g),phase_mean=float(primary[:,k].mean()),same_sign_contribution=same,wrong_sign_contribution=wrong,
        wrong_sign_point_fraction=float(sign[:,k].mean()),phase_delta_partition={name:float(v.mean()) for name,v in terms.items()},
        diagnostic_correlations=[dict(metric=name,phase_rho=rho(v[:,k],primary[:,k]),
            partial_phase_rho=partial(v[:,k],primary[:,k],mag[:,k]),mean=float(v[:,k].mean())) for name,v in [('LP_signed_vector',vector),('phase_secant',slope)]]))
    for j,ff in enumerate(f):byfreq.append(dict(case=src.name,group=str(g),frequency=float(ff),wrong_sign_rate=float(sign[:,k,j].mean()),phase_mse=float(err[:,k,j].mean())))
for name,data in [('followup_rows',rows),('frequency_sign',byfreq)]:
    with (out/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
(out/'FOLLOWUP.json').write_text(json.dumps(aggregate,indent=2),encoding='utf-8')
# Confirm the same scores on common initial and baseline-final populations.
common=[]
for seed in seeds:
    with np.load(src/'runs'/'baseline'/f'seed_{seed:03d}'/'final_population.npz') as saved:final=saved['log_parameters'].copy()
    for context,pars in [('initial',e.initial_population(cfg,int(seed))),('baseline_final',final)]:
        zz=impedance(f,10**pars);pp=np.angle(zz);aa=20*np.log10(abs(zz));coords=np.stack([aa,pp],axis=-1)/st
        dd=np.diff(coords,axis=-2)/h[:,None]
        vv=np.sqrt(np.sum(h*np.sum((dd-dyt)**2,axis=-1),axis=-1)/np.sum(h*np.sum(dyt*dyt,axis=-1)))
        ss=np.sqrt(np.sum(h*(dd[...,1]-dyt[...,1])**2,axis=-1)/np.sum(h*dyt[...,1]**2))
        ph=np.mean((pp-pt)**2,axis=-1)
        for name,values in [('LP_signed_vector',vv),('phase_secant',ss)]:
            common.append(dict(case=src.name,seed=int(seed),context=context,metric=name,phase_rho=rho(values,ph)))
with (out/'followup_common.csv').open('w',encoding='utf-8-sig',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(common[0]));w.writeheader();w.writerows(common)
(out/'followup_scales.json').write_text(json.dumps(dict(target_range=st.tolist(),
    phase_fraction_of_target_derivative_energy=float(np.sum(h*dyt[...,1]**2)/np.sum(h*np.sum(dyt*dyt,axis=-1)))),indent=2))
print(json.dumps(aggregate[:4],indent=2))

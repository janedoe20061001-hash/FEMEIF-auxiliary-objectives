import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,csv,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'frozen'));sys.path.insert(0,str(HERE/'analysis_toolkit'))
from model import Model
from adaptive import Adaptive,spectra
from refined_grid import improved_dense_grid,mode_candidates
from structure import target_catalog,classify_events,amp_events,diagnostic
OLD=HERE.parent/'mse_structure_step1_20261008'
OUT=HERE.parents[1]/'output/mse_structure_step2_20261009'
def label(s):return ('s'+format(s,'g')).replace('.','p')
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.item()),encoding='utf8');q.replace(p)
def csvout(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
class Target(Model):
    def __init__(self,s=1):
        super().__init__('case1');self.scale=s
        self.record=json.loads(json.dumps(self.record));p=np.asarray(self.record['target_parameters'],float);p[4]*=s;p[5]/=s
        self.record['target_parameters']=p.tolist();self.target=self.network(self.f,p)[0]
        self.record['target_real']=self.target.real.tolist();self.record['target_imag']=self.target.imag.tolist()
        self.amp=20*np.log10(abs(self.target));self.phase=np.angle(self.target)
def base_grid(m):
    # Target-specific moderate grid; candidates adapt separately during scoring.
    xx=[np.log10(m.f)]
    for fc,width in mode_candidates(m.record,np.asarray(m.record['target_parameters'])):
        c=np.log10(fc);d=width/np.log(10)
        if 2<=c<=6:
            xx.append(np.linspace(max(2,c-d),min(6,c+d),17))
            off=np.geomspace(max(d,1e-12),max(d,.5),17)
            for sign in [-1,1]:xx.append(np.clip(c+sign*off,2,6))
    return 10**np.unique(np.concatenate(xx))
def evaluator(m):
    with np.load(HERE/'targets'/label(m.scale)/'grid.npz') as d:return Adaptive(m.record,d['frequency'])
def windows(cat):
    p=next(e for e in cat if e['event_id']=='P3');w=p['width_3db_decades']
    return {'core':[10**(p['x']-w/2),10**(p['x']+w/2)],'pair':[23000,26000],'broad':[10000,10**4.5]}
def regional(m,p,cat,lev=2):
    win=windows(cat);a=Adaptive(m.record,m.f)
    f=improved_dense_grid(m.record,np.atleast_2d(p),lev)
    x=np.unique(np.r_[np.log10(f),a.edges,*[np.log10(v) for v in win.values()]])
    z=spectra(10**x,p)[0];t=spectra(10**x,m.record['target_parameters'])[0]
    residual=np.array([(20*np.log10(abs(z)/abs(t)))**2,(np.angle(z)-np.angle(t))**2])
    midpoint=(x[1:]+x[:-1])/2;cell=np.clip(np.searchsorted(a.edges,midpoint,side='right')-1,0,a.nold-1)
    masses={'common_logf':np.diff(x)/(x[-1]-x[0]),'training_cell_reference':np.diff(x)/np.diff(a.edges)[cell]/a.nold}
    rows=[]
    for weighting,mass in masses.items():
        vals=(residual[:,:-1]+residual[:,1:])/2*mass
        total=vals.sum(axis=1)
        for name,(lo,hi) in win.items():
            mask=(midpoint>=np.log10(lo))&(midpoint<=np.log10(hi));wt=float(mass[mask].sum());q=vals[:,mask].sum(axis=1)
            rows.append(dict(window=name,weighting=weighting,weight_fraction=wt,global_mag=float(total[0]),global_phase=float(total[1]),mag_contribution=float(q[0]),phase_contribution=float(q[1]),mag_error_fraction=float(q[0]/total[0]) if total[0]>0 else None,phase_error_fraction=float(q[1]/total[1]) if total[1]>0 else None,local_amp_rmse_db=float(np.sqrt(q[0]/wt)),local_phase_rmse_deg=float(np.rad2deg(np.sqrt(q[1]/wt)))))
    z=m.network(m.f,p)[0];res=np.array([(20*np.log10(abs(z))-m.amp)**2,(np.angle(z)-m.phase)**2]);total=res.mean(axis=1)
    for name,(lo,hi) in win.items():
        mask=(m.f>=lo)&(m.f<=hi);wt=float(mask.mean());q=(res*mask).mean(axis=1)
        rows.append(dict(window=name,weighting='training_discrete',weight_fraction=wt,global_mag=float(total[0]),global_phase=float(total[1]),mag_contribution=float(q[0]),phase_contribution=float(q[1]),mag_error_fraction=float(q[0]/total[0]) if total[0]>0 else None,phase_error_fraction=float(q[1]/total[1]) if total[1]>0 else None,local_amp_rmse_db=float(np.sqrt(q[0]/wt)) if wt else None,local_phase_rmse_deg=float(np.rad2deg(np.sqrt(q[1]/wt))) if wt else None))
    return rows

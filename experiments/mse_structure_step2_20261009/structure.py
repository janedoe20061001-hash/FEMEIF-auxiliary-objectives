import numpy as np
from scipy.signal import find_peaks,peak_widths
from scipy.optimize import linear_sum_assignment
from adaptive import spectra
from refined_grid import improved_dense_grid
from diagnostics import diagnose
def amp_events(x,z):
    y=20*np.log10(abs(z));result=[]
    for kind,sign in [('peak',1),('valley',-1)]:
        ix,pr=find_peaks(sign*y,prominence=1)
        for j,i in enumerate(ix):
            width=None
            if pr['prominences'][j]>=3:
                ww=peak_widths(sign*y,np.asarray([i]),rel_height=3/pr['prominences'][j],prominence_data=(pr['prominences'][j:j+1],pr['left_bases'][j:j+1],pr['right_bases'][j:j+1]))
                width=float(np.interp(ww[3][0],np.arange(len(x)),x)-np.interp(ww[2][0],np.arange(len(x)),x))
            result.append(dict(kind=kind,x=float(x[i]),height_db=float(y[i]),width_3db_decades=width,prominence_db=float(pr['prominences'][j])))
    return result

def target_catalog(m):
    f=improved_dense_grid(m.record,np.empty((0,18)),4);ev=amp_events(np.log10(f),spectra(f,m.record['target_parameters'])[0])
    for kind,letter in [('peak','P'),('valley','V')]:
        rr=sorted([e for e in ev if e['kind']==kind],key=lambda e:e['x'])
        for i,e in enumerate(rr):e['event_id']=letter+str(i+1);e['frequency_hz']=float(10**e['x'])
    return ev

def classify_events(target,fit):
    rows=[]
    for kind in ['peak','valley']:
        tt=[e for e in target if e['kind']==kind];ff=[e for e in fit if e['kind']==kind]
        cost=np.full((len(tt),len(ff)+len(tt)),.250001)
        for i,a in enumerate(tt):
            for j,b in enumerate(ff):cost[i,j]=abs(a['x']-b['x']) if abs(a['x']-b['x'])<=.25 else 1e6
        rr,cc=linear_sum_assignment(cost);match={i:j for i,j in zip(rr,cc) if j<len(ff) and cost[i,j]<=.25}
        for i,a in enumerate(tt):
            b=ff[match[i]] if i in match else None
            shift=b['x']-a['x'] if b else None;height=b['height_db']-a['height_db'] if b else None
            width=float(np.log2(b['width_3db_decades']/a['width_3db_decades'])) if b and b['width_3db_decades'] and a['width_3db_decades'] else None
            complete=bool(b is not None and abs(shift)<=.02 and abs(height)<=3 and width is not None and abs(width)<=1)
            rows.append(dict(event_id=a['event_id'],kind=kind,target_frequency_hz=a['frequency_hz'],target_width_3db_decades=a['width_3db_decades'],target_height_db=a['height_db'],matched=b is not None,complete=complete,fit_frequency_hz=float(10**b['x']) if b else None,shift_decades=shift,height_delta_db=height,width_log2_ratio=width,position_bad=bool(b is not None and abs(shift)>.02),height_bad=bool(b is not None and abs(height)>3),width_bad=bool(b is not None and (width is None or abs(width)>1))))
    return rows

def diagnostic(m,p,target):
    prev=None;stable=False
    for lev in [1,2,4,8]:
        f=improved_dense_grid(m.record,np.atleast_2d(p),lev);z=spectra(f,p)[0];t=spectra(f,m.record['target_parameters'])[0]
        diag,bands,_=diagnose(f,t,z);ee=classify_events(target,amp_events(np.log10(f),z))
        if prev:
            d0,e0=prev
            stable=all(d0[k]==diag[k] for k in ['amp_missed','amp_extra','phase_missed','phase_extra']) and all(abs(d0[k]-diag[k])/max(abs(diag[k]),floor)<=.01 for k,floor in [('amplitude_rmse_db',.1),('phase_rmse_deg',.5)]) and all(a['matched']==b['matched'] and a['complete']==b['complete'] for a,b in zip(e0,ee))
            if stable:break
        prev=(diag,ee)
    row={k:diag[k] for k in ['amplitude_rmse_db','phase_rmse_deg','amp_missed','amp_extra','phase_missed','phase_extra','worst_band_phase_rmse_deg','amp_mean_abs_shift_decades','amp_mean_abs_width_log2_ratio','amp_missed_prom_0p5','amp_missed_prom_2p0']}
    row.update(grid_stable=stable,refinement=lev,amp_incomplete=sum(not e['complete'] for e in ee),amp_position_bad=sum(e['position_bad'] for e in ee),amp_height_bad=sum(e['height_bad'] for e in ee),amp_3db_width_bad=sum(e['width_bad'] for e in ee),phase_bad_bands=sum(b['phase_rmse_deg']>15 for b in bands))
    return row,ee,bands

import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,csv,hashlib,time,argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
from scipy.signal import find_peaks,peak_widths
from scipy.optimize import linear_sum_assignment
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'frozen'))
from model import Model
sys.path.insert(0,str(HERE/'toolkit'))
from refined_grid import improved_dense_grid
from diagnostics import diagnose,events
from adaptive import Adaptive,spectra

def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item()),encoding='utf8');q.replace(p)
def csvout(p,rows):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

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
    row={k:diag[k] for k in ['amplitude_rmse_db','phase_rmse_deg','amp_missed','amp_extra','phase_missed','phase_extra','worst_band_phase_rmse_deg','amp_mean_abs_shift_decades','amp_mean_abs_width_log2_ratio']}
    row.update(grid_stable=stable,refinement=lev,amp_incomplete=sum(not e['complete'] for e in ee),amp_position_bad=sum(e['position_bad'] for e in ee),amp_height_bad=sum(e['height_bad'] for e in ee),amp_3db_width_bad=sum(e['width_bad'] for e in ee),phase_bad_bands=sum(b['phase_rmse_deg']>15 for b in bands))
    return row,ee,bands

def job(arg):
    arm,seed,runs,out=arg;m=Model('case1');catalog=target_catalog(m);d=Path(runs)/arm/f'seed_{seed}';rows=[];ers=[];brs=[];frontrows=[];front_events=[];front_bands=[];checked=0
    summary=json.loads((d/'summary.json').read_text());assert summary['status']=='complete' and summary['evaluations']==200000
    x0=np.load(d/'initial_log_parameters.npy');np.testing.assert_array_equal(x0,np.random.default_rng(seed).uniform(m.lower,m.upper,(200,18)))
    with np.load(d/'trajectory.npz') as data:
        for g,xs,fs in zip(data['generations'],data['log_parameters'],data['training_metrics']):
            for j,rule in enumerate(['min_wsmse','balanced_minmax']):
                p=10**xs[j];diag,ee,bands=diagnostic(m,p,catalog);base=dict(arm=arm,seed=seed,generation=int(g),rule=rule)
                rows.append(dict(**base,train_mag=float(fs[j,0]),train_phase=float(fs[j,1]),**diag));ers.extend(dict(**base,**e) for e in ee);brs.extend(dict(**base,**b) for b in bands)
    with np.load(HERE/'target_grid.npz') as f:a=Adaptive(m.record,f['frequency'])
    for g in [300,1000]:
        with np.load(d/f'generation_{g:03}.npz') as data:
            x=data['log_parameters'];F=data['optimization_objective'];assert x.shape==(200,18) and np.all(x>=m.lower) and np.all(x<=m.upper)
            original=m.score(x);independent=spectra(m.f,10**x);np.testing.assert_allclose(m.spectrum(x),independent,rtol=1e-9,atol=1e-9);np.testing.assert_allclose(original,data['original_metrics'],rtol=1e-10,atol=1e-8)
            actual=original[:,:2] if arm=='original' else a.score(x);np.testing.assert_allclose(F,actual,rtol=1e-10,atol=1e-8)
            dom=np.all(F[:,None,:]<=F[None,:,:],axis=2)&np.any(F[:,None,:]<F[None,:,:],axis=2);front=np.flatnonzero(~dom.any(axis=0));np.testing.assert_array_equal(front,np.sort(data['front_indices']));checked+=len(x)
            ix=int(data['selected_index']);bx=int(data['balanced_index']);assert ix==int(front[np.argmin(F[front,0]+m.alpha*F[front,1])]);assert bx==int(front[np.argmin(np.maximum(F[front,0],m.alpha*F[front,1]))])
            if arm=='original' and g==300:
                with np.load(HERE/'old300'/f'{seed}.npz') as old:np.testing.assert_array_equal(x,old['log_parameters'])
            for index in front:
                diag,ee,bands=diagnostic(m,10**x[index],catalog);base=dict(arm=arm,seed=seed,generation=g,index=int(index),is_primary=bool(index==ix),is_balanced=bool(index==bx))
                frontrows.append(dict(**base,train_mag=float(F[index,0]),train_phase=float(F[index,1]),**diag));front_events.extend(dict(**base,**e) for e in ee);front_bands.extend(dict(**base,**b) for b in bands)
    result=dict(arm=arm,seed=seed,population_individuals_checked=checked,representatives=rows,events=ers,bands=brs,front=frontrows,front_events=front_events,front_bands=front_bands,summary=summary)
    dump(Path(out)/'chunks'/f'{arm}_{seed}.json',result);return arm,seed

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=6);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((HERE/'CONFIG.json').read_text());catalog=target_catalog(Model('case1'));dump(a.output/'TARGET_EVENTS.json',catalog);pending=[]
    for arm in cfg['arms']:
        for s in cfg['seeds']:
            if not (a.output/'chunks'/f'{arm}_{s}.json').exists():pending.append((arm,s,str(a.runs),str(a.output)))
    start=time.time();done=200-len(pending)
    for name,h in json.loads((HERE/'ANALYSIS_SOURCES.json').read_text()).items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==h,name
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        active={}
        while pending or active:
            ready=[j for j in pending if (Path(j[2])/j[0]/f'seed_{j[1]}'/'summary.json').exists()]
            for j in ready[:max(0,a.workers-len(active))]:active[pool.submit(job,j)]=j;pending.remove(j)
            if not active:
                status=a.runs/'status.json'
                if status.exists() and json.loads(status.read_text())['status']=='failed':raise RuntimeError('Training failed; diagnostic waits do not replace failed seeds')
                time.sleep(10);continue
            finished,_=wait(active,timeout=10,return_when=FIRST_COMPLETED)
            for fut in finished:
                arm,s=fut.result();active.pop(fut);done+=1;dump(a.output/'STATUS.json',dict(status='running',completed=done,total=200,seconds=time.time()-start));print(json.dumps(dict(classified=done,arm=arm,seed=s)),flush=True)
    data=[json.loads((a.output/'chunks'/f'{arm}_{s}.json').read_text()) for arm in cfg['arms'] for s in cfg['seeds']]
    for key,name in [('representatives','stage_diagnostics.csv'),('events','stage_events.csv'),('bands','stage_bands.csv'),('front','front_diagnostics.csv'),('front_events','front_events.csv'),('front_bands','front_bands.csv')]:csvout(a.output/name,[row for d in data for row in d[key]])
    dump(a.output/'RUN_SUMMARIES.json',[d['summary'] for d in data]);dump(a.output/'VALIDATION.json',dict(runs=200,full_population_individuals_recomputed=sum(d['population_individuals_checked'] for d in data),representative_rows=sum(len(d['representatives']) for d in data),front_rows=sum(len(d['front']) for d in data),unstable_representatives=sum(not row['grid_stable'] for d in data for row in d['representatives']),unstable_front_rows=sum(not row['grid_stable'] for d in data for row in d['front']),original300_all_exact=True));dump(a.output/'STATUS.json',dict(status='complete',completed=200,total=200,seconds=time.time()-start))
if __name__=='__main__':main()

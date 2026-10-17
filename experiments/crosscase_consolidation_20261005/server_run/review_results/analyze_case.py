"""Audit complete runs, independent formulas, common-population correlations."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
from pathlib import Path
import sys,json,csv,hashlib,importlib.util,time,argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.stats import rankdata

def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csvout(p,rows):
    if not rows:return
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def correlation(a,b):
    if np.ptp(a)==0 or np.ptp(b)==0:return None,None
    return float(np.corrcoef(a,b)[0,1]),float(np.corrcoef(rankdata(a),rankdata(b))[0,1])

def initialize(case_path):
    global case,root,e,cfg,f,z,imp,groups,objects,allmetrics,highK,geom,derivative_matrix
    case=Path(case_path);root=case.parent;os.chdir(case);sys.path.insert(0,str(case))
    import experiment as e
    from global_metrics import GlobalObjectives,GROUPS
    from length_metrics import GlobalObjectives as Length
    cfg,f=e.config(case/'config.json');groups=e.groups(cfg)
    if case.name=='case1':
        from legacy_objectives import impedance,TARGET
        imp=impedance;z=imp(f,TARGET)[0]
    else:
        from case2 import impedance,load_target
        imp=impedance;_,z=load_target(cfg['target_data'])
    geom=Length(f,target_z=z);highK=np.load(case/'trend_kernel_check.npy')
    objects={e.group_name(g):GlobalObjectives(f,aux=g,target_z=z) for g in groups}
    # Independent derivative operator: fit a degree-2 interpolant to three nodes.
    x=np.log10(f);derivative_matrix=np.zeros((len(f),len(f)))
    for i in range(len(f)):
        ix=np.arange(3) if i==0 else np.arange(len(f)-3,len(f)) if i==len(f)-1 else np.arange(i-1,i+2)
        dx=x[ix]-x[i]
        derivative_matrix[i,ix]=np.linalg.solve(np.stack([np.ones(3),dx,dx*dx]),[0,1,0])
    allmetrics=[];seen=set()
    for directory in ['global','geometry','logpolar','refinement','directional','interval']:
        folder=root/'analysis_libs'/directory;sys.path.insert(0,str(folder))
        mod=load('historical_'+directory,folder/'global_metrics.py')
        # Each singleton is accepted even in modules with a frozen group list.
        for mid,name,label in mod.CATALOG:
            if mid in seen or '+' in name:continue
            obj=mod.GlobalObjectives(f,aux=(name,),target_z=z)
            allmetrics.append((mid,name,obj));seen.add(mid)
    from legacy_objectives import Objectives as Legacy,CATALOG
    retained={'I-03','I-06','II-01','II-02'}
    removed={'I-01','I-02','I-04','I-05','I-07','II-11'}
    for mid,name,label in CATALOG:
        if mid in retained|removed:
            allmetrics.append((mid,name,Legacy(f,aux=(name,),target_z=z)))
    for mid,name in [('T-01','all_interval_mean_residual_logpolar'),('G-01-LP','bidirectional_residual_logpolar')]:
        allmetrics.append((mid,name,GlobalObjectives(f,aux=(name,),target_z=z)))

def independent(spectra,group):
    # Reconstruct losses without calling any production metric helper.
    amp=10*np.log10(spectra.real*spectra.real+spectra.imag*spectra.imag)
    phase=np.arctan2(spectra.imag,spectra.real)
    target_amp=10*np.log10(z.real*z.real+z.imag*z.imag);target_phase=np.arctan2(z.imag,z.real)
    values=[np.mean((amp-target_amp)**2,axis=1),np.mean((phase-target_phase)**2,axis=1)]
    for name in group:
        if name=='reactance_crossing_direction_consistency':
            ix=np.flatnonzero(np.signbit(z.imag[:-1])!=np.signbit(z.imag[1:]))
            events=(geom.u[ix]+geom.u[ix+1])/2
            direction=np.where(z.imag[ix]<z.imag[ix+1],1,-1)
            edges=np.r_[0,(events[:-1]+events[1:])/2,1]
            loss=[]
            for row in spectra:
                ids=np.flatnonzero(np.signbit(row.imag[:-1])!=np.signbit(row.imag[1:]))
                loc=(geom.u[ids]+geom.u[ids+1])/2;sg=np.where(row.imag[ids]<row.imag[ids+1],1,-1)
                loss.append(np.mean([not np.any((loc>=edges[j])&(loc<=edges[j+1])&(sg==d)) for j,d in enumerate(direction)]) if len(direction) else 0)
            values.append(np.array(loss))
        elif name in ['characteristic_frequency_shift_error_decades','transition_speed_relative_error']:
            g=derivative_matrix@z;eps=.05*np.median(abs(g)[abs(g)>0])
            if name.startswith('characteristic'):
                delta=spectra-z;den=abs(g)**2+eps**2
                shift=(delta.real*g.real+delta.imag*g.imag)/den;weight=abs(g)**2/den
                values.append((abs(shift)*weight).sum(axis=1)/weight.sum())
            else:
                fitted=spectra@derivative_matrix.T
                values.append(np.mean(abs(abs(fitted)-abs(g))/(abs(g)+eps),axis=1))
        elif name=='cumulative_length_logpolar':
            coords=np.stack([amp,target_phase[None,:]+phase-target_phase[None,:]],axis=-1)/geom.scale
            d=np.diff(coords,axis=1);ds=np.sqrt(np.diff(geom.u)**2+np.sum(d*d,axis=-1))
            tcoords=np.stack([target_amp,target_phase],axis=-1)/geom.scale
            td=np.sqrt(np.diff(geom.u)**2+np.sum(np.diff(tcoords,axis=0)**2,axis=-1))
            err=(np.cumsum(ds,axis=1)-np.cumsum(td))/sum(td)
            err=np.column_stack([np.zeros(len(err)),err]);gauss=np.array([.5-.5/np.sqrt(3),.5+.5/np.sqrt(3)])
            v=(1-gauss)*err[:,:-1,None]+gauss*err[:,1:,None]
            values.append(np.sqrt(np.sum(v*v*np.diff(geom.u)[:,None]/2,axis=(1,2))))
        else:
            r=np.stack([amp-target_amp,phase-target_phase],axis=-1)/geom.scale
            if name=='all_interval_mean_residual_logpolar':
                values.append(np.sqrt(np.maximum(np.einsum('nic,ij,njc->n',r,highK,r,optimize=True),0)))
            elif name=='bidirectional_residual_logpolar':
                c=np.zeros_like(r)
                for j,h in enumerate(np.diff(geom.u)):
                    c[:,j+1]=c[:,j]+h*(r[:,j]+r[:,j+1])/2
                density=(np.sum(c*c,axis=2)+np.sum((c[:,-1:, :]-c)**2,axis=2))/2
                values.append(np.sum(np.diff(geom.u)*(density[:,:-1]+density[:,1:])/2,axis=1))
            else:raise AssertionError(name)
    return np.column_stack(values)

def job(seed):
    mf=json.loads((case/'runs/manifest.json').read_text());expected=e.initial_population(cfg,seed)
    rows=[];curves=[];parameters=[];histories=[];within=[];selected=[];maxprod=0.;maxind=0.;maxt=0.
    for group in groups:
        name=e.group_name(group);rd=case/'runs'/name/f'seed_{seed:03d}'
        record=json.loads((rd/'summary.json').read_text())
        assert record['fingerprint']==mf['fingerprint'] and record['status']=='complete'
        assert record['seed']==seed and record['group']==name
        assert record['generations']==300 and record['evaluations']==300*cfg['population']
        init=np.load(rd/'initial_log_parameters.npy');np.testing.assert_array_equal(init,expected)
        assert hashlib.sha256(init.tobytes()).hexdigest()==record['initial_population_sha256']
        with np.load(rd/'final_population.npz') as saved:
            x=saved['log_parameters'];F=saved['objectives'];idx=int(saved['selected_index'])
            assert x.shape==expected.shape and F.shape==(len(x),2+len(group))
            assert np.isfinite(x).all() and np.all(x>=e.LOWER) and np.all(x<=e.UPPER)
            spectra=imp(f,10**x);again=objects[name].evaluate_spectra(spectra)
            np.testing.assert_allclose(F,again,rtol=1e-11,atol=1e-11)
            maxprod=max(maxprod,float(np.max(abs(F-again))))
            ind=independent(spectra,group)
            tolerance=1e-6 if 'all_interval_mean_residual_logpolar' in group else 1e-8
            np.testing.assert_allclose(F,ind,rtol=tolerance,atol=2e-9)
            if 'all_interval_mean_residual_logpolar' in group:maxt=max(maxt,float(np.max(abs(F[:,2]-ind[:,2])/np.maximum(abs(ind[:,2]),1e-14))))
            else:maxind=max(maxind,float(np.max(abs(F-ind))))
            assert idx==int(np.argmin(F[:,0]+cfg['final_selection']['phase_weight']*F[:,1]))
            selected_fields=e.select_wsmse(F,cfg['final_selection']['phase_weight'])
            for k,v in selected_fields.items():np.testing.assert_allclose(record[k],v,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(saved['selected_primary_mse'],F[idx,:2],rtol=0,atol=0)
            full=e.nd_indices(F);primary=e.nd_indices(F[:,:2])
            np.testing.assert_array_equal(saved['full_front_indices'],full);np.testing.assert_array_equal(saved['primary_front_indices'],primary)
            zz=spectra[idx];curves.append(zz);parameters.append(x[idx])
            rows.append(dict(case=case.name,seed=seed,group=name,magnitude=F[idx,0],phase=F[idx,1],
                wsmse=record['wsmse_selected_value'],seconds=record['elapsed_seconds'],
                full_front_fraction=len(full)/len(x),primary_front_fraction=len(primary)/len(x)))
            for k,metric in enumerate(group):
                for j,primary_name in enumerate(['magnitude','phase']):
                    p,s=correlation(F[:,k+2],F[:,j]);within.append(dict(case=case.name,seed=seed,context='own_final',metric=metric,primary=primary_name,pearson=p,spearman=s))
            if not group:
                final_baseline=spectra;baselineF=F;baseline_index=idx
        with (rd/'history.csv').open(encoding='utf-8-sig',newline='') as fh:h=list(csv.DictReader(fh))
        assert len(h)==300
        assert [int(v['generation']) for v in h]==list(range(1,301))
        assert [int(v['evaluations']) for v in h]==list(np.arange(1,301)*cfg['population'])
        histories.append(np.array([[float(v['wsmse_selected_magnitude_mse']),float(v['wsmse_selected_phase_mse'])] for v in h]))
    correlations=[]
    for context,spectra in [('initial',imp(f,10**expected)),('baseline_final',final_baseline)]:
        m=objects['baseline'].evaluate_spectra(spectra)
        p,s=correlation(m[:,0],m[:,1]);correlations.append(dict(case=case.name,seed=seed,context=context,id='M-01_vs_M-02',metric='primary_pair',primary='phase',pearson=p,spearman=s))
        for mid,name,obj in allmetrics:
            v=obj.evaluate_spectra(spectra)
            np.testing.assert_allclose(v[:,:2],m,rtol=1e-12,atol=1e-12)
            for j,primary in enumerate(['magnitude','phase']):
                p,s=correlation(v[:,2],m[:,j]);correlations.append(dict(case=case.name,seed=seed,context=context,id=mid,metric=name,primary=primary,pearson=p,spearman=s))
            if context=='baseline_final':selected.append(dict(case=case.name,seed=seed,id=mid,metric=name,loss=float(v[baseline_index,2]),magnitude=float(m[baseline_index,0]),phase=float(m[baseline_index,1])))
    return dict(seed=seed,rows=rows,curves=np.array(curves),parameters=np.array(parameters),histories=np.array(histories),correlations=correlations,selected=selected,within=within,maxprod=maxprod,maxind=maxind,maxt=maxt)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('case');ap.add_argument('--workers',type=int,default=12);a=ap.parse_args()
    c=Path(a.case).resolve();out=c/'analysis';out.mkdir(exist_ok=True)
    sys.path.insert(0,str(c));from trend_kernel import kernel
    cfg0=json.loads((c/'config.json').read_text());mf=json.loads((c/'runs/manifest.json').read_text())
    u=np.log10(mf['frequency_hz']);u=(u-u[0])/(u[-1]-u[0]);np.save(c/'trend_kernel_check.npy',kernel(u,32))
    # Also verify the manifest and frozen source before any summaries are accepted.
    frozen=json.loads((c.parent/'FROZEN.json').read_text())['files_sha256']
    for name,digest in frozen.items():assert hashlib.sha256((c.parent/name).read_bytes()).hexdigest()==digest,name
    results=[];t=time.time()
    with ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(str(c),)) as pool:
        futures=[pool.submit(job,s) for s in range(2000,2100)]
        for future in as_completed(futures):
            results.append(future.result())
            print(json.dumps(dict(case=c.name,seeds_audited=len(results),seconds=time.time()-t)),flush=True)
    results.sort(key=lambda r:r['seed'])
    for key in ['rows','correlations','selected','within']:csvout(out/f'{key}.csv',[v for r in results for v in r[key]])
    np.savez_compressed(out/'selected_curves.npz',frequency=np.array(mf['frequency_hz']),
        curves=np.array([r['curves'] for r in results]),parameters=np.array([r['parameters'] for r in results]),
        histories=np.array([r['histories'] for r in results]),seeds=np.arange(2000,2100),
        groups=np.array(['+'.join(g) if g else 'baseline' for g in cfg0['groups']]))
    dump(out/'audit.json',dict(status='passed',runs=600,individuals=600*cfg0['population'],
        all_initial_populations_verified=True,all_300_generations_and_evaluation_counts_verified=True,
        full_objectives_and_selection_recomputed=True,independent_formula_check=True,
        max_production_abs=max(r['maxprod'] for r in results),max_independent_abs_non_T01=max(r['maxind'] for r in results),
        max_T01_q16_vs_q32_relative=max(r['maxt'] for r in results),seconds=time.time()-t))
if __name__=='__main__':main()

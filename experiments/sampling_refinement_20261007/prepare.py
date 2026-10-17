"""Build target-only grids and freeze controlled interpolation/true-data arms."""
from pathlib import Path
import sys,json,csv,hashlib,shutil,tarfile
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/log_radius_distance_20261007'))
import analyze as prev
IDS=['case1','rlc_background_c3','rlc_parallel_c3','rlc_series_c3','three_parallel_units_c3','three_series_units_c3','two_parallel_units_c3','two_series_units_c3','rc_series_c1','rlc_bridge_c2']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def compress(x,a,p,forced,ta,tp):
    keep=set(np.searchsorted(x,forced).tolist());fixed=sorted(keep);stack=list(zip(fixed[:-1],fixed[1:]))
    while stack:
        i,j=stack.pop()
        if j<=i+1:continue
        xx=x[i+1:j];t=(xx-x[i])/(x[j]-x[i]);err=np.maximum(abs(a[i+1:j]-(a[i]+t*(a[j]-a[i])))/ta,abs(p[i+1:j]-(p[i]+t*(p[j]-p[i])))/tp)
        k=int(np.argmax(err))
        if err[k]>1:
            k+=i+1;keep.add(k);stack.extend([(i,k),(k,j)])
    return np.asarray(sorted(keep))
def cell_weights(x,original):
    edges=np.r_[original[0],(original[:-1]+original[1:])/2,original[-1]];b=np.searchsorted(edges,(x[:-1]+x[1:])/2,side='right')-1;b=np.clip(b,0,len(original)-1)
    mass=np.diff(x)/np.diff(edges)[b]/len(original);w=np.zeros(len(x));w[:-1]+=mass/2;w[1:]+=mass/2
    cells=np.bincount(b,weights=mass,minlength=len(original));np.testing.assert_allclose(cells,1/len(original),rtol=1e-10,atol=1e-12)
    assert (w>=0).all();np.testing.assert_allclose(w.sum(),1,rtol=1e-12);return w,cells
def main():
    frozen=HERE/'frozen_scalar';frozen.mkdir(exist_ok=True);src=ROOT/'experiments/primary_loss_comparison_20261007'
    for name in ['model.py','run.py','circuits.json']:
        shutil.copyfile(src/name,frozen/name)
    shutil.copytree(src/'frozen',frozen/'frozen',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (HERE/'grids').mkdir(exist_ok=True);records=[];checks=[]
    diagnostics=list(csv.DictReader((ROOT/'output/optimized_curve_structure_20261007/per_seed_diagnostics.csv').open(encoding='utf-8-sig')))
    for ident in IDS:
        m=prev.Model(ident);r=m.record;assert r['target_parameters'] is not None;ori=np.log10(m.f);edges=np.r_[ori[0],(ori[:-1]+ori[1:])/2,ori[-1]];forced=np.unique(np.r_[ori,edges]);p=np.asarray(r['target_parameters'])
        reference=prev.improved_dense_grid(r,np.empty((0,len(p))),2);xx=np.unique(np.r_[np.log10(reference),forced]);z=m.network(10**xx,p)[0];a=20*np.log10(abs(z));theta=np.rad2deg(np.angle(z))
        ff=prev.improved_dense_grid(r,np.empty((0,len(p))),4);zz=m.network(ff,p)[0];aa=20*np.log10(abs(zz));pp=np.rad2deg(np.angle(zz));tol_a=.1;tol_p=.5
        for attempt in range(6):
            ix=compress(xx,a,theta,forced,tol_a,tol_p);x=xx[ix];gap_a=float(np.max(abs(aa-np.interp(np.log10(ff),x,a[ix]))));gap_p=float(np.max(abs(pp-np.interp(np.log10(ff),x,theta[ix]))))
            if gap_a<=.2 and gap_p<=1.:break
            tol_a/=2;tol_p/=2
        else:raise RuntimeError((ident,'grid validation failed',gap_a,gap_p))
        f=10**x;weights,cells=cell_weights(x,ori);true=m.network(f,p)[0];amp=np.interp(x,ori,m.amp);phase=np.interp(x,ori,m.phase);interpolated=10**(amp/20)*np.exp(1j*phase)
        np.testing.assert_allclose(m.network(m.f,p)[0],m.target,rtol=1e-10,atol=1e-8)
        np.testing.assert_allclose(np.interp(ori,x,amp),m.amp,rtol=1e-10,atol=1e-10);np.testing.assert_allclose(np.interp(ori,x,phase),m.phase,rtol=1e-10,atol=1e-10)
        np.savez_compressed(HERE/'grids'/f'{ident}.npz',frequency=f,weights=weights,target_true=true,target_interpolated=interpolated,original_frequency=m.f,original_target=m.target,log_cell_edges=edges,cell_total_weights=cells)
        item=dict(circuit=ident,original_points=len(m.f),dense_points=len(f),amplitude_max_holdout_gap_db=gap_a,phase_max_holdout_gap_deg=gap_p,compression_tolerance_db=tol_a,compression_tolerance_deg=tol_p,original_gap_db=float(next(d['target_sampling_gap_max_db'] for d in diagnostics if d['circuit']==ident)),cell_weights_preserved=True)
        checks.append(item);records.append(dict(circuit=ident,seeds=list(range(r['seed_start'],r['seed_start']+r['seed_count'])),population=r['population'],generations=300,alpha=r['alpha']))
        print(json.dumps(item),flush=True)
    runs=sum(len(r['seeds'])*6 for r in records);budget=sum(len(r['seeds'])*6*r['population']*300 for r in records);assert runs==1680
    config=dict(records=records,arms=['dense_interpolated','dense_true'],groups=['wsmse','mean_distance','dual_mse'],planned_runs=runs,planned_evaluations=budget,workers=28)
    (HERE/'CONFIG.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    files=[HERE/'PROTOCOL.md',HERE/'prepare.py',HERE/'run_sampling.py',HERE/'CONFIG.json',*list((HERE/'grids').glob('*.npz')),*list(frozen.rglob('*'))];files=[p for p in files if p.is_file() and '__pycache__' not in str(p)]
    report=dict(passed=True,target_only_grids=True,planned_runs=runs,planned_evaluations=budget,grid_checks=checks,sha256={p.relative_to(HERE).as_posix():sha(p) for p in files})
    (HERE/'PREPARATION.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with tarfile.open(HERE/'optimization_bundle.tar.gz','w:gz') as tar:
        for p in [*files,HERE/'PREPARATION.json']:tar.add(p,arcname=p.relative_to(HERE).as_posix())
    print(json.dumps(dict(bundle_sha256=sha(HERE/'optimization_bundle.tar.gz'),bytes=(HERE/'optimization_bundle.tar.gz').stat().st_size,planned_runs=runs,planned_evaluations=budget)))
if __name__=='__main__':main()

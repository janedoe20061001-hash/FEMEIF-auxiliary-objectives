"""Verify complete archive, baseline reproducibility, and independent formulas.

No fitting or selection changes are made. Independent quadratures use two-point
Gauss rules, not the centered-moment or Simpson routines in the objective code.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import argparse,csv,hashlib,json,tarfile,sys
import numpy as np
from scipy.spatial.distance import cdist

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--reuse-extracted',action='store_true',help='Read-only hash verification of an already extracted archive')
args=parser.parse_args()
run=root/'experiments/curve3d_objectives_20261003'
archive=run/'server_run/formal_results.tar.gz'
expected=(run/'server_run/formal_results.sha256').read_text().split()[0]
actual=hashlib.sha256(archive.read_bytes()).hexdigest()
assert actual==expected
status=json.loads((run/'server_run/formal_status.json').read_text())
assert status['status']=='complete' and status['completed_runs']==2200 and status['failed_runs']==0
assert status['evaluations']==115500000
out=run/'server_run/raw_results'
if out.exists() and not args.reuse_extracted:raise FileExistsError(out)
reproduction={}
with tarfile.open(archive,'r:gz') as tf:
    members=tf.getmembers()
    for member in members:
        path=Path(member.name)
        assert not path.is_absolute() and '..' not in path.parts and member.isfile()
    summaries=[m for m in members if '/runs/' in m.name and m.name.endswith('/summary.json')]
    assert len(summaries)==2200
    for case in ['case1','case2']:
        audit=json.load(tf.extractfile(f'{case}/audit.json'))
        assert audit['status']=='passed' and audit['runs']==1100
        manifest=json.load(tf.extractfile(f'{case}/runs/manifest.json'))
        for filename,digest in manifest['code_sha256'].items():
            archived=tf.extractfile(f'{case}/{filename}').read()
            assert hashlib.sha256(archived).hexdigest()==digest
            assert archived==(run/case/filename).read_bytes()
        assert json.load(tf.extractfile(f'{case}/config.json'))==json.loads((run/case/'config.json').read_text(encoding='utf-8'))
        with (root/f'experiments/{case}_dual_pdf18/server_run/selected_individuals.csv').open(encoding='utf-8-sig',newline='') as fh:
            old={int(r['seed']):r for r in csv.DictReader(fh) if r['group']=='baseline'}
        errors=[]
        for seed in range(100):
            r=json.load(tf.extractfile(f'{case}/runs/baseline/seed_{seed:03d}/summary.json'))
            for key in ['wsmse_selected_magnitude_mse','wsmse_selected_phase_mse','wsmse_selected_value']:
                value=r[key];previous=float(old[seed][key])
                np.testing.assert_allclose(value,previous,rtol=1e-12,atol=1e-13)
                errors.append(abs(value-previous))
        reproduction[case]=dict(seeds=100,max_abs_difference=max(errors),matches_historical_baseline=True)
    if out.exists():
        # Never overwrite an existing extraction, including after a failed check.
        for member in members:
            saved=out/member.name
            assert saved.is_file() and not saved.is_symlink() and saved.stat().st_size==member.size
            assert hashlib.sha256(saved.read_bytes()).digest()==hashlib.sha256(tf.extractfile(member).read()).digest()
    else:tf.extractall(out,filter='data')

# Import only circuit evaluators; metric formulas below are independent.
sys.path.insert(0,str(out/'case1'))
from legacy_objectives import impedance as circuit1,TARGET
sys.path.insert(0,str(out/'case2'))
from case2 import impedance as circuit2,load_target

def path(z,u,scale,coord,offset,extent):
    xy=np.stack([z.real,z.imag],axis=-1)
    if coord=='asinh':xy=np.arcsinh(xy/scale)
    xy=(xy-offset)/extent
    return np.concatenate([np.broadcast_to(u,xy.shape[:-1])[...,None],xy],axis=-1)

def moment(p,h):
    # Independent Gauss integration: exact for quadratic outer products.
    t=np.array([.5-.5/np.sqrt(3),.5+.5/np.sqrt(3)])
    sample=(1-t[:,None])*p[...,:-1,None,:]+t[:,None]*p[...,1:,None,:]
    mu=np.sum(sample*h[:,None,None]/2,axis=(-3,-2))
    centered=sample-mu[...,None,None,:]
    return np.einsum('...igj,...igk,i->...jk',centered,centered,h/2)

def projected_area(p):
    # Shoelace on the closed polygon, computed separately in each plane.
    next_p=np.roll(p,-1,axis=-2)
    return .5*np.stack([np.sum(p[..., :,j]*next_p[..., :,k]-p[..., :,k]*next_p[..., :,j],axis=-1)
                       for j,k in [(1,2),(2,0),(0,1)]],axis=-1)

def arc_integral(p,t):
    def knots(v):
        s=np.r_[0,np.cumsum(np.sqrt(np.sum(np.diff(v,axis=0)**2,axis=1)))];return s/s[-1]
    a=knots(p);b=knots(t);union=np.unique(np.r_[a,b]);h=np.diff(union)
    gauss=union[:-1,None]+h[:,None]*np.array([.5-.5/np.sqrt(3),.5+.5/np.sqrt(3)])
    error=np.stack([np.interp(gauss,a,p[:,j])-np.interp(gauss,b,t[:,j]) for j in range(3)],axis=-1)
    return np.sum(np.sum(error*error,axis=-1)*h[:,None]/2)

checked=0;individuals=0;max_abs=0.;max_primary_abs=0.;max_primary_relative=0.
for case in ['case1','case2']:
    case_root=out/case;cfg=json.loads((case_root/'config.json').read_text(encoding='utf-8'))
    if case=='case1':
        frequency=np.geomspace(100,1e6,180);target=circuit1(frequency,TARGET)[0];circuit=circuit1
    else:
        frequency,target=load_target({**cfg['target_data'],'file':str(case_root/cfg['target_data']['file'])});circuit=circuit2
    logf=np.log10(frequency);u=(logf-logf[0])/(logf[-1]-logf[0]);h=np.diff(u)
    w=np.r_[h[0]/2,(h[:-1]+h[1:])/2,h[-1]/2];ww=w[:,None]*w[None,:]
    scale=np.median(abs(target));refs={}
    for coord in ['linear','asinh']:
        xy=np.column_stack([target.real,target.imag])
        if coord=='asinh':xy=np.arcsinh(xy/scale)
        offset=xy.min(axis=0);extent=xy.max(axis=0)-offset
        t=path(target,u,scale,coord,offset,extent)
        refs[coord]=(offset,extent,t)
    for folder in sorted((case_root/'runs').glob('*/seed_*')):
        group=folder.parent.name
        with np.load(folder/'final_population.npz') as saved:
            z=circuit(frequency,10**saved['log_parameters']);stored=saved['objectives']
        individuals+=len(stored)
        primary=np.column_stack([np.mean((20*np.log10(np.maximum(abs(z),1e-300))-20*np.log10(abs(target)))**2,axis=1),
                                 np.mean((np.angle(z)-np.angle(target))**2,axis=1)])
        difference=abs(stored[:,:2]-primary)
        max_primary_abs=max(max_primary_abs,float(difference.max()))
        max_primary_relative=max(max_primary_relative,float(np.max(difference/np.maximum(abs(stored[:,:2]),np.finfo(float).tiny))))
        # Circuit evaluation is repeated across Linux and Windows. Tiny differences
        # in exponentiation/trigonometry can be amplified around high-Q resonances.
        # Record the maxima explicitly; tolerance is negligible versus reported MSE.
        np.testing.assert_allclose(stored[:,:2],primary,rtol=1e-9,atol=1e-11)
        if group!='baseline':
            family,coord=group.rsplit('_',1);offset,extent,t=refs[coord]
            p=path(z,u,scale,coord,offset,extent)
            if family=='chords':
                dt=cdist(t,t);denominator=np.sum(ww*dt**2)
                value=np.array([np.sqrt(np.sum(ww*(cdist(v,v)-dt)**2)/denominator) for v in p])
            elif family=='covariance':
                ct=moment(t,h);value=np.sqrt(np.sum((moment(p,h)-ct)**2,axis=(-2,-1)))/np.sqrt(np.sum(ct*ct))
            elif family=='area':value=np.sqrt(np.sum((projected_area(p)-projected_area(t))**2,axis=-1))
            elif family=='srvf':
                vt=np.diff(t,axis=0)/h[:,None];v=np.diff(p,axis=1)/h[:,None]
                qt=vt/np.sum(vt*vt,axis=-1)[:,None]**.25;q=v/np.sum(v*v,axis=-1)[...,None]**.25
                value=np.sum(np.sum((q-qt)**2,axis=-1)*h,axis=-1)/np.sum(np.sqrt(np.sum(vt*vt,axis=-1))*h)
            elif family=='arclength':value=np.array([arc_integral(v,t) for v in p])
            else:raise AssertionError(group)
            np.testing.assert_allclose(stored[:,2],value,rtol=2e-9,atol=1e-10)
            max_abs=max(max_abs,float(np.max(abs(stored[:,2]-value))))
        checked+=1
    print(case,'independent formulas passed',flush=True)
assert checked==2200
assert individuals==385000
record=dict(status='passed',sha256=actual,archive_bytes=archive.stat().st_size,
            completed_runs=2200,extracted_directory=str(out),baseline_reproduction=reproduction,
            frozen_source_and_configuration_verified=True,
            independently_checked_final_individuals=individuals,
            independent_primary_max_abs_difference=max_primary_abs,
            independent_primary_max_relative_difference=max_primary_relative,
            cross_platform_primary_tolerance=dict(rtol=1e-9,atol=1e-11),
            independent_all_final_population_formula_checks=True,independent_max_abs_difference=max_abs)
(run/'server_run/download_audit.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record))

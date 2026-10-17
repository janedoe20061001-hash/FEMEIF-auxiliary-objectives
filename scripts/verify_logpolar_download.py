"""Streaming archive checks and independent full-population formula verification."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import argparse,csv,hashlib,json,tarfile,sys
import numpy as np
from scipy.spatial.distance import cdist

root=Path(__file__).resolve().parents[1];run=root/'experiments/curve3d_logpolar_native_20261003'
parser=argparse.ArgumentParser();parser.add_argument('--reuse-extracted',action='store_true');args=parser.parse_args()
archive=run/'server_run/formal_results.tar.gz';out=run/'server_run/raw_results'
actual=hashlib.sha256(archive.read_bytes()).hexdigest()
assert actual==(run/'server_run/formal_results.sha256').read_text().split()[0]
status=json.loads((run/'server_run/formal_status.json').read_text())
assert status['status']=='complete' and status['completed_runs']==1400 and status['failed_runs']==0 and status['evaluations']==73500000
reuse=out.exists()
if reuse and not args.reuse_extracted:raise FileExistsError(out)
summaries=0
with tarfile.open(archive,'r|gz') as tf:
    for member in tf:
        name=Path(member.name)
        assert member.isfile() and not name.is_absolute() and '..' not in name.parts
        if reuse:
            saved=out/name
            assert saved.is_file() and not saved.is_symlink() and saved.stat().st_size==member.size
            assert hashlib.sha256(saved.read_bytes()).digest()==hashlib.sha256(tf.extractfile(member).read()).digest()
        else:tf.extract(member,out,filter='data')
        if '/runs/' in member.name and member.name.endswith('/summary.json'):summaries+=1
assert summaries==1400
assert (out/'RESULTS.json').read_bytes()==(run/'RESULTS.json').read_bytes()
reproduction={}
for case in ['case1','case2']:
    audit=json.loads((out/case/'audit.json').read_text());assert audit['status']=='passed' and audit['runs']==700
    manifest=json.loads((out/case/'runs/manifest.json').read_text())
    for filename,digest in manifest['code_sha256'].items():
        data=(out/case/filename).read_bytes()
        assert hashlib.sha256(data).hexdigest()==digest
        if filename not in ['orientation_native.so','build_native.json']:
            assert data==(run/case/filename).read_bytes()
    build=json.loads((out/case/'build_native.json').read_text())
    assert build['source_sha256']==hashlib.sha256((out/case/'orientation_native.c').read_bytes()).hexdigest()
    assert build['binary_sha256']==hashlib.sha256((out/case/'orientation_native.so').read_bytes()).hexdigest()
    assert build['flags']==['-O3','-fno-fast-math','-ffp-contract=off','-shared','-fPIC']
    assert json.loads((out/case/'config.json').read_text())==json.loads((run/case/'config.json').read_text(encoding='utf-8'))
    with (root/f'experiments/{case}_dual_pdf18/server_run/selected_individuals.csv').open(encoding='utf-8-sig',newline='') as fh:
        old={int(r['seed']):r for r in csv.DictReader(fh) if r['group']=='baseline'}
    errors=[]
    for seed in range(100):
        record=json.loads((out/case/f'runs/baseline/seed_{seed:03d}/summary.json').read_text())
        for key in ['wsmse_selected_magnitude_mse','wsmse_selected_phase_mse','wsmse_selected_value']:
            previous=float(old[seed][key]);value=record[key]
            np.testing.assert_allclose(value,previous,rtol=1e-12,atol=1e-13);errors.append(abs(value-previous))
    reproduction[case]=dict(seeds=100,max_abs_difference=max(errors),matches_historical_baseline=True)
print('Archive, frozen sources, complete summaries and historical baselines verified',flush=True)
equivalence=json.loads((out/'native_equivalence.json').read_text())
assert equivalence['status']=='passed' and len(equivalence['checks'])==2
assert all(row['bitwise_equal'] and row['max_abs_difference']==0 for row in equivalence['checks'])

sys.path.insert(0,str(out/'case1'))
from legacy_objectives import impedance as circuit1,TARGET
sys.path.insert(0,str(out/'case2'))
from case2 import impedance as circuit2,load_target

def moment(p,h):
    t=np.array([.5-.5/np.sqrt(3),.5+.5/np.sqrt(3)])
    sample=(1-t[:,None])*p[...,:-1,None,:]+t[:,None]*p[...,1:,None,:]
    mu=np.sum(sample*h[:,None,None]/2,axis=(-3,-2));centered=sample-mu[...,None,None,:]
    return np.einsum('...igj,...igk,i->...jk',centered,centered,h/2)

def projected_area(p):
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

checked=individuals=0;max_primary_abs=max_primary_relative=max_aux_abs=0.
for case in ['case1','case2']:
    case_root=out/case;cfg=json.loads((case_root/'config.json').read_text())
    if case=='case1':frequency=np.geomspace(100,1e6,180);target=circuit1(frequency,TARGET)[0];circuit=circuit1
    else:frequency,target=load_target({**cfg['target_data'],'file':str(case_root/cfg['target_data']['file'])});circuit=circuit2
    logf=np.log10(frequency);u=(logf-logf[0])/(logf[-1]-logf[0]);h=np.diff(u)
    w=np.r_[h[0]/2,(h[:-1]+h[1:])/2,h[-1]/2];ww=w[:,None]*w[None,:]
    xy=np.column_stack([20*np.log10(abs(target)),np.angle(target)]);offset=xy.min(axis=0);extent=np.ptp(xy,axis=0)
    t=np.column_stack([u,(xy-offset)/extent]);dt=cdist(t,t);ct=moment(t,h)
    ii,jj=np.triu_indices(len(u),1);pair_weight=w[ii]*w[jj]
    delta_t=t[jj]-t[ii];norm_t=np.sqrt(np.sum(delta_t*delta_t,axis=1))
    for folder in sorted((case_root/'runs').glob('*/seed_*')):
        group=folder.parent.name
        with np.load(folder/'final_population.npz') as saved:
            z=circuit(frequency,10**saved['log_parameters']);stored=saved['objectives']
        individuals+=len(stored)
        primary=np.column_stack([np.mean((20*np.log10(np.maximum(abs(z),1e-300))-20*np.log10(abs(target)))**2,axis=1),
                                 np.mean((np.angle(z)-np.angle(target))**2,axis=1)])
        diff=abs(stored[:,:2]-primary);max_primary_abs=max(max_primary_abs,float(diff.max()))
        max_primary_relative=max(max_primary_relative,float(np.max(diff/np.maximum(abs(stored[:,:2]),np.finfo(float).tiny))))
        np.testing.assert_allclose(stored[:,:2],primary,rtol=1e-9,atol=1e-11)
        if group!='baseline':
            family=group.rsplit('_',1)[0]
            xy=np.stack([20*np.log10(abs(z)),np.angle(z)],axis=-1);xy=(xy-offset)/extent
            p=np.concatenate([np.broadcast_to(u,xy.shape[:-1])[...,None],xy],axis=-1)
            if family=='chords':value=np.array([np.sqrt(np.sum(ww*(cdist(v,v)-dt)**2)/np.sum(ww*dt**2)) for v in p])
            elif family=='covariance':value=np.sqrt(np.sum((moment(p,h)-ct)**2,axis=(-2,-1)))/np.sqrt(np.sum(ct*ct))
            elif family=='area':value=np.sqrt(np.sum((projected_area(p)-projected_area(t))**2,axis=-1))
            elif family=='srvf':
                vt=np.diff(t,axis=0)/h[:,None];v=np.diff(p,axis=1)/h[:,None]
                qt=vt/np.sum(vt*vt,axis=-1)[:,None]**.25;q=v/np.sum(v*v,axis=-1)[...,None]**.25
                value=np.sum(np.sum((q-qt)**2,axis=-1)*h,axis=-1)/np.sum(np.sqrt(np.sum(vt*vt,axis=-1))*h)
            elif family=='arclength':value=np.array([arc_integral(v,t) for v in p])
            elif family=='orientation':
                values=[]
                for v in p:
                    delta=v[jj]-v[ii];norm=np.sqrt(np.sum(delta*delta,axis=1))
                    cosine=np.sum(delta*delta_t,axis=1)/(norm*norm_t)
                    values.append(np.sum(pair_weight*2*(1-cosine))/pair_weight.sum())
                value=np.array(values)
            else:raise AssertionError(family)
            np.testing.assert_allclose(stored[:,2],value,rtol=2e-9,atol=1e-10)
            max_aux_abs=max(max_aux_abs,float(np.max(abs(stored[:,2]-value))))
        checked+=1
    print(case,'independent formulas passed',flush=True)
assert checked==1400 and individuals==245000
record=dict(status='passed',sha256=actual,archive_bytes=archive.stat().st_size,completed_runs=1400,
    extracted_directory=str(out),baseline_reproduction=reproduction,frozen_source_and_configuration_verified=True,
    independently_checked_final_individuals=individuals,independent_all_final_population_formula_checks=True,
    independent_primary_max_abs_difference=max_primary_abs,independent_primary_max_relative_difference=max_primary_relative,
    independent_max_abs_difference=max_aux_abs,cross_platform_primary_tolerance=dict(rtol=1e-9,atol=1e-11))
(run/'server_run/download_audit.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print(json.dumps(record))

"""Download/source integrity plus independent formulas on every final individual."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import argparse,csv,hashlib,json,tarfile,sys
import numpy as np
from scipy.stats import wilcoxon

root=Path(__file__).resolve().parents[1];run=root/'experiments/curve3d_interval_20261003'
parser=argparse.ArgumentParser();parser.add_argument('--reuse-extracted',action='store_true');args=parser.parse_args()
archive=run/'server_run/formal_results.tar.gz';out=run/'server_run/raw_results'
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
assert digest==(run/'server_run/formal_results.sha256').read_text().split()[0]
status=json.loads((run/'server_run/formal_status.json').read_text())
assert status['status']=='complete' and status['completed_runs']==800 and status['evaluations']==42000000 and status['failed_runs']==0
reuse=out.exists()
if reuse and not args.reuse_extracted:raise FileExistsError(out)
summaries=0
with tarfile.open(archive,'r|gz') as tf:
    for member in tf:
        path=Path(member.name)
        assert member.isfile() and not path.is_absolute() and '..' not in path.parts
        if reuse:
            saved=out/path;assert saved.is_file() and not saved.is_symlink() and saved.stat().st_size==member.size
            assert hashlib.sha256(saved.read_bytes()).digest()==hashlib.sha256(tf.extractfile(member).read()).digest()
        else:tf.extract(member,out,filter='data')
        if '/runs/' in member.name and member.name.endswith('/summary.json'):summaries+=1
assert summaries==800
for name in ['RESULTS.json','PARENT_COMPARISONS.json','FROZEN.json']:
    assert (out/name).read_bytes()==(run/name).read_bytes()
frozen=json.loads((out/'FROZEN.json').read_text())
for name,sha in frozen['files_sha256'].items():
    normalized=Path(name.replace('\\','/'))
    assert hashlib.sha256((out/normalized).read_bytes()).hexdigest()==sha
    assert (out/normalized).read_bytes()==(run/normalized).read_bytes()
for case in ['case1','case2']:
    case_root=out/case;cfg=json.loads((case_root/'config.json').read_text())
    assert cfg['seeds']==dict(start=1100,count=100)
    old=json.loads((root/f'experiments/curve3d_refinement_20261003/{case}/config.json').read_text(encoding='utf-8'))
    for k in cfg:
        if k not in ['protocol','seeds','groups']:assert cfg[k]==old[k],(case,k)
    audit=json.loads((case_root/'audit.json').read_text());assert audit['status']=='passed' and audit['runs']==400
    mf=json.loads((case_root/'runs/manifest.json').read_text())
    for name,sha in mf['code_sha256'].items():assert hashlib.sha256((case_root/name).read_bytes()).hexdigest()==sha
print('Archive, frozen sources, configurations, all completion markers verified',flush=True)
checks=json.loads((out/'software_checks.json').read_text());assert checks['status']=='passed'
assert all(r['old_controls_bitwise_equal'] and r['max_abs_difference']==0 for r in checks['control_equivalence'])
sys.path.insert(0,str(out/'case1'));from legacy_objectives import impedance as circuit1,TARGET,LOWER as LO1,UPPER as HI1
sys.path.insert(0,str(out/'case2'));from case2 import impedance as circuit2,load_target,LOWER as LO2,UPPER as HI2
gauss=np.array([.5-.5/np.sqrt(3),.5+.5/np.sqrt(3)])
def integrate_square(e,h):
    a=e[...,:-1,None,:];b=e[...,1:,None,:]
    samples=(1-gauss[:,None])*a+gauss[:,None]*b
    return np.sum(np.sum(samples*samples,axis=-1)*h[:,None]/2,axis=(-2,-1))
def prefixes(p):
    # Shoelace in all three projection planes, closing each prefix to p_1.
    output=[]
    for j,k in [(1,2),(2,0),(0,1)]:
        edge=p[..., :-1,j]*p[...,1:,k]-p[..., :-1,k]*p[...,1:,j]
        cumulative=np.concatenate([np.zeros_like(edge[..., :1]),np.cumsum(edge,axis=-1)],axis=-1)
        closing=p[..., :,j]*p[..., :1,k]-p[..., :,k]*p[..., :1,j]
        output.append(.5*(cumulative+closing))
    return np.stack(output,axis=-1)
def lengths(p):
    delta=p[...,1:,:]-p[...,:-1,:];edge=np.sqrt(np.sum(delta*delta,axis=-1))
    return np.concatenate([np.zeros_like(edge[..., :1]),np.cumsum(edge,axis=-1)],axis=-1)
def arc_integral(p,t):
    a=lengths(p);a=a/a[-1];b=lengths(t);b=b/b[-1]
    union=np.unique(np.r_[a,b]);h=np.diff(union);points=union[:-1,None]+h[:,None]*gauss
    err=np.stack([np.interp(points,a,p[:,j])-np.interp(points,b,t[:,j]) for j in range(3)],axis=-1)
    return np.sum(np.sum(err*err,axis=-1)*h[:,None]/2)
checked=individuals=0;max_primary_abs=max_primary_relative=max_aux_abs=0.
for case in ['case1','case2']:
    folder=out/case;cfg=json.loads((folder/'config.json').read_text())
    if case=='case1':f=np.geomspace(100,1e6,180);target=circuit1(f,TARGET)[0];circuit=circuit1;lower,upper=LO1,HI1;nvar=18
    else:f,target=load_target({**cfg['target_data'],'file':str(folder/cfg['target_data']['file'])});circuit=circuit2;lower,upper=LO2,HI2;nvar=15
    x=np.log10(f);u=(x-x[0])/(x[-1]-x[0]);h=np.diff(u)
    xy=np.column_stack([20*np.log10(abs(target)),np.angle(target)])
    sample=(1-gauss[:,None])*xy[:-1,None,:]+gauss[:,None]*xy[1:,None,:]
    mu=np.sum(sample*h[:,None,None]/2,axis=(0,1));sigma=np.sqrt(np.sum((sample-mu)**2*h[:,None,None]/2,axis=(0,1)))
    refs={'logpolar':(xy.min(axis=0),np.ptp(xy,axis=0)),'stdpolar':(mu,sigma)}
    for directory in sorted((folder/'runs').glob('*/seed_*')):
        seed=int(directory.name.split('_')[1]);assert 1100<=seed<1200
        np.testing.assert_array_equal(np.load(directory/'initial_log_parameters.npy'),
                                     np.random.default_rng(seed).uniform(lower,upper,(cfg['population'],nvar)))
        record=json.loads((directory/'summary.json').read_text())
        assert record['evaluations']==300*cfg['population'] and record['generations']==300
        with (directory/'history.csv').open(encoding='utf-8-sig',newline='') as fh:
            history=list(csv.DictReader(fh));assert len(history)==300 and int(history[-1]['evaluations'])==300*cfg['population']
        with np.load(directory/'final_population.npz') as saved:
            z=circuit(f,10**saved['log_parameters']);stored=saved['objectives'];idx=int(saved['selected_index'])
            assert idx==int(np.argmin(stored[:,0]+cfg['final_selection']['phase_weight']*stored[:,1]))
        primary=np.column_stack([np.mean((20*np.log10(np.maximum(abs(z),1e-300))-20*np.log10(abs(target)))**2,axis=1),
                                  np.mean((np.angle(z)-np.angle(target))**2,axis=1)])
        error=abs(stored[:,:2]-primary);max_primary_abs=max(max_primary_abs,float(error.max()))
        max_primary_relative=max(max_primary_relative,float(np.max(error/np.maximum(abs(stored[:,:2]),np.finfo(float).tiny))))
        np.testing.assert_allclose(stored[:,:2],primary,rtol=1e-9,atol=1e-11)
        group=directory.parent.name
        names=[] if group=='baseline' else group.split('+')
        assert stored.shape==(cfg['population'],2+len(names))
        if names:
            offset,scale=refs['logpolar']
            t=np.column_stack([u,(xy-offset)/scale]);candidate=np.stack([20*np.log10(abs(z)),np.angle(z)],axis=-1)
            candidate=(candidate-offset)/scale;p=np.concatenate([np.broadcast_to(u,candidate.shape[:-1])[...,None],candidate],axis=-1)
            for column,name in enumerate(names,2):
                family=name.rsplit('_',1)[0]
                if family=='cumulative_area':value=np.sqrt(integrate_square(prefixes(p)-prefixes(t),h))
                elif family in ['cumulative_length','symmetric_length']:
                    bp,bt=lengths(p),lengths(t);err=(bp-bt)/bt[-1]
                    forward=integrate_square(err[...,None],h)
                    if family=='cumulative_length':value=np.sqrt(forward)
                    else:
                        # Independently form remaining length, not e_N-e.
                        backward=((bp[...,-1:]-bp)-(bt[-1]-bt))/bt[-1]
                        value=np.sqrt((forward+integrate_square(backward[...,None],h))/2)
                elif family in ['axis_variation','direction_variation']:
                    delta=p[...,1:,1:]-p[...,:-1,1:];dt=t[1:,1:]-t[:-1,1:]
                    def prefix(d):return np.concatenate([np.zeros_like(d[...,:1,:]),np.cumsum(d,axis=-2)],axis=-2)
                    vp,vt=prefix(abs(delta)),prefix(abs(dt));total=vt[-1]
                    if family=='axis_variation':error=(vp-vt)/total
                    else:
                        # Jordan identity rather than max used in production.
                        disp=p[...,1:]-p[...,:1,1:];target_disp=t[:,1:]-t[:1,1:]
                        positive=(vp+disp)/2;negative=(vp-disp)/2
                        tp=(vt+target_disp)/2;tn=(vt-target_disp)/2
                        error=np.concatenate([(positive-tp)/total,(negative-tn)/total],axis=-1)
                    value=np.sqrt(integrate_square(error,h)/2)
                elif family=='interval_length':
                    bp,bt=lengths(p),lengths(t);err=(bp-bt)/bt[-1]
                    samples=(1-gauss)*err[...,:-1,None]+gauss*err[...,1:,None]
                    weights=h[:,None]/2
                    average=np.sum(samples*weights,axis=(-2,-1))
                    value=np.sqrt(2*np.sum((samples-average[...,None,None])**2*weights,axis=(-2,-1)))
                elif family=='hellinger_length':
                    # Independent speed-density quadrature, not the segment-root implementation.
                    vp=(p[...,1:,:]-p[...,:-1,:])/h[:,None]
                    vt=(t[1:,:]-t[:-1,:])/h[:,None]
                    sp=np.sqrt(np.sum(vp*vp,axis=-1));st=np.sqrt(np.sum(vt*vt,axis=-1))
                    value=np.sqrt(np.sum(h*(np.sqrt(sp)-np.sqrt(st))**2,axis=-1)/np.sum(h*st))
                else:raise AssertionError(family)
                np.testing.assert_allclose(stored[:,column],value,rtol=2e-9,atol=1e-10)
                max_aux_abs=max(max_aux_abs,float(np.max(abs(stored[:,column]-value))))
        checked+=1;individuals+=len(stored)
    print(case,'all independent final population formula checks passed',flush=True)
assert checked==800 and individuals==140000
rows=json.loads((out/'RESULTS.json').read_text(encoding='utf-8'))
parent_rows=json.loads((out/'PARENT_COMPARISONS.json').read_text(encoding='utf-8'))
all_rows=rows+parent_rows;assert len(rows)==6 and len(parent_rows)==4
parent_names={'R-03-C':'cumulative_area_logpolar','R-05-C':'cumulative_length_logpolar'}
bootstrap=np.random.default_rng(20261003).integers(0,100,(10000,100))
fields=['wsmse_selected_magnitude_mse','wsmse_selected_phase_mse','wsmse_selected_value']
max_stat_difference=0.
for index,r in enumerate(all_rows):
    comparator='baseline' if index<len(rows) else parent_names[r['parent']]
    def values(group):
        data=[json.loads((out/r['case']/f'runs/{group}/seed_{s:03d}/summary.json').read_text()) for s in range(1100,1200)]
        return np.array([[v[k] for k in fields] for v in data])
    a,b=values(r['group']),values(comparator)
    np.testing.assert_allclose(r['auxiliary_mean'],a.mean(axis=0),rtol=1e-13)
    np.testing.assert_allclose(r['baseline_mean'],b.mean(axis=0),rtol=1e-13)
    pct=100*(b.mean(axis=0)-a.mean(axis=0))/b.mean(axis=0)
    ci=np.quantile(100*(1-a[bootstrap].mean(axis=1)/b[bootstrap].mean(axis=1)),[.025,.975],axis=0).T
    np.testing.assert_allclose(r['reduction_pct'],pct,rtol=1e-11,atol=1e-11)
    np.testing.assert_allclose(r['reduction_pct_bootstrap95'],ci,rtol=1e-12,atol=1e-12)
    max_stat_difference=max(max_stat_difference,float(abs(np.array(r['reduction_pct'])-pct).max()))
    p=[float(wilcoxon(a[:,i]-b[:,i],method='approx').pvalue) if np.any(a[:,i]!=b[:,i]) else 1. for i in range(2)]
    np.testing.assert_allclose(r['wilcoxon_two_sided_p'],p,rtol=1e-12,atol=1e-14)
pvals=np.array([p for r in all_rows for p in r['wilcoxon_two_sided_p']]);assert len(pvals)==20
# Independent explicit Holm step-down loop (rather than cumulative vector code).
corrected=np.zeros(20);last=0.
for rank,index in enumerate(np.argsort(pvals)):
    last=max(last,min(1.,(20-rank)*pvals[index]));corrected[index]=last
for index,r in enumerate(all_rows):
    np.testing.assert_allclose(r['holm_adjusted_p_20_primary_comparisons'],corrected[index*2:index*2+2],rtol=1e-13)
record=dict(status='passed',sha256=digest,archive_bytes=archive.stat().st_size,completed_runs=checked,
    independent_final_individuals=individuals,independent_primary_max_abs_difference=max_primary_abs,
    independent_primary_max_relative_difference=max_primary_relative,independent_aux_max_abs_difference=max_aux_abs,
    all_initial_populations_verified=True,all_histories_verified=True,final_selection_verified=True,
    frozen_sources_verified=True,original_non_group_non_seed_settings_unchanged=True,
    old_controls_bitwise_equal_on_both_full_initial_populations=True,new_seed_block=[1100,1199],
    all_10_comparisons_mean_percent_bootstrap_wilcoxon_holm_independently_verified=True,
    mean_percentage_max_abs_difference=max_stat_difference)
(run/'server_run/download_audit.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record))

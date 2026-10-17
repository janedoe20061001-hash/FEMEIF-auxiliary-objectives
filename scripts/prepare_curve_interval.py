"""Prepare and freeze two exact length refinements; no optimization in this script."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import shutil
import tarfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'experiments/curve3d_shortlist_20261003'
DST = ROOT/'experiments/curve3d_interval_20261003'
if (DST/'FROZEN.json').exists():
    raise RuntimeError('Already frozen')
source_frozen = json.loads((SRC/'FROZEN.json').read_text(encoding='utf-8'))
for name, digest in source_frozen['files_sha256'].items():
    assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==digest, name
recovery = json.loads((SRC/'PACKAGING_RECOVERY.json').read_text())
assert hashlib.sha256((SRC/recovery['file']).read_bytes()).hexdigest()==recovery['sha256']

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

gm = load('interval_metrics', DST/'global_metrics.py')
old = load('previous_metrics', SRC/'global_metrics.py')
shutil.copy2(SRC/'curve_math.py', DST/'curve_math.py')
for case in ['case1','case2']:
    a, b = SRC/case, DST/case
    b.mkdir(exist_ok=True)
    for p in a.iterdir():
        if p.is_file() and p.suffix in ['.py','.txt']:
            shutil.copy2(p,b/p.name)
    if case=='case2':
        shutil.copytree(a/'data',b/'data')
        shutil.copy2(a/'analysis_config.json',b/'analysis_config.json')
    shutil.copy2(DST/'global_metrics.py',b/'global_metrics.py')
    cfg = json.loads((a/'config.json').read_text())
    cfg.update(seeds=dict(start=1100,count=100), groups=[list(g) for g in gm.GROUPS],
               protocol='curve3d_interval_20261003_v1')
    (b/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')

audit=(SRC/'audit_case.py').read_text(encoding='utf-8').replace('range(1000,1100)','range(1100,1200)')
(DST/'audit_case.py').write_text(audit,encoding='utf-8')
runner=(SRC/'run_formal.py').read_text(encoding='utf-8').replace('completed_runs=600','completed_runs=800').replace('evaluations=31500000','evaluations=42000000')
(DST/'run_formal.py').write_text(runner,encoding='utf-8')
report=(SRC/'finish_report.py').read_text(encoding='utf-8')
report=report.replace("from global_metrics import CATALOG\nCOMPARISONS=[('symmetric_length_logpolar','cumulative_length_logpolar')]",'from global_metrics import CATALOG,COMPARISONS')
for a,b in [('12-test','20-test'),("audit['runs']==300","audit['runs']==400"),
    ('range(1000,1100)','range(1100,1200)'),('len(p)==12','len(p)==20'),('p_12_primary','p_20_primary'),
    ('固定三组：累计弧长候选复核','全区间与逐段弧长：正式实验'),
    ('两个 Case 各 3 组，每组 100 个新配对种子（1000–1099）；600 次完整运行，31,500,000 次评估。',
     '两个 Case 各 4 组，每组 100 个新配对种子（1100–1199）；800 次完整运行，42,000,000 次评估。'),
    ('所有 8 项对基线和 4 项对累计弧长','所有 12 项对基线和 8 项对累计弧长'),
    ('本轮仅比较两个固定弧长候选；不消除平移、噪声与不可达目标的问题，不增加额外指标。',
     '本轮检验全区间长度和逐段长度平方根两个改法；二者不消除平移、噪声与不可达目标的问题。')]:
    assert a in report,a
    report=report.replace(a,b)
(DST/'finish_report.py').write_text(report,encoding='utf-8')

verifier=(ROOT/'scripts/verify_curve_shortlist.py').read_text(encoding='utf-8')
for a,b in [('curve3d_shortlist_20261003','curve3d_interval_20261003'),('==600','==800'),
    ('==31500000','==42000000'),('==105000','==140000'),('start=1000,count=100','start=1100,count=100'),
    ("audit['runs']==300","audit['runs']==400"),('1000<=seed<1100','1100<=seed<1200'),
    ('range(1000,1100)','range(1100,1200)'),('len(rows)==4 and len(parent_rows)==2','len(rows)==6 and len(parent_rows)==4'),
    ('len(pvals)==12','len(pvals)==20'),('np.zeros(12)','np.zeros(20)'),('(12-rank)','(20-rank)'),
    ('p_12_primary','p_20_primary'),('new_seed_block=[1000,1099]','new_seed_block=[1100,1199]'),
    ('all_6_comparisons','all_10_comparisons')]:
    assert a in verifier,a
    verifier=verifier.replace(a,b)
addition="""                elif family=='interval_length':
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
"""
assert verifier.count('                else:raise AssertionError(family)')==1
verifier=verifier.replace('                else:raise AssertionError(family)',addition+'                else:raise AssertionError(family)')
verify_path=ROOT/'scripts/verify_curve_interval.py'
if verify_path.exists():raise FileExistsError(verify_path)
verify_path.write_text(verifier,encoding='utf-8')

# Formula checks on nonuniform frequency intervals, not fitted test problems.
u=np.array([0.,.07,.35,.6,1.]);h=np.diff(u)
e=np.array([[0.,.2,-.1,.3,.15],[0.,.05,.2,-.2,.4]])
g=np.array([.5-.5/np.sqrt(3),.5+.5/np.sqrt(3)])
samples=((1-g)*e[:,:-1,None]+g*e[:,1:,None]).reshape(2,-1)
weights=np.repeat(h/2,2)
direct=np.sqrt(np.sum((samples[:,:,None]-samples[:,None,:])**2*weights[:,None]*weights[None,:],axis=(1,2)))
np.testing.assert_allclose(gm.interval_loss(e,h),direct,rtol=1e-13,atol=1e-14)
np.testing.assert_allclose(gm.interval_loss(2*u,h),2/np.sqrt(6),rtol=1e-14)
reversed_e=e[:,-1:]-e[:,::-1]
np.testing.assert_allclose(gm.interval_loss(e,h),gm.interval_loss(reversed_e,h[::-1]),rtol=1e-13)
fraction=.37
u_split=np.sort(np.r_[u,u[:-1]+fraction*h])
e_split=np.stack([np.interp(u_split,u,row) for row in e])
np.testing.assert_allclose(gm.interval_loss(e,h),gm.interval_loss(e_split,np.diff(u_split)),rtol=1e-13)
target_lengths=np.array([.2,.6,.5,.9]);candidate_lengths=np.array([[.3,.4,.8,.7],[.6,.5,.1,.3]])
def split(x):return np.stack([fraction*x,(1-fraction)*x],axis=-1).reshape(x.shape[:-1]+(-1,))
np.testing.assert_allclose(gm.hellinger_loss(candidate_lengths,target_lengths),
    gm.hellinger_loss(split(candidate_lengths),split(target_lengths)),rtol=1e-13)
mass=candidate_lengths.sum(axis=1);tm=target_lengths.sum()
affinity=np.sum(np.sqrt((candidate_lengths/mass[:,None])*(target_lengths/tm)),axis=1)
decomposition=(np.sqrt(mass/tm)-1)**2+2*np.sqrt(mass/tm)*(1-affinity)
np.testing.assert_allclose(gm.hellinger_loss(candidate_lengths,target_lengths)**2,decomposition,rtol=1e-13)
# Equal segment lengths do not determine the direction of a curve.
p0=np.column_stack([u,np.array([0.,.2,-.1,.1,.3]),np.zeros(len(u))])
p1=p0.copy();p1[:,1]*=-1
l0=np.linalg.norm(np.diff(p0,axis=0),axis=1);l1=np.linalg.norm(np.diff(p1,axis=0),axis=1)
assert gm.hellinger_loss(l1,l0)==0
q0=np.diff(p0,axis=0)/np.sqrt(h*l0)[:,None]
q1=np.diff(p1,axis=0)/np.sqrt(h*l1)[:,None]
assert np.sum(h*np.sum((q1-q0)**2,axis=1))/l0.sum()>0

c1=load('interval_c1',DST/'case1/legacy_objectives.py')
c2=load('interval_c2',DST/'case2/case2.py')
checks=[];diagnostics=[];max_independent=0.
for case,mod in [('case1',c1),('case2',c2)]:
    cfg=json.loads((DST/case/'config.json').read_text())
    if case=='case1':
        f=np.geomspace(100,1e6,180);z=mod.impedance(f,mod.TARGET)[0]
    else:f,z=mod.load_target({**cfg['target_data'],'file':str(DST/case/cfg['target_data']['file'])})
    x=np.random.default_rng(1100).uniform(mod.LOWER,mod.UPPER,(cfg['population'],len(mod.LOWER)))
    spectra=mod.impedance(f,10**x)
    previous=old.GlobalObjectives(f,aux=(gm.LENGTH,),target_z=z).evaluate_spectra(spectra)
    current=gm.GlobalObjectives(f,aux=(gm.LENGTH,),target_z=z).evaluate_spectra(spectra)
    np.testing.assert_array_equal(previous,current)
    for name in gm.AUX:
        obj=gm.GlobalObjectives(f,aux=(name,),target_z=z)
        values=obj.evaluate_spectra(spectra)
        np.testing.assert_array_equal(values[:,:2],previous[:,:2])
        np.testing.assert_array_equal(obj.evaluate_spectra(z),np.zeros((1,3)))
        path=obj.path(spectra);delta=np.diff(path,axis=1)
        lengths=np.sqrt(np.sum(delta*delta,axis=-1));lt=np.sqrt(np.sum(np.diff(obj.p,axis=0)**2,axis=-1))
        bt=np.r_[0.,np.cumsum(lt)]
        bp=np.column_stack([np.zeros(len(path)),np.cumsum(lengths,axis=1)])
        error=(bp-bt)/bt[-1]
        values_g=(1-g)*error[:,:-1,None]+g*error[:,1:,None]
        weights=obj.du[:,None]/2
        if name==gm.LENGTH:independent=np.sqrt(np.sum(weights*values_g**2,axis=(1,2)))
        elif name==gm.INTERVAL:
            mean=np.sum(weights*values_g,axis=(1,2))
            independent=np.sqrt(2*np.sum(weights*(values_g-mean[:,None,None])**2,axis=(1,2)))
        else:
            sp=np.sqrt(np.sum((delta/obj.du[:,None])**2,axis=-1))
            st=np.sqrt(np.sum((np.diff(obj.p,axis=0)/obj.du[:,None])**2,axis=-1))
            independent=np.sqrt(np.sum(obj.du*(np.sqrt(sp)-np.sqrt(st))**2,axis=1)/np.sum(obj.du*st))
            # Check exact decomposition of the historical SRVF squared discrepancy.
            that=delta/lengths[...,None];tt=np.diff(obj.p,axis=0)/lt[:,None]
            qhat=delta/np.sqrt(obj.du*lengths)[...,None]
            qt=np.diff(obj.p,axis=0)/np.sqrt(obj.du*lt)[:,None]
            srv=np.sum(obj.du*np.sum((qhat-qt)**2,axis=-1),axis=1)/lt.sum()
            direction=np.sum(2*np.sqrt(lengths*lt)*(1-np.sum(that*tt,axis=-1)),axis=1)/lt.sum()
            np.testing.assert_allclose(srv,values[:,2]**2+direction,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(values[:,2],independent,rtol=1e-12,atol=1e-12)
        max_independent=max(max_independent,float(np.max(abs(values[:,2]-independent))))
    checks.append(dict(case=case,full_initial_population=cfg['population'],frequency_points=len(f),
                       old_controls_bitwise_equal=True,max_abs_difference=0.))
    diagnostics.append(dict(case=case,**obj.target_summary()))

software=dict(status='passed',no_optimization_performed=True,control_equivalence=checks,
    primary_objectives_bitwise_unchanged=True,zero_error=True,interval_double_integral_identity=True,
    interval_reversal_and_subdivision_invariance=True,hellinger_subdivision_invariance=True,
    hellinger_total_mass_shape_identity=True,srvf_length_direction_identity=True,
    length_cannot_distinguish_direction_counterexample=True,independent_max_abs_difference=max_independent,
    target_only_diagnostics=diagnostics)
(DST/'software_checks.json').write_text(json.dumps(software,ensure_ascii=False,indent=2),encoding='utf-8')
protocol=dict(protocol='curve3d_interval_20261003_v1',groups=[list(g) for g in gm.GROUPS],seeds=dict(start=1100,count=100),
    comparisons=[(n,'baseline') for n in gm.AUX]+gm.COMPARISONS,holm_primary_tests=20,planned_runs=800,
    planned_evaluations=42000000,bootstrap_repetitions=10000,bootstrap_seed=20261003,
    source_frozen_sha256=hashlib.sha256((SRC/'FROZEN.json').read_bytes()).hexdigest(),
    local_verifier_sha256=hashlib.sha256(verify_path.read_bytes()).hexdigest(),new_measurement_validation=False)
(DST/'PROTOCOL.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
for p in list(DST.glob('*.py'))+list(DST.glob('case*/*.py'))+[verify_path]:ast.parse(p.read_text(encoding='utf-8'))
files={p.relative_to(DST).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DST.rglob('*'))
       if p.is_file() and '__pycache__' not in p.parts}
(DST/'FROZEN.json').write_text(json.dumps(dict(protocol=protocol['protocol'],files_sha256=files),indent=2),encoding='utf-8')
archive=ROOT/'tmp/curve3d_interval_20261003_code.tar.gz'
with tarfile.open(archive,'w:gz') as tf:
    for p in sorted(DST.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:tf.add(p,arcname=p.relative_to(DST).as_posix())
print(json.dumps(dict(status='frozen',archive=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                     checks=software),ensure_ascii=False))

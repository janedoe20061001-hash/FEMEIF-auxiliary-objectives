"""Mechanically reuse audited full-budget engines; freeze before any fitting."""
from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import sys
import tarfile
import numpy as np

root = Path(__file__).resolve().parents[1]
src = root / 'experiments/curve3d_directional_20261003'
dst = root / 'experiments/curve3d_shortlist_20261003'
if (dst / 'FROZEN.json').exists():
    raise RuntimeError('Already frozen; refusing edits')
frozen = json.loads((src / 'FROZEN.json').read_text(encoding='utf-8'))
for name, digest in frozen['files_sha256'].items():
    assert hashlib.sha256((src / name.replace('\\', '/')).read_bytes()).hexdigest() == digest, name
groups = [[], ['cumulative_length_logpolar'], ['symmetric_length_logpolar']]
shutil.copy2(src / 'global_metrics.py', dst / 'global_metrics.py')
shutil.copy2(src / 'curve_math.py', dst / 'curve_math.py')
for case in ('case1', 'case2'):
    source, target = src / case, dst / case
    target.mkdir(exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and path.suffix in ('.py', '.txt'):
            shutil.copy2(path, target / path.name)
    if case == 'case2':
        shutil.copytree(source / 'data', target / 'data')
        shutil.copy2(source / 'analysis_config.json', target / 'analysis_config.json')
    cfg = json.loads((source / 'config.json').read_text(encoding='utf-8'))
    cfg.update(seeds=dict(start=1000, count=100), groups=groups,
               protocol='curve3d_shortlist_20261003_v1')
    (target / 'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')

audit = (src / 'audit_case.py').read_text(encoding='utf-8').replace('range(200,300)', 'range(1000,1100)')
audit = audit.replace("cfg,freq=e.config(case/'config.json');root=case/'runs'",
    "cfg,freq=e.config(case/'config.json');root=case/'runs'\nCATALOG=[row for row in CATALOG if row[1] in {e.group_name(g) for g in e.groups(cfg)}]")
(dst / 'audit_case.py').write_text(audit, encoding='utf-8')
runner = (src / 'run_formal.py').read_text(encoding='utf-8')
runner = runner.replace('completed_runs=1400', 'completed_runs=600').replace('evaluations=73500000', 'evaluations=31500000')
runner = runner.replace("        for case in ['case1','case2']:",
    "        import hashlib\n        frozen=json.loads((root/'FROZEN.json').read_text())\n        for name,digest in frozen['files_sha256'].items():\n            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name\n        assert json.loads((root/'software_checks.json').read_text())['status']=='passed'\n        for case in ['case1','case2']:")
(dst / 'run_formal.py').write_text(runner, encoding='utf-8')
report = (src / 'finish_report.py').read_text(encoding='utf-8')
report = report.replace('from global_metrics import COMPARISONS,CATALOG',
    "from global_metrics import CATALOG\nCOMPARISONS=[('symmetric_length_logpolar','cumulative_length_logpolar')]")
for a, b in [('44-test', '12-test'), ("audit['runs']==700", "audit['runs']==300"),
             ('range(200,300)', 'range(1000,1100)'), ('len(p)==44', 'len(p)==12'),
             ('p_44_primary', 'p_12_primary'), ('第四轮：方向、端点对称与双辅助正式实验', '固定三组：累计弧长候选复核'),
             ('两个 Case 各 7 组，每组 100 个新配对种子（200–299）；1,400 次完整运行，73,500,000 次评估。',
              '两个 Case 各 3 组，每组 100 个新配对种子（1000–1099）；600 次完整运行，31,500,000 次评估。'),
             ('所有 24 项对基线和 20 项对父组的主目标比较统一 Holm 修正；',
              '所有 8 项对基线和 4 项对累计弧长的主目标比较统一 Holm 修正；'),
             ('三个变体不消除平移、噪声与不可达目标的问题；方向分解有部分主误差信息重合，双辅助也可能变差。',
              '本轮仅比较两个固定弧长候选；不消除平移、噪声与不可达目标的问题，不增加额外指标。')]:
    report = report.replace(a, b)
(dst / 'finish_report.py').write_text(report, encoding='utf-8')

verifier = (root / 'scripts/verify_curve_directional.py').read_text(encoding='utf-8')
for a, b in [('curve3d_directional_20261003', 'curve3d_shortlist_20261003'),
             ('==1400', '==600'), ('==73500000', '==31500000'), ('==245000', '==105000'),
             ('start=200,count=100', 'start=1000,count=100'), ("audit['runs']==700", "audit['runs']==300"),
             ('200<=seed<300', '1000<=seed<1100'), ('range(200,300)', 'range(1000,1100)'),
             ('len(rows)==12 and len(parent_rows)==10', 'len(rows)==4 and len(parent_rows)==2'),
             ('len(pvals)==44', 'len(pvals)==12'), ('np.zeros(44)', 'np.zeros(12)'),
             ('(44-rank)', '(12-rank)'), ('p_44_primary', 'p_12_primary'),
             ('new_seed_block=[200,299]', 'new_seed_block=[1000,1099]'),
             ('all_22_comparisons', 'all_6_comparisons')]:
    verifier = verifier.replace(a, b)
new_verifier = root / 'scripts/verify_curve_shortlist.py'
if new_verifier.exists():
    raise FileExistsError(new_verifier)
new_verifier.write_text(verifier, encoding='utf-8')

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

old = load('shortlist_old', src / 'global_metrics.py')
new = load('shortlist_new', dst / 'global_metrics.py')
c1 = load('shortlist_c1', dst / 'case1/legacy_objectives.py')
c2 = load('shortlist_c2', dst / 'case2/case2.py')
checks = []
gauss = np.array([.5-.5/np.sqrt(3), .5+.5/np.sqrt(3)])
max_gauss_difference = 0.
for case, model, lo, hi, nvar in [('case1', c1.impedance, c1.LOWER, c1.UPPER, 18),
                                 ('case2', c2.impedance, c2.LOWER, c2.UPPER, 15)]:
    cfg = json.loads((dst / case / 'config.json').read_text(encoding='utf-8'))
    if case == 'case1':
        f = np.geomspace(100, 1e6, 180)
        z = model(f, c1.TARGET)[0]
    else:
        f, z = c2.load_target({**cfg['target_data'], 'file': str(dst / case / cfg['target_data']['file'])})
    x = np.random.default_rng(1000).uniform(lo, hi, (cfg['population'], nvar))
    spectra = model(f, 10**x)
    for name in [g[0] for g in groups if g]:
        a = old.GlobalObjectives(f, aux=(name,), target_z=z)
        b = new.GlobalObjectives(f, aux=(name,), target_z=z)
        values = b.evaluate_spectra(spectra)
        np.testing.assert_array_equal(a.evaluate_spectra(spectra), values)
        np.testing.assert_array_equal(b.evaluate_spectra(z), np.zeros((1,3)))
        p = b.path(spectra)
        increments = p[:,1:,:] - p[:,:-1,:]
        lengths = np.concatenate([np.zeros((len(p),1)), np.cumsum(np.sqrt(np.sum(increments**2,axis=-1)),axis=-1)],axis=-1)
        e = (lengths-b.cl)/b.length
        def integral(err):
            nodes = (1-gauss)*err[:,:-1,None] + gauss*err[:,1:,None]
            return np.sum(nodes**2*b.du[:,None]/2,axis=(1,2))
        value = np.sqrt(integral(e) if name == new.LENGTH else .5*(integral(e)+integral(e[:,-1:]-e)))
        np.testing.assert_allclose(value, values[:,2], rtol=1e-12, atol=1e-12)
        max_gauss_difference=max(max_gauss_difference,float(np.max(abs(value-values[:,2]))))
    checks.append(dict(case=case, full_initial_population=cfg['population'], frequency_points=len(f),
                       old_controls_bitwise_equal=True, max_abs_difference=0.))
software = dict(status='passed', no_optimization_performed=True, zero_error=True,
                original_and_symmetric_losses_unchanged=True, independent_gaussian_integrals=True,
                independent_gauss_max_abs_difference=max_gauss_difference, control_equivalence=checks)
(dst / 'software_checks.json').write_text(json.dumps(software, indent=2), encoding='utf-8')
protocol = dict(protocol='curve3d_shortlist_20261003_v1', groups=groups, seeds=dict(start=1000,count=100),
    comparisons=[['cumulative_length_logpolar','baseline'],['symmetric_length_logpolar','baseline'],
                 ['symmetric_length_logpolar','cumulative_length_logpolar']], holm_primary_tests=12,
    planned_runs=600, planned_evaluations=31500000, bootstrap_repetitions=10000, bootstrap_seed=20261003,
    new_measurement_validation=False, noise_levels_not_invented=True,
    source_protocol='curve3d_directional_20261003_v1', source_frozen_sha256=hashlib.sha256((src/'FROZEN.json').read_bytes()).hexdigest(),
    local_verifier_sha256=hashlib.sha256(new_verifier.read_bytes()).hexdigest())
(dst / 'PROTOCOL.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
files = {p.relative_to(dst).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted(dst.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
(dst / 'FROZEN.json').write_text(json.dumps(dict(protocol=protocol['protocol'],files_sha256=files),indent=2),encoding='utf-8')
archive = root / 'tmp/curve3d_shortlist_20261003_code.tar.gz'
with tarfile.open(archive, 'w:gz') as tf:
    for p in sorted(dst.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:
            tf.add(p,arcname=p.relative_to(dst).as_posix())
print(json.dumps(dict(status='frozen',code_archive=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),software_checks=software),ensure_ascii=False))

"""Copy existing engines unchanged except objective definitions and source audit."""
from pathlib import Path
import hashlib
import json
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'experiments/global_objectives_20261003'
shared=['experiment.py','launch_batch.py','summarize.py','final_selection.py',
        'report_results.py','requirements.txt','requirements-lock.txt']
for case in ['case1','case2']:
    source=ROOT/f'experiments/{case}_dual_pdf18'
    target=DEST/case
    if (target/'runs').exists():
        raise RuntimeError('Never rebuild a running protocol')
    target.mkdir(parents=True,exist_ok=True)
    files=shared+(['case2.py','stability.py','analysis_config.json'] if case=='case2' else [])
    for name in files:
        shutil.copy2(source/name,target/name)
    shutil.copy2(source/'objectives.py',target/'legacy_objectives.py')
    shutil.copy2(DEST/'global_metrics.py',target/'global_metrics.py')
    if case=='case2':
        (target/'data').mkdir(exist_ok=True)
        shutil.copy2(source/'data/targetImpedance_case2.csv',target/'data/targetImpedance_case2.csv')
    cfg=json.loads((source/'config.json').read_text(encoding='utf-8'))
    cfg['features']={}
    cfg['groups']=[[],['bidirectional_cumulative_relative'],['max_interval_cumulative_relative'],['relative_complex_l2']]
    cfg['protocol']='global_objectives_20261003_v1'
    (target/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    text=(target/'experiment.py').read_text(encoding='utf-8')
    text=text.replace('"experiment.py","objectives.py",','"experiment.py","objectives.py","global_metrics.py","legacy_objectives.py",')
    text=text.replace('Feature constants must match the PDF definitions','Global auxiliary metrics have no tunable feature constants')
    (target/'experiment.py').write_text(text,encoding='utf-8')
    if case=='case1':
        wrapper='''from global_metrics import AUX, CATALOG, DEFAULTS, GlobalObjectives
from legacy_objectives import TARGET, LOWER, UPPER, impedance
import numpy as np
class Objectives(GlobalObjectives):
    def __init__(self,frequency,settings=None,aux=(),transform="loss",epsilon=None,target_z=None):
        target=impedance(frequency,TARGET)[0] if target_z is None else target_z
        super().__init__(frequency,settings,aux,transform,epsilon,target)
    def evaluate(self,x):
        return self.evaluate_spectra(impedance(self.f,10**np.atleast_2d(x)))
'''
    else:
        wrapper='from global_metrics import AUX, CATALOG, DEFAULTS, GlobalObjectives\nObjectives=GlobalObjectives\n'
    (target/'objectives.py').write_text(wrapper,encoding='utf-8')
    provenance={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in files+['objectives.py','config.json']}
    (target/'SOURCE.json').write_text(json.dumps({'source':str(source.relative_to(ROOT)),'sha256':provenance},indent=2),encoding='utf-8')

archive=ROOT/'tmp/global_objectives_20261003_code.tar.gz'
with tarfile.open(archive,'w:gz') as tf:
    for p in sorted(DEST.rglob('*')):
        if p.is_file() and not any(v in p.parts for v in ['__pycache__','runs','server_run','prepared']):
            tf.add(p,arcname=str(p.relative_to(DEST)))
print(archive)
print(hashlib.sha256(archive.read_bytes()).hexdigest())

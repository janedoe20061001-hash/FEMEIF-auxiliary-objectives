"""Verify sample counts, circuit-level denominators, frozen metric formulas and artifacts."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import csv
import hashlib
import importlib.util
import json
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/parameter_curve_population_20261006'
sys.path.append(str(ROOT/'tmp/curve_refinement_deps'))


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec)
    import sys; sys.modules[name]=mod; spec.loader.exec_module(mod); return mod


class Page(HTMLParser):
    def __init__(self):
        super().__init__(); self.links=[]; self.ids=set(); self.tables=[]; self.table=None; self.cells=0
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a: self.ids.add(a['id'])
        if tag=='a' and 'href' in a: self.links.append(a['href'])
        if tag=='img': self.links.append(a['src'])
        if tag=='table': self.table=[]
        if tag=='tr': self.cells=0
        if tag in ('td','th'): self.cells+=1
    def handle_endtag(self,tag):
        if tag=='tr' and self.table is not None: self.table.append(self.cells)
        if tag=='table': self.tables.append(self.table); self.table=None


info=json.loads((OUT/'INTEGRITY.json').read_text(encoding='utf-8'))
assert info['completed'] and info['all_checks_passed'] and info['baseline_circuits']==2304
protocol=json.loads((OUT/'data/PROTOCOL.json').read_text(encoding='utf-8'))
for file,sha in protocol['source_sha256'].items(): assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==sha,file
for file,sha in info['artifacts_sha256'].items(): assert hashlib.sha256((OUT/file).read_bytes()).hexdigest()==sha,file
refs=json.loads((OUT/'data/reference_circuits.json').read_text(encoding='utf-8'))
assert len({r['id'] for r in refs})==2304
strata=Counter((r['family'],r['split'],r['damping_bin'],r['spacing_bin']) for r in refs)
assert len(strata)==288 and set(strata.values())=={8}
assert len({(r['family'],tuple(r['parameters'])) for r in refs})==2304
counts=Counter(); undefined=0
with (OUT/'data/perturbations.csv').open(encoding='utf-8-sig',newline='') as f:
    for r in csv.DictReader(f):
        counts[r['circuit_id']]+=1
        undefined+=int(r['I03']=='')
        for k in ('magnitude_rmse_db','phase_rmse_deg','relative_complex_rmse'):
            assert np.isfinite(float(r[k])) and float(r[k])>=0,(r['circuit_id'],k)
assert sum(counts.values())==207360==info['perturbed_spectra']
for r in refs:
    has_l=any(n.startswith('L') for n in r['names']); has_c=any(n.startswith('C') for n in r['names'])
    assert counts[r['id']]==12*(len(r['names'])+2+int(has_l and has_c))
with (OUT/'data/rule_summary.csv').open(encoding='utf-8-sig',newline='') as f:
    summary=list(csv.DictReader(f))
assert all(int(r['circuits'])==72 and 0<=float(r['support_fraction'])<=1 for r in summary)
with (OUT/'data/parameter_rule_summary.csv').open(encoding='utf-8-sig',newline='') as f:
    params=list(csv.DictReader(f))
assert all(int(r['circuits'])==72 for r in params)
assert all(r['supported_circuits']==r['circuits'] for r in params if r['port_role']=='port_series' and r['parameter'].startswith('R'))
assert all(r['port_role']=='internal' for r in params if r['family']=='rlc_bridge')
supp=json.loads((OUT/'data/SUPPLEMENT_PROTOCOL.json').read_text(encoding='utf-8'))
for file,sha in {**supp['source_sha256'],**supp['input_sha256']}.items():
    assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==sha,file
with (OUT/'data/numerical_checks.csv').open(encoding='utf-8-sig',newline='') as f: checks=list(csv.DictReader(f))
assert len(checks)==info['numerical_checks'] and all(r['passed']=='True' for r in checks)
pop=module('population_verifier_source',Path(__file__).with_name('run_population.py'))
legacy=module('population_verifier_legacy',ROOT/'experiments/crosscase_consolidation_20261005/case1/legacy_objectives.py')
length=module('population_verifier_length',ROOT/'experiments/crosscase_consolidation_20261005/case1/length_metrics.py')
f=np.geomspace(*pop.BAND,513); metric_checks=0; i03_checks=0
for family,title,tree in pop.FAMILIES:
    record=next(r for r in refs if r['family']==family and r['split']=='B' and r['damping_bin']==1 and r['spacing_bin']==1)
    model=pop.load_record(record); meta,p,deltas=pop.plan(model); selected=[j for j,m in enumerate(meta) if m['level_pct']==10 and m['sign']==1]
    pp=p[selected]; dd=deltas[selected]; actual=pop.audit(model,pp,dd); target=model.response(f)[0]; z=model.response(f,pp)[0]
    obj=legacy.Objectives(f,aux=('characteristic_frequency_shift_error_decades','transition_speed_relative_error'),target_z=target)
    old=obj.evaluate_spectra(z)
    for j,k in enumerate(('sampled_magnitude_mse_db2','sampled_phase_mse_rad2','II01','II02')):
        np.testing.assert_allclose(actual[k],old[:,j],atol=1e-10,rtol=1e-9); metric_checks+=len(pp)
    length_obj=length.GlobalObjectives(f,aux=('cumulative_length_logpolar',),target_z=target)
    np.testing.assert_allclose(actual['cumulative_length'],length_obj.evaluate_spectra(z)[:,2],atol=1e-10,rtol=1e-9); metric_checks+=len(pp)
    if len(obj.cross_events) and len(obj.turn_events):
        cross_obj=legacy.Objectives(f,aux=('reactance_crossing_direction_consistency',),target_z=target)
        np.testing.assert_allclose(actual['I03'],cross_obj.evaluate_spectra(z)[:,2],atol=1e-12); i03_checks+=len(pp)
page=Page(); page.feed((OUT/'index.html').read_text(encoding='utf-8'))
for link in page.links:
    if link.startswith('#'): assert link[1:] in page.ids,link
    elif not link.startswith(('http://','https://')): assert (OUT/link).is_file(),link
assert all(len(set(t))==1 for t in page.tables)
validation=dict(passed=True,baseline_circuits=2304,perturbed_spectra=sum(counts.values()),strata=len(strata),
    circuits_per_stratum=8,independent_batches=2,undefined_I03_rows=undefined,metric_values_compared_with_frozen_code=metric_checks,
    I03_values_compared_with_frozen_code=i03_checks,html_tables=len(page.tables),html_links=len(page.links),
    source_and_output_hashes_verified=True,verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
print(json.dumps(validation,ensure_ascii=False))
(OUT/'VALIDATION.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
info['artifacts_sha256']['VALIDATION.json']=hashlib.sha256((OUT/'VALIDATION.json').read_bytes()).hexdigest()
(OUT/'INTEGRITY.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')

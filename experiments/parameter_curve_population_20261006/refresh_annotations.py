"""Repair bridge location annotation; reuse all numerical spectra and checks unchanged."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/parameter_curve_population_20261006'
source=Path(__file__).with_name('run_population.py')
spec=importlib.util.spec_from_file_location('population_annotation_refresh',source)
pop=importlib.util.module_from_spec(spec); sys.modules[spec.name]=pop; spec.loader.exec_module(pop)


def rows(file):
    with (OUT/'data'/file).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))


parameter_summary=rows('parameter_rule_summary.csv')
for r in parameter_summary:
    if r['family']=='rlc_bridge': r['port_role']='internal'
    for k in ('level_pct','circuits','supported_circuits','support_fraction'): r[k]=float(r[k]) if k in ('level_pct','support_fraction') else int(r[k])
pop.csvout(OUT/'data/parameter_rule_summary.csv',parameter_summary)
summary=rows('rule_summary.csv')
for r in summary:
    for k in ('level_pct','circuits','supported_circuits','support_fraction'): r[k]=float(r[k]) if k in ('level_pct','support_fraction') else int(r[k])
checks=rows('numerical_checks.csv')
for r in checks: r['passed']=r['passed']=='True'
refs=json.loads((OUT/'data/reference_circuits.json').read_text(encoding='utf-8'))
raw_examples=json.loads((OUT/'data/counterexamples.json').read_text(encoding='utf-8'))
examples=[dict(model=pop.load_record(e['reference']),row=e['original_row'],type=e['kind'],strength=e['strength']) for e in raw_examples]
info=json.loads((OUT/'INTEGRITY.json').read_text(encoding='utf-8'))
report=pop.make_report(summary,[],parameter_summary,examples,checks,refs,info['perturbed_spectra'],info['elapsed_seconds'])
(OUT/'REPORT.md').write_text(report,encoding='utf-8'); pop.make_html(OUT,report,examples)
protocol=json.loads((OUT/'data/PROTOCOL.json').read_text(encoding='utf-8'))
source_key=str(source.relative_to(ROOT)); snapshot=OUT/'data/calculation_source_before_role_fix.py'
old=protocol['source_sha256'][source_key]
assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==old
protocol['source_sha256'][str(snapshot.relative_to(ROOT))]=old
protocol['source_sha256'][source_key]=hashlib.sha256(source.read_bytes()).hexdigest()
protocol['post_calculation_annotation_correction']='Bridge edges are internal, not port-parallel; spectra, rules, support counts and numerical checks unchanged. Exact calculation source archived.'
pop.dump(OUT/'data/PROTOCOL.json',protocol)
info['artifacts_sha256']={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='INTEGRITY.json'}
pop.dump(OUT/'INTEGRITY.json',info)
print('Bridge position annotations and reports corrected; numerical data preserved.')

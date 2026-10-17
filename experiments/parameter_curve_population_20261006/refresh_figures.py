"""Regenerate summary figures from saved results, without repeating simulations."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/parameter_curve_population_20261006'
source=Path(__file__).with_name('run_population.py')
spec=importlib.util.spec_from_file_location('population_figure_refresh',source)
pop=importlib.util.module_from_spec(spec); sys.modules[spec.name]=pop; spec.loader.exec_module(pop)
with (OUT/'data/rule_summary.csv').open(encoding='utf-8-sig',newline='') as f:
    summary=list(csv.DictReader(f))
for r in summary:
    for k in ('level_pct','support_fraction'): r[k]=float(r[k])
refs=json.loads((OUT/'data/reference_circuits.json').read_text(encoding='utf-8'))
pop.style()
pop.summary_plots(summary,refs,OUT)
protocol=json.loads((OUT/'data/PROTOCOL.json').read_text(encoding='utf-8'))
protocol['source_sha256'][str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
protocol['post_calculation_presentation_correction']='Explicit colorbar axes avoid overlap; saved numerical results unchanged.'
pop.dump(OUT/'data/PROTOCOL.json',protocol)
info=json.loads((OUT/'INTEGRITY.json').read_text(encoding='utf-8'))
info['artifacts_sha256']={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='INTEGRITY.json'}
pop.dump(OUT/'INTEGRITY.json',info)
print('Summary figures regenerated; numerical data preserved.')

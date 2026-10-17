"""Artifact QA: links, tables, rows, recorded checks and input/output hashes."""
from pathlib import Path
from html.parser import HTMLParser
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/parameter_curve_mechanism_20261006'


class Page(HTMLParser):
    def __init__(self):
        super().__init__(); self.links=[]; self.ids=set(); self.tables=[]; self.table=None; self.cells=0

    def handle_starttag(self, tag, attrs):
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
protocol=json.loads((OUT/'data/PROTOCOL.json').read_text(encoding='utf-8'))
assert info['completed'] and info['all_checks_passed'] and info['optimizer_runs']==0
for file,digest in protocol['source_sha256'].items():
    assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==digest,file
for file,digest in info['artifacts_sha256'].items():
    assert hashlib.sha256((OUT/file).read_bytes()).hexdigest()==digest,file
with (OUT/'data/perturbations.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
assert len(rows)==info['perturbed_spectra']==1932
assert len({(r['circuit'],r['direction'],r['level_pct'],r['sign']) for r in rows})==len(rows)
with (OUT/'data/numerical_checks.csv').open(encoding='utf-8-sig',newline='') as f: checks=list(csv.DictReader(f))
assert len(checks)==info['numerical_checks']==486
assert all(r['passed']=='True' and float(r['value'])<float(r['limit']) for r in checks)
page=Page(); page.feed((OUT/'index.html').read_text(encoding='utf-8'))
for link in page.links:
    if link.startswith('#'): assert link[1:] in page.ids,link
    elif not link.startswith(('https://','http://')): assert (OUT/link).is_file(),link
for counts in page.tables: assert len(set(counts))==1,counts
assert len(page.tables)>=15
print(json.dumps(dict(passed=True,perturbed_spectra=len(rows),numerical_checks=len(checks),
    figures=info['figures'],html_tables=len(page.tables),html_links=len(page.links),
    source_and_output_hashes_verified=True),ensure_ascii=False))

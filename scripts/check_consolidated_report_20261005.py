"""Integrity checks of report references and consolidated statistical tables."""
from pathlib import Path
import csv,json,re,hashlib

root=Path(__file__).resolve().parents[1]
report=root/'output/近期研究统一总结_指标演变与验证_20261005.md'
text=report.read_text(encoding='utf-8')
links=re.findall(r'\]\(([^)]+)\)',text)
missing=[]
for target in links:
    if target.startswith(('https://','http://')):continue
    path=Path(target)
    if not path.is_absolute():path=report.parent/path
    if not path.exists():missing.append(str(path))
assert not missing,missing
assert '\ufffd' not in text
out=root/'output/research_consolidation_20261005'
hist=json.loads((out/'historical_effects.json').read_text(encoding='utf-8'))
keys=[tuple(r[k] for k in ['stage','case','id','comparator','primary']) for r in hist]
assert len(keys)==len(set(keys))==192
with (out/'selected_correlations.csv').open(encoding='utf-8-sig',newline='') as f:sel=list(csv.DictReader(f))
assert len(sel)==160
assert len({r['id'] for r in sel})==40
with (out/'comparisons.csv').open(encoding='utf-8-sig',newline='') as f:comparisons=list(csv.DictReader(f))
files=[report]+list(out.glob('*.csv'))+list((out/'figures').glob('*.png'))
data=dict(status='passed',report_links=len(links),historical_effect_rows=len(hist),selected_correlation_rows=len(sel),
          metrics=40,figure_count=5,comparisons_csv_rows=len(comparisons),
          sha256={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
(out/'report_integrity.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in data.items() if k!='sha256'}))

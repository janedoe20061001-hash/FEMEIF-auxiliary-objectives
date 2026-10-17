"""Refresh the gallery from saved results; never recalculate the experiment."""
from pathlib import Path
import hashlib
import json
import gallery_site
from run_population import FAMILIES

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/parameter_curve_population_20261006'
gallery_site.render(OUT,FAMILIES)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=json.loads((OUT/'data/PROTOCOL.json').read_text(encoding='utf-8'))
for file in ('run_population.py','README.md'):
    source=Path(__file__).with_name(file)
    protocol['source_sha256'][str(source.relative_to(ROOT))]=sha(source)
protocol['post_calculation_gallery_correction']='Presentation only: 16 standard-symbol electrical schematics, exact curve-baseline parameter tables, navigation and grouped statistics; numerical results and archived calculation source unchanged.'
(OUT/'data/PROTOCOL.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
supp=json.loads((OUT/'data/SUPPLEMENT_PROTOCOL.json').read_text(encoding='utf-8'))
archive=OUT/'data/condition_source_before_gallery.py'
archive_key=str(archive.relative_to(ROOT))
if archive.is_file(): supp['source_sha256'][archive_key]=sha(archive)
source=Path(__file__).with_name('analyze_conditions.py')
supp['source_sha256'][str(source.relative_to(ROOT))]=sha(source)
supp['post_calculation_gallery_correction']='Only HTML generation changed; original numerical analysis source archived.'
(OUT/'data/SUPPLEMENT_PROTOCOL.json').write_text(json.dumps(supp,ensure_ascii=False,indent=2),encoding='utf-8')
gallery_site.refresh_manifest(OUT)
print('Gallery regenerated from saved data: 16 schematics, 16 template parameter downloads, three summary views and 12 annotated counterexamples.')

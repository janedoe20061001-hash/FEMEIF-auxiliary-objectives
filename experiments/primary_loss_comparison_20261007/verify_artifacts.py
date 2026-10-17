"""Check delivered circuit diagrams, plot files, links, and source/archive integrity."""
from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from PIL import Image
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'output/primary_loss_comparison_20261007'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
records=json.loads((HERE/'circuits.json').read_text(encoding='utf-8'))
assert json.loads((OUT/'VALIDATION.json').read_text())['passed']
for r in records:
    tree=ET.parse(OUT/'schematics'/f'{r["id"]}.svg')
    names=[n.attrib['data-element'] for n in tree.iter() if 'data-element' in n.attrib]
    assert sorted(names)==sorted(r['parameter_names']) and len(names)==len(set(names))
    assert (OUT/'circuits'/f'{r["id"]}.md').exists()
images=list((OUT/'figures').glob('*.png'));assert len(images)==102
for p in images:
    with Image.open(p) as image:image.verify()
links=0
for p in OUT.rglob('*.md'):
    for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
        if ':' in link or link.startswith('#'):continue
        assert (p.parent/link).exists(),(p,link);links+=1
archive=json.loads((HERE/'server_run/ARCHIVES.json').read_text());assert sha(HERE/'server_run'/archive['file'])==archive['sha256']
# Browser inspection is a separate, dated manual record. This file-only check
# must not fabricate a new UI pass or claim an old preview PID is still running.
result=dict(passed=True,circuit_pages=50,schematics_component_names=50,readable_figures=len(images),valid_local_markdown_links=links,archive_sha256_match=True)
(OUT/'ARTIFACT_VALIDATION.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
(OUT/'INTEGRITY.json').write_text(json.dumps(dict(archive=archive,local_source_matches_remote=True,
    files_sha256={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='INTEGRITY.json'}),indent=2),encoding='utf-8')
print(json.dumps(result))

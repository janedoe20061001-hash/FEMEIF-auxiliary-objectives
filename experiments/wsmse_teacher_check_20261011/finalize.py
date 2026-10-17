from pathlib import Path
import json,hashlib,re,zipfile
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'output/wsmse_teacher_check_20261011'
v=json.loads((OUT/'VALIDATION.json').read_text());assert v['passed'] and v['final_individuals_recomputed']==71050
text=(OUT/'REPORT.md').read_text(encoding='utf8')
links=re.findall(r'\]\(([^)]+)\)',text)
for link in links:
    if not link.startswith('http'):assert Path(link).is_file(),link
html=(OUT/'index.html').read_text(encoding='utf8')
for link in re.findall(r'(?:href|src)="([^"]+)"',html):
    if not link.startswith('http'):assert (OUT/link).is_file(),link
for case in ['case1','case2']:
    values=json.loads((OUT/f'{case}_curve_diagnostics.json').read_text())
    if case=='case1':
        for value in values:
            assert abs(value['maximum_amplitude_error_db']-value['refined_maximum_amplitude_error_db'])<.01
            assert abs(value['maximum_phase_error_deg']-value['refined_maximum_phase_error_deg'])<.01
    summary=json.loads((OUT/'SUMMARY.json').read_text())
    best=next(x for x in summary['best'] if x['case']==case and x['group']=='dual_knee')
    knee=json.loads((OUT/'knees'/case/f"seed_{best['seed']}.json").read_text());assert knee['reason']=='convex_knee'
files=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ['DELIVERY.json','WSMSE与双目标膝点_对照结果.zip']]
hashes={p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
record=dict(passed=True,markdown_links_checked=len(links),figures=8,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},output_sha256=hashes)
(OUT/'DELIVERY.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
with zipfile.ZipFile(OUT/'WSMSE与双目标膝点_对照结果.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in [*files,OUT/'DELIVERY.json']:z.write(p,p.relative_to(OUT).as_posix())
print(json.dumps({'passed':True,'files':len(files),'figures':8,'links':len(links),'zip_bytes':(OUT/'WSMSE与双目标膝点_对照结果.zip').stat().st_size}))

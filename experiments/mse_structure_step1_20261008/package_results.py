"""Assemble exact frozen inputs and verify downloaded archives, without credentials."""
from pathlib import Path
import hashlib,json,shutil,tarfile,re
HERE=Path(__file__).resolve().parent;SR=HERE/'server_run';OUT=HERE.parents[1]/'output'/'mse_structure_step1_20261008'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    expected={'training_results.tar.gz':'4a2cefdbae5e26b86896a38f59e463a32e67962033f3b22a9d3f1bbe1425ffbe','analysis_results.tar.gz':'b3601b76fe561e59f2ded8cbff1b2c8e7a03053508a86948dfffa7cd3b78fad5','quadrature_results.tar.gz':'1cce987216ad9399c971bfb10ac68945334b293f6efcbe124ebb247c072d85be'}
    for f,h in expected.items():assert sha(SR/f)==h,f
    training=json.loads((SR/'retrieved/SOURCES.json').read_text());analysis=json.loads((SR/'retrieved_analysis/ANALYSIS_SOURCES.json').read_text());dest=SR/'frozen_inputs';dest.mkdir(exist_ok=True)
    for name,h in {**training,**analysis}.items():
        source=(SR/'retrieved'/name) if name in ['native.so','PREFLIGHT.json','BUILD.json'] else HERE/name
        assert sha(source)==h,name
        q=dest/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,q)
    shutil.copy2(SR/'retrieved/SOURCES.json',dest/'SOURCES.json');shutil.copy2(SR/'retrieved_analysis/ANALYSIS_SOURCES.json',dest/'ANALYSIS_SOURCES.json')
    for name in ['check_analysis_v2.py','check_quadrature.py','check_quadrature_v2.py','summarize.py','detail_stats.py','plot_results.py','package_results.py']:
        shutil.copy2(HERE/name,dest/name)
    sources={p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file() and p.name!='FINAL_SOURCES.json'};(dest/'FINAL_SOURCES.json').write_text(json.dumps(sources,indent=2),encoding='utf8')
    with tarfile.open(SR/'frozen_inputs.tar.gz','w:gz') as t:
        for p in sorted(dest.rglob('*')):
            if p.is_file():t.add(p,arcname=p.relative_to(dest).as_posix())
    manifests=dict(archives={f:dict(bytes=(SR/f).stat().st_size,sha256=sha(SR/f)) for f in [*expected,'frozen_inputs.tar.gz']},frozen_inputs=sources,report_artifacts={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['MANIFEST.json','summary_stdout.json']})
    (OUT/'MANIFEST.json').write_text(json.dumps(manifests,ensure_ascii=False,indent=2),encoding='utf8')
    for p in OUT.glob('*.md'):
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf8')):
            if link.startswith('D:/'):assert Path(link).exists(),(p.name,link)
    assert json.loads((SR/'retrieved_quadrature/quadrature/VALIDATION.json').read_text())['passed']
    assert json.loads((SR/'retrieved/runs/status.json').read_text())['completed']==200
    (OUT/'MANIFEST.json').write_text(json.dumps(manifests,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(archive_hashes_verified=len(expected),frozen_source_files=len(sources),report_artifacts=len(manifests['report_artifacts'])),ensure_ascii=False))
if __name__=='__main__':main()

from pathlib import Path
import tarfile,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2];archive=Path(sys.argv[1]);expected=sys.argv[2];kind=sys.argv[3];h=hashlib.sha256(archive.read_bytes()).hexdigest();assert h==expected
with tarfile.open(archive) as t:t.extractall(ROOT,filter='data')
manifest=json.loads((ROOT/f'{kind}_export_hashes.json').read_text());original={}
for name,v in manifest.items():
    p=ROOT/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==v,name
    original[name]=v
    if p.suffix=='.md':p.write_text(p.read_text(encoding='utf8').replace('/root/autodl-tmp/curve_analysis_workspace_20261008',ROOT.as_posix()),encoding='utf8')
out=ROOT/'output'/('dual_trajectory_20261008' if kind=='trajectory' else 'sampling_refinement_20261007')
(out/'DELIVERY.json').write_text(json.dumps(dict(archive_sha256=h,server_content_sha256=original,local_content_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in manifest},markdown_links_relocated=True),indent=2),encoding='utf8')
print(json.dumps(dict(passed=True,files=len(manifest),report=str(out/'REPORT.md'))))

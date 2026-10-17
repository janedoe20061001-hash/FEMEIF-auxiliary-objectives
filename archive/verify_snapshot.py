"""Verify the published reading snapshot; uses only the Python standard library."""
from pathlib import Path
import hashlib, json, re, sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
def main():
    manifest=json.loads((ROOT/'archive/SNAPSHOT_MANIFEST.json').read_text(encoding='utf-8'))
    failures=[]
    for item in manifest['files']:
        p=ROOT/item['path']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
            failures.append('hash: '+item['path'])
    links=0
    for p in ROOT.rglob('*.md'):
        if '.git' in p.parts: continue
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8-sig')):
            target=target.strip('<>')
            url=urlsplit(target)
            if url.scheme or target.startswith('#'):continue
            location=unquote(url.path)
            if location and not (p.parent/location).resolve().exists():
                failures.append('link: '+p.relative_to(ROOT).as_posix()+' -> '+target)
            links+=1
    result={'files_checked':len(manifest['files']),'internal_links_checked':links,'failures':failures}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return bool(failures)
if __name__=='__main__':sys.exit(main())

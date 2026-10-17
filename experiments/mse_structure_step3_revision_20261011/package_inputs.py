from core import *
import zipfile

def main():
    files=[ROOT/name for name in hashes()]
    files+=list(HERE.glob('*.py'))
    files+=[ROOT/'docs/三维曲线指标_新一轮候选与实验约定_20261011.md',OUT/'ISSUES.md']
    package=OUT/'server_inputs.zip'
    with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(files)):
            assert p.is_file(),p
            z.write(p,p.relative_to(ROOT).as_posix())
    dump(OUT/'PACKAGE.json',dict(path=str(package),bytes=package.stat().st_size,sha256=hashlib.sha256(package.read_bytes()).hexdigest(),files=len(set(files)),scientific_source_hashes=hashes()))
    print(json.dumps(dict(files=len(set(files)),bytes=package.stat().st_size,sha256=hashlib.sha256(package.read_bytes()).hexdigest())),flush=True)

if __name__=='__main__':main()

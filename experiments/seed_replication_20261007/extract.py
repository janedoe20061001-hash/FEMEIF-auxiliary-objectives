from pathlib import Path
import tarfile,json,hashlib
HERE=Path(__file__).resolve().parent
p=HERE/'server_run/results.tar.gz';h=hashlib.sha256(p.read_bytes()).hexdigest()
assert h=='d95951bee3600cd4043def97dde2893be94a07724b0c76f2464a3c220538ee05'
dest=HERE/'server_run/retrieved';dest.mkdir(exist_ok=False)
with tarfile.open(p) as t:t.extractall(dest,filter='data')
prep=json.loads((HERE/'PREPARATION.json').read_text(encoding='utf-8'))
for name,digest in prep['sha256'].items():
    assert hashlib.sha256((dest/name).read_bytes()).hexdigest()==digest
(HERE/'server_run/ARCHIVES.json').write_text(json.dumps({'file':'results.tar.gz','sha256':h,'bytes':p.stat().st_size,'remote_directory':'/root/autodl-tmp/seed_replication_20261007_v2','source_hashes_verified':True},indent=2),encoding='utf-8')
print(json.dumps({'archive_sha256_verified':True,'source_hashes_verified':True,'retrieved':str(dest)}))

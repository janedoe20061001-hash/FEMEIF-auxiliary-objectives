from pathlib import Path
import csv
import hashlib
import json
import tarfile
import numpy as np

root=Path(__file__).resolve().parents[1]
run=root/'experiments/global_objectives_20261003'
archive=run/'server_run/formal_results.tar.gz'
expected=(run/'server_run/formal_results.sha256').read_text().split()[0]
actual=hashlib.sha256(archive.read_bytes()).hexdigest()
assert actual==expected
out=run/'server_run/raw_results'
if out.exists():raise FileExistsError(out)
reproduction={}
with tarfile.open(archive,'r:gz') as tf:
    files=tf.getmembers()
    for member in files:
        path=Path(member.name)
        assert not path.is_absolute() and '..' not in path.parts and member.isfile()
    summaries=[m for m in files if '/runs/' in m.name and m.name.endswith('/summary.json')]
    assert len(summaries)==800
    for case in ['case1','case2']:
        with (root/f'experiments/{case}_dual_pdf18/server_run/selected_individuals.csv').open(encoding='utf-8-sig',newline='') as fh:
            old={int(r['seed']):r for r in csv.DictReader(fh) if r['group']=='baseline'}
        errors=[]
        for seed in range(100):
            r=json.load(tf.extractfile(f'{case}/runs/baseline/seed_{seed:03d}/summary.json'))
            for key in ['wsmse_selected_magnitude_mse','wsmse_selected_phase_mse','wsmse_selected_value']:
                expected_value=float(old[seed][key]);value=r[key]
                np.testing.assert_allclose(value,expected_value,rtol=1e-12,atol=1e-13)
                errors.append(abs(value-expected_value))
        reproduction[case]=dict(seeds=100,max_abs_difference=max(errors),matches_historical_baseline=True)
    tf.extractall(out,filter='data')
record=dict(status='passed',sha256=actual,archive_bytes=archive.stat().st_size,
            completed_runs=800,extracted_directory=str(out),baseline_reproduction=reproduction)
(run/'server_run/download_audit.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record))

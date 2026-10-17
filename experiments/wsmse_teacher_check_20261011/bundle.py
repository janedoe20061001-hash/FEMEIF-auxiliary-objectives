from pathlib import Path
import tarfile, json, hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
files=[HERE/'check.py',HERE/'PROTOCOL.md',ROOT/'targetImpedance_case2.csv']
base=ROOT/'experiments/dual_trajectory_20261008'
files += [base/n for n in ['SOURCES.json','run.py','CONFIG.json','PROTOCOL.md']]
files += [p for p in (base/'frozen_dual').rglob('*') if p.is_file() and '__pycache__' not in str(p)]
for case in ['case1','case2']:
    for seed in range(2300,2400):
        scalar=ROOT/f'experiments/primary_loss_comparison_20261007/server_run/retrieved/runs/{case}/wsmse/seed_{seed}'
        dual=base/f'server_run/retrieved/runs/{case}/seed_{seed}'
        files += [scalar/n for n in ['summary.json','initial_log_parameters.npy','final_population.npz']]
        files += [dual/n for n in ['summary.json','initial_log_parameters.npy','generation_300.npz']]
manifest={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(HERE/'INPUT_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
files.append(HERE/'INPUT_MANIFEST.json')
with tarfile.open(HERE/'server_bundle.tar.gz','w:gz') as t:
    for p in files:t.add(p,arcname=p.relative_to(ROOT).as_posix())
p=HERE/'server_bundle.tar.gz'
print(json.dumps({'files':len(files),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))

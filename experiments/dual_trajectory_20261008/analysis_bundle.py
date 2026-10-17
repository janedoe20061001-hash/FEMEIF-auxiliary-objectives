from pathlib import Path
import json,tarfile
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
config=json.loads((ROOT/'experiments/sampling_refinement_20261007/CONFIG.json').read_text());ids=[r['circuit'] for r in config['records']]
files=[]
for rel in ['experiments/sampling_refinement_20261007','experiments/optimized_curve_structure_20261007','experiments/seed_replication_20261007','experiments/dual_trajectory_20261008']:
    d=ROOT/rel
    for p in d.rglob('*'):
        if not p.is_file():continue
        relative=p.relative_to(d)
        if any(x in relative.parts for x in ['server_run','__pycache__','runs']) or p.suffix=='.gz':continue
        if rel.endswith('seed_replication_20261007') and relative.parts[0] not in ['analyze.py','frozen_scalar']:continue
        files.append(p)
files.extend(ROOT/'output/optimized_curve_structure_20261007/selected'/f'{c}.npz' for c in ids)
files.extend(ROOT/'output/primary_loss_comparison_20261007/schematics'/f'{c}.svg' for c in ids+['case2'])
with tarfile.open(HERE/'analysis_bundle.tar.gz','w:gz') as tar:
    for p in files:tar.add(p,arcname=p.relative_to(ROOT).as_posix())
print(json.dumps(dict(bytes=(HERE/'analysis_bundle.tar.gz').stat().st_size,files=len(files))))

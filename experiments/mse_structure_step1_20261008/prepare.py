from pathlib import Path
import sys,json,shutil,hashlib,tarfile
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def main():
    r=next(x for x in json.loads((ROOT/'experiments/optimized_curve_structure_20261007/circuits.json').read_text(encoding='utf8')) if x['id']=='case1')
    frozen=HERE/'frozen';(frozen/'frozen/case1').mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/'experiments/optimized_curve_structure_20261007/model.py',frozen/'model.py')
    shutil.copy2(ROOT/'experiments/optimized_curve_structure_20261007/frozen/case1/legacy_objectives.py',frozen/'frozen/case1/legacy_objectives.py')
    (frozen/'circuits.json').write_text(json.dumps([r],ensure_ascii=False,indent=2),encoding='utf8')
    shutil.copy2(ROOT/'experiments/sampling_refinement_20261007/grids/case1.npz',HERE/'target_grid.npz')
    old=HERE/'old300';old.mkdir(exist_ok=True)
    for seed in range(2300,2400):shutil.copy2(ROOT/f'experiments/dual_trajectory_20261008/server_run/retrieved/runs/case1/seed_{seed}/generation_300.npz',old/f'{seed}.npz')
    cfg=dict(case='case1',seeds=list(range(2300,2400)),arms=['original','adaptive'],generations=1000,population=200,snapshots=list(range(20,1001,20)),full_populations=[300,1000],planned_runs=200,planned_candidate_evaluations=40000000,bootstrap_rng=2026100802,plot_seeds=[2307,2312,2321],workers=24)
    (HERE/'CONFIG.json').write_text(json.dumps(cfg,indent=2),encoding='utf8')
    sources=[p for p in HERE.rglob('*') if p.is_file() and p.name not in ['SOURCES.json','ANALYSIS_SOURCES.json'] and p.suffix in ['.py','.c','.json','.npz','.md'] and '__pycache__' not in str(p) and 'server_run' not in str(p)]
    (HERE/'SOURCES.json').write_text(json.dumps({p.relative_to(HERE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2),encoding='utf8')
    with tarfile.open(HERE/'bundle.tar.gz','w:gz') as tar:
        for p in [*sources,HERE/'SOURCES.json']:tar.add(p,arcname=p.relative_to(HERE).as_posix())
    print(json.dumps({'runs':cfg['planned_runs'],'evaluations':cfg['planned_candidate_evaluations'],'bundle_bytes':(HERE/'bundle.tar.gz').stat().st_size}))
if __name__=='__main__':main()

"""Freeze all task parameters and producer/input hashes before new scoring."""
from metric import *
import zipfile,shutil

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    tasks=json.loads((ROOT/'output/geometry_followup_20261016/SCORING_TASKS.json').read_text())['tasks']
    assert len(tasks)==756
    cfg=dict(primary='JV',ablations=['JM','JC'],angles=ANGLES.tolist(),weights=[.5,.5],atol=1e-4,rtol=1e-3,polygon_levels=[64,128,256,512,1024],workers=32,
        regression_not_holdout=True,development_min=9,development_extra_min=5,regression_min=18,regression_family_min=6,regression_extra_min=7,max_cost_ratio=1.25,optimization_started=False)
    dump(HERE/'config.json',cfg)
    shutil.copyfile(ROOT/'docs/复轨迹方向变化分布_实验方案_20261017.md',HERE/'PROTOCOL.md')
    frozen=dict(tasks=tasks,source_tasks_sha256=sha(ROOT/'output/geometry_followup_20261016/SCORING_TASKS.json'),old_bank_sha256=sha(ROOT/'output/geometry_followup_20261016/HOLDOUT_BANK.json'),scope='All 36 existing pairs are development/regression; no new holdout')
    dump(OUT/'TASKS.json',frozen)
    files=[]
    for folder in ['experiments/arx_light_20261015','experiments/arx_structure_diagnosis_20261016','experiments/geometry_followup_20261016']:
        files+=list((ROOT/folder).glob('*.py'))+list((ROOT/folder).glob('*.json'))
    old=ROOT/'experiments/arx_light_20261015'
    files+=list(old.glob('*.c'))+[old/'PROTOCOL.md']
    files += [p for sub in ['frozen','inputs'] for p in (old/sub).rglob('*') if p.is_file() and p.suffix not in ('.pyc','.so') and '__pycache__' not in str(p)]
    files+=list(HERE.glob('*.py'))+[HERE/'config.json',HERE/'PROTOCOL.md',OUT/'TASKS.json']
    # Labels and old scores are immutable comparators, not used in score evaluation.
    files += [ROOT/'output/geometry_followup_20261016/HOLDOUT_BANK.json',ROOT/'output/arx_structure_diagnosis_20261016/SCORING_BANK.json',ROOT/'output/geometry_followup_20261016/PAIR_DIRECTIONS.csv']
    files=sorted(set(files));manifest={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    dump(OUT/'SOURCE_MANIFEST.json',dict(files=manifest,analytic_checks=analytic_checks(),frozen_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()))
    with zipfile.ZipFile(OUT/'source.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(ROOT).as_posix())
        z.write(OUT/'SOURCE_MANIFEST.json',(OUT/'SOURCE_MANIFEST.json').relative_to(ROOT).as_posix())
    print(json.dumps(dict(tasks=len(tasks),files=len(files),source_zip_bytes=(OUT/'source.zip').stat().st_size,sha256=sha(OUT/'source.zip'))))
if __name__=='__main__':main()

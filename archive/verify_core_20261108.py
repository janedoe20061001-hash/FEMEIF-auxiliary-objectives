"""Read-only publication checks and fixed auxiliary replays, not new optimization."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
COHORTS = [
    ('有效指标_L1_COARSE_20261028', 'L1_COARSE', 96, (23, 140, 20)),
    ('L1_COARSE_扩大参数验证_20261028', 'L1_COARSE', 96, (21, 137, 21)),
    ('有效指标_JOINT_POWER_20261028', 'JOINT_POWER', 120, (19, 133, 18)),
]

def require(condition, message):
    if not condition:
        raise ValueError(message)

def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def main():
    import numpy
    import scipy
    sources = load(ROOT/'archive/CORE_SOURCES_20261108.json')
    for row in sources['files']:
        p = ROOT/row['path']
        require(p.is_file(), 'Missing published source: '+row['path'])
        require(hashlib.sha256(p.read_bytes()).hexdigest() == row['published_sha256'], 'Hash mismatch: '+row['path'])
        if p.suffix != '.md':
            require(row['source_sha256'] == row['published_sha256'], 'Scientific source changed: '+row['path'])
    checks = []
    for folder, method, count, expected in COHORTS:
        directory = ROOT/'output'/folder
        with (directory/'evidence/PER_SEED.csv').open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        require(len(rows) == count, folder+': unexpected trial count')
        require(len({(r['target'], r['method'], r['seed']) for r in rows}) == count, folder+': duplicate trial')
        require(set(Counter((r['target'], r['method']) for r in rows).values()) == {8}, folder+': unbalanced cohort')
        require(all(r['accepted'] == 'True' and r['strict_stable'] == 'True' for r in rows), folder+': unexpected representative audit')
        main_rows = [r for r in rows if r['method'] == method]
        require(len(main_rows) == 24 and len({r['target'] for r in main_rows}) == 3, folder+': target layout')
        totals = (sum(r['p3_complete'] == 'True' for r in main_rows), sum(int(r['strict_peak_count']) for r in main_rows), sum(r['all_six'] == 'True' for r in main_rows))
        require(totals == expected, folder+': summary mismatch')
        require(sum(r['p3_complete'] == 'True' for r in rows if r['method'] == 'MAG_PHA') == 0, folder+': baseline mismatch')
        require(all(float(r['extra_amplitude']) == 0 and float(r['extra_phase']) == 0 for r in main_rows), folder+': unexpected extra event')
        if method == 'JOINT_POWER':
            require(sum(r['p3_complete'] == 'True' for r in rows if r['method'] == 'L1_COARSE') == 16, 'Same-cohort L1 comparison mismatch')
        # The original replay script writes LOCAL_REPLAY.json. Keep that mutation
        # confined to a disposable copy; never overwrite historical evidence.
        with tempfile.TemporaryDirectory(prefix='femeif-core-') as tmp:
            copy = Path(tmp)
            for p in directory.iterdir():
                if p.is_file() and p.suffix in {'.py', '.json'}:
                    shutil.copyfile(p, copy/p.name)
            process = subprocess.run([sys.executable, '-B', str(copy/'verify_replay.py')], cwd=copy, text=True, capture_output=True)
            require(process.returncode == 0, folder+': replay failed\n'+process.stdout+process.stderr)
            replay = load(copy/'LOCAL_REPLAY.json')
            require(replay['passed'] and replay['representatives'] == 24 and replay['permutation_checks'] == 24, folder+': incomplete replay')
        checks.append({'folder': folder, 'trials_read': count, 'method': method, 'p3': totals[0], 'strict_peaks': totals[1], 'all_six': totals[2], 'auxiliary_replay': replay})
    # ASCII JSON survives Windows shell pipe encoding without corrupting paths.
    print(json.dumps({'passed': True, 'publication_date': '2026-11-08', 'python': platform.python_version(), 'numpy': numpy.__version__, 'scipy': scipy.__version__, 'source_files_checked': len(sources['files']), 'trials_read': sum(x['trials_read'] for x in checks), 'fixed_representatives_replayed': 72, 'branch_permutations_checked': 72, 'cohorts': checks, 'scope': 'Recomputed CSV summaries and fixed auxiliary values only; no new optimization, mode extraction, strict curve integration, or timing benchmark.'}, ensure_ascii=True, indent=2))

if __name__ == '__main__':
    main()

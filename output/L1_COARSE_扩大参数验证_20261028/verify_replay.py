from pathlib import Path
import json
import numpy as np
from observable_axis import ObservableAxis
HERE=Path(__file__).resolve().parent
targets=json.loads((HERE/'observed_targets.json').read_text(encoding='utf-8'))
rows=json.loads((HERE/'confirmed_representatives.json').read_text(encoding='utf-8'))
metrics={t['target_id']:ObservableAxis([complex(*p) for p in t['poles_hz']],[complex(*a) for a in t['residues_ohm_hz']],t['lower_log10_rlc'],t['upper_log10_rlc'],method='L1_COARSE') for t in targets}
errors=[]
for r in rows:
    value=metrics[r['target_id']](r['log10_rlc']);expected=np.asarray(r['expected_auxiliary']);error=abs(value-expected)
    assert np.all(error<=1e-6+1e-3*abs(expected))
    permutation=metrics[r['target_id']](r['log10_rlc'][::-1]);assert np.all(abs(value-permutation)<=1e-6+1e-3*abs(value));errors.append(float(np.max(error)))
result=dict(passed=True,representatives=len(rows),permutation_checks=len(rows),maximum_absolute_score_error=max(errors),scope='Auxiliary-score replay; strict curve verdicts are frozen independent records, not recomputed by this script')
(HERE/'LOCAL_REPLAY.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

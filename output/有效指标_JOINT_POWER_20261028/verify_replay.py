from pathlib import Path
import json,hashlib
import numpy as np
from observable_power import ObservablePower
HERE=Path(__file__).resolve().parent
targets=json.loads((HERE/'observed_targets.json').read_text(encoding='utf-8'))
rows=json.loads((HERE/'confirmed_representatives.json').read_text(encoding='utf-8'))
engines={t['target_id']:ObservablePower([complex(*p) for p in t['poles_hz']],[complex(*a) for a in t['residues_ohm_hz']],t['lower_log10_rlc'],t['upper_log10_rlc'],'JOINT_POWER') for t in targets}
errors=[]
for r in rows:
    value=engines[r['target_id']](r['log10_rlc']);expected=np.asarray(r['expected_auxiliary']);error=abs(value-expected)
    assert np.all(error<=1e-9+1e-10*abs(expected))
    perm=engines[r['target_id']](r['log10_rlc'][::-1]);assert np.all(abs(value-perm)<=1e-9+1e-10*abs(value));errors.append(float(np.max(error)))
result=dict(passed=True,representatives=len(rows),permutation_checks=len(rows),maximum_absolute_error=max(errors),public_api_sha256=hashlib.sha256((HERE/'observable_power.py').read_bytes()).hexdigest(),scope='Auxiliary replay of fixed representatives; curve verdicts remain independent frozen evidence')
(HERE/'LOCAL_REPLAY.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))

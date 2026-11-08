from pathlib import Path
import json
import numpy as np
from observable_axis import ObservableAxis
HERE=Path(__file__).resolve().parent
target=json.loads((HERE/'observed_target.json').read_text(encoding='utf-8'))
example=json.loads((HERE/'example_candidate.json').read_text(encoding='utf-8'))
metric=ObservableAxis([complex(*v) for v in target['poles_hz']],[complex(*v) for v in target['residues_ohm_hz']],target['lower_log10_rlc'],target['upper_log10_rlc'],method='L1_COARSE')
value=metric(example['log10_rlc']);expected=np.asarray(example['expected_auxiliary'])
passed=bool(np.all(abs(value-expected)<=1e-6+1e-3*abs(expected)))
assert passed,'Standalone score disagrees with frozen representative'
print(json.dumps(dict(target=example['target_id'],seed=example['seed'],auxiliary=value.tolist(),expected=expected.tolist(),passed=passed),indent=2))

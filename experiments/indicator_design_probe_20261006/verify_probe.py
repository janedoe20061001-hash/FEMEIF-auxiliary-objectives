"""Independent score algebra and matched-pair audit of all 400 saved pools."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/indicator_design_probe_20261006'
sys.path.append(str(ROOT/'tmp/curve_refinement_deps'))
spec=importlib.util.spec_from_file_location('phase_probe_check',Path(__file__).with_name('probe.py'))
probe=importlib.util.module_from_spec(spec);sys.modules[spec.name]=probe;spec.loader.exec_module(probe)
summary=json.loads((OUT/'SUMMARY.json').read_text(encoding='utf-8'))
manifest=json.loads((OUT/'INTEGRITY.json').read_text(encoding='utf-8'))
provenance=json.loads((OUT/'data/PROVENANCE.json').read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for file,h in manifest['artifacts_sha256'].items(): assert sha(OUT/file)==h,file
for file,h in {**provenance['source_sha256'],**provenance['input_sha256']}.items(): assert sha(ROOT/file)==h,file
assert provenance['protocol_sha256']==sha(Path(__file__).with_name('PROTOCOL.md'))
with (OUT/'data/seed_ranking.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
row_lookup={(r['case'],r['context'],int(r['seed'])):r for r in rows}
checks=0;pairs_checked=0;total=0
for case in ('case1','case2'):
    e=probe.Evaluator(case)
    assert e.saved['groups'].tolist()==probe.GROUPS
    np.testing.assert_array_equal(e.saved['seeds'],np.arange(2000,2100))
    for context in ('initial','final'):
        for seed in range(2000,2100):
            with np.load(OUT/'pools'/f'{case}_{context}_{seed}.npz') as saved:
                scores=saved['scores'];pairs=saved['pairs'];pars=saved['log_parameters'];groups=saved['group_index']
                assert saved['score_names'].tolist()==probe.COLS and np.isfinite(scores).all() and (scores>=0).all()
                assert len(np.unique(pairs))==pairs.size
                total+=len(scores); pairs_checked+=len(pairs)
                if len(pairs):
                    good,bad=pairs.T
                    ratio=np.maximum(scores[good,:2],scores[bad,:2])/np.minimum(scores[good,:2],scores[bad,:2])
                    assert np.max(ratio)<=1.10+1e-12
                    assert np.all(scores[bad,6]>=1.20*scores[good,6]-1e-14)
                record=row_lookup[(case,context,seed)]
                assert int(record['pairs'])==len(pairs) and int(record['individuals'])==len(scores)
                for name,col in [('II02',3),('D_phi',4),('D_LP',5),('phase_MSE',1)]:
                    if len(pairs):
                        a=scores[pairs[:,0],col];b=scores[pairs[:,1],col]
                        tie=abs(a-b)<=1e-12*np.maximum(1,np.maximum(abs(a),abs(b)))
                        actual=np.where(tie,.5,a<b).mean(); assert abs(actual-float(record[name+'_correct_fraction']))<1e-14
                # Independent phase-secant algebra on fixed start/middle/end individuals in every pool.
                ix=sorted(set([0,len(pars)//2,len(pars)-1]));z=e.spectrum(pars[ix]);phase=np.angle(z)
                residual=phase-e.phase
                numerator=np.sum(np.diff(residual,axis=1)**2/e.h,axis=1)
                denominator=np.sum(np.diff(e.phase)**2/e.h)
                dphi=np.sqrt(numerator/denominator)
                np.testing.assert_allclose(dphi,scores[ix,4],rtol=1e-10,atol=1e-10)
                worst=np.sqrt(np.max(np.stack([np.mean(residual[:,e.bins==j]**2,axis=1) for j in range(8)],axis=1),axis=1))
                np.testing.assert_allclose(worst,scores[ix,6],atol=1e-12,rtol=1e-10)
                np.testing.assert_allclose(e.score(z),scores[ix],atol=1e-9,rtol=1e-9)
                checks+=len(ix)
assert total==summary['evaluated_individuals'] and pairs_checked==summary['matched_pairs']
assert len(list((OUT/'pools').glob('*.npz')))==400
for item in summary['ranking']:
    rr=[r for r in rows if r['case']==item['case'] and r['context']==item['context'] and r['eligible']=='True']
    assert len(rr)==item['eligible_seeds']
    for name in ('II02','D_phi','D_LP','phase_MSE'):
        assert abs(np.mean([float(r[name+'_correct_fraction']) for r in rr])-item[name+'_seed_mean'])<1e-14
supplement_groups=0
if (OUT/'data/SUPPLEMENT_PROTOCOL.json').is_file():
    protocol=json.loads((OUT/'data/SUPPLEMENT_PROTOCOL.json').read_text(encoding='utf-8'))
    for file,h in protocol['sources'].items(): assert sha(ROOT/file)==h,file
    for file,h in protocol['inputs'].items(): assert sha(OUT/file)==h,file
    with (OUT/'data/sensitivity_seed_rows.csv').open(encoding='utf-8-sig',newline='') as f: supplemental_rows=list(csv.DictReader(f))
    sensitivity=json.loads((OUT/'SENSITIVITY.json').read_text(encoding='utf-8'))
    assert sensitivity['main_gate_unchanged'] and summary['gate']['passed'] is False
    for item in sensitivity['ranking_sensitivity']:
        rr=[r for r in supplemental_rows if r['case']==item['case'] and r['context']==item['context'] and r['condition']==item['condition'] and r['eligible']=='True']
        assert len(rr)==item['seeds'] and sum(int(r['pairs']) for r in rr)==item['pairs']
        for name in ('II02','D_phi','phase_MSE'):
            assert abs(np.mean([float(r[name]) for r in rr])-item[name])<1e-14
        supplement_groups+=1
result=dict(passed=True,pools=400,individuals=total,all_nonoverlapping_matched_pairs_checked=pairs_checked,
            independently_recomputed_individuals=checks,signed_phase_score_and_local_quality_verified=True,
            seed_ranking_recomputed=True,supplemental_ranking_groups_recomputed=supplement_groups,hashes_verified=True,verifier_sha256=sha(Path(__file__)))
(OUT/'VALIDATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
manifest['artifacts_sha256']={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='INTEGRITY.json'}
(OUT/'INTEGRITY.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))

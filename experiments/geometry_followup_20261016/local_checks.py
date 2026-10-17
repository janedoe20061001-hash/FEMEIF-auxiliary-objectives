"""Independent local replay, source alignment and physical model checks."""
from common import *
from core import checked_nodal
from concurrent.futures import ProcessPoolExecutor,as_completed
MODELS={}

def replay(file):
    r=json.loads(Path(file).read_text());tid=r['target']
    if tid not in MODELS:
        MODELS[tid]=Geometry(target(tid));MODELS[tid].fast=None
    m=MODELS[tid];label=next(iter(r['windows']));p=np.array(r['parameters']);want=next(v['scores'] for v in r['windows'][label]['levels'] if v['order']==32 and v['refinement']==1)
    got=m.block(p,label,32,1);checked={k:abs(got[k]-want[k])<=1e-4+1e-3*abs(want[k]) for k in got}
    rel=max(abs(got[k]-want[k])/max(abs(want[k]),1e-30) for k in got)
    return dict(id=r['id'],passed=all(checked.values()),max_relative_difference=rel,Root=got['Root'],B=got['B'])

def main(workers):
    old=json.loads((ROOT/'output/arx_structure_diagnosis_20261016/SCORING_BANK.json').read_text());b=json.loads((OUT/'HOLDOUT_BANK.json').read_text());tasks=json.loads((OUT/'SCORING_TASKS.json').read_text());scores=list((OUT/'scores').glob('*.json'))
    assert b['complete'] and len(b['selected'])==24 and len(scores)==756
    assert sha(OUT/'HOLDOUT_BANK.json')==tasks['holdout_sha256'];assert sha(HERE/'common.py')==tasks['common_sha256'];assert sha(HERE/'config.json')==tasks['config_sha256']
    assert b['selection_code_sha256']==sha(HERE/'bank.py')
    assert not json.loads((OUT/'SELECTION_AMENDMENT_STATUS.json').read_text())['applied_to_primary_bank']
    excluded={np.asarray(v[s]['log_parameters'],dtype='<f8').tobytes() for v in old['pairs'] for s in ['better','worse']};used=set();sources=[]
    assert len({r['seed'] for r in b['selected']})==12
    for tid in target_ids():
        rr=[r for r in b['selected'] if r['target']==tid];assert len(rr)==8 and len({r['seed'] for r in rr})==8
    for family in ['missing_structure','existing_distortion','extra_structure']:assert sum(r['family']==family for r in b['selected'])==8
    for r in b['selected']:
        data=target(r['target']);f=DATA/r['source_stage'];assert sha(f)==r['source_stage_sha256']
        with np.load(f) as a:X=a['X'];F=a['F'][:,:2]
        for side in ['better','worse']:
            lp=np.array(r[side]['log_parameters']);i=r[side+'_index'];np.testing.assert_array_equal(lp,X[i]);np.testing.assert_array_equal(F[i],r[side+'_primary'])
            key=np.asarray(lp,dtype='<f8').tobytes();assert key not in used and key not in excluded;used.add(key)
        va,vb=np.array(r['better_primary']),np.array(r['worse_primary']);assert np.all(abs(va-vb)/np.maximum(np.maximum(va,vb),CFG['audit']['base_atol'])<=.05)
        if r['family']!='extra_structure':
            a=next(v for v in r['better']['events'] if v['event_id']==r['region']);bad=next(v for v in r['worse']['events'] if v['event_id']==r['region']);e=next(v for v in data['record']['events'] if v['event_id']==r['region'])
            assert a['complete'] and abs(a['shift_decades'])<=e['width_3db_decades']
            if r['family']=='missing_structure':assert not bad['matched']
            else:
                assert bad['matched'] and not bad['complete'];q=max(abs(bad['shift_decades'])/.02,abs(bad['height_delta_db'])/3,abs(np.log(bad['width_ratio']))/np.log(2));assert q>=2
        else:
            assert sum(v['complete'] for v in r['better']['events'])==sum(v['complete'] for v in r['worse']['events'])
            block=next(v for v in data['record']['blocks'] if v['label']==r['region']);assert not block['event']
            assert not any(block['lo']<=v['x']<=block['hi'] for v in r['better']['fit'])
            assert any(v['event_id'] in r['worse']['extra_ids'] and block['lo']<=v['x']<=block['hi'] for v in r['worse']['fit'])
        sources.append(dict(pair_id=r['pair_id'],source_verified=True,labels_verified=True))
    values=[json.loads(p.read_text()) for p in scores];assert all(r['numerical_passed'] and r['implementation_sha256']==sha(HERE/'common.py') for r in values)
    assert sum(len(r['windows']) for r in values)==2124
    c=ControlCircuit();formula_checks=[]
    for r in values:
        if r['target']!='controlled':continue
        p=np.array(r['parameters']);fc=[1/(2*np.pi*np.sqrt(l*cc)) for _,l,cc in p.reshape(-1,3)];freq=np.unique(np.r_[np.geomspace(100,1e6,81),fc])
        expected=np.zeros(len(freq),complex)
        for rr,ll,cc in p.reshape(-1,3):
            f0=1/(2*np.pi*np.sqrt(ll*cc));q=rr*np.sqrt(cc/ll);expected+=rr/(1+1j*q*(freq/f0-f0/freq))
        actual=c.response(freq,p)[0];np.testing.assert_allclose(actual,expected,rtol=1e-8,atol=1e-8)
        formula_checks.append(float(np.max(abs(actual-expected)/np.maximum(abs(expected),1e-30))))
    old_checks=[]
    for pair in old['pairs']:
        for side in ['better','worse']:
            r=json.loads((OUT/'scores'/f"dev_{pair['pair_id']}_{side}.json").read_text());new=next(v['scores'] for v in r['windows'][pair['region']]['levels'] if v['order']==64 and v['refinement']==2)
            path=ROOT/'output/arx_structure_diagnosis_20261016/metric_retrieved/output/arx_structure_diagnosis_20261016/metric_scores'/f"{pair[side]['uid']}_{pair['region']}.json";prior=json.loads(path.read_text())['scores']
            keys=['J0','J0_W','ARX_L','ARX_S_L','Local_ARX','H','J0_local_weighted'];ok=all(abs(new[k]-prior[k])<=1e-8+1e-8*abs(prior[k]) for k in keys);assert ok
            old_checks.append(dict(id=r['id'],passed=ok,max_relative=max(abs(new[k]-prior[k])/max(abs(prior[k]),1e-30) for k in keys)))
    real=[str(p) for p in scores if not p.name.startswith('control_')];replays=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(replay,p) for p in real]):replays.append(f.result())
    assert len(replays)==72 and all(r['passed'] for r in replays)
    curvefile=OUT/'INDEPENDENT_NATIVE_CURVES.json';native=json.loads(curvefile.read_text());curve_checks=[]
    for r in native['curves']:
        data=target(r['target']);c=data['circuit'];freq=np.array(r['frequency_hz']);p=np.array(r['parameters']);wanted=np.array(r['real'])+1j*np.array(r['imag']);z=c.response(freq,p)[0]
        np.testing.assert_allclose(z,wanted,rtol=CFG['audit']['nodal_rtol'],atol=CFG['audit']['nodal_atol']);nodal=checked_nodal(c,freq,p)
        curve_checks.append(dict(id=r['id'],native_matches_original=True,nodal=nodal))
    result=dict(complete=True,pairs_source_and_strata_verified=sources,strict_unmatched_definition_retained=True,
        all_score_records=756,window_records=2124,all_numerical_checks_passed=True,controlled_formula_checks=684,max_formula_relative_error=max(formula_checks),
        original_scores_replayed=old_checks,local_metric_replays=replays,local_metric_max_relative=max(r['max_relative_difference'] for r in replays),
        native_curve_and_nodal_checks=curve_checks,source_common_sha256=sha(HERE/'common.py'),holdout_bank_sha256=sha(OUT/'HOLDOUT_BANK.json'))
    dump(OUT/'LOCAL_CHECKS.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['pairs_source_and_strata_verified','original_scores_replayed','local_metric_replays','native_curve_and_nodal_checks']}),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);main(p.parse_args().workers)

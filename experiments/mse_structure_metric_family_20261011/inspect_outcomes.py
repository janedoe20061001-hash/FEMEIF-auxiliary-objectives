"""Supplement old gates with continuous position units and locate extra events."""
from core import *

def main():
    rows=[];extras=[]
    for ident in CFG['targets']:
        data=target(ident);catalog={e['event_id']:e for e in data['catalog']}
        for method in CFG['methods']:
            for seed in CFG['seeds']:
                d=json.loads((OUT/f'analysis/details/{ident}_{method}_{seed}_primary.json').read_text(encoding='utf-8'))
                for e in d['focus']:
                    width=catalog[e['event_id']]['width_3db_decades'];shift=e['shift_decades']
                    rows.append(dict(target=ident,method=method,seed=seed,event=e['event_id'],old_gate_complete=e['complete'],height_delta_db=e['height_delta_db'],width_ratio=e['width_ratio'],shift_decades=shift,signed_shift_target_widths=None if shift is None else shift/width,absolute_shift_target_widths=None if shift is None else abs(shift)/width,position_gate_target_widths=.02/width))
                if ident=='three0p25' and method=='MAG_PHA_BRIDGE':
                    p=10**np.array(d['log_parameters']);x=data['quadrature'].reference_grid(p,4);z,_=data['circuit'].response(10**x,p)
                    events=nd.events(x,z,data['config'])
                    extras.append(dict(seed=seed,extra_events=[e for e in events if e['event_id'] in d['diagnostic']['extra_events']]))
    load('family_outcome_csv',HERE/'analyze.py').csvout(OUT/'analysis/CONTINUOUS_FOCUS_POSITION.csv',rows)
    dump(OUT/'analysis/THREE_LEVEL_BRIDGE_EXTRA_EVENTS.json',dict(records=extras,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),refinement=4,original_gates_unchanged=True))
    dump(OUT/'analysis/CONTINUOUS_POSITION_PROVENANCE.json',dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),records=len(rows),purpose='diagnostic of old fixed position threshold in target-width units; no changed pass flags or selections'))
    print('Continuous position and extra-event checks ready',flush=True)

if __name__=='__main__':main()

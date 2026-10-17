"""Describe fixed paired results; no reselection or new optimization."""
from core import *
import csv
from statistics import fmean


def main():
    with (OUT/'analysis/REPRESENTATIVES.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    primary=[r for r in rows if r['rule']=='primary']
    comparisons=[('MAG_PHA',m) for m in CFG['methods'][1:]]
    comparisons += [('MAG_PHA_MAX','MAG_PHA_MEAN'),('MAG_PHA_MEAN','MAG_PHA_CHORD'),
                    ('MAG_PHA_MEAN','MAG_PHA_INTEGRAL')]
    paired=[]; totals=[]
    for target_id in CFG['targets']:
        for before,after in comparisons:
            these=[]
            for seed in CFG['seeds']:
                a=next(r for r in primary if r['target']==target_id and r['method']==before and r['seed']==str(seed))
                b=next(r for r in primary if r['target']==target_id and r['method']==after and r['seed']==str(seed))
                ag=a['focus_joint_ok']=='True';bg=b['focus_joint_ok']=='True'
                both=float(b['mag'])<=float(a['mag']) and float(b['pha'])<=float(a['pha'])
                rec=dict(target=target_id,before=before,after=after,seed=seed,
                         before_joint_ok=ag,after_joint_ok=bg,repaired=not ag and bg,
                         broken=ag and not bg,mag_delta=float(b['mag'])-float(a['mag']),
                         pha_delta=float(b['pha'])-float(a['pha']),
                         local_phase_delta_deg=float(b['local_phase_rmse_deg'])-float(a['local_phase_rmse_deg']),
                         both_base_not_worse=both,repaired_and_both_base_not_worse=not ag and bg and both,
                         before_geometry_audit_ok=a['geometry_stable']=='True' and a['geometry_training_ok']=='True',
                         after_geometry_audit_ok=b['geometry_stable']=='True' and b['geometry_training_ok']=='True')
                these.append(rec);paired.append(rec)
            totals.append(dict(target=target_id,before=before,after=after,n=len(these),
                               before_ok=sum(r['before_joint_ok'] for r in these),after_ok=sum(r['after_joint_ok'] for r in these),
                               repaired=sum(r['repaired'] for r in these),broken=sum(r['broken'] for r in these),
                               both_base_not_worse=sum(r['both_base_not_worse'] for r in these),
                               repaired_and_both_base_not_worse=sum(r['repaired_and_both_base_not_worse'] for r in these),
                               mean_mag_delta=fmean(r['mag_delta'] for r in these),mean_pha_delta=fmean(r['pha_delta'] for r in these),
                               local_phase_improved=sum(r['local_phase_delta_deg']<0 for r in these)))
    path=OUT/'analysis/PAIRED_COMPARISONS.csv'
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    dump(OUT/'analysis/COMPARISON_SUMMARY.json',totals)
    print(json.dumps(totals,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

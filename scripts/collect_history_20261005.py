"""Read-only consolidation of published experiment summaries and source hashes."""
from pathlib import Path
import json,csv,hashlib

root=Path(__file__).resolve().parents[1]
out=root/'output/research_consolidation_20261005';out.mkdir(exist_ok=True)
stages=[('G','全局残差候选','global_objectives_20261003'),
 ('D1','三维实虚部坐标','curve3d_objectives_20261003'),
 ('D2','改为幅相坐标','curve3d_logpolar_native_20261003'),
 ('R','全过程累计与尺度对照','curve3d_refinement_20261003'),
 ('Q','方向分解与组合','curve3d_directional_20261003'),
 ('V','固定少量候选复核','curve3d_shortlist_20261003'),
 ('I','全区间与逐段弧长','curve3d_interval_20261003')]
rows=[];source_hashes={};rounds=[]
for tag,label,name in stages:
    folder=root/'experiments'/name
    cfg=json.loads((folder/'case1/config.json').read_text(encoding='utf-8'))
    rounds.append(dict(stage=tag,label=label,directory=name,seeds=cfg['seeds'],groups_per_case=len(cfg['groups']),
                       runs=2*len(cfg['groups'])*cfg['seeds']['count']))
    for file in ['RESULTS.json','PARENT_COMPARISONS.json']:
        p=folder/file
        if not p.exists():continue
        source_hashes[p.relative_to(root).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        data=json.loads(p.read_text(encoding='utf-8'))
        for r in data:
            holm=next(r[k] for k in r if k.startswith('holm_adjusted'))
            for i,primary in enumerate(['magnitude','phase']):
                rows.append(dict(stage=tag,stage_label=label,directory=name,case=r['case'],id=r['id'],metric=r['group'],label=r['label'],
                    comparator='baseline' if file=='RESULTS.json' else r['parent'],primary=primary,
                    seed_start=cfg['seeds']['start'],seed_count=cfg['seeds']['count'],
                    reduction_pct=r['reduction_pct'][i],ci_low=r['reduction_pct_bootstrap95'][i][0],ci_high=r['reduction_pct_bootstrap95'][i][1],
                    holm_p=holm[i],supported_improvement=r['reduction_pct'][i]>0 and holm[i]<.05,
                    baseline_mean=r['baseline_mean'][i],candidate_mean=r['auxiliary_mean'][i],source=p.relative_to(root).as_posix()))
with (out/'historical_effects.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'historical_effects.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
(out/'historical_rounds.json').write_text(json.dumps(rounds,ensure_ascii=False,indent=2),encoding='utf-8')
(out/'historical_sources_sha256.json').write_text(json.dumps(source_hashes,indent=2),encoding='utf-8')
print(json.dumps(dict(rounds=rounds,effect_rows=len(rows)),ensure_ascii=True))

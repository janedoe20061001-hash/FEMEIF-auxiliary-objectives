from pathlib import Path
import json,sys,csv,numpy as np
root=Path('D:/develop/FEMEIF/output/complex_area_20261014');sys.path.insert(0,str(root/'review_inputs/experiments/complex_area_20261014'))
from core import target,diag,CFG,knee
out=[]
for seed in CFG['seeds']:
 ident='case1';data=target(ident);folder=root/'review_inputs/output/complex_area_20261014/jobs'/ident/'MAG_PHA'/f'seed_{seed}';audit=json.loads((folder/'audit.json').read_text())
 with np.load(root/'analysis/populations'/f'case1_MAG_PHA_{seed}.npz') as a:
  X=a['X'];F=a['F'];local=a['local_complex_error'];idx=int(a['selected_index']);allowed=np.all(F[:,:2]<=1.1*F[idx,:2]+1e-14,axis=1);ii=np.flatnonzero(allowed)[np.argmin(local[allowed])]
 diag2,curve=diag.diagnostic(data['quadrature'],X[ii],data['record']['events'],data['record']['phase_events'])
 row=dict(seed=seed,original_selected_index=idx,diagnostic_candidate_index=int(ii),original_local_error=float(local[idx]),candidate_local_error=float(local[ii]),
  original_complete=sum(e['complete'] for e in audit['amplitude_events']),candidate_complete=sum(e['complete'] for e in diag2['events']),
  original_extras=len(audit['extra_amplitude_events']),candidate_extras=len(diag2['extra_events']),candidate_diagnostic_stable=diag2['stable'],
  mag_ratio=float(diag2['objective'][0]/audit['objective'][0]),pha_ratio=float(diag2['objective'][1]/audit['objective'][1]))
 _,sel=knee(F[:,:2]);row.update(candidate_in_MAG_PHA_front=bool(ii in sel['front_indices']),candidate_dominated_by_original=bool(np.all(F[idx,:2]<=F[ii,:2]) and np.any(F[idx,:2]<F[ii,:2])),candidate_log_parameters=X[ii].tolist(),candidate_amplitude_events=diag2['events'],candidate_phase_events=diag2['phase_events'],candidate_objective=diag2['objective']);out.append(row);print({k:row[k] for k in ('seed','original_complete','candidate_complete','candidate_in_MAG_PHA_front')},flush=True)
(root/'SELECTOR_STRUCTURE_DIAGNOSIS.json').write_text(json.dumps(dict(scope='Read-only diagnostic alternatives from saved population, not final reselection',records=out),ensure_ascii=False,indent=2),encoding='utf-8')
# Loss allocation on the final selected candidates, all windows kept.
con=[]
for ident in ['case1','mixed_resonators/base','mixed_resonators/lc_reciprocal_8p0']:
 data=target(ident);blocks=data['record']['blocks']
 for method in ['J0','A','ARX']:
  for seed in CFG['seeds']:
   a=json.loads((root/'review_inputs/output/complex_area_20261014/jobs'/ident/method/f'seed_{seed}'/'audit.json').read_text());v=np.array(a['auxiliary']['window_scores']);total=sum(v)
   for b,s in zip(blocks,v):con.append(dict(target=ident,method=method,seed=seed,label=b['label'],low_hz=10**b['lo'],high_hz=10**b['hi'],score=float(s),fraction=float(s/total) if total else 0))
with (root/'analysis/AUXILIARY_WINDOW_CONTRIBUTIONS.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(con[0]));w.writeheader();w.writerows(con)

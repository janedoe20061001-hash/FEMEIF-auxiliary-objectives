from pathlib import Path
import ast,json,hashlib,sys,csv
import numpy as np
from scipy import linalg
from scipy.optimize import linear_sum_assignment
ROOT=Path('D:/develop/FEMEIF');OUT=ROOT/'output/arx_light_20261015';CODE=ROOT/'experiments/arx_light_20261015';sys.path.insert(0,str(CODE))
from core import target,CFG
old=ROOT/'output/complex_joint_causes_20261013/investigate.py';tree=ast.parse(old.read_text(encoding="utf-8"))
functions=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('descriptor','groups')],type_ignores=[])
ns=dict(np=np,linalg=linalg,WREF=2*np.pi*1e4);exec(compile(functions,str(old),'exec'),ns)
rows=[];errors=[]
for ident in ['mixed_resonators/base','mixed_resonators/lc_reciprocal_8p0']:
 d=target(ident);c=d['circuit'];tm=ns['descriptor'](c,c.p);tg=ns['groups'](tm);hi=max((g for g in tg if g['osc']),key=lambda g:g['freq']);ti=max(hi['ix'],key=lambda j:tm['poles'][j].imag);tr=tm['res'][ti]
 for method in CFG['methods']:
  for seed in CFG['seeds']:
   path=OUT/'retrieved/output/arx_light_20261015/jobs'/ident/method/f'seed_{seed}'/'generation_0300.npz'
   with np.load(path) as a:p=10**a['X'][int(a['selected_index'])]
   try:
    cm=ns['descriptor'](c,p);gg=ns['groups'](cm)
    cost=np.array([[abs(np.log10(max(g['freq'],1e-20)/max(q['freq'],1e-20)))+(0 if g['osc']==q['osc'] else 10) for q in gg] for g in tg]);ri,cj=linear_sum_assignment(cost);matching=dict(zip(ri,cj));cg=gg[matching[tg.index(hi)]]
    if not cg['osc']:raise ValueError('Target oscillatory mode matched an aperiodic candidate; modal ratios undefined')
    ci=max(cg['ix'],key=lambda j:cm['poles'][j].imag);cr=cm['res'][ci]
    rows.append(dict(target=ident,method=method,seed=seed,frequency_hz=cg['freq'],target_frequency_hz=hi['freq'],frequency_abs_log_error=abs(np.log(cg['freq']/hi['freq'])),
      decay_ratio=cg['alpha']/hi['alpha'],residue_abs_ratio=abs(cr/tr),residue_complex_relative_error=abs(cr-tr)/abs(tr),
      visibility_ratio=(abs(cr)/cg['alpha'])/(abs(tr)/hi['alpha']),nearest_zero_distance_over_decay=float(min(abs(cm['zeros']-cm['poles'][ci]))/abs(cm['poles'][ci].real)),
      reconstruction_error=cm['error'],node_error=cm['node_error'],eigen_residual=cm['eig_residual']))
   except Exception as e:errors.append(dict(target=ident,method=method,seed=seed,error=repr(e)))
with (OUT/'analysis/MODAL_REVIEW.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'MODAL_REVIEW_CHECKS.json').write_text(json.dumps(dict(source_file=str(old),source_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),decompositions=len(rows),failures=errors,max_reconstruction_error=max(r['reconstruction_error'] for r in rows),note='Additional port modal diagnosis; frequency-based assignment can be ambiguous. No parameter optimization.'),ensure_ascii=False,indent=2),encoding='utf-8')
print('modal review',len(rows),'failures',errors)
for ident in ['mixed_resonators/base','mixed_resonators/lc_reciprocal_8p0']:
 b={r['seed']:r for r in rows if r['target']==ident and r['method']=='MAG_PHA'}
 for method in CFG['methods'][1:]:
  rr=[r for r in rows if r['target']==ident and r['method']==method];counts={}
  for key in ['frequency_abs_log_error','residue_complex_relative_error']:
   counts[key]=sum(r[key]<b[r['seed']][key] for r in rr)
  for key in ['decay_ratio','visibility_ratio']:
   counts[key]=sum(abs(np.log(r[key]))<abs(np.log(b[r['seed']][key])) for r in rr)
  print(ident,method,counts)

from review_analysis import *
from core import diag,BASECFG

ident='mixed_resonators/base';data=target(ident);d=DATA/'jobs'/ident/'MAG_PHA/seed_46003'
with np.load(d/'generation_0300.npz') as a:lp=a['X'][int(a['selected_index'])]
rows=[]
for level in [1,2,4,8,16,32]:
    score,x,z,ref=data['quadrature'].reference(lp,level);fit=diag.events(x,z,BASECFG);events,extra=diag.match(data['record']['events'],fit,BASECFG)
    _,dz=data['circuit'].response(10**x,10**lp);phase=diag.phase_events(x,z,dz,BASECFG);pe,px=diag.phase_match(data['record']['phase_events'],phase,BASECFG)
    rows.append(dict(level=level,points=len(x),score=score.tolist(),amplitude_events=events,extra_amplitude=extra,fit_amplitude=fit,phase_events=pe,extra_phase=px,fit_phase=phase))
dump(OUT/'SUPPLEMENTAL_FAILED_STRUCTURE_CHECK.json',dict(original_accepted=False,thresholds_unchanged=True,scope='Read-only supplemental diagnosis; does not change original audit',rows=rows))

"""Reconstruct the existing 800 Case1 representative solutions, no new fitting."""
import sys,csv,json,os,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
source=ROOT.parent/'crosscase_consolidation_20261005'/'case1';sys.path.insert(0,str(source));os.chdir(source)
import experiment as e
from legacy_objectives import Objectives,impedance,TARGET
from length_metrics import GlobalObjectives as Length
cfg,f=e.config(source/'config.json');target=impedance(f,TARGET)[0];pt=np.angle(target)
with (ROOT/'historical_case1_selected.csv').open(encoding='utf-8-sig',newline='') as fh:rows=list(csv.DictReader(fh))
assert len(rows)==800
params=np.array([[float(r[f'{letter}{i}']) for i in range(1,7) for letter in ['R','L','C']] for r in rows]);z=impedance(f,params)
names=['reactance_crossing_direction_consistency','characteristic_frequency_shift_error_decades','transition_speed_relative_error']
obj=Objectives(f,aux=names,target_z=target);F=obj.evaluate_spectra(z)
np.testing.assert_allclose(F[:,0],[float(r['wsmse_selected_magnitude_mse']) for r in rows],rtol=1e-11,atol=1e-11)
np.testing.assert_allclose(F[:,1],[float(r['wsmse_selected_phase_mse']) for r in rows],rtol=1e-11,atol=1e-11)
L=Length(f,aux=('cumulative_length_logpolar',),target_z=target).evaluate_spectra(z)[:,2]
phi=np.angle(z);pe=(phi-pt)**2;wrong=np.signbit(z.imag)!=np.signbit(target.imag)
u=np.log10(f);u=(u-u[0])/(u[-1]-u[0]);h=np.diff(u);dt=np.diff(pt)/h
ps=np.sqrt(np.sum(h*(np.diff(phi,axis=-1)/h-dt)**2,axis=-1)/np.sum(h*dt**2))
records=[]
for i,r in enumerate(rows):
    records.append(dict(seed=int(r['seed']),group=r['group'],magnitude=F[i,0],phase=F[i,1],I03=F[i,2],II01=F[i,3],II02=F[i,4],R05=L[i],
        phase_secant=ps[i],wrong_sign_contribution=np.mean(pe[i]*wrong[i]),same_sign_contribution=np.mean(pe[i]*(~wrong[i])),
        wrong_sign_points=wrong[i].mean()))
with (ROOT/'historical_diagnostics.csv').open('w',encoding='utf-8-sig',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
summary=[]
for g in dict.fromkeys(r['group'] for r in records):
    rr=[r for r in records if r['group']==g]
    summary.append(dict(group=g,**{k:float(np.mean([r[k] for r in rr])) for k in rr[0] if k not in ['seed','group']}))
(ROOT/'historical_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(ROOT/'historical_audit.json').write_text(json.dumps(dict(runs=800,primary_mse_verified=True,no_optimization=True,
    original_csv_sha256=hashlib.sha256((ROOT/'historical_case1_selected.csv').read_bytes()).hexdigest()),indent=2))
print(json.dumps(summary,indent=2))

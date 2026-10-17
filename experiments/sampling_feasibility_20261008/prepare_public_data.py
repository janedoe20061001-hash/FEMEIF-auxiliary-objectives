"""Validate public numerical data; keep originals and provenance intact."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/public_impedance_zenodo_10794584'
OUT=ROOT/'output/sampling_feasibility_20261008'
def main():
    record=json.loads((DATA/'record.json').read_text(encoding='utf8'))
    files={f['key']:f for f in record['files']};rows=[]
    (DATA/'normalized').mkdir(exist_ok=True)
    for p in sorted((DATA/'raw').glob('*.json')):
        blob=p.read_bytes();d=json.loads(blob)
        checksum=hashlib.md5(blob).hexdigest();assert files[p.name]['checksum']=='md5:'+checksum
        f=np.asarray(d['f'],dtype=float);re=np.asarray(d['z_real'],dtype=float);im=np.asarray(d['z_imag'],dtype=float);z=re+1j*im
        assert all(a.ndim==1 and len(a)==len(f) and np.isfinite(a).all() for a in [f,re,im])
        assert (f>0).all() and len(np.unique(f))==len(f) and (abs(z)>0).all()
        np.testing.assert_allclose(d['omega'],2*np.pi*f,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(d['z_modulus'],abs(z),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(d['z_phase'],np.angle(z),rtol=1e-12,atol=1e-12)
        rounded=np.asarray([complex(v.replace('i','j')) for v in d['z']])
        string_ok=bool(np.allclose(rounded,z,rtol=1e-6,atol=1e-12))
        string_abs=float(np.max(abs(rounded-z)))
        string_rel=float(np.max(abs(rounded-z)/abs(z)))
        dest=DATA/'normalized'/p.with_suffix('.csv').name
        with dest.open('w',encoding='utf8',newline='') as fh:
            w=csv.writer(fh);w.writerow(['frequency_hz','z_real_ohm','z_imag_ohm','magnitude_ohm','phase_rad','phase_deg'])
            for i in np.argsort(f):w.writerow([f[i],re[i],im[i],abs(z[i]),np.angle(z[i]),np.rad2deg(np.angle(z[i]))])
        rows.append(dict(file=p.name,points=len(f),frequency_min_hz=float(min(f)),frequency_max_hz=float(max(f)),negative_real_points=int((re<0).sum()),source_url=f'https://zenodo.org/records/10794584/files/{p.name}?download=1',md5=checksum,sha256=hashlib.sha256(blob).hexdigest(),normalized_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),numeric_fields_consistent=True,redundant_string_pass_1e6=string_ok,redundant_string_max_abs_ohm=string_abs,redundant_string_max_relative=string_rel,normalized_fields='z_real and z_imag; original z string retained in raw JSON',role='priority_RLC_measurement' if p.stem=='rlc_circuit' else 'reserve_electrochemical_measurement'))
        if p.stem=='rlc_circuit':
            ix=np.argsort(f);fig,axs=plt.subplots(2,1,sharex=True,figsize=(8,5),layout='constrained')
            axs[0].semilogx(f[ix],20*np.log10(abs(z[ix])),'o-',ms=3);axs[0].set_ylabel('Magnitude (dB re 1 ohm)')
            axs[1].semilogx(f[ix],np.rad2deg(np.angle(z[ix])),'o-',ms=3);axs[1].set_ylabel('Phase (degree)');axs[1].set_xlabel('Frequency (Hz)')
            for ax in axs:ax.grid(True,alpha=.2)
            fig.suptitle('Public measured RLC: 81 original points; lines are visual guides')
            fig.savefig(OUT/'public_rlc_measured.png',dpi=160);plt.close(fig)
    meta=record['metadata']
    result=dict(record_url='https://zenodo.org/records/10794584',doi=meta['doi'],title=meta['title'],authors=[a['name'] for a in meta['creators']],publication_date=meta['publication_date'],license=meta['license']['id'],record_sha256=hashlib.sha256((DATA/'record.json').read_bytes()).hexdigest(),spectra=len(rows),points=sum(r['points'] for r in rows),datasets=rows,scope='data preparation only; no new optimization or quality/causality proof')
    (DATA/'MANIFEST.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    (OUT/'PUBLIC_DATA_VALIDATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()

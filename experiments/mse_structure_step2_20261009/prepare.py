from common import *
import platform,time
cfg=json.loads((HERE/'CONFIG.json').read_text());catalogs={};metadata=[]
for s in cfg['target_scales']:
    m=Target(s);p=np.asarray(m.record['target_parameters']);x=np.log10(p);assert np.all(x>=m.lower) and np.all(x<=m.upper)
    cat=target_catalog(m);assert len([e for e in cat if e['kind']=='peak'])==6 and len([e for e in cat if e['kind']=='valley'])==5
    d=HERE/'targets'/label(s);d.mkdir(parents=True,exist_ok=True);grid=base_grid(m);np.savez_compressed(d/'grid.npz',frequency=grid)
    dump(d/'record.json',m.record);dump(d/'events.json',cat);catalogs[label(s)]=cat
    np.testing.assert_allclose(m.network(m.f,p),spectra(m.f,p),rtol=1e-10,atol=1e-9)
    a=evaluator(m);zero=a.score(x)[0];assert np.max(zero)<1e-15
    e={r['event_id']:r for r in cat};p3=e['P3'];v2=e['V2'];fc=1/(2*np.pi*np.sqrt(p[4]*p[5]));q=p[3]*np.sqrt(p[5]/p[4]);
    z=spectra(np.array([23000,25000,26000,30000]),p)[0]
    metadata.append(dict(scale=s,R2=p[3],L2=p[4],C2=p[5],branch_center_hz=fc,branch_bandwidth_hz=fc/q,P3_frequency_hz=p3['frequency_hz'],P3_height_db=p3['height_db'],P3_width_hz=10**(p3['x']+p3['width_3db_decades']/2)-10**(p3['x']-p3['width_3db_decades']/2),P3_width_decades=p3['width_3db_decades'],V2_frequency_hz=v2['frequency_hz'],V2_height_db=v2['height_db'],V2_width_decades=v2['width_3db_decades'],gap_hz=p3['frequency_hz']-v2['frequency_hz'],phase_23000_deg=np.angle(z[0],deg=True),phase_25000_deg=np.angle(z[1],deg=True),phase_26000_deg=np.angle(z[2],deg=True),amp_30000_db=20*np.log10(abs(z[3])),true_L2_fraction=(x[4]-m.lower[4])/(m.upper[4]-m.lower[4]),true_C2_fraction=(x[5]-m.lower[5])/(m.upper[5]-m.lower[5]),base_points=len(grid),zero_mag=zero[0],zero_phase=zero[1]))
dump(OUT/'TARGET_EVENTS.json',catalogs);csvout(OUT/'TARGET_CONDITIONS.csv',metadata);dump(OUT/'PROTOCOL.json',cfg)
sources={p.relative_to(HERE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['frozen','analysis_toolkit','targets'] for p in (HERE/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)}
for name in ['adaptive.py','structure.py','common.py','CONFIG.json']:sources[name]=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
dump(HERE/'INPUT_HASHES.json',sources)
print(json.dumps(dict(targets=len(metadata),metadata=metadata),ensure_ascii=False))

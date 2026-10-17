from pathlib import Path
import ast, shutil, json
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'mse_structure_step1_20261008'
for rel in ['frozen/model.py','frozen/circuits.json','frozen/frozen/case1/legacy_objectives.py','adaptive.py','analysis_toolkit/diagnostics.py','analysis_toolkit/refined_grid.py','analysis_toolkit/frozen/population_source.py']:
    dst=HERE/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OLD/rel,dst)
source=(OLD/'classify_v2.py').read_text(encoding='utf8')
names=['amp_events','target_catalog','classify_events','diagnostic']
prefix='import numpy as np\nfrom scipy.signal import find_peaks,peak_widths\nfrom scipy.optimize import linear_sum_assignment\nfrom adaptive import spectra\nfrom refined_grid import improved_dense_grid\nfrom diagnostics import diagnose\n'
(HERE/'structure.py').write_text(prefix+'\n\n'.join(ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names)+'\n',encoding='utf8')
cfg=dict(target_scales=[.25,.5,1,2,4,8],run_scales=[.25,1,8],seeds=list(range(31000,31008)),generations=300,population=200,workers=4,plot_seeds=[31000,31003,31006],bootstrap_seed=2026100902,old_stages=[20,100,300,1000],pair_window_hz=[23000,26000],broad_window_hz=[10000,10**4.5],core_rule='P3 center +/- half target 3dB log-frequency width',close_scores=[.02,.05,.10],score_floors=[.01,(.5*3.141592653589793/180)**2],meaningful_shape_rule='matched status differs OR height difference >=6 dB OR width ratio between candidates >=2',match_rule='same kind within .25 decade; complete shift <=.02 decade,height <=3dB,width ratio .5..2',primary_rule='minimum MAG+25.12*PHA on nondominated front',secondary_rule='minimum max(MAG,25.12*PHA) on nondominated front',primary_outcome='P3 and V2 complete on SAME representative curve',scope='Exploratory 8 paired seeds, no auxiliary objective',sampling='Pure Python adaptive, original 180 log-cells each mass 1/180',source_step1=str(OLD),protocol_frozen_before_new_results=True)
(HERE/'CONFIG.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf8')
print('Copied frozen numerical sources; fixed protocol and 24-run pilot.')

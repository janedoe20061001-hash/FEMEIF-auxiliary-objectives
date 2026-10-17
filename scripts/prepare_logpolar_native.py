"""Version numerical implementation, preserve v1; all metric definitions unchanged."""
from pathlib import Path
import json,shutil,tarfile,hashlib
root=Path(__file__).resolve().parents[1]
source=root/'experiments/curve3d_logpolar_20261003'
target=root/'experiments/curve3d_logpolar_native_20261003'
for p in source.rglob('*'):
    if not p.is_file() or any(part in p.parts for part in ['__pycache__','server_run','runs','figures']):continue
    if p.suffix not in ['.py','.c','.json','.md','.txt','.csv']:continue
    relative=p.relative_to(source);dest=target/relative
    if (target/'case1/runs').exists() or (target/'case2/runs').exists():raise RuntimeError('Never rebuild a running protocol')
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
core=(target/'global_metrics.py').read_text(encoding='utf-8')
core=core.replace('from scipy.spatial.distance import pdist','from scipy.spatial.distance import pdist\nfrom native_backend import weighted_terms,native_available')
start=core.index('    def orientation_loss(');end=core.index('    def evaluate_spectra(',start)
core=core[:start]+'''    def orientation_loss(self,p,chunk_size=32):
        result=[]
        for start in range(0,len(p),chunk_size):
            terms=weighted_terms(p[start:start+chunk_size],self.i,self.j,self.directions,self.pair_weights)
            result.extend(np.sum(terms,axis=-1)/self.pair_weights.sum())
        return np.asarray(result)

'''+core[end:]
core=core.replace("return dict(frequency_points=len(self.f),frequency_coordinate", "return dict(orientation_backend='native_c' if native_available else 'numpy_reference',frequency_points=len(self.f),frequency_coordinate")
(target/'global_metrics.py').write_text(core,encoding='utf-8')
readme=(target/'README.md').read_text(encoding='utf-8')
readme+='\n数值版本、原慢版本保留范围与全预算重新运行说明见 [NUMERICAL_VERSION.md](NUMERICAL_VERSION.md)。\n'
(target/'README.md').write_text(readme,encoding='utf-8')
for case in ['case1','case2']:
    folder=target/case
    cfg=json.loads((folder/'config.json').read_text(encoding='utf-8'))
    cfg['protocol']='curve3d_logpolar_20261003_native_v2'
    (folder/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    engine=(folder/'experiment.py').read_text(encoding='utf-8')
    engine=engine.replace('"global_metrics.py","curve_math.py",','"global_metrics.py","curve_math.py","native_backend.py","orientation_native.c","orientation_native.so","build_native.json",')
    (folder/'experiment.py').write_text(engine,encoding='utf-8')
    shutil.copy2(target/'global_metrics.py',folder/'global_metrics.py')
    for name in ['native_backend.py','orientation_native.c']:shutil.copy2(target/name,folder/name)
report=(target/'finish_report.py').read_text(encoding='utf-8').replace("'.txt','.log']","'.txt','.log','.c','.so']")
(target/'finish_report.py').write_text(report,encoding='utf-8')
archive=root/'tmp/curve3d_logpolar_native_20261003_code.tar.gz'
with tarfile.open(archive,'w:gz') as tf:
    for p in sorted(target.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:tf.add(p,arcname=str(p.relative_to(target)))
print(archive);print(hashlib.sha256(archive.read_bytes()).hexdigest())

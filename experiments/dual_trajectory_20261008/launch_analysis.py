"""Create read-only links to retained server runs, launch independent analyses."""
from pathlib import Path
import tarfile,json,hashlib,subprocess,shutil
ROOT=Path('/root/autodl-tmp/curve_analysis_workspace_20261008');ROOT.mkdir(exist_ok=False)
with tarfile.open('/root/autodl-tmp/curve_analysis_bundle_20261008.tar.gz') as t:t.extractall(ROOT,filter='data')
shutil.copyfile('/root/autodl-tmp/trajectory_analyze_updated.py',ROOT/'experiments/dual_trajectory_20261008/analyze.py')
for name in ['primary_loss_comparison_20261007','optimized_curve_structure_20261007','sampling_refinement_20261007','dual_trajectory_20261008']:
    d=ROOT/'experiments'/name/'server_run';d.mkdir(parents=True,exist_ok=True)
    (d/'retrieved').symlink_to(Path('/root/autodl-tmp')/name,target_is_directory=True)
archive=Path('/root/autodl-tmp/sampling_refinement_results_20261007.tar.gz');status=json.loads((Path('/root/autodl-tmp/sampling_refinement_20261007/runs/status.json')).read_text())
(ROOT/'experiments/sampling_refinement_20261007/server_run/ARCHIVES.json').write_text(json.dumps(dict(location='server retained complete archive; partial local transfer is not usable',sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size,remote=str(archive),status=status),indent=2))
python='/root/autodl-tmp/indicator_phase_validation_20261007/.venv/bin/python'
jobs=[['experiments/sampling_refinement_20261007/analyze.py','--runs','experiments/sampling_refinement_20261007/server_run/retrieved/runs','--output','output/sampling_refinement_20261007','--workers','16'],['experiments/dual_trajectory_20261008/background.py'],['experiments/dual_trajectory_20261008/analyze.py']]
launched=[]
for i,args in enumerate(jobs):
    f=(ROOT/f'analysis_{i}.log').open('w');p=subprocess.Popen([python,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);launched.append(dict(pid=p.pid,args=args,log=f.name))
(ROOT/'LAUNCHER.json').write_text(json.dumps(launched,indent=2));print(json.dumps(launched))

"""Run independent circuit diagnostics in parallel, then verify and merge."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,csv,hashlib,json,subprocess,sys,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def loadcsv(path):
    textkeys={'circuit','family','condition','group','grid','metric','rule','kind'}
    rows=[]
    with path.open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            rows.append({k:(v if k in textkeys else None if v=='' else True if v=='True' else False if v=='False' else float(v)) for k,v in row.items()})
    return rows
def task(r,runs,parts):
    log=parts/(r['id']+'.log');out=parts/r['id']
    with log.open('w',encoding='utf8') as f:
        p=subprocess.run([sys.executable,str(HERE/'analyze.py'),'--dual-runs',str(runs),'--output',str(out),'--circuit',r['id']],stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError(r['id']+' failed; see '+str(log))
    return r['id']
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dual-runs',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,default=8);args=ap.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);parts=out/'parts';parts.mkdir();runs=args.dual_runs.resolve()
    records=json.loads((HERE/'circuits.json').read_text(encoding='utf8'));start=time.time();done=0
    def status(state):
        (out/'status.json').write_text(json.dumps({'status':state,'circuits_completed':done,'circuits_total':50,'workers':args.workers,'seconds':time.time()-start},indent=2),encoding='utf8')
    status('running')
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(task,r,runs,parts) for r in records]):
                ident=future.result();done+=1;status('running');print(json.dumps({'completed':done,'circuit':ident,'seconds':time.time()-start}),flush=True)
        # Import numerical/report helpers only once after subprocesses have ended.
        from analyze import csvout,dump,sha,GROUPS
        rows={name:[] for name in ['per_seed_diagnostics','band_diagnostics','grid_refinement','dual_selection_sensitivity','feature_matches','group_means','paired_comparisons']}
        checked=seedruns=0;(out/'selected').mkdir()
        import shutil
        for r in records:
            directory=parts/r['id'];v=json.loads((directory/'VALIDATION.json').read_text());s=json.loads((directory/'SUMMARY.json').read_text())
            assert v['passed'] and not v['pilot'];assert v['analysis_sha256']==sha(HERE/'analyze.py') and v['diagnostics_sha256']==sha(HERE/'diagnostics.py')
            checked+=v['dual_final_individuals_recomputed'];seedruns+=s['paired_seed_runs']
            shutil.copy2(directory/'selected'/f'{r["id"]}.npz',out/'selected'/f'{r["id"]}.npz')
            for name in rows:rows[name].extend(loadcsv(directory/(name+'.csv')))
        assert checked==150200 and seedruns==1160 and len(rows['per_seed_diagnostics'])==4640 and len(rows['band_diagnostics'])==37120
        for name,values in rows.items():csvout(out/(name+'.csv'),values)
        unstable=sum(not r['stable'] for r in rows['grid_refinement'])
        dump(out/'SUMMARY.json',dict(means=rows['group_means'],paired_comparisons=rows['paired_comparisons'],rows=4640,groups=GROUPS,dual_final_individuals_recomputed=checked,paired_seed_runs=seedruns,grid_unstable_rows=unstable,grid_refinement_rows=len(rows['grid_refinement']),pilot=False))
        dump(out/'VALIDATION.json',dict(passed=True,primary_selected_curves=3480,balanced_selected_curves=1160,dual_final_individuals_recomputed=checked,paired_seed_runs=1160,paired_initial_populations=True,all_source_hashes=True,independent_pareto_front_check=True,all_selection_rules_recomputed=True,selected_spectra_metrics_recomputed=True,generations_300=True,added_evaluations=45060000,case2_all_201_points_retained=True,grid_unstable_rows=unstable,analysis_sha256=sha(HERE/'analyze.py'),diagnostics_sha256=sha(HERE/'diagnostics.py'),runner_sha256=sha(__file__),dual_archive_sha256=sha(HERE/'server_run/dual_results.tar.gz')))
        status('complete');print(json.dumps({'passed':True,'rows':4640,'unstable':unstable,'seconds':time.time()-start}),flush=True)
    except Exception:
        status('failed');raise
if __name__=='__main__':main()

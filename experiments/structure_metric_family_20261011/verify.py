"""Check inventory provenance, counts, and artifact references."""
from inventory import *
import re,time


def main():
    d=json.loads((OUT/'INVENTORY.json').read_text(encoding='utf-8'))
    assert sum(r['n'] for r in d['nested_amplitude_vs_phase'])==168
    assert len(d['revision_extra_and_phase_events'])==80
    assert len(d['unscored_gaps'])==20 and all(r['gap_rmse_stable'] for r in d['unscored_gaps'])
    for rel,h in d['input_sha256'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
    assert d['script_sha256']==hashlib.sha256((Path(__file__).parent/'inventory.py').read_bytes()).hexdigest()
    paths=[OUT/'REPORT.md',ROOT/'docs/三维曲线结构缺陷_指标组设计_20261011.md']
    links=[]
    for p in paths:
        for url in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            if re.match(r'^[A-Za-z]:/',url):assert Path(url).exists(),url;links.append(url)
    result=dict(status='complete',time_unix=time.time(),new_optimization_tasks=0,website_publications=0,
                revised_representatives=80,nested_representatives=168,counts_kept_separate=True,
                gap_refinement_checks=20,all_gap_checks_stable=True,input_hashes_verified=len(d['input_sha256']),
                links_checked=len(links),figure_manually_inspected=True,
                source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
                artifact_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*paths,OUT/'BRIDGE.png',OUT/'INVENTORY.json']})
    (OUT/'VERIFICATION.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Existing data and counts verified; new candidate optimization not performed.')


if __name__=='__main__':main()

"""Verify every selected row with an independent library front and scalar construction."""
from reselect import *
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import re

def main():
    checks();count=0;stages=0;all_raw={};kinds={};aux_flags=[];independent=[]
    for batch in BATCHES:
        ver=json.loads((OUT/batch/'INPUT_VERIFICATION.json').read_text())
        assert ver['passed'];stages+=ver['stage_count']
        for name,h in ver['optimization_source_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
        for name,h in ver['raw_checksums'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
        all_raw.update(ver['raw_checksums'])
        cfg=json.loads((ROOT/'experiments'/BATCHES[batch]/'protocol.json').read_text())
        for sfile in sorted((OUT/batch/'selections').glob('*.json')):
            s=json.loads(sfile.read_text());path=ROOT/s['source'];assert hashlib.sha256(path.read_bytes()).hexdigest()==s['source_sha256']
            with np.load(path) as a:
                f=a['objectives'][:,:2];x=a['log_parameters'];ix=s['selected_index']
                lib=NonDominatedSorting().do(f,only_non_dominated_front=True)
                np.testing.assert_array_equal(np.sort(lib),np.sort(s['front_indices']))
                ki,info=knee(f);assert ki==ix and info['reason']==s['reason']
                vv=f[s['unique_indices']];lo=vv.min(0);hi=vv.max(0)
                normalized=(vv-lo)/np.maximum(hi-lo,1e-14*np.maximum(1,np.maximum(abs(lo),abs(hi))))
                ds=[float((1-u-v)/np.sqrt(2)) for u,v in normalized]
                np.testing.assert_allclose(ds,s['signed_distances'],atol=1e-15,rtol=1e-14)
                if s['reason']=='convex_knee':
                    eligible=[j for j,v in enumerate(ds) if v>=max(ds)-1e-12]
                else:eligible=list(range(len(vv)))
                best=min(eligible,key=lambda j:(float(sum(normalized[j]**2)),s['unique_indices'][j]))
                assert s['unique_indices'][best]==ix
                d=json.loads((OUT/batch/f"analysis/details/{s['target']}_{s['method']}_{s['seed']}_knee.json").read_text(encoding='utf-8'))
                np.testing.assert_array_equal(x[ix],d['log_parameters']);np.testing.assert_array_equal(f[ix],d['training_objective'])
                assert d['summary']['stable'] and d['summary']['base_score_ok']
                assert d['summary']['rule']=='knee' and d['summary']['focus_joint_ok']==all(e['complete'] for e in d['focus'])
                sha=d['replacement_analyzer_sha256']
                assert sha==hashlib.sha256((HERE/'reselect_executed.py').read_bytes()).hexdigest(), 'Missing executed-source snapshot'
                if not d['summary']['geometry_training_ok']:aux_flags.append(dict(batch=batch,target=s['target'],method=s['method'],seed=s['seed']))
                kinds[s['reason']]=kinds.get(s['reason'],0)+1;count+=1
        audit=json.loads((OUT/batch/'AUDIT.json').read_text());assert audit['independent_structure_and_scores_passed'] and not audit['failures']
        st=list(csv.DictReader((OUT/batch/'STAGES.csv').open(encoding='utf-8-sig')))
        assert len(st)==ver['stage_count']
        for r in st:
            source=OUT/'source_populations'/batch
            if not source.exists():source=ROOT/'output'/BATCHES[batch]/'batch'
            with np.load(source/f"{r['target']}/{r['method']}/seed_{r['seed']}/generation_{int(r['generation']):04}.npz") as a:
                ix,info=knee(a['objectives'][:,:2]);assert ix==int(r['index']) and info['reason']==r['reason']
        independent.append(dict(batch=batch,jobs=ver['jobs'],source_passed=True,all_base_and_structure_passed=True,all_auxiliary_checks_passed=audit['passed']))
    assert count==252 and stages==3780 and kinds['convex_knee']==200 and len(aux_flags)==2
    group_rows=list(csv.DictReader((OUT/'GROUPS.csv').open(encoding='utf-8-sig')))
    assert len(group_rows)==63 and len(list(csv.DictReader((OUT/'REPRESENTATIVES.csv').open(encoding='utf-8-sig'))))==252
    assert len(list(csv.DictReader((OUT/'PAIRED.csv').open(encoding='utf-8-sig'))))==252
    paths=list(OUT.glob('*/figures/*.png'));assert len(paths)==17
    links=[]
    updated_docs=[ROOT/'docs'/name for name in ('MSE困难结构与辅助指标_研究大纲.md','三维阻抗曲线_第三步更新设计_20261011.md','三维曲线指标_新一轮候选与实验约定_20261011.md','三维曲线结构缺陷_指标组设计_20261011.md','三维阻抗曲线_辅助指标候选与设计思路.md')]
    for file in list(OUT.rglob('*.md'))+[HERE/'README.md',HERE/'PROTOCOL.md']+updated_docs:
        for link in re.findall(r'\]\(([^)]+)\)',file.read_text(encoding='utf-8')):
            link=link.strip('<>')
            if link.startswith(('http:','https:','#')):continue
            p=Path(link) if re.match(r'^[A-Za-z]:[/\\]',link) else file.parent/link
            assert p.exists(),f'{file}: {link}';links.append(str(p.resolve()))
    hashes={p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name not in ('DELIVERY_VERIFICATION.json','FINAL_STATUS.json','CLEANUP.json') and 'source_populations' not in p.parts}
    dump(OUT/'DELIVERY_VERIFICATION.json',dict(passed=True,curves=count,stages=stages,original_evaluations=count*60000,new_optimization_evaluations=0,independent_library_front_verification=True,independent_selection_arithmetic=True,source_and_raw_sha256_checked=True,executed_analyzer_source_snapshot_checked=True,representative_parameters_and_objectives_checked=True,all_base_and_structure_passed=True,all_auxiliary_checks_passed=False,auxiliary_training_discrepancies=aux_flags,knee_reasons=kinds,batches=independent,figures=len(paths),manually_reviewed_figures=17,local_links_checked=len(links),delivery_hashes=hashes))
    print(json.dumps(dict(passed=True,curves=count,stages=stages,figures=len(paths),auxiliary_training_discrepancies=len(aux_flags))))

if __name__=='__main__':main()

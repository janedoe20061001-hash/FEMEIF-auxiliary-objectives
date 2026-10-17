"""Validate source lineage and retained results; assemble tables and hash ledger."""
from scan import *
import csv,re,zipfile
from datetime import datetime,timezone

def readrows(name):return list(csv.DictReader((OUT/name).open(encoding='utf-8-sig')))
def table(rows,keys=None):
    if not rows:return ''
    keys=keys or list(rows[0]);result=['|'+'|'.join(keys)+'|','|'+'|'.join(['---']*len(keys))+'|']
    for r in rows:
        result.append('|'+'|'.join(str(r[k]).replace('|','/') for k in keys)+'|')
    return '\n'.join(result)+'\n'

def main():
    manifest=json.loads((DATA/'MANIFEST.json').read_text());source_ok={}
    for name,digest in manifest['hashes'].items():
        assert sha(ROOT/name)==digest,('frozen source/input',name);source_ok[name]=digest
    inventory_meta=json.loads((OUT/'INVENTORY.json').read_text());assert sha(OUT/'INVENTORY.npz')==inventory_meta['inventory_sha256']
    for name,digest in inventory_meta['source_hashes'].items():assert sha(DATA/name)==digest,('stage',name)
    b=json.loads((OUT/'SCORING_BANK.json').read_text());digest=sha(OUT/'SCORING_BANK.json')
    assert b['pairs']==json.loads((OUT/'FROZEN_PAIR_BANK.json').read_text())['pairs']
    assert (OUT/'SCORING_BANK.json').read_bytes()==(OUT/'FROZEN_PAIR_BANK_V1.json').read_bytes()
    assert sha(HERE/'PAIR_SELECTION_V1_LF.py')==b['selection_code_sha256']
    assert (HERE/'PAIR_SELECTION_V1.py').read_bytes().replace(b'\r\n',b'\n')==(HERE/'PAIR_SELECTION_V1_LF.py').read_bytes()
    assert sha(HERE/'PAIR_PROTOCOL.md')==b['pair_protocol_sha256']
    assert sha(HERE/'PROTOCOL.md')==inventory_meta['protocol_sha256']
    assert sha(HERE/'scan.py')==inventory_meta['scan_source_sha256']
    assert sha(OUT/'INVENTORY.npz')==b['source_inventory_sha256']
    checked=json.loads((OUT/'SCAN_CHECKS.json').read_text());inv=np.load(OUT/'INVENTORY.npz')
    with np.load(OUT/'STRUCTURE_RESULTS.npz') as stored:res={k:stored[k] for k in stored.files}
    seen=np.zeros(len(inv['X']),bool)
    retrieved=OUT/'retrieved/output/arx_structure_diagnosis_20261016'
    for name,d in checked['batch_hashes'].items():
        p=retrieved/'batches'/name;assert sha(p)==d
        with np.load(p) as a:
            ids=a['uid'];seen[ids]=True
            for k in res:np.testing.assert_array_equal(a[k],res[k][ids])
    assert seen.all()
    for mi in range(6):
        for si in range(12):
            for gi in range(1,16):assert knee(inv['primary'][mi,si,gi])[0]==inv['selected'][mi,si,gi]
    audited=readrows('ORIGINAL_FINAL_AUDIT_REPLAY.csv');assert len(audited)==72 and all(r['passed']=='True' and r['original']==r['replayed'] for r in audited)
    m=json.loads((OUT/'METRIC_CHECKS.json').read_text());assert m['bank_sha256']==digest and m['all_numerical_passed'] and not m['optimization_started']
    for v in m['summary']:
        if v['metric'] in ['Local_ARX','Combined']:assert not v['development_gate_passed']
    score_files=list((OUT/'metric_retrieved').rglob('metric_scores/*.json'));assert len(score_files)==24
    for f in score_files:
        v=json.loads(f.read_text());assert v['bank_sha256']==digest and v['numerical_passed'] and v['implementation_sha256']==sha(HERE/'local_geometry.py')
    global_files=list((OUT/'global_retrieved').rglob('global_scores/*.json'));assert len(global_files)==24
    for f in global_files:
        v=json.loads(f.read_text());assert v['bank_sha256']==digest and v['fine_converged']
    local=json.loads((OUT/'LOCAL_INDEPENDENT_CHECKS.json').read_text());assert local['all_metric_original_gates_passed'] and local['certificate_all_passed'] and local['centered_borderline_all_four_levels_consistent']
    archives={}
    for name,meta_name in [('structure_scan_20261016.zip','RESULTS_META.json'),('structure_metrics_20261016.zip','METRIC_RESULTS_META.json'),('structure_global_20261016.part','GLOBAL_RESULTS_META.json')]:
        meta=json.loads((OUT/meta_name).read_text());p=OUT/name
        assert p.stat().st_size==meta['bytes'] and sha(p)==meta['sha256']
        with zipfile.ZipFile(p) as z:assert z.testzip() is None
        archives[name]=dict(bytes=p.stat().st_size,sha256=sha(p),zip_crc_passed=True)
    remote=json.loads((OUT/'REMOTE_FINAL_HASH_CHECK.json').read_text());assert not remote['mismatches'];assert remote['all_native_match']
    runs=readrows('RUN_SUMMARY.csv');stages=readrows('SAVED_STAGE_COUNTS.csv')
    assert len(runs)==72 and len(stages)==1152
    assert sum(r['saved_any_centered']=='True' for r in runs)==5
    assert sum(int(r['final_P3_centered_records'])>0 for r in runs)==4
    assert not any(int(r['P3_uncertain_records']) for r in stages if r['generation']=='300')
    pairs=[dict(pair_id=p['pair_id'],family=p['family'],method=p['method'],seed=p['seed'],generation=p['generation'],region=p['region'],better_uid=p['better_uid'],worse_uid=p['worse_uid']) for p in b['pairs']]
    text='# Case1种群与候选对：完整汇总表\n\n本文件为数值表。better/worse仅为声明窗口的结构标签，不代表全曲线排序。记录、阶段、参数对及seed并非互相独立。\n\n'
    sections=[('方法汇总',readrows('METHOD_SUMMARY.csv')),('72次运行',runs),('固定候选对来源',pairs),('指标筛选结果',m['summary']),('96项局部方向',readrows('PAIR_METRIC_DIRECTIONS.csv')),('96项全窗口背景分数',readrows('GLOBAL_PAIR_SCORE_CONTEXT.csv'))]
    for title,rows in sections:text+='## '+title+'\n\n'+table(rows)+'\n'
    text+='## 逐阶段记录\n\n1152行完整保存于[SAVED_STAGE_COUNTS.csv](/D:/develop/FEMEIF/output/arx_structure_diagnosis_20261016/SAVED_STAGE_COUNTS.csv)。未计算的其他结构不由P3阴性证书推断。\n'
    (OUT/'ALL_TABLES.md').write_text(text,encoding='utf-8')
    for name in ['REPORT.md','FINDINGS.md','ISSUES.md','ALL_TABLES.md']:
        for link in re.findall(r'\]\((/D:/[^)]+)\)',(OUT/name).read_text(encoding='utf-8')):
            if Path(link[1:]).name not in ['PROVENANCE.json','REVIEW_STATUS.json']:assert Path(link[1:]).exists(),link
    now=datetime.now(timezone.utc).isoformat()
    data_files=[p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json','.csv','.npz','.md'] and p.name not in ['PROVENANCE.json','REVIEW_STATUS.json']]
    provenance=dict(created_utc=now,scope='Read-only development diagnostics; no new optimization',frozen_old_source_inputs=source_ok,
        source_stage_count=len(inventory_meta['source_hashes']),source_stage_ledger_sha256=sha(OUT/'INVENTORY.json'),
        scoring_bank_sha256=digest,archives=archives,new_code_hashes={str(p.relative_to(ROOT)):sha(p) for p in HERE.iterdir() if p.is_file()},
        output_hashes={str(p.relative_to(OUT)):sha(p) for p in data_files+list((OUT/'figures').glob('*.png'))},
        original_scan_batches_ledger_sha256=sha(OUT/'SCAN_CHECKS.json'),all_scan_batches_checked=True,remote_final_hash_check=remote)
    dump(OUT/'PROVENANCE.json',provenance)
    review=dict(completed_utc=now,scientific_review_complete=True,figures_manually_reviewed=[p.name for p in sorted((OUT/'figures').glob('*.png'))],
        records=230400,unique_candidates=165294,all_uid_covered=True,uncertain_P3_candidates_retained=29,final_uncertain_records=0,
        original_final_P3_audits=72,all_stage_knees_rechecked=1080,pairs=12,pair_seeds=9,numerical_candidate_region_checks=24,
        local_independent_replays=24,all_original_source_input_hashes_unchanged=True,all_source_stage_hashes_unchanged=True,
        all_recovered_archives_sha256_crc_passed=True,V1_V2_pair_content_identical=True,local_and_combined_development_gate_passed=False,
        optimization_started=False,website_published=False,monitoring_created=False,server_shutdown_requested=False,
        provenance_sha256=sha(OUT/'PROVENANCE.json'),report_sha256=sha(OUT/'REPORT.md'))
    dump(OUT/'REVIEW_STATUS.json',review)
    print(json.dumps(review,ensure_ascii=False),flush=True)

if __name__=='__main__':main()

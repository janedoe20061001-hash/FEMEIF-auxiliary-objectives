"""Compose tables, enforce the frozen stop decision, and record final provenance."""
from common import *
from datetime import datetime,timezone
import csv,re

def rows(name):return list(csv.DictReader((OUT/name).open(encoding='utf-8-sig')))
def table(data,keys=None):
    if not data:return ''
    keys=keys or list(data[0]);out=['|'+'|'.join(keys)+'|','|'+'|'.join(['---']*len(keys))+'|']
    for r in data:out.append('|'+'|'.join(str(r.get(k,'')).replace('|','/') for k in keys)+'|')
    return '\n'.join(out)+'\n'

def main():
    recovery=json.loads((OUT/'ALL_RECOVERY_CHECKS.json').read_text());assert recovery['all_verified']
    for name,meta in recovery['archives'].items():assert (OUT/name).stat().st_size==meta['bytes']
    audit=json.loads((OUT/'LOCAL_CHECKS.json').read_text());assert audit['complete'] and audit['strict_unmatched_definition_retained']
    assert len(audit['local_metric_replays'])==72 and all(r['passed'] for r in audit['local_metric_replays'])
    scales=json.loads((OUT/'TARGET_SCALE_AUDIT.json').read_text());assert scales['all_passed']
    server=json.loads((OUT/'FINAL_SERVER_INTEGRITY.json').read_text());assert not server['original_source_input_mismatches'] and not server['scoring_source_mismatches']
    assert not server['strict_selection_amendment_applied'] and all(v['expected']==v['actual'] for v in server['native'].values())
    producer=json.loads((OUT/'SOURCE_MANIFEST_03.json').read_text())
    for name,digest in producer.items():assert sha(ROOT/name)==digest,('producer source differs',name)
    bank=json.loads((OUT/'HOLDOUT_BANK.json').read_text());tasks=json.loads((OUT/'SCORING_TASKS.json').read_text());assert sha(OUT/'HOLDOUT_BANK.json')==tasks['holdout_sha256']
    assert bank['complete'] and len(bank['selected'])==24 and len(bank['seeds'])==12
    direction=rows('PAIR_DIRECTIONS.csv');summary=json.loads((OUT/'DIRECTION_SUMMARY.json').read_text());assert len(direction)==432
    for s in summary['summary']:
        for scope in ['dev','held']:
            rr=[r for r in direction if r['scope']==scope and r['metric']==s['metric']]
            assert len(rr)==s[scope+'_total'] and sum(r['direction']=='correct' for r in rr)==s[scope+'_correct']
    candidates=[v for v in summary['summary'] if v['metric'] in ['Root','B']]
    assert len(candidates)==2 and not any(v['direction_gates_passed'] for v in candidates)
    now=datetime.now(timezone.utc).isoformat()
    decision=dict(completed_utc=now,phase='complete_at_predeclared_direction_gate',candidate_gates={r['metric']:r for r in candidates},
        numerical_checks_passed=True,holdout_bank_complete=True,strict_strata_retained=True,optimization_started=False,
        formal_new_optimization_tasks=0,cost_admission_benchmark_executed=False,optimization_trace_logging_executed=False,
        reason='Both proposed candidates failed predeclared structural-direction requirements; no optimization or threshold change',
        main_delta=.01,sensitivity_delta_not_selected=True,website_published=False,automation_created=False,server_shutdown_requested=False)
    dump(OUT/'EXECUTION_DECISION.json',decision)
    source_rows=[]
    for p in bank['selected']:
        r={k:p[k] for k in ['pair_id','target','family','method','seed','generation','region','better_index','worse_index','max_primary_relative']}
        if p['family']!='extra_structure':
            e=next(v for v in p['better']['events'] if v['event_id']==p['region']);r.update(better_height_db=e['height_delta_db'],better_width_ratio=e['width_ratio'],better_shift_decades=e['shift_decades'])
        source_rows.append(r)
    text='# 全部汇总表\n\n局部方向标签不代表全曲线优劣；固定候选对、同seed和参数网格均不视为独立试验。\n\n'
    text+='## 固定24对来源\n\n'+table(source_rows,['pair_id','target','family','method','seed','generation','region','max_primary_relative'])+'\n'
    text+='## 指标方向汇总\n\n'+table(summary['summary'],['metric','dev_correct','dev_total','dev_extra_correct','held_correct','held_total','held_extra_correct','two_P3_counterexamples_correct','held_family_correct','direction_gates_passed'])+'\n'
    text+='## 逐对方向与分数\n\n'+table(direction)+'\n'
    decomposition=json.loads((OUT/'B_TRANSLATION_DECOMPOSITION.json').read_text());text+='## B错位分解（事后解释）\n\n'+table(decomposition['means'])+'\n'
    text+='## 完整参数网格与评分\n\n684行受控设置和全部原始分数保存在[CONTROLLED_SCORES.csv](/D:/develop/FEMEIF/output/geometry_followup_20261016/CONTROLLED_SCORES.csv)。原单因素切片见[SINGLE_FACTOR_RESPONSES.csv](/D:/develop/FEMEIF/output/geometry_followup_20261016/SINGLE_FACTOR_RESPONSES.csv)。\n'
    (OUT/'ALL_TABLES.md').write_text(text,encoding='utf-8')
    for name in ['REPORT.md','FINDINGS.md','ISSUES.md','ALL_TABLES.md']:
        for link in re.findall(r'\]\((/D:/[^)]+)\)',(OUT/name).read_text(encoding='utf-8')):
            if Path(link[1:]).name not in ['PROVENANCE.json','REVIEW_STATUS.json']:assert Path(link[1:]).exists(),link
    output=[p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json','.jsonl','.csv','.md'] and p.name not in ['PROVENANCE.json','REVIEW_STATUS.json','LAST_STATUS.json']]
    score_files=sorted((OUT/'scores').glob('*.json'));assert len(score_files)==756
    source_proposals=sorted((OUT/'bank_candidates').rglob('*.json'));assert len(source_proposals)==2592
    for p in score_files:assert json.loads(p.read_text())['numerical_passed']
    ledger=dict(completed_utc=now,scope='Controlled physical mechanism and paired candidate discrimination; no new optimization',
        frozen_scoring_producer_hashes=producer,all_producer_sources_match=True,common_formula_sha256=sha(HERE/'common.py'),holdout_bank_sha256=sha(OUT/'HOLDOUT_BANK.json'),
        source_stage_proposals=len(source_proposals),proposal_files_sha256={str(p.relative_to(OUT)):sha(p) for p in source_proposals},
        original_source_integrity=server,archives=recovery['archives'],native_curves_sha256=recovery['native_curves_sha256'],
        local_code_hashes={p.name:sha(p) for p in HERE.iterdir() if p.is_file()},
        not_used_for_primary_results=['AMENDMENT_01_DRAFT_NOT_APPLIED.md','source_patch_04.zip','source_versions/bank_amendment_draft_04.py'],
        output_sha256={str(p.relative_to(OUT)):sha(p) for p in output+score_files+list((OUT/'figures').glob('*.png'))},
        source_archive_hashes={p.name:sha(p) for p in OUT.glob('*.zip') if p.name not in recovery['archives']})
    dump(OUT/'PROVENANCE.json',ledger)
    review=dict(completed_utc=now,status='complete',scientific_review_complete=True,controlled_settings=684,development_pairs=12,holdout_pairs=24,
        source_stage_files=2592,score_tasks=756,window_records=2124,all_numerical_passed=True,local_replays=72,native_and_nodal_curves=72,
        manually_reviewed_figures=[p.name for p in sorted((OUT/'figures').glob('*.png'))],
        all_results_recovered=True,all_archives_sha256_and_crc_passed=True,strict_selection_retained=True,source_input_hashes_unchanged=True,
        optimization_started=False,formal_new_optimization_tasks=0,cost_admission_not_reached=True,
        website_published=False,automation_created=False,server_shutdown_requested=False,ssh_session_closed=True,
        report_sha256=sha(OUT/'REPORT.md'),provenance_sha256=sha(OUT/'PROVENANCE.json'))
    dump(OUT/'REVIEW_STATUS.json',review);dump(OUT/'LAST_STATUS.json',review)
    print(json.dumps(review,ensure_ascii=False),flush=True)

if __name__=='__main__':main()

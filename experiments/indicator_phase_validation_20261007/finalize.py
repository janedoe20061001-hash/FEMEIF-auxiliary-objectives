"""Add a result-led reading guide without modifying the archived server analysis."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'output/indicator_phase_validation_20261007'
SAVED=HERE/'server_run/retrieved'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
s=json.loads((OUT/'SUMMARY.json').read_text())
manifest=json.loads((SAVED/'runs/manifest.json').read_text())
for p in ('run.py','analyze.py','PROTOCOL.md'):
    assert sha(HERE/p)==sha(SAVED/p)
for p,h in manifest['frozen_sha256'].items(): assert sha(HERE/p)==h
intro=['# 相位斜率指标：服务器优化验证结果（2026-10-07）','',
       '**结论：这轮结果不支持用 D_phi 替换 II-02。** 在固定预算和固定选解规则下，候选的综合拟合误差在两个 Case 上都更高；相位误差没有可靠改善。','',
       '| 候选相对原 I-03＋II-02 | Case1 | Case2 |','|---|---:|---:|']
for metric,label in [('wsmse','平均综合误差变化'),('magnitude_mse','平均幅值 MSE 变化'),('phase_mse','平均相位 MSE 变化')]:
    cc=[next(c for c in s['comparisons'] if c['case']==case and c['metric']==metric and c['treatment']=='signed_phase_pair' and c['reference']=='old_pair') for case in ('case1','case2')]
    intro.append(f"| {label} | {cc[0]['relative_change_of_means_pct']:+.1f}% | {cc[1]['relative_change_of_means_pct']:+.1f}% |")
intro+=['','综合误差的配对均值差区间在两个 Case 均高于零，Case2 区间下界接近零，证据较弱。相位 MSE 的区间均跨零，不能把均值增加直接说成已经证实相位退步。局部最差频带误差也未显示可靠收益。','']
case2={m['group']:m for m in s['means'] if m['case']=='case2'}
change=100*(case2['signed_phase_pair']['D_phi']/case2['old_pair']['D_phi']-1)
intro += [f"一个直接的指标设计警示：Case2 候选组的 D_phi 均值下降了 {abs(change):.1f}%，但综合误差增加了 32.9%。这说明降低带方向相位导数误差，仍不能保证按现有规则选出的解更接近目标。它支持把该指标暂留为诊断项，不支持直接作为 II-02 的替代目标。",'',
          '本轮只有两个电路、每组 20 个种子，结论限于本预算与本选解方法；不能外推为该方向在所有电路上无效。关于损失原因的解释仍属假设，未做独立因果实验。','',
          '以下是服务器生成的详细统计与曲线。上面的解读是在计算完成后添加的，没有修改预登记方案、公式、种子或原始统计。','']
original=(SAVED/'analysis/REPORT.md').read_text(encoding='utf-8').split('\n',1)[1]
(OUT/'REPORT.md').write_text('\n'.join(intro)+original,encoding='utf-8')
(OUT/'SERVER_CHECK.json').write_bytes((SAVED/'SERVER_CHECK.json').read_bytes())
integrity=dict(server_archive=json.loads((HERE/'server_run/ARCHIVES.json').read_text()),
    local_matches_server_protocol_and_code=True,local_matches_all_frozen_source_hashes=True,
    artifacts_sha256={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='INTEGRITY.json'})
(OUT/'INTEGRITY.json').write_text(json.dumps(integrity,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(local_source_match=True,artifact_count=len(integrity['artifacts_sha256']))))

from common import *
from summarize import csvread
import re,subprocess
assert json.loads((OUT/'AUDIT.json').read_text())['passed']
assert json.loads((OUT/'FULL_POPULATION_VALIDATION.json').read_text())['passed']
full=csvread(OUT/'analysis/FULL_POPULATION_STRUCTURE.csv');front=csvread(OUT/'analysis/FRONT.csv');by={(float(r['scale']),int(r['seed']),int(r['index'])):r for r in full}
for r in front:
    b=by[(float(r['scale']),int(r['seed']),int(r['index']))];assert b['is_front']=='True'
    for k in ['P3_complete','V2_complete','joint_complete']:assert b[k]==r[k]
representatives=csvread(OUT/'analysis/REPRESENTATIVES.csv')
for r in representatives:
    if int(r['generation'])!=300:continue
    s=float(r['scale']);seed=int(r['seed'])
    with np.load(HERE/'batch'/label(s)/f'seed_{seed}'/'generation_300.npz') as data:ix=int(data['selected_index'] if r['rule']=='min_wsmse' else data['balanced_index'])
    b=by[(s,seed,ix)]
    for k in ['P3_complete','V2_complete','joint_complete','P3_matched','V2_matched']:assert b[k]==r[k]
with (OUT/'FINALIZE_LOG.txt').open('a',encoding='utf8') as log:
    for script in ['plots.py','report.py']:
        result=subprocess.run([sys.executable,str(HERE/script)],stdout=log,stderr=log);assert result.returncode==0
audit=json.loads((OUT/'AUDIT.json').read_text());audit.update(full_population_structure_candidates=4800,full_population_unstable=0,full_population_front_and_representatives_consistent=True,scoring_probes_passed=True);dump(OUT/'AUDIT.json',audit)
p=OUT/'AUDIT.md';text=p.read_text(encoding='utf8');extra='\n## 末代完整种群补查\n\n300代所有24×200=4800个候选的P3/V2结构经过2/4档（必要时8档）密网格复核，分类稳定。前沿、两种末代代表的分类与既有诊断逐条一致。三档均没有组合合格的末代保留曲线；宽目标2个seed有P3合格，V2均未合格。未检查所有被淘汰的历史候选，不作“搜索全过程从未出现正确解”的断言。\n\n42条补充评分诊断采用2/4/8档独立积分；4/8档相对收敛要求0.1%，绝对下限MAG 1e-8、PHA 1e-10。自适应评分最大相对差约0.205%/0.194%；全部参数在原范围内。这是探索性机制诊断，非新随机优化。\n'
if '## 末代完整种群补查' not in text:p.write_text(text+extra,encoding='utf8')
for doc in ['第二步实验方案_困难条件与本地运行.md','MSE困难结构与辅助指标_研究大纲.md']:
    p=HERE.parents[1]/'docs'/doc;text=p.read_text(encoding='utf8');marker='## 2026年10月9日：第二步首批探索已完成'
    if marker not in text:p.write_text(text+'\n\n'+marker+'\n\n已有100seed结果分析、六档目标核验、三档×8seed×300代本地探索与末代4800候选补查已完成。通俗报告：[第二步结果](D:/develop/FEMEIF/output/mse_structure_step2_20261009/REPORT.md)。\n\n当前支持的机制线索是极窄峰同程度高度/宽度失真对整体MAG/PHA增量较小；这轮三档代表都未恢复好峰谷组合，不能宣布纯峰宽导致优化难度的规律。下一次优先考虑3个未知参数与18个未知参数的搜索条件诊断，再验证针对性辅助指标。报告只在本地保存。\n',encoding='utf8')
manifest={}
for p in HERE.rglob('*'):
    if p.is_file() and not any(x in p.parts for x in ['.venv','__pycache__','smoke','timing']):manifest[p.relative_to(HERE).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
dump(OUT/'SOURCE_MANIFEST.json',manifest)
report=OUT/'REPORT.md';text=report.read_text(encoding='utf8');paths=[path for path in re.findall(r'\]\(([^)]+)\)',text) if path.startswith(('D:/','C:/'))];bad=[p for p in paths if not Path(p).exists()];assert not bad,bad
dump(OUT/'DELIVERY_CHECK.json',dict(passed=True,report_bytes=report.stat().st_size,local_links_checked=len(paths),missing_links=bad,report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),full_population_diagnostics_included=True))
print('Final report and local links verified')

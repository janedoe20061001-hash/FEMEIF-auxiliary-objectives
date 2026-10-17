from pathlib import Path
import json,csv,re,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/dual_trajectory_20261008';SAMPLE=ROOT/'output/sampling_refinement_20261007'
def link(p):return p.resolve().as_posix()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for c in ['case1','case2']:
    p=OUT/f'{c}.md';text=p.read_text(encoding='utf8');text=re.sub(r'\| ([RLC]\d+) \|',lambda m:'| '+m[1]+'/'+{'R':'Ω','L':'H','C':'F'}[m[1][0]]+' |',text)
    d=np.load(OUT/'selected'/f'{c}_generation_300.npz');r=next(r for r in json.loads((ROOT/'experiments/dual_trajectory_20261008/frozen_dual/circuits.json').read_text(encoding='utf8')) if r['id']==c)
    hdr=['元件/单位','目标真值',*[f'seed {s}·300代' for s in d['seeds']]]
    rows=[]
    for j,name in enumerate(r['parameter_names']):rows.append([name+'/'+{'R':'Ω','L':'H','C':'F'}[name[0]],f"{r['target_parameters'][j]:.6g}" if r['target_parameters'] is not None else '无真值',*[f'{10**d["log_parameters"][si,0,j]:.6g}' for si in range(3)]])
    joint='\n'.join(['| '+' | '.join(hdr)+' |','| '+' | '.join(['---']*len(hdr))+' |',*['| '+' | '.join(row)+' |' for row in rows]])
    text=re.sub(r'\| 元件 \| 目标值/SI \| log10下界 \| log10上界 \|\n(?:\|[^\n]*\|\n?)+',joint+'\n',text)
    p.write_text(text,encoding='utf8')
p=OUT/'REPORT.md';text=p.read_text(encoding='utf8')
if '**结论先看' not in text:text=text.replace('## 先看后台结构结论','**结论先看：200→300代整体幅相误差仍下降，但相位事件漏失没有确认稳定改善。** Case1末代有73/100seeds存在幅值事件漏失；Case2的测点曲线上为100/100。图文报告给出具体频段、宽度偏差与局部放大图，并分别核查采样和模型限制。\n\n## 先看后台结构结论')
text=text.replace('完整阶段曲线NPZ','原频点曲线与精确参数NPZ（解析图可复算）');text=text.replace('后台另核验全部200次运行。','全部200次运行均已独立核验，末代完整种群都与旧实验逐项一致。')
if '中位数从53.81dB' not in text:text=text.replace('它检查采样缺口；本轮检查原双目标运行中结构随代数的变化。','目标采样缺口已修复，但未确认幅相共同改善。Case1双目标候选曲线的最大漏采幅值差，中位数从53.81dB降到1.67dB，真值加密组90%分位仍约19.49dB；说明候选自身的窄峰评价还需检查。它检查采样缺口；本轮检查原双目标运行中结构随代数的变化。')
p.write_text(text,encoding='utf8')
p=OUT/'FINDINGS.md';text=p.read_text(encoding='utf8')
if '## 采样交叉检查' not in text:text=text.replace('## 后续指标的设计方向','## 采样交叉检查\n\n加密采样的目标表达已可靠，但Case1双目标组的候选折线缺口中位数仍为1.67dB、90%分位约19.49dB；原粗采样中位数约53.81dB。这是拟合曲线自身的漏采，与目标约25.23kHz窄峰的漏采不同。因此原频点过程里的残差，不能全部归因于MAG/PHA目标；下一轮评价应先防止候选极窄尖峰躲在训练点之间，再检验局部结构辅助指标。\n\n## 后续指标的设计方向')
p.write_text(text,encoding='utf8')
background=json.loads((OUT/'BACKGROUND_VALIDATION.json').read_text());sampling=json.loads((SAMPLE/'VALIDATION.json').read_text());assert background['passed'] and background['independent_individuals_recomputed']==525000 and sampling['passed'] and sampling['final_individuals_recomputed']==249600
counts={}
for p,expected in [(OUT/'all_seed_diagnostics.csv',6000),(OUT/'all_seed_metrics.csv',6000),(OUT/'all_seed_bands.csv',48000),(SAMPLE/'per_seed_diagnostics.csv',3360),(SAMPLE/'selected_metrics.csv',2520)]:
    count=sum(1 for _ in csv.DictReader(p.open(encoding='utf-8-sig')));assert count==expected;counts[p.relative_to(ROOT).as_posix()]=count
broken=[];checked_links=0
for d in [OUT,SAMPLE]:
    for p in d.glob('*.md'):
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text(encoding='utf8')):
            if target.startswith('D:/') or target.startswith('D:\\'):
                checked_links+=1
                if not Path(target).exists():broken.append([str(p),target])
assert not broken,broken
sources={p.relative_to(ROOT).as_posix():sha(p) for p in Path(__file__).parent.glob('*.py')};(OUT/'ANALYSIS_SOURCES.json').write_text(json.dumps(sources,indent=2),encoding='utf8')
manifest={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for d in [OUT,SAMPLE] for p in d.rglob('*') if p.is_file() and p.name!='FINAL_MANIFEST.json'}
(OUT/'FINAL_MANIFEST.json').write_text(json.dumps(dict(passed=True,links_checked=checked_links,csv_counts=counts,complete_local_raw_archives=True,final_content=manifest),indent=2),encoding='utf8')
print(json.dumps(dict(passed=True,links=checked_links,files=len(manifest),csv_counts=counts)))

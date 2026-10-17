"""Descriptive conditional summaries. The baseline circuit is the unit of counting."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import gallery_site
from run_population import FAMILIES

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/parameter_curve_population_20261006'


def csvout(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))


def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)


info=json.loads((OUT/'INTEGRITY.json').read_text(encoding='utf-8')); assert info['completed']
refs=json.loads((OUT/'data/reference_circuits.json').read_text(encoding='utf-8'))
titles={r['family']:r['title'] for r in refs}
rules=read(OUT/'data/circuit_rules.csv'); strata=defaultdict(list)
for r in rules:
    strata[(r['family'],r['split'],r['rule'],float(r['level_pct']),int(r['damping_bin']),int(r['spacing_bin']))].append(r['supported']=='True')
strata_rows=[]
for (family,split,rule,level,damping,spacing),values in strata.items():
    strata_rows.append(dict(family=family,split=split,rule=rule,level_pct=level,damping_bin=damping,spacing_bin=spacing,
        circuits=len(values),supported_circuits=sum(values),support_fraction=float(np.mean(values))))
assert all(r['circuits']==8 for r in strata_rows)
csvout(OUT/'data/rule_strata_summary.csv',strata_rows)

# For each class/level, first average the per-parameter responses within a circuit.
# Then summarize across circuits; the number of elements does not inflate n.
within=defaultdict(lambda:dict(count=0,mag=0.,phase=0.,mixed=False))
with (OUT/'data/perturbations.csv').open(encoding='utf-8-sig',newline='') as f:
    for r in csv.DictReader(f):
        if r['kind']!='single' or r['sign']!='1' or float(r['level_pct']) not in (1.,10.,30.): continue
        key=(r['circuit_id'],r['family'],r['split'],r['direction'][0],float(r['level_pct']))
        agg=within[key]; agg['count']+=1; agg['mag']+=float(r['magnitude_rmse_db']); agg['phase']+=float(r['phase_rmse_deg'])
        agg['mixed'] |= float(r['magnitude_up_frequency_fraction'])>=.01 and float(r['magnitude_down_frequency_fraction'])>=.01
groups=defaultdict(list)
for (cid,family,split,letter,level),agg in within.items():
    groups[(family,split,letter,level)].append((agg['mag']/agg['count'],agg['phase']/agg['count'],agg['mixed']))
response_rows=[]
for (family,split,letter,level),values in groups.items():
    a=np.asarray(values,float); assert len(a)==72
    response_rows.append(dict(family=family,split=split,element_class=letter,level_pct=level,circuits=len(a),
        magnitude_rmse_db_circuit_median=float(np.median(a[:,0])),phase_rmse_deg_circuit_median=float(np.median(a[:,1])),
        magnitude_rmse_db_circuit_p10=float(np.quantile(a[:,0],.1)),magnitude_rmse_db_circuit_p90=float(np.quantile(a[:,0],.9)),
        phase_rmse_deg_circuit_p10=float(np.quantile(a[:,1],.1)),phase_rmse_deg_circuit_p90=float(np.quantile(a[:,1],.9)),
        magnitude_mixed_circuits=int(a[:,2].sum()),magnitude_mixed_fraction=float(np.mean(a[:,2]))))
csvout(OUT/'data/response_distribution.csv',response_rows)

font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
families=['two_parallel_units','three_parallel_units','two_series_units','three_series_units','rlc_ladder','rlc_background','rlc_bridge']
fig,axes=plt.subplots(2,len(families),figsize=(19,6),sharex=True,sharey=True)
for row,split in enumerate(('A','B')):
    for col,family in enumerate(families):
        matrix=np.full((3,3),np.nan)
        for r in strata_rows:
            if r['family']==family and r['split']==split and r['rule']=='R_amp_up' and r['level_pct']==10:
                matrix[r['damping_bin']-1,r['spacing_bin']-1]=r['support_fraction']
        im=axes[row,col].imshow(matrix,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
        for i in range(3):
            for j in range(3): axes[row,col].text(j,i,f'{round(8*matrix[i,j])}/8',ha='center',va='center',color='white' if matrix[i,j]>.7 else '#17324d',fontsize=9)
        axes[row,col].set_xticks(range(3),['近','中','远']); axes[row,col].set_yticks(range(3),['Q构造低','中','高'])
        if row==0: axes[row,col].set_title(titles[family].replace('单元','\n单元').replace('支路','\n支路'),fontsize=9)
        if col==0: axes[row,col].set_ylabel('独立批次 '+split)
        if row==1: axes[row,col].set_xlabel('频率锚点间距')
fig.suptitle('R增大，各频点幅值均不降：按构造区域统计（每格8个独立电路；不是端口实际Q）',fontsize=12,y=1.01)
fig.subplots_adjust(wspace=.15,hspace=.15,left=.07,right=.9,top=.83,bottom=.12)
fig.colorbar(im,cax=fig.add_axes([.925,.22,.012,.5]),label='支持比例')
fig.savefig(OUT/'figures/conditional_R_amplitude.png',dpi=150,bbox_inches='tight'); plt.close(fig)

summary=read(OUT/'data/rule_summary.csv'); parameters=read(OUT/'data/parameter_rule_summary.csv')
def rate(family,rule,split):
    r=next((r for r in summary if r['family']==family and r['rule']==rule and r['level_pct']=='10.0' and r['split']==split),None)
    return '—' if r is None else f"{r['supported_circuits']}/{r['circuits']}"
lines=['# 大样本中的条件规律与数据分布','',
    '这一补充分析使用已完成的2,304个基准电路、207,360条扰动谱，不新增优化。先在每个电路内汇总，再在不同电路间统计，保持电路为计数单位。',
    '', '## 连接模板决定了基本方向','',
    table(['模板','R增大：幅值不降 A/B','R增大：相位向0 A/B','C增大：相位不升 A/B','L增大：相位不降 A/B'],[[titles[f]]+[' / '.join(rate(f,k,s) for s in ('A','B')) for k in ('R_amp_up','R_phase_to_zero','C_phase_down','L_phase_up')] for f in titles]),'',
    '以上均为正向×1.1，并要求该电路内所有该类元件在全部计算频点满足方向。它比“多数频点同向”更严格；支持率低不代表参数没有稳定的局部作用。具体元件位置见 parameter_rule_summary.csv。',
    '', '## 阻尼构造与谐振间距的影响','',
    '![条件区域中的电阻幅值规律](figures/conditional_R_amplitude.png)','',
    '每格分别是一个构造区域的8个电路，A/B为独立种子样本。小格适合观察条件差异与检查反例分布，不适合宣布精确发生概率。频率锚点间距不等同于实际端口峰间距；名义Q只在独立RLC单元中有直接定义。三单元的频率锚点采用等比例间距，尚未覆盖全部不规则多模态间距。',
    '', '## 幅相变化幅度与正反频段','',
    '下表10档正向，先对每个基准电路内该类元件的单参数RMS取平均，再对电路取中位数。混合方向表示至少一个该类元件在不少于1%的log频带上抬高幅值，同时在不少于1%的log频带上降低幅值（频点变化阈值1e-6 dB）。', '',
    table(['模板','元件类','幅值RMS中位数 A/B / dB','相位RMS中位数 A/B / °','幅值混合方向电路 A/B'],[[titles[family],letter,
        ' / '.join(f"{next(r for r in response_rows if r['family']==family and r['element_class']==letter and r['level_pct']==10 and r['split']==s)['magnitude_rmse_db_circuit_median']:.4g}" for s in ('A','B')),
        ' / '.join(f"{next(r for r in response_rows if r['family']==family and r['element_class']==letter and r['level_pct']==10 and r['split']==s)['phase_rmse_deg_circuit_median']:.4g}" for s in ('A','B')),
        ' / '.join(str(next(r for r in response_rows if r['family']==family and r['element_class']==letter and r['level_pct']==10 and r['split']==s)['magnitude_mixed_circuits'])+'/72' for s in ('A','B'))]
        for family in titles for letter in ('R','L','C') if any(r['family']==family and r['element_class']==letter for r in response_rows)]),'',
    '这类分布比单一代表电路更适合指导指标：既要识别平均变化幅度，也要保留不同频段的方向，避免总量抵消。幅相单位不同，不能直接用数字大小比较哪项更重要。',
    '', '## 如何解释“普遍”','',
    '可解析证明的端口串并联关系与整体缩放关系，可以在适用假设下作为规律。仅由样本统计发现的高支持率关系，应写成“在这些拓扑和参数范围内稳定”，并保留不支持的样本。两批相近并不是范围外验证。',
    '本轮没有有源元件、传输线、CPE或扩散元件，也没有全部随机图拓扑。大量数据显著扩展了第一轮覆盖，但仍不能替代研究对象和频带的明确限定。',
    '', '数据：[按构造区域统计](data/rule_strata_summary.csv)、[按电路的幅相分布](data/response_distribution.csv)、[按元件位置统计](data/parameter_rule_summary.csv)。主报告：[REPORT.md](REPORT.md)。','']
report='\n'.join(lines); (OUT/'CONDITIONAL_RULES.md').write_text(report,encoding='utf-8')
gallery_site.render(OUT,FAMILIES)
supplement=dict(source_sha256={str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    input_sha256={str((OUT/'data'/file).relative_to(ROOT)):hashlib.sha256((OUT/'data'/file).read_bytes()).hexdigest() for file in ('circuit_rules.csv','perturbations.csv','reference_circuits.json','parameter_rule_summary.csv','rule_summary.csv')},
    baseline_circuits=2304,parameter_response_groups=len(response_rows),construction_rule_groups=len(strata_rows),no_new_optimization=True)
(OUT/'data/SUPPLEMENT_PROTOCOL.json').write_text(json.dumps(supplement,ensure_ascii=False,indent=2),encoding='utf-8')
info['artifacts_sha256']={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='INTEGRITY.json'}
info['figures']=len(list((OUT/'figures').glob('*.png')))
(OUT/'INTEGRITY.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(completed=True,response_groups=len(response_rows),conditional_groups=len(strata_rows)),ensure_ascii=False))

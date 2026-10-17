"""Explain the three teacher questions using independently audited stored data."""
from pathlib import Path
import sys,json,csv,importlib.util,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/primary_loss_comparison_20261007';A=OUT/'audit'
r=json.loads((A/'AUDIT.json').read_text());assert r['passed']
records=json.loads((ROOT/'experiments/primary_loss_comparison_20261007/circuits.json').read_text(encoding='utf-8'))
def load_module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
# Independent target conversion and synthetic target provenance.
sys.path.insert(0,str(ROOT/'experiments/primary_loss_comparison_20261007'))
from model import response
c1=records[0];f=np.asarray(c1['frequency_hz']);target=np.asarray(c1['target_real'])+1j*np.asarray(c1['target_imag'])
p=np.asarray(c1['target_parameters']).reshape(6,3);s=2j*np.pi*f
zt=np.sum(p[:,0,None]*s*p[:,1,None]/(p[:,0,None]+s*p[:,1,None]+p[:,0,None]*s**2*p[:,1,None]*p[:,2,None]),axis=0)
np.testing.assert_allclose(zt,target,rtol=1e-12,atol=1e-9)
cfg=json.loads((ROOT/'experiments/primary_loss_comparison_20261007/frozen/case2/config.json').read_text());path=ROOT/'experiments/primary_loss_comparison_20261007/frozen/case2'/cfg['target_data']['file']
assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['target_data']['sha256']
d=np.genfromtxt(path,delimiter=',',names=True,encoding='utf-8-sig');c2=records[1]
np.testing.assert_array_equal(d['frequency'],c2['frequency_hz']);expected=d['mag']*(np.cos(d['pha']*np.pi/180)+1j*np.sin(d['pha']*np.pi/180))
np.testing.assert_allclose(expected,np.asarray(c2['target_real'])+1j*np.asarray(c2['target_imag']),rtol=1e-13,atol=1e-9)
# Check measured-case loss concentration per frequency, without changing samples.
contributions=[]
for ident in ['case1','case2']:
    with np.load(OUT/'selected'/f'{ident}.npz') as data:
        ff=data['frequency'];tt=data['target'];zz=data['curves'];aa=(20*np.log10(abs(zz)/abs(tt)))**2;pp=(np.angle(zz)-np.angle(tt))**2
        edges=np.linspace(np.log10(ff[0]),np.log10(ff[-1]),9)
        for gi,g in enumerate(['wsmse','mean_distance']):
            for k,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
                mask=(np.log10(ff)>=lo)&(np.log10(ff)<hi if k<7 else np.log10(ff)<=hi)
                contributions.append(dict(circuit=ident,group=g,band=k+1,lo_hz=float(10**lo),hi_hz=float(10**hi),points=int(mask.sum()),amplitude_mean_contribution=float(aa[:,gi,mask].sum()/len(ff)/len(zz)),phase_mean_contribution=float(pp[:,gi,mask].sum()/len(ff)/len(zz)),amplitude_total_share=float(aa[:,gi,mask].sum()/aa[:,gi].sum()),phase_total_share=float(pp[:,gi,mask].sum()/pp[:,gi].sum())))
with (A/'component_band_contributions.csv').open('w',encoding='utf-8-sig',newline='') as h:
    w=csv.DictWriter(h,fieldnames=list(contributions[0]));w.writeheader();w.writerows(contributions)
# Recheck curve-distance example off-grid with the previously stabilized grid.
sys.path.insert(0,str(ROOT/'experiments/optimized_curve_structure_20261007'))
from refined_grid import improved_dense_grid
with np.load(OUT/'selected/case1.npz') as data:fit=data['log_parameters'][0];orig=data['metrics'][0]
dense=[]
for refinement in [1,2]:
    ff=improved_dense_grid(c1,10**fit,refinement);tt=response(c1['tree'],c1['parameter_names'],ff,c1['target_parameters'])[0];zz=response(c1['tree'],c1['parameter_names'],ff,10**fit)
    x=np.log10(ff);values=np.trapezoid(abs(zz-tt),x,axis=1)/(x[-1]-x[0]);dense.append(dict(refinement=refinement,points=len(ff),wsmse_group_mean_distance_logweighted_ohm=float(values[0]),distance_group_mean_distance_logweighted_ohm=float(values[1])))
np.testing.assert_allclose([dense[0][k] for k in ['wsmse_group_mean_distance_logweighted_ohm','distance_group_mean_distance_logweighted_ohm']],[dense[1][k] for k in ['wsmse_group_mean_distance_logweighted_ohm','distance_group_mean_distance_logweighted_ohm']],rtol=.01)
r['target_provenance_independently_verified']=True;r['case2_all_201_rows_and_degree_to_radian_conversion_verified']=True;r['seed2300_dense_logfrequency_distance']=dense
r['numerical_tolerance_note']='Independent network arithmetic agreed at atol 1e-8 plus rtol 2e-7 for all final-population metrics; maximum relative difference uses a 1e-8 denominator floor and occurs for near-zero losses. Selected spectra maximum relative discrepancy is 1.70e-11. Scientific table values are reproduced.'
r['explanation_code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(A/'AUDIT.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
means={(x['circuit'],x['group']):x for x in json.loads((OUT/'SUMMARY.json').read_text(encoding='utf-8'))['means']}
def fmt(v):return f'{v:.6g}'
lines=['# 老师三项问题的独立核查','',
'核查覆盖全部2,320次单目标运行、300,400个末代个体，独立重写了电路求值与误差公式。逐seed选解、进化历史最小值、选中曲线、四项误差和100个seeds均值均通过核对；没有重新优化或更改原结果。','',
'## 1. 平均距离到底是什么','',
'它是同频点对应的原始复阻抗距离：D = (1/N) Σᵢ sqrt[(Re Zᵢ − Re Tᵢ)² + (Im Zᵢ − Im Tᵢ)²]，单位Ω。','',
'若写成坐标(fᵢ, Re Zᵢ, Im Zᵢ)，同一频点的频率差为零，所以得到相同结果。没有频率对齐、最近点搜索、曲线间最短距离，也没有归一化。这里称为“同频复阻抗平均距离”最清楚；仅说“三维曲线距离”容易引入其他定义。','',
'WSMSE=M_A+αM_φ，其中M_A=mean[(20 log₁₀(abs Z/abs T))²]，M_φ=mean[(arg Z−arg T)²]，相位使用rad主值，α运行前固定。D与WSMSE在坐标尺度与误差聚合上均不同。','',
'在本次300代对照的两个Case中，WSMSE组的幅值/相位MSE与多数结构诊断更好；距离组的原始Ω距离更好。不能省略评价标准后宣布一种目标普遍更好。','',
'## 2. 为什么两个Case的幅值、相位比例接近','',
'老师观察到的接近确实存在。距离组Case1/Case2的幅值MSE比为1.96514，相位MSE比为1.99033。这两个同分量的跨Case比值不受共同单位换算影响。也可考察每个Case的幅值MSE/相位MSE数值比，分别为86.4212和87.5290；后者跨越dB²与rad²，数值会随单位选择改变，不能作为幅值/相位贡献百分比。','',
'| 电路/距离组选解 | 幅值MSE / dB² | 相位MSE / rad² | 两列数值比 | 每seed数值比10%–90%范围 |','|---|---:|---:|---:|---|']
for c in ['case1','case2']:
    m=means[(c,'mean_distance')];st=next(x for x in r['amp_phase_statistics'] if x['circuit']==c and x['group']=='mean_distance')
    lines.append(f'| {c} | {fmt(m["magnitude_mse_db2"])} | {fmt(m["phase_mse_rad2"])} | {fmt(st["amp_over_phase_ratio"])} | {st["seed_ratio_p10"]:.2f}–{st["seed_ratio_p90"]:.2f} |')
lines+=['','两项误差分别从复阻抗的模长与atan2相位独立计算，没有按固定比例生成或复用另一Case的值。距离组逐seed的两项误差相关系数分别为0.948和0.963，说明它们在这些解中共同变化，但相关性不能证明均值比接近的机制。','',
'按另一种理解，把距离组/WSMSE组作为“恶化倍数”：Case1幅值5.556倍、相位4.871倍；Case2幅值5.094倍、相位3.917倍，并不是相同倍率。','',
'核查已排除结果复制、角度单位混用、错组选解与汇总错误；目前没有证明为何两个Case的均值比恰好接近，不能把“巧合”或某种普遍规律当作已证实原因。仅两个Case不足以建立这种规律。逐seed数据和逐频带平方误差贡献均可下载复算。','',
'## 3. 第三张图为什么看起来距离组更差，却算出距离更小','',
'第三张图是Case1/seed2300，上面汇总表是100个seeds均值。核查该图本身的同一seed：','',
'| seed2300选解 | 幅值MSE / dB² | 相位MSE / rad² | WSMSE | 同频平均距离 / Ω |','|---|---:|---:|---:|---:|']
for gi,g in enumerate(['WSMSE组','距离组']):lines.append('| '+g+' | '+' | '.join(fmt(v) for v in orig[gi])+' |')
lines+=['','原图的幅值纵轴是dB、相位纵轴是度数，无法直接展示实部/虚部的欧姆残差。距离组在高频结构上确实更差，但它在其余频带的绝对欧姆误差降低更多。180个原频点中，它在128个点的距离更小。分带贡献如下，每项均为“该频带距离之和/180”，所以四行可直接相加得到总D：','',
'| 原频带 | 频点数 | WSMSE组贡献 / Ω | 距离组贡献 / Ω |','|---|---:|---:|---:|']
for b in r['seed2300_bands']:lines.append(f'| {b["lo_hz"]:g}–{b["hi_hz"]:g} Hz | {b["points"]} | {b["wsmse_contribution_to_overall_mean_ohm"]:.6f} | {b["mean_distance_contribution_to_overall_mean_ohm"]:.6f} |')
lines+=['','低于100 kHz，WSMSE组贡献5.36159Ω，距离组0.508487Ω；100 kHz以上，WSMSE组1.30020Ω，距离组1.45307Ω。前面降低约4.85310Ω，后面增加约0.152872Ω，净降低4.70023Ω。','',
'| 具体频点 | 目标模长 / Ω | WSMSE组距离 / Ω | 距离组距离 / Ω |','|---|---:|---:|---:|']
for row in r['seed2300_largest_distance_advantages'][:3]:lines.append(f'| {row["frequency_hz"]:.3f} Hz | {row["target_magnitude_ohm"]:.6f} | {row["wsmse_distance_ohm"]:.6f} | {row["distance_distance_ohm"]:.6f} |')
lines+=['','![逐点距离与累计贡献](audit/case1_seed2300_distance_audit.png)','',
'为检查原180点采样是否改变这个seed的距离排序，对同一两个选中解，另外采用解析模型和两档加密网格检查log频率平均距离，得到下表。此积分与训练频点等权D是两个定义，不能混为同一列。','',
'| 诊断网格 | 点数 | WSMSE组log频率平均距离 / Ω | 距离组log频率平均距离 / Ω |','|---|---:|---:|---:|']
for row in dense:lines.append(f'| 加密{row["refinement"]} | {row["points"]} | {row["wsmse_group_mean_distance_logweighted_ohm"]:.6f} | {row["distance_group_mean_distance_logweighted_ohm"]:.6f} |')
lines+=['','两档复核后，这个seed的欧姆距离排序仍相同。采样遗漏尖峰确实存在，需要修复；它没有推翻这张图的距离计算结果。','',
'## 核查结论与改进','',
'原计算和汇总在核查容差内成立，原始数值保留。表述和展示需要更明确：注明同频对应、原始Ω、采样方式、seed与跨seed均值；单seed曲线旁列出该seed自己的误差，并补充真实欧姆残差。下一轮应同时控制采样，再比较局部结构指标。','',
'[机器核查记录](audit/AUDIT.json) · [独立复算逐seed数据](audit/recomputed_seed_metrics.csv) · [独立复算均值](audit/recomputed_means.csv) · [seed2300逐频点复阻抗及距离](audit/case1_seed2300_points.csv) · [seed2300逐频带贡献](audit/case1_seed2300_bands.csv) · [比例与相关性](audit/amp_phase_ratios.csv) · [两Case分量频带贡献](audit/component_band_contributions.csv) · [原始单目标报告](REPORT.md)','']
(OUT/'AUDIT.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'passed':True,'target_provenance':True,'dense':dense},ensure_ascii=False,indent=2))

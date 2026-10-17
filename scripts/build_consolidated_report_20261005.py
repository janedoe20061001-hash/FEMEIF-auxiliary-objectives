"""Render verified server results and historical reports into one research record."""
from pathlib import Path
import csv,json,math,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/crosscase_consolidation_20261005'
RAW=EXP/'server_run/review_results'
OUT=ROOT/'output/research_consolidation_20261005';OUT.mkdir(exist_ok=True)
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
assert json.loads((EXP/'server_run/download_audit.json').read_text())['status']=='passed'
assert json.loads((RAW/'analysis/COMPLETE.json').read_text())['status']=='passed'
def jread(p):return json.loads(p.read_text(encoding='utf-8'))
def cread(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def writecsv(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def link(p,label):return f'[{label}]({p.as_posix()})'
def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])+'\n'
def pct(x):return f'{x:+.2f}%'
def pair(v):return ' / '.join(pct(x) for x in v[:2])
def num(x):return '未定义' if x in [None,''] else f'{float(x):+.3f}'
names={'baseline':'双主目标基线',
 'reactance_crossing_direction_consistency+characteristic_frequency_shift_error_decades':'I-03+II-01 方向＋频移',
 'reactance_crossing_direction_consistency+transition_speed_relative_error':'I-03+II-02 方向＋速度',
 'cumulative_length_logpolar':'R-05-C 累计弧长',
 'all_interval_mean_residual_logpolar':'T-01 全区间平均残差',
 'bidirectional_residual_logpolar':'G-01-LP 同坐标累计残差'}
short={g:n.split(' ')[0] for g,n in names.items()};short['baseline']='Baseline'
groups=list(names);colors=['#687588','#0067a6','#e17a27','#11998e','#b94776','#8b73b5']
cmp=jread(RAW/'analysis/comparisons.json');gs=jread(RAW/'analysis/groups.json')
hist=jread(OUT/'historical_effects.json');rounds=jread(OUT/'historical_rounds.json')
corr=cread(RAW/'analysis/correlation_summary.csv');sel=cread(RAW/'analysis/selected_correlations.csv')
diagnostics=jread(RAW/'analysis/target_diagnostics.json');bands=cread(RAW/'analysis/band_errors.csv')
for fn in ['comparisons.csv','correlation_summary.csv','selected_correlations.csv','frequency_error_profiles.csv','band_errors.csv','target_diagnostics.json']:
    shutil.copy2(RAW/'analysis'/fn,OUT/fn)
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})

# One consistent graph: same seeds, same baseline, two primary losses.
fig,axs=plt.subplots(1,2,figsize=(13,5.8),layout='constrained')
for ax,case in zip(axs,['case1','case2']):
    for k,g in enumerate(groups[1:]):
        r=next(r for r in cmp if r['case']==case and r['group']==g and r['comparator']=='baseline')
        for m,offset,color in [(0,-.12,'#0075a8'),(1,.12,'#d86926')]:
            value=r['reduction_pct'][m];lo,hi=r['bootstrap95'][m]
            y=k+offset;ax.plot([lo,hi],[y,y],color=color,lw=2)
            ax.scatter([value],[y],color=color,marker='o' if r['holm60'][m]<.05 else 's',s=40,
                       label=['幅值 MSE','相位 MSE'][m] if k==0 else None)
    ax.axvline(0,color='#888',lw=1);ax.set_yticks(range(5),[short[g] for g in groups[1:]])
    ax.invert_yaxis();ax.set_title(case.upper()+' · 100 个新配对种子');ax.set_xlabel('相对基线 MSE 均值下降（%）');ax.grid(axis='x',alpha=.18);ax.legend()
fig.suptitle('2026-10-05 统一比较：点为均值比例，线为未校正 95% 区间\n圆点表示该分量通过 60 项 Holm 校正；方点表示未通过',fontsize=13)
fig.savefig(FIG/'new_comparison.png',dpi=170);plt.close(fig)

# Main trajectory: coordinates -> full cumulative process -> refinements.
keys=[('G','G-01'),('G','G-02'),('G','G-05'),('D1','D-03-LIN'),('D1','D-03-ASH'),('D2','D-03-LP'),
 ('D1','D-05-LIN'),('D1','D-05-ASH'),('D2','D-05-LP'),('R','D-03-LP'),('R','R-03-C'),
 ('R','D-05-LP'),('R','R-05-C'),('R','R-03-S'),('R','R-05-S'),('Q','R-05-C'),('Q','Q-05-B'),
 ('Q','Q-05-A'),('Q','Q-05-D'),('Q','Q-COMB'),('V','R-05-C'),('V','Q-05-B'),('I','R-05-C'),('I','G-05-I'),('I','G-05-H')]
mat=[];labels=[]
for stage,mid in keys:
    row=[]
    for case in ['case1','case2']:
        for primary in ['magnitude','phase']:
            r=next(r for r in hist if r['stage']==stage and r['id']==mid and r['case']==case and r['primary']==primary and r['comparator']=='baseline')
            row.append(r['reduction_pct'])
    mat.append(row);labels.append(stage+'  '+mid)
fig,ax=plt.subplots(figsize=(10,12),layout='constrained');im=ax.imshow(mat,cmap='RdYlGn',norm=TwoSlopeNorm(vmin=-80,vcenter=0,vmax=80),aspect='auto')
ax.set_yticks(range(len(labels)),labels);ax.set_xticks(range(4),['Case1 幅值','Case1 相位','Case2 幅值','Case2 相位'])
for i,row in enumerate(mat):
    for k,v in enumerate(row):ax.text(k,i,f'{v:+.2f}',ha='center',va='center',fontsize=9,color='white' if abs(v)>55 else '#192d3e')
ax.set_title('探索演变中的主目标效果（%）\n各行相对本轮基线；不同轮次不能直接排名',pad=14)
fig.colorbar(im,ax=ax,label='MSE 均值下降（正数改善）',shrink=.65);fig.savefig(FIG/'history_evolution.png',dpi=170);plt.close(fig)

chosen=['I-03','II-01','II-02','G-01','G-02','D-03-LIN','D-03-LP','R-03-C','D-05-LIN','D-05-LP',
        'R-05-C','Q-05-B','Q-05-A','Q-05-D','G-05-I','G-05-H','G-01-LP','T-01']
def selected_value(mid,case,primary,kind='spearman'):
    return next(r[kind] for r in sel if r['id']==mid and r['case']==case and r['primary']==primary)
mat=np.array([[float(selected_value(mid,c,p)) for c in ['case1','case2'] for p in ['magnitude','phase']] for mid in chosen])
fig,ax=plt.subplots(figsize=(10,9),layout='constrained');im=ax.imshow(mat,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto')
ax.set_yticks(range(len(chosen)),chosen);ax.set_xticks(range(4),['Case1 幅值','Case1 相位','Case2 幅值','Case2 相位'])
for i,row in enumerate(mat):
    for k,v in enumerate(row):ax.text(k,i,f'{v:+.3f}',ha='center',va='center',fontsize=9,color='white' if abs(v)>.67 else '#192d3e')
ax.set_title('改造前后的相关性：同一批基线选中解\n每 Case 100 个独立种子，Spearman；相关性不等于优化收益',pad=14)
fig.colorbar(im,ax=ax,label='Spearman ρ',shrink=.7);fig.savefig(FIG/'correlation_evolution.png',dpi=170);plt.close(fig)

profiles=cread(RAW/'analysis/frequency_error_profiles.csv')
fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
for row,case in enumerate(['case1','case2']):
    for j,primary in enumerate(['magnitude_mse','phase_mse']):
        ax=axs[row,j]
        for k,g in enumerate(groups):
            r=[r for r in profiles if r['case']==case and r['group']==g]
            ax.plot([float(x['frequency_hz']) for x in r],[float(x[primary]) for x in r],label=short[g],color=colors[k],lw=1.3)
        ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('频率（Hz）');ax.set_ylabel(['幅值平方误差（dB²）','相位平方误差（rad²）'][j]);ax.set_title(case.upper());ax.grid(alpha=.15)
axs[0,0].legend(fontsize=8,ncol=2)
fig.suptitle('最终同一个体的逐频点误差：100 次运行均值\n完整原频率网格；纵轴对数仅用于显示',fontsize=13)
fig.savefig(FIG/'frequency_error_profiles.png',dpi=170);plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
for ax,case in zip(axs,['case1','case2']):
    d=np.load(RAW/case/'analysis/selected_curves.npz');alpha=25.12 if case=='case1' else 17.38
    h=d['histories'];ws=h[...,0]+alpha*h[...,1]
    for k,g in enumerate(groups):ax.plot(np.arange(1,301),ws[:,k,:].mean(axis=0),label=short[g],color=colors[k])
    ax.set_yscale('log');ax.set_xlabel('代数');ax.set_ylabel('每代按固定 WSMSE 选解的平均值');ax.set_title(case.upper());ax.grid(alpha=.15)
axs[0].legend(fontsize=8,ncol=2);fig.suptitle('收敛轨迹（描述性）：不能用综合误差替代两项主误差判断')
fig.savefig(FIG/'convergence.png',dpi=170);plt.close(fig)

# Legacy single-auxiliary descriptive table, never mixed with front-mean paper results.
import importlib.util
spec=importlib.util.spec_from_file_location('catalog',ROOT/'experiments/case1_dual_pdf18/objectives.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
excluded=set(jread(ROOT/'scripts/metric_catalog_selection.json')['excluded'])
legacy_rows=[]
for mid,name,label in legacy.CATALOG:
    vals=[]
    for case in ['case1','case2']:
        rr=cread(ROOT/f'experiments/{case}_dual_pdf18/server_run/paired_summary.csv')
        r=next(r for r in rr if r['front']=='wsmse_selected' and r['group']==name)
        vals.extend([-float(r['magnitude_change_pct']),-float(r['phase_change_pct'])])
    legacy_rows.append(dict(id=mid,name=label,excluded=mid in excluded,case1_mag=vals[0],case1_phase=vals[1],case2_mag=vals[2],case2_phase=vals[3]))
writecsv(OUT/'legacy_single_effects.csv',legacy_rows)

parts=[]
def add(s=''):parts.append(s.strip()+'\n')
add('# FEMEIF 近期研究统一总结：指标删选、设计演变、实验效果与相关性')
add('日期：2026-10-05。涵盖旧局部指标筛选、近期七轮全局/三维指标探索、整体走势理论审查，以及本次服务器上的统一比较。所有百分比均以“正数表示 MSE 下降、负数表示变差”呈现。')
add('## 1. 当前最重要的判断')
add('1. 已从当前候选讲稿中移除六项预设事件数量指标；历史公式、代码与结果保留。删除依据是数量门槛缺乏足够解释，并非这六项实验都无效。保留项仍包含检测阈值、稳定化常数与坐标约定，不能统称“完全无主观参数”。\n2. 新指标的主要进展是：从实部/虚部几何改为幅值/相位几何，再从最终汇总量改为沿频率的累计过程。累计弧长 R-05-C 是近期几何方向中最值得保留的候选；更多复杂变体没有可靠地超过它。\n3. 本次 1,200 次正式运行把两种局部组合、累计弧长、T-01 放在相同种子和预算下比较。Case2 的方向＋速度、累计弧长保持较大均值收益；二者未分出两项主误差都更好的赢家。Case1 的相位表现更依赖种子，先前方向＋频移的优势没有完整重复。\n4. T-01 已完成实现、数值核验和首轮正式实验。它并非旧指标改名，但没有明确超过同坐标旧累计残差；Case2 幅值明显落后于局部方向＋速度和累计弧长。当前不建议加入核心目标，也不继续堆叠它与其他辅助项。\n5. 本次统一重算了 40 项误差（其中含六项已删指标，仅作审查）的相关性。部分改造提高了误差排序的一致性，但高相关并不保证加入优化有效；G-05-H 与 T-01 都提供了直接反例。')
add('这轮六组全部直接比较共 60 项主误差检验，统一 Holm 校正。没有候选同时在两个 Case 的两个主误差上获得明确改善支持；这与“均值多数下降”是不同结论。历史各轮的检验族大小不同，本报告不把显著性标签直接当作跨轮排名。')
add('## 2. 到底删掉了什么，还剩哪些人为选择')
add(table(['移除编号','旧定义','处理'],[
 ['I-01','全局转折数量至少 5 个','从当前目录移除，旧结果保留'],['I-02','目标转折区域覆盖 5-of-6','同上'],
 ['I-04','全局电抗过零至少 10 次','同上'],['I-05','全局转折数量至少 4 个','同上'],
 ['I-07','全局电抗过零至少 8 次','同上'],['II-11','目标转折区域覆盖 4-of-6','同上']]))
add('依据是 '+link(ROOT/'scripts/metric_catalog_selection.json','2026-10-02 保留/排除清单')+'。当时保留 19 项包含两个主误差、12 项中文 PDF 辅助指标和 5 项英文论文指标。这是展示与候选范围的清理；旧实验代码未被删除，也未把旧统计改写成“删掉后的新实验”。')
add('代码还会把要求事件数按目标实际检测数封顶。Case2 中仅检测到 2 个目标转折，因此 I-01/I-05 等价、I-02/II-11 等价；这反映当前检测配置的结果，不意味着物理系统只有两个转折。I-04 与 I-07 在已有 Case2 单项实验中结果相同，但目标有 9 次过零，其封顶后阈值为 9 和 8，不能仅凭运行结果相同就宣称两个公式代数等价。')
add(table(['保留项/环节','仍存在的数值选择','应怎样表述'],[
 ['转折检测','目标角度 95% 分位、候选比例 0.20、事件间隔 4 个索引','检测配置；固定事件数删除后仍需交代'],
 ['I-03 过零方向','目标实际过零、区域边界及离散符号判定','没有预设 8/10 次要求；损失本身不使用转折角度阈值'],
 ['I-06 PTC','邻点差 0.2 dB、γ=0.41','检测/匹配设置'],
 ['II-01 / II-02','目标非零速度中位数的 5% 稳定项','会改变平坦频段权重，不只是机器舍入常数'],
 ['II-03 / II-04','DTW 窗口 4；Soft-DTW 窗口 10、γ=0.10','匹配范围和软化设置'],
 ['II-05 / II-08 / II-09','Hausdorff 95% 分位；局部惩罚权重 0.10','分位/权重选择，尚不能称无参数'],
 ['英文论文指标','SS 的 THk=8、β=0.4、μ=0.5；LFP 阈值 1.6；DTW 60 点','本地冻结实现；不能当成当前 Case2 的物理最优值'],
 ['新几何/累计指标','log 频率、目标范围或标准差尺度、欧氏合并、主值相位','没有指定峰数或过零数，仍有建模约定'],
 ['T-01 数值积分','16 阶 Gauss；预先要求相邻阶相对变化 <1e-6','数值精度设置，经收敛检查；不是趋势宽度或事件门槛']]))
add('因此，“删掉数量先验”“保留可解释的建模选择”“核验数值精度”是三个不同层面。不能把所有带数字的定义一概删除，也不能把其余常数藏在实现里。')
add('## 3. 探索路径：先设计什么，后来为什么改')
add('### 3.1 前序局部指标、组合与鲁棒性工作')
add('前序工作先完成旧 18 项辅助指标的单项评估，再做组合筛选、独立种子复验、组成项比较、参数稳定性和模拟噪声/缺失频点验证。当前六项数量先验的删选不抹掉这些历史结果；单项完整表见附表 A。')
add(table(['前序工作','历史主要结果','本轮如何接续'],[
 ['Case1 固定局部组合复验','I-03+II-01 幅值/相位下降 30.639%/23.629%，两项调整区间支持；I-03+II-02 相位下降仅 2.924%','两种组合均放入新批次，检验能否重复'],
 ['Case2 组合冠军复验','I-03+II-02 幅值/相位下降 38.374%/12.480%，原检验口径下双项支持','作为工作对照，与累计弧长直接比较'],
 ['Case2 鲁棒性','已测低/中模拟噪声和减半采样下有效；高模拟噪声备选 P-03+P-05；四分之一频点或连续缺段无可靠候选','保留适用边界，本轮不声称完成新测量泛化验证'],
 ['参数分析、可辨识性和对偶讨论','拟合质量、参数集中程度与参数真值恢复是不同问题','不将更低误差或更集中参数写成结构可辨识性证明'],
 ['自动电路生成','已有思路与文献探索','未在本轮扩展为电路拓扑生成实验']]))
add('历史精确统计和背景分别见 '+link(ROOT/'experiments/case1_dual_validation/server_run/STABILITY.md','Case1 固定组合稳定性')+'、'+link(ROOT/'experiments/CASE2_REMAINING_EXPERIMENTS_SUMMARY.md','Case2 迁移及鲁棒性总表')+'、'+link(ROOT/'output/跨对话汇总_下一阶段工作_20261004.md','前次跨对话汇总')+'。这些历史成绩不能与本轮新种子百分比直接排成名次。')
add('### 3.2 七轮全局/几何候选')
add(table(['阶段','问题与方案','正式运行数','种子'],[[r['stage']+' '+r['label'],
 {'G':'G-01 双向累计残差、G-02 最坏连续区间、G-05 相对复误差',
 'D1':'距离、协方差、面积、SRVF、弧长对应形状；LIN/ASH 两种坐标',
 'D2':'五类几何量改用 dB/相位 LP 坐标，新增全频率对方向差',
 'R':'面积/长度改为全过程累计；另以目标标准差替换范围尺度',
 'Q':'两端对称、分轴总变化、分轴上升/下降、面积＋弧长组合',
 'V':'只复核原累计弧长和两端对称累计弧长',
 'I':'所有频率区间的弧长误差、逐段弧长平方根误差'}[r['stage']],r['runs'],f"{r['seeds']['start']}–{r['seeds']['start']+99}"] for r in rounds]))
add('上述七轮共 8,600 次完整运行；部分轮次复用了相同种子编号并重跑基线，因此不能称作 8,600 个独立新随机条件。本次新增 1,200 次，使用新种子 2000–2099。早期中止的 302 次慢实现运行不纳入正式效果表。')
add('### 3.3 最早的全局候选并没有全部进入实验')
add('最初提出 G-01 至 G-08：累计残差、最坏区间、变化量 Wasserstein、幅值相关形状、相对复误差、导纳误差、Fréchet 距离、不确定度加权。正式比较的是 G-01、G-02、G-05；其余因丢失方向/总量、坐标与匹配问题或缺少测量不确定度依据等保留在理论候选阶段。不能把“提出过”记为“实验通过”。')
add('### 3.4 三维几何的两次关键改造')
add('第一步：保留 log 频率轴，把另外两轴从 Re/Im 的线性或 asinh 坐标，改为目标尺度固定的 dB 幅值/主值相位。这使部分指标更贴近原主目标的坐标，但它不是新增测量信息，也没有消除尺度选择。\n\n第二步：从终点面积或按归一化弧长对应的位置，改成每个原频率处的累计面积/累计实际长度。尤其 D-05-LP 测的是“按各自归一化弧长对应后的形状”；R-05-C 测的是“走到同一频率时累计长度差多少”，二者不是同一个长度公式的小改名。')
add('### 3.5 后续变体针对的缺口')
add(table(['新版本','父方案/动机','实际改变'],[
 ['R-03-C','D-03-LP 仅比较最终面积','逐频率比较累计有向面积，保留中途积累过程'],
 ['R-05-C','D-05-LP 的弧长归一化与重新对应','同频率比较累计实际长度，保留总长差'],
 ['R-03-S / R-05-S','目标范围是否不合适','只换目标标准差尺度；结果不支持普遍升级'],
 ['Q-05-B','只从低频端累计是否偏向一端','前向/反向累计误差等权；总长一致时与原版相同'],
 ['Q-05-A / Q-05-D','长度不保留坐标方向','分轴累计绝对变化；进一步拆成累计上升/下降'],
 ['Q-COMB','面积与长度能否互补','两个辅助目标同时优化；不是一个新的标量损失'],
 ['G-05-I','固定端点导致局部偏差持续传播','比较任意两频率间弧长差，等价于中心化累计误差'],
 ['G-05-H','累计长度可能抵消；SRVF 含方向项','逐段平方根长度差，保留长度部分；外层变换也改变了'],
 ['T-01','连续频段是否持续偏高/偏低','将残差积分除以频段宽度，汇总所有连续区间平均偏差']]))
add('分轴上升/下降可由总变化量与起点位移重构，不能称完全独立信息；G-05-I 也是旧累计误差的重新分配。数学解释成立并不要求一定将指标加入优化。')
add('### 3.6 几个关键改造的公式区别')
add(r'''令 $u$ 为归一化 log 频率，$y$ 为按目标范围归一化的 dB 幅值/主值相位坐标，$p=(u,y)$。沿原节点连线的累计长度记为 $\ell(u)$，目标总长记为 $L^*$，令 $e(u)=[\ell(u)-\ell^*(u)]/L^*$。

$$R\text{-}05\text{-}C=\left[\int_0^1e(u)^2\,du\right]^{1/2},\qquad
G\text{-}05\text{-}I=\left[2\int_0^1(e(u)-\bar e)^2\,du\right]^{1/2},\quad\bar e=\int_0^1e(u)\,du.$$

原累计弧长逐频率比较已经走过的长度；全区间版比较两个端点之间的长度差，在该连续定义下等价于中心化累计误差。D-05-LP 则先按各自总弧长归一化，再比较对应位置的形状：改成 R-05-C 同时恢复了原频率对应和总长度信息。

逐段长度为 $d_i$ 时，$G\text{-}05\text{-}H=[\sum_i(\sqrt{d_i}-\sqrt{d_i^*})^2/L^*]^{1/2}$，直接比较逐段长度分配。它与 SRVF 的带方向向量不同，不能称包含完整方向信息。

T-01 的输入改为同频率坐标残差 $r(u)=y(u)-y^*(u)$，令 $A(u)=\int_0^u r(t)dt$：

$$T\text{-}01=\left[2\int_0^1\int_a^1\left\|\frac{A(b)-A(a)}{b-a}\right\|^2\,db\,da\right]^{1/2}.$$

同坐标 G-01-LP 使用 $\frac12\int_0^1[\|A(u)\|^2+\|A(1)-A(u)\|^2]du$ 的原节点梯形离散，无外层开方。T-01 增加区间跨度归一化和所有连续端点对，但所有端点对等权意味着跨度密度为 $2(1-h)$，短区间更多；它没有预先指定峰数，也不等于独立提取慢变趋势。''')
add('完整理论定义和解析反例见 '+link(ROOT/'output/整体走势候选_详细公式与解析反例_20261004.md','2026-10-04 公式稿')+'；代码/近邻方法审查见 '+link(ROOT/'output/整体走势指标_代码文献审查与筛选结论_20261004.md','理论筛选结论')+'。这些文件保持当时的理论阶段记录，本轮新增的实验判断以本报告为准。')
add('## 4. 改造前后对主目标的效果')
add('以下均为“幅值 MSE / 相位 MSE”下降百分比。跨轮表用于追踪过程；判断某个改造是否升级，应优先看同轮对父方案的直接配对比较。')
add('![指标探索的主要演变](research_consolidation_20261005/figures/history_evolution.png)')
def history_pair(stage,mid,case,comparator='baseline'):
    return [next(r for r in hist if r['stage']==stage and r['id']==mid and r['case']==case and r['primary']==p and r['comparator']==comparator) for p in ['magnitude','phase']]
add('### 4.1 同轮父子直接比较')
parentrows=[]
for stage,mid,parent in [('R','R-03-C','D-03-LP'),('R','R-05-C','D-05-LP'),
 ('R','R-03-S','D-03-LP'),('R','R-05-S','D-05-LP'),('Q','Q-05-B','R-05-C'),
 ('Q','Q-05-A','R-05-C'),('Q','Q-05-D','R-05-C'),('Q','Q-COMB','R-05-C'),('Q','Q-COMB','R-03-C'),
 ('V','Q-05-B','R-05-C'),('I','G-05-I','R-05-C'),('I','G-05-H','R-05-C')]:
    a=history_pair(stage,mid,'case1',parent);b=history_pair(stage,mid,'case2',parent)
    supported=['C'+str(i+1)+' '+p for i,v in enumerate([a,b]) for p,r in zip(['幅值','相位'],v) if r['supported_improvement']]
    parentrows.append([stage+' '+mid,parent,pair([r['reduction_pct'] for r in a]),pair([r['reduction_pct'] for r in b]),'、'.join(supported) or '无明确增量改善'])
add(table(['轮次/改造','比较对象','Case1','Case2','通过本轮校正的改善分量'],parentrows))
add('可确认的局部进展是：累计面积相对终点面积在 R 轮 Case1 有双项增益；累计弧长相对归一化弧长对应形状在 R 轮 Case2 有双项增益。它们均未形成跨两个 Case 的通用升级。Q 轮对称弧长看起来更好，但 V 轮新种子复核未重复出明确增量。G-05-H 在 Case1 及相对原累计弧长的直接比较中表现较差，退出优先候选。')
add('### 4.2 原累计弧长跨轮表现')
add(table(['种子区块','Case1 幅值/相位下降','Case2 幅值/相位下降','证据说明'],[
 [str(next(r['seed_start'] for r in hist if r['stage']==stage))+'–'+str(next(r['seed_start'] for r in hist if r['stage']==stage)+99),
 pair([r['reduction_pct'] for r in history_pair(stage,'R-05-C','case1')]),
 pair([r['reduction_pct'] for r in history_pair(stage,'R-05-C','case2')]),
 'Case1 '+('双项支持' if all(r['supported_improvement'] for r in history_pair(stage,'R-05-C','case1')) else '双项支持不足')+'；Case2 双项支持'] for stage in ['R','Q','V','I']]))
add('这些旧轮次保持各自预设统计口径。本次 2000–2099 的新结果见下一节；不把重复使用的旧随机条件合并伪装成更多独立证据。')
add('## 5. 本次服务器实验：六组统一直接比较')
add('两 Case 各 600 次运行，全部成功；共 6,300 万次评价。原主目标、频点、电路、边界、算法和固定选解方式沿用历史设置。T-01 和 G-01-LP 使用相同的目标范围归一化 dB/相位残差；二者汇总公式及外层变换不同。')
newrows=[]
for g in groups[1:]:
    rs=[next(r for r in cmp if r['case']==case and r['group']==g and r['comparator']=='baseline') for case in ['case1','case2']]
    newrows.append([names[g],pair(rs[0]['reduction_pct']),pair(rs[1]['reduction_pct']),
      ' / '.join(f'{p:.4g}' for p in rs[0]['holm60']),' / '.join(f'{p:.4g}' for p in rs[1]['holm60'])])
add(table(['方案','Case1 幅值/相位下降','Case2 幅值/相位下降','Case1 Holm p','Case2 Holm p'],newrows))
add('![新一轮统一比较](research_consolidation_20261005/figures/new_comparison.png)')
add('### 5.1 直接比较回答了什么')
add('- Case2：累计弧长相对方向＋速度，幅值均值变差约 6.01%，相位均值改善约 1.32%；两项直接比较均未通过校正。当前不能说累计弧长已经替代局部核心组合，也不能反向宣称局部组合全面支配它。\n- Case1：累计弧长相对方向＋速度，相位改善约 25.43%，通过本轮校正；幅值均值变差约 9.29%，但没有明确差异支持。这是指标侧重点不同的证据，不能称两项共同升级。\n- T-01 相对累计弧长：Case1 幅值改善约 5.43%、相位变差约 9.07%，均未明确；Case2 幅值变差约 45.02%，通过校正，相位约变差 2.67%，未明确。\n- T-01 相对方向＋速度：Case2 幅值变差约 53.73%，通过校正；相位没有明确差异。\n- T-01 与同坐标 G-01-LP：两个 Case 均无明确的双项增量优势。公式不同已经证明，作为新增辅助目标的价值尚未证明。')
add('这些结论不将不显著解释为等效。本轮 Case2 相位的均值收益依然存在，但 60 项校正后的支持不足；此前较小检验族及其他种子区块的支持继续按原样保留。')
add('### 5.2 T-01 从理论候选走到哪一步')
add('它现在已通过实现和正式实验，但结果支持将其降为研究记录中的备选，而非加入核心优化。它仍只描述连续区间的平均偏差，没有提取独立的慢变趋势。若未来重新考虑，需要新的使用场景或机制理由；本轮没有理由继续增加长区间阈值、调权重或与现有方案堆叠。')
add('## 6. 改造前后的相关系数：统一样本后再看')
add('本节把 40 项损失计算在这次相同的基线样本上：每 Case 的初始种群、基线最终种群各 100 个种子；另外每种子按原 WSMSE 选同一个解，得到每 Case 100 个独立代表解。这里没有改变旧公式，没有为不同指标选择不同代表解。')
add('主表给出代表解之间的 Spearman；Pearson、种群内中位数/四分位数和有效种子数均在附表。相关系数是无量纲关联，不能读成 MSE 降幅。')
add(table(['指标','Case1 幅值/相位 ρ','Case2 幅值/相位 ρ'],[[mid,' / '.join(num(selected_value(mid,'case1',p)) for p in ['magnitude','phase']),' / '.join(num(selected_value(mid,'case2',p)) for p in ['magnitude','phase'])] for mid in chosen]))
add('![同样本相关性对比](research_consolidation_20261005/figures/correlation_evolution.png)')
add('### 6.1 这些相关性怎样解释')
add('- 坐标改造确实改变了误差排序。例如 Case2 的 D-05-LIN 与幅值/相位为负相关，D-05-LP 转为正相关；这说明 LP 表示在当前样本中更贴近主误差。\n- 将 D-05-LP 改为 R-05-C 后，两个 Case 的代表解相关性均提高，与累计弧长值得保留的历史结果方向一致，但仍不能据此证明机制。\n- G-05-H 的相关性高于原累计弧长的部分分量，而历史优化表现反而更差；T-01 在 Case1 幅值上的 ρ 约 0.933，却没有带来明确的双项优化收益。高相关无法单独用作筛选标准。\n- R-03-C 在 Case2 相关性很高，但历史优化收益并不稳定。因此“能评价最终解好坏”和“加入进化后能把搜索引向更好结果”必须分开检验。')
add('### 6.2 为什么最终种群内部会出现负相关')
ctxrows=[]
for case in ['case1','case2']:
    for context in ['initial','baseline_final']:
        r=next(r for r in corr if r['case']==case and r['context']==context and r['id']=='M-01_vs_M-02')
        ctxrows.append([case,context,num(r['pearson_median']),num(r['spearman_median'])])
add(table(['Case','样本','幅值/相位 Pearson 中位数','幅值/相位 Spearman 中位数'],ctxrows))
add('双主目标优化后的种群含有明显的幅值—相位权衡。某指标在该种群中与幅值负相关、与相位正相关，并不等于它在所有参数空间都损害幅值。初始种群、基线最终种群和每次最终代表解回答不同问题，不能混成一张无说明的“相关性排名”。')
add('常数指标的相关系数记为“未定义”，不是 0。早期 Case2 围绕一个拟合参考解的 1% 参数扰动实验里，若干事件指标呈常值；该局部结果不能代替这里较广样本上的相关性，更不能证明这些事件指标在整个优化中无效。完整旧扰动结论见 '+link(ROOT/'experiments/case2_physical_perturbation/results/REPORT.md','扰动分析')+'。')
add('## 7. 两个 Case 为何不同：事实、线索和解释边界')
targetrows=[]
for d in diagnostics:
    t=d['preflight']['local_target'];targetrows.append([d['case'],d['points'],f"{d['frequency_range'][0]:g}–{d['frequency_range'][1]:g}",
     f"{d['magnitude_range'][0]:g}–{d['magnitude_range'][1]:g}",t['target_turns'],t['target_crossings'],f"{d['max_principal_phase_jump_deg']:.2f}°"])
add(table(['Case','频点','频率 Hz','幅值 Ω','当前检测转折数','过零数','最大相邻主值相位差'],targetrows))
add('Case1 是同一电路模型生成的合成目标，Case2 是实测目标；模型维数、频率覆盖、预算和尺度也不同。Case1 的六个转折分布较广；Case2 的转折检测集中在前端，不能据此认定局部结构完全被捕获。II-01 是切线投影得到的局部频移近似，不是真正逐个峰匹配。')
add('### 7.1 本次误差主要落在哪些频段')
br=[]
for case in ['case1','case2']:
    rs=[r for r in bands if r['case']==case and r['group']=='baseline'];total=np.array([sum(float(r[k]) for r in rs) for k in ['magnitude_contribution','phase_contribution']])
    for r in rs:
        share=np.array([float(r[k]) for k in ['magnitude_contribution','phase_contribution']])/total*100
        br.append([case,r['band'],f"{float(r['frequency_min']):.4g}–{float(r['frequency_max']):.4g}",f'{share[0]:.2f}%',f'{share[1]:.2f}%'])
add(table(['Case','等宽 log 频率段','原点频率范围 Hz','基线总幅值误差贡献','基线总相位误差贡献'],br))
add('三个频段仅用于描述误差分布，不参与目标计算。Case2 的误差几乎都集中在最高 log 频率三分之一，方向＋速度和累计弧长的主要收益也发生在那里。Case1 的误差更分散；本轮方向＋速度虽降低幅值，但中、高频相位均值上升，抵消了其他收益。这解释了“收益发生在哪里”，尚未证明“为什么该辅助项会改变搜索”。')
add('![逐频点误差剖面](research_consolidation_20261005/figures/frequency_error_profiles.png)')
add('### 7.2 不能把所有差异归因于相位跳变或一个测量点')
d=diagnostics[1]
add('Case2 在 16 MHz 存在超出当前正参数被动模型相位域的原点，16–17.2 MHz 的相邻主值相位差为 184.9°。本次它占基线总相位误差约 '+f"{d['outside_phase_contribution_by_group']['baseline']['fraction']*100:.2f}%"+'；各方案中约 '+f"{min(v['fraction'] for v in d['outside_phase_contribution_by_group'].values())*100:.2f}%–{max(v['fraction'] for v in d['outside_phase_contribution_by_group'].values())*100:.2f}%"+'。它不是总误差的大多数，但可能影响其他频段的折衷。Case1 同样存在很陡的相位变化，因此“大相邻差值”本身不能当作唯一原因。没有删除原点、展开相位或更改训练数据来改善这批成绩。')
add('### 7.3 搜索机制仍是待验证解释')
add('更多辅助目标会改变非支配关系；最终前沿占比与收敛曲线已保留，但这些描述不能单独证明“选择压力减弱”是某组变差的原因。需要区分公式侧重点、尺度影响、可行域限制与优化随机性。当前证据足以缩小候选，却不足以给所有失败统一归因。')
add('![收敛轨迹](research_consolidation_20261005/figures/convergence.png)')
add('## 8. 指标体系与下一步决定')
add(table(['角色','当前安排','依据/边界'],[
 ['主目标','幅值 MSE、相位 MSE 始终保留','最后仍按两个主误差判断效果'],
 ['Case2 核心工作对照','I-03+II-02','历史多轮与鲁棒性证据较丰富；本轮均值方向保持，相位校正支持不足'],
 ['几何方向优先对照','原 R-05-C 累计弧长','跨轮 Case2 较稳定；本轮未明确替代局部组合，Case1 仍不稳定'],
 ['Case1 局部备选','I-03+II-01','历史有双项支持，本轮相位改善仅 3.08%；不能固化为通用冠军'],
 ['诊断而非默认加目标','频移、DTW、局部最坏误差、频段误差剖面','可解释最终解的不同缺陷；任何新增门控阈值均需另行定义验证'],
 ['降为备选记录','T-01、对称/全区间弧长、分轴变体','当前无明确的跨 Case 增量优势'],
 ['退出优先候选','逐段弧长平方根 G-05-H','历史优化效果差，高相关也不能挽救'],
 ['理论暂缓','T-02、FSV 平滑趋势、能量距离/Wasserstein 等','尚缺优先研究理由或可解释的尺度/应用依据']]))
add('后续最有价值的是新阻抗曲线、重复测量或真实噪声依据上的验证，以及针对 Case1 不稳定性的明确机制假设。继续在同两条曲线上改小公式、看均值挑赢家，收益有限。现有 Case2 模拟高噪声备选 P-03+P-05 和缺段/极稀疏下失效的结论继续保留，但不能把某个模拟噪声幅度或采样比例当作所有设备的普适阈值。')
add('## 9. 正确性、复现与文件')
audit=jread(RAW/'analysis/COMPLETE.json');archives=jread(EXP/'server_run/ARCHIVES.json')
add('本次检查全部 1,200 个初始种群、每次 300 代历史、合计 6,300 万次评价的计数、210,000 个最终个体的目标、非支配索引及固定 WSMSE 选解。生产公式复算最大差约 6.94e-18；独立公式（除 T-01 高阶求积对照）最大绝对差约 8.53e-14；T-01 的 16/32 阶评分最大相对差约 2.08e-8。常数、线性、非均匀网格、倒序/细分和独立自适应积分检查均通过。')
add('完整原始包 94,112,224 字节，下载后 SHA256 与服务器一致，包含 1,200 份运行完成记录和 1,200 个最终种群：`'+next(r['sha256'] for r in archives if r['file']=='formal_results.tar.gz')+'`。全部实验已结束，服务器没有设置继续运行的自动任务。')
add('\n'.join('- '+link(p,label) for p,label in [
 (EXP/'PROTOCOL.md','本轮启动前冻结方案'),(EXP/'server_run/review_results/analysis/comparisons.csv','本轮全部直接比较、区间与 60 项校正'),
 (OUT/'historical_effects.csv','七轮历史效果和父子比较全表'),(OUT/'correlation_summary.csv','40 项损失的初始/最终种群 Pearson 与 Spearman'),
 (OUT/'selected_correlations.csv','100 个基线代表解的 Pearson 与 Spearman'),(OUT/'legacy_single_effects.csv','旧 18 项单辅助描述性结果'),
 (EXP/'server_run/formal_results.tar.gz','完整原始结果包'),(EXP/'server_run/review_results.tar.gz','轻量代码与分析包'),
 (EXP/'server_run/download_audit.json','下载完整性校验'),(RAW/'analysis/COMPLETE.json','全量复算审计')]))
add('## 附表 A：旧 18 项单辅助结果与删选状态')
add('下表沿用 seeds 0–99 的同一个体 WSMSE 结果，统一换成正数表示下降。这里仅列描述性均值；不补造原报告没有的显著性结论。英文论文原 Case1 部分报告使用最终前沿均值，未混入这张表。')
add(table(['编号','名称','当前目录','Case1 幅值/相位下降','Case2 幅值/相位下降'],[[r['id'],r['name'],'移除' if r['excluded'] else '保留',pair([r['case1_mag'],r['case1_phase']]),pair([r['case2_mag'],r['case2_phase']])] for r in legacy_rows]))
add('## 附表 B：七轮新指标全部结果（含失败项）')
for round in rounds:
    stage=round['stage'];add('### '+stage+' '+round['label'])
    mids=list(dict.fromkeys(r['id'] for r in hist if r['stage']==stage and r['comparator']=='baseline'))
    rr=[]
    for mid in mids:
        a=history_pair(stage,mid,'case1');b=history_pair(stage,mid,'case2')
        rr.append([mid,a[0]['label'],pair([r['reduction_pct'] for r in a]),pair([r['reduction_pct'] for r in b]),
                   '/'.join(f"{r['holm_p']:.3g}" for r in a),'/'.join(f"{r['holm_p']:.3g}" for r in b)])
    add(table(['指标','含义','Case1 降幅','Case2 降幅','Case1 Holm p','Case2 Holm p'],rr))
add('## 附表 C：本轮相对基线的区间和原始/校正检验')
rr=[]
for r in cmp:
    if r['comparator']!='baseline':continue
    for k,p in enumerate(['幅值','相位']):
        rr.append([r['case'],short[r['group']],p,pct(r['reduction_pct'][k]),
                   '['+', '.join(pct(x) for x in r['bootstrap95'][k])+']',f"{r['p_raw'][k]:.4g}",f"{r['holm60'][k]:.4g}"])
add(table(['Case','方案','主误差','均值下降','配对 95% 区间','原始 p','Holm p'],rr))
add('## 附表 D：全部 40 项的代表解相关性')
allmids=list(dict.fromkeys(r['id'] for r in sel))
rr=[]
for mid in allmids:
    rr.append([mid]+[' / '.join(num(selected_value(mid,c,p,k)) for p in ['magnitude','phase']) for c in ['case1','case2'] for k in ['pearson','spearman']])
add(table(['指标','C1 Pearson 幅/相','C1 Spearman 幅/相','C2 Pearson 幅/相','C2 Spearman 幅/相'],rr))

report=ROOT/'output/近期研究统一总结_指标演变与验证_20261005.md'
report.write_text('\n'.join(parts),encoding='utf-8')
print(json.dumps(dict(report=str(report),characters=len(report.read_text(encoding='utf-8')),figures=len(list(FIG.glob('*.png'))),metrics=len(allmids)),ensure_ascii=True))

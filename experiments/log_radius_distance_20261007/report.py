"""Render a local Markdown report and exactly two representative scientific plots."""
from pathlib import Path
import sys,json,csv,hashlib,shutil
import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'output/log_radius_distance_20261007'
sys.path.insert(0,str(HERE))
import analyze as ana
LABELS={'wsmse':'WSMSE','mean_distance':'原 Ω 距离','logradius_distance':'老师式对数半径距离'}
ENLABELS=['Optimize WSMSE','Optimize mean distance (ohm)','Optimize log-radius distance']
COLORS=['#2674ad','#df8834','#29936f']
def n(v):return f'{float(v):.6g}'
def table(headers,rows):
    esc=lambda v:str(v).replace('|',r'\|')
    return '\n'.join(['| '+' | '.join(map(esc,headers))+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(esc,r))+' |' for r in rows]])
def link(p):return p.resolve().as_posix()
def put(p,txt):p.write_text(txt+'\n',encoding='utf-8')
def main():
    s=json.loads((OUT/'SUMMARY.json').read_text(encoding='utf-8'));v=json.loads((OUT/'VALIDATION.json').read_text(encoding='utf-8'));assert v['passed']
    records={r['id']:r for r in json.loads((HERE/'frozen_scalar/circuits.json').read_text(encoding='utf-8'))}
    struct=list(csv.DictReader((OUT/'per_seed_structure.csv').open(encoding='utf-8-sig')));bands=list(csv.DictReader((OUT/'band_diagnostics.csv').open(encoding='utf-8-sig')))
    (OUT/'figures').mkdir(exist_ok=True);(OUT/'schematics').mkdir(exist_ok=True)
    examples=[];rep=[]
    for c in ana.CASES:
        r=records[c];d=np.load(OUT/'selected'/f'{c}.npz');m=d['metrics'];seeds=d['seeds']
        # Nine ranks: three physical evaluation metrics for each of three groups.
        ranks=np.column_stack([rankdata(m[:,gi,k],method='average') for gi in range(3) for k in [0,1,3]])
        score=ranks.mean(axis=1);center=float(np.median(score));ix=int(np.argmin(abs(score-center)));seed=int(seeds[ix]);p=10**d['log_parameters'][ix]
        rep.append(dict(circuit=c,seed=seed,mean_rank=float(score[ix]),median_mean_rank=center,metric_columns=[0,1,3],tie_rule='smallest_seed'))
        if c=='case1':
            f=ana.improved_dense_grid(r,p,refinement=2);target=ana.ind.independent_spectra(c,f,r['target_parameters'])[0];z=ana.ind.independent_spectra(c,f,p)
        else:f=d['frequency'];target=d['target'];z=d['curves'][ix]
        fig,ax=plt.subplots(2,1,figsize=(11,7.1),sharex=True,layout='constrained')
        for gi in range(3):
            ax[0].semilogx(f,20*np.log10(abs(z[gi])),color=COLORS[gi],lw=1.35,label=ENLABELS[gi]);ax[1].semilogx(f,np.rad2deg(np.angle(z[gi])),color=COLORS[gi],lw=1.35)
        ax[0].semilogx(f,20*np.log10(abs(target)),'k--',lw=1.4,label='Target');ax[1].semilogx(f,np.rad2deg(np.angle(target)),'k--',lw=1.4)
        if c=='case1':
            ax[0].scatter(d['frequency'],20*np.log10(abs(d['target'])),s=7,c='#666666',alpha=.5,label='Original target samples',zorder=4)
        ax[0].set_ylabel('Magnitude (dB re 1 ohm)');ax[1].set_ylabel('Phase (degree)');ax[1].set_xlabel('Frequency (Hz)')
        for a in ax:a.grid(True,which='major',alpha=.18);a.set_xlim(f[0],f[-1])
        ax[0].legend(fontsize=9,loc='best');ax[0].set_title(f'{c}: median-rank representative seed {seed}'+ ('; analytic dense grid' if c=='case1' else '; original measured points'))
        fig.savefig(OUT/'figures'/f'{c}.png',dpi=170);plt.close(fig)
        shutil.copyfile(ROOT/'output/primary_loss_comparison_20261007/schematics'/f'{c}.svg',OUT/'schematics'/f'{c}.svg')
        cols=['参数（SI单位）','目标值' if c=='case1' else '目标值','WSMSE选解','原 Ω 距离选解','老师式距离选解']
        paramrows=[]
        for j,name in enumerate(r['parameter_names']):
            unit={'R':'Ω','L':'H','C':'F'}[name[0]];truth=n(r['target_parameters'][j]) if r['target_parameters'] is not None else '测量目标，无已知真值'
            paramrows.append([f'{name} / {unit}',truth,*[n(p[gi,j]) for gi in range(3)]])
        ex=[f'# {c}：典型 seed {seed}',f'选择规则：九个seed内排名的平均秩={n(score[ix])}，种子中位平均秩={n(center)}。不是选出某目标优势最大的图。',f'![电路结构]({link(OUT/"schematics"/f"{c}.svg")})',('六个并联 RLC 单元串联，18个参数。' if c=='case1' else '三个 C∥(L+R)∥R 单元，加串联 C4、R7、L4，15个参数；目标为201个测量点。'),table(cols,paramrows),f'![原幅相曲线]({link(OUT/"figures"/f"{c}.png")})',('实线和虚线使用解析加密网格；灰点是原180个目标频点。加密曲线未参与优化，不能拿这张图对应的密集误差替换训练误差。' if c=='case1' else '曲线连接原201个频点，未构造测量点之间的目标真值。'),table(['组','幅值MSE / dB²','相位MSE / rad²','WSMSE','DΩ / Ω','Dlog'],[[LABELS[g],*[n(x) for x in m[ix,gi]]] for gi,g in enumerate(ana.GROUPS)]),'本表仅对应这个seed；主报告统计所有100个seed，不能用此例代替总体。',f'[返回报告]({link(OUT/"REPORT.md")})']
        put(OUT/f'{c}_example.md','\n\n'.join(ex));examples.append(f'[{c}：seed {seed}，电路、参数及曲线]({link(OUT/f"{c}_example.md")})')
    (OUT/'REPRESENTATIVES.json').write_text(json.dumps(rep,ensure_ascii=False,indent=2),encoding='utf-8')
    means=table(['Case','优化目标','幅值MSE / dB²','相位MSE / rad²','WSMSE','原距离 / Ω','新距离 Dlog'],[[r['circuit'],LABELS[r['group']],*[n(r[k]) for k in ana.COLS]] for r in s['means']])
    comparisons=table(['Case','对照','指标','新组相对变化','新组−对照：95%区间','新组更低的seed数'],[[r['circuit'],LABELS[r['baseline']],{'magnitude_mse_db2':'幅值MSE','phase_mse_rad2':'相位MSE','mean_distance_ohm':'原 Ω 距离','mean_logradius_distance':'新距离 Dlog'}[r['metric']],f"{100*(r['new_over_baseline']-1):+.2f}%",f"{n(r['new_minus_baseline'])} [{n(r['ci95'][0])}, {n(r['ci95'][1])}]",f"{r['new_lower_seeds']}/100"] for r in s['paired_comparisons'] if r['metric']!='wsmse'])
    dense=table(['Case / 网格','组','幅值RMSE / dB','相位RMSE / °','漏幅值峰谷','新增幅值峰谷','漏相位事件','新增相位事件','最坏频带相位RMSE / °','log f加权原距 / Ω'],[[r['circuit']+(' / 加密' if r['grid']=='analytic_dense' else ' / 原点'),LABELS[r['group']],*[n(r[k]) for k in ['amplitude_rmse_db','phase_rmse_deg','amp_missed','amp_extra','phase_missed','phase_extra','worst_band_phase_rmse_deg','mean_distance_logweighted_ohm']]] for r in s['structure_means']])
    dc=table(['网格','对照','指标','新组−对照','配对95%区间'],[[r['circuit']+'/'+r['grid'],LABELS[r['baseline']],{'amp_missed':'漏幅值峰谷数','phase_missed':'漏相位事件数','amplitude_rmse_db':'幅值RMSE/dB','phase_rmse_deg':'相位RMSE/°'}[r['metric']],n(r['new_minus_baseline']),f"[{n(r['ci95'][0])}, {n(r['ci95'][1])}]"] for r in s['structure_comparisons'] if r['metric'] in ['amp_missed','phase_missed','amplitude_rmse_db','phase_rmse_deg'] and ((r['circuit']=='case1' and r['grid']=='analytic_dense') or r['circuit']=='case2')])
    near=table(['Case','组','目标 |A|≤1dB 点数','这些点相位RMSE / °','这些点幅值RMSE / dB','A符号相反的点数','异号点相位RMSE / °'],[[r['circuit'],LABELS[r['group']],*[n(r[k]) if r[k] is not None else '不适用' for k in ['target_near_zero_db_points','near_zero_db_phase_rmse_deg','near_zero_db_amplitude_rmse_db','opposite_logradius_sign_points','opposite_sign_phase_rmse_deg']]] for r in s['special_regions']])
    br=[]
    for c in ana.CASES:
        for gi,g in enumerate(ana.GROUPS):
            arr=[b for b in bands if b['circuit']==c and b['group']==g];vals=[]
            for k in range(1,9):
                rr=[b for b in arr if int(b['band'])==k];vals.append((k,float(np.mean([float(b['amplitude_rmse_db']) for b in rr])),float(np.mean([float(b['phase_rmse_deg']) for b in rr])),rr[0]))
            for field,ix in [('幅值',1),('相位',2)]:
                k,x,y,r=max(vals,key=lambda q:q[ix]);br.append([c,LABELS[g],field,f"{n(r['lo_hz'])}–{n(r['hi_hz'])} Hz",n(x if field=='幅值' else y)])
    bandsummary=table(['Case','组','误差类型','组平均误差最高的频带','该频带平均RMSE（dB或°）'],br)
    # Obtain the exact finite-budget deltas for the narrative, without changing selection.
    ratio={(r['circuit'],r['baseline'],r['metric']):r['new_over_baseline'] for r in s['paired_comparisons']}
    pct=lambda c,b,k:f'{100*(ratio[(c,b,k)]-1):+.1f}%'
    stable=s['unstable_dense_rows']==0
    lines=['# 老师式对数半径坐标：单目标实验结果',
      '**在这两个Case、相同300代预算下，新坐标距离没有带来一致的幅相改善。Case1幅值和相位MSE均值高于两组对照；与原Ω距离组的差值区间包含零，不能认定稳定变差。Case2幅相误差低于原Ω距离组，仍高于WSMSE组。它自身的距离下降了，不能据此认为原阻抗曲线拟合更好。**',
      '2026-10-07。Case1、Case2各100个配对seed；完成200次新优化，复用400次相同seed的对照结果。全部600条选解曲线在后台比较，只绘制两个典型seed。未更新网站。',
      '## 1. 严格采用老师提出的坐标',
      r'\[A=20\log_{10}(|Z|/1\Omega),\quad\theta=\arg Z,\quad X=A\sin\theta,\quad Y=A\cos\theta\]',
      r'\[D_{\log}=\frac1N\sum_i\sqrt{(X_i-X_i^*)^2+(Y_i-Y_i^*)^2}\]',
      'θ以弧度进入三角函数。负A保留，不取绝对值、不加偏移、不截断，也不补权重。频点与目标一一对应，频率残差为零；交换sin/cos两根轴不改变距离。X、Y是变换坐标，不是物理Re Z、Im Z。Dlog按对数幅值的dB数值计算，不是Ω单位，不能与DΩ数值大小直接比较。',
      '对照为WSMSE=M_A+αM_θ（Case1 α=25.12，Case2 α=17.38），以及原始实虚部的平均Ω距离。M_A用dB幅值差，M_θ用原主值相位的弧度差平方，沿用原程序、没有展开相位或另做环绕修正。新目标无α。每个seed只按本组目标从末代选最小个体，再对同一个体计算五项误差，最后跨100个seed等权平均。',
      '## 2. 五项误差：所有seed的均值',means,
      f"相对WSMSE：Case1幅值MSE变化{pct('case1','wsmse','magnitude_mse_db2')}、相位MSE变化{pct('case1','wsmse','phase_mse_rad2')}；Case2分别{pct('case2','wsmse','magnitude_mse_db2')}、{pct('case2','wsmse','phase_mse_rad2')}。相对原Ω距离组：Case1分别{pct('case1','mean_distance','magnitude_mse_db2')}、{pct('case1','mean_distance','phase_mse_rad2')}；Case2分别{pct('case2','mean_distance','magnitude_mse_db2')}、{pct('case2','mean_distance','phase_mse_rad2')}。",
      '下面给出配对差值。区间为10,000次seed块bootstrap（RNG=20261007）；它描述随机优化种子的变化，不描述测量误差。新组−对照为正表示新组误差更大，区间包含零时不足以认定均值差异方向。',comparisons,
      '## 3. 曲线结构与误差位置',
      'Case1加密诊断显示结构有取舍：相对原Ω距离组，新组平均少漏0.89个幅值峰谷、0.67个相位事件，但最坏频带相位RMSE从90.67°升至104.55°。相对WSMSE，新组漏失更多、整体幅相误差更大。Case2的新组相对原Ω距离组少漏3.18个幅值事件、3.26个相位事件，仍未达到WSMSE组的幅相水平。',
      'Case1原采样平均Ω距离在新组为2.899Ω、原距离组为2.561Ω；加密后log f加权平均Ω距离分别为6.820Ω、7.843Ω，排序发生变化。两者评价频点与权重不同，不是计算矛盾。这个结果说明，除了主目标，采样是否覆盖窄峰也会改变评价。',
      'Case1分别检查原180频点和解析加密网格；Case2只检查原201个测点。结构表中的RMSE先按每个seed的log f区间积分计算，再平均seed，因此不能直接开上表的MSE均值平方根得到。漏失/新增是每条曲线的平均事件数，允许小数。',dense,
      ('300条Case1加密诊断全部通过两档网格检查：四个积分误差相对变化≤1%，幅相漏失/新增事件数一致。' if stable else f"有{s['unstable_dense_rows']}条Case1诊断未通过网格稳定性检查。加密值仅供探索，不据此判断精确优势；逐条标记保留在grid_refinement.csv。"),dc,
      '幅值峰谷显著度阈值1dB，相位峰谷5°；同类型事件在0.25 decade以内配对。事件漏失数量依赖阈值和匹配容差，它是结构线索，不是物理谐振个数的证明。Case2测量起伏尤其不能直接解释为电路谐振。',
      '新组最集中的幅相偏差仍在高频：Case1为316kHz–1MHz，Case2为17.24MHz–110MHz。这说明整体平均距离仍可能容许末端曲线结构明显偏离；这些频带应成为后续指标设计的检查对象。',bandsummary,'频带是八个等log f宽度区间；表中按每个频带跨seed的平均RMSE取最大。逐seed最坏频带在per_seed_structure.csv中，全部频带在band_diagnostics.csv中。',
      '## 4. 这个变换改变了什么',
      r'每个频点的平方距离满足：\[d_{\log}^2=(A-A^*)^2+2AA^*[1-\cos(\theta-\theta^*)].\]',
      '在候选接近目标且A≠0时，局部近似为 dlog²≈(ΔA)²+(A*)²(Δθ)²。因此相位权重随目标对数幅值变化；这不是固定α的WSMSE，而且每点开根号再平均，误差聚合方式也不同。此次实验同时包含这两项变化，不能将结果差异完全归因于“用了对数坐标”。',
      '当|Z|=1Ω时A=0，所有相位映射到原点，无法由这两个坐标区分相位。A与A*异号时，上式相位项为负，必须使用完整向量距离，不能将其理解为幅相误差的正权加和。一般复数域还存在符号翻转加180°的重合；本实验被动模型实部为正，不能把一般域的重合示例直接当成当前电路已出现的问题。',near,
      'Case1原频点有4个目标点满足|A|≤1dB；Case2没有这些点，也没有对数半径异号点。Case1新组在这4点的相位RMSE低于原Ω距离组、高于WSMSE组。新组每条曲线平均16.66个点发生A异号，其区域相位RMSE平均122.78°；三组的异号区域并非同一组点，这个条件统计用于定位问题，不能直接作固定区域的配对优劣判断。这里能确认变换的数学退化及各组局部误差，尚不能证明它是Case1全局MSE升高的主要原因。参考阻抗固定为1Ω；更改参考会改变相位权重与退化位置，不属于本轮实验。',
      '## 5. 两个典型例子',
      '典型seed按预登记规则选取：在每个Case内，对三组各自的幅值MSE、相位MSE、Ω距离分别给100个seed排名（共9项），取平均秩最接近其中位数的seed，精确并列取较小seed。不是按某组优势选图。',
      '\n'.join('- '+x for x in examples),
      '## 6. 如何向老师汇报与下一步',
      '> 老师，我按照20log₁₀(|Z|/1Ω)作为半径，乘sinθ、cosθ后计算同频点平均欧氏距离，已经做了两个Case各100个seed的受控对照。新目标本身降低了，但Case1原幅相误差明显高于WSMSE，与原Ω距离组还没有稳定差异；Case2幅相误差比原始Ω距离好、仍不及WSMSE。新组有些峰谷保留得更多，但Case1最坏频带的相位偏差更大。这个坐标可以作为优化目标，但会随对数幅值改变相位权重，1Ω处还会丢失相位区分。当前结果还不足以认定它更适合阻抗拟合。',
      '如果继续设计指标，优先做能分清原因的消融：①固定α下对(A, √αθ)的平均欧氏距离，与WSMSE比较以区分平方聚合和逐点距离聚合；②老师式坐标下平均平方距离，与当前逐点开根号平均比较；③继续分析全部选解的漏峰、相位平台和最坏频带，检查平均误差掩盖的局部结构。这里仅提出后续实验，本轮未追加这些目标。',
      '## 7. 条件、核验与可复算资料',
      'Case1种群200、Case2种群150，均300代，seed为2026100700–2026100799。NSGA2单目标方式，log10参数编码、SBX(prob=0.9, eta=1)、PM(prob=1, eta=1, prob_var=0.1/0.08)。固定边界、目标、原频点、初始化和算法随机seed，与旧两组逐项一致。新组共10,500,000次评价，服务器正式运行46.48秒；分析耗时另计。200次新运行均成功。',
      '独立公式复算全部35,000个末代个体；核对预算、参数边界、初始种群、哈希、五项指标、选解和每代最优历史。新目标同时用三角坐标公式及实虚部分量公式交叉计算；两组旧选解和四项指标保持不变。下载归档SHA256已核验。',
      'Case1原采样在约25.23kHz漏过一个真实窄峰：原180点折线低估峰高约40.49dB。本轮保留这一采样缺口，未把采样修复混入目标变更。加密诊断有助于发现训练点之间的问题，但不能凭两个Case证明普遍规律。Case1真参数在边界内，理论上三种目标都可达到零误差；本报告比较的是有限预算搜索效果，不是三种目标的全局最优。Case2没有测点之间的目标真值。',
      '\n'.join(f'- [{label}]({link(path)})' for label,path in [('预登记方案',HERE/'PROTOCOL.md'),('五项误差与区间JSON',OUT/'SUMMARY.json'),('600条逐seed指标CSV',OUT/'selected_metrics.csv'),('逐seed结构诊断',OUT/'per_seed_structure.csv'),('全部频带误差',OUT/'band_diagnostics.csv'),('网格稳定性检查',OUT/'grid_refinement.csv'),('独立复算记录',OUT/'VALIDATION.json'),('下载哈希与来源',HERE/'server_run/ARCHIVES.json'),('复算说明',HERE/'README.md')])]
    put(OUT/'REPORT.md','\n\n'.join(lines))
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'analyze.py',HERE/'report.py',ROOT/'experiments/seed_replication_20261007/analyze.py',ROOT/'experiments/optimized_curve_structure_20261007/diagnostics.py',ROOT/'experiments/optimized_curve_structure_20261007/refined_grid.py']}
    (OUT/'ANALYSIS_SOURCES.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print(json.dumps({'report':link(OUT/'REPORT.md'),'representatives':rep,'dense_stable':stable},ensure_ascii=False,indent=2))
if __name__=='__main__':main()

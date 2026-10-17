from core import *
from analyze import LABEL,METHODS
import re,zipfile,csv,shutil
sys.path.append(str(ROOT/'tmp/curve_refinement_deps'))
import markdown
SUMMARY=json.loads((OUT/'SUMMARY.json').read_text());POLISH=json.loads((OUT/'POLISH_SUMMARY.json').read_text())

def find(condition,case,method,g=1000,sel=None):
    sel=sel or ('knee' if method=='dual_nsga' else 'minimum')
    return next(x for x in SUMMARY if (x['condition'],x['case'],x['method'],x['selection'],x['generation'])==(condition,case,method,sel,g))

def main():
    for name in ['PROTOCOL.md','VERDICT.md']:shutil.copy2(HERE/name,OUT/name)
    lines=['# 老师疑点的实验核验','', (HERE/'VERDICT.md').read_text(encoding='utf8'),'','## 正式实验与选解规则','',
           '服务器完成800次1000代优化，共135,000,000次候选评价。Full条件使用现有Case1的180点和Case2的201点；每Case比较WSMSE当前标量框架、WSMSE传统GA、MAG/PHA双目标。Case2另做10kHz–30MHz范围内108个真实测点的频段代理实验。每单元100seed（2300–2399），全部使用同seed同初始种群、相同log参数边界、SBX和PM；没有删除不良seed。传统GA按论文明确的锦标赛、父子合并留最优机制构建，并非缺失的原作者源码复现。','',
           '双目标每seed先选膝点，再在100个代表中按统一WSMSE取最佳。膝点规则在实验前固定：前沿两目标min/max归一化后，选向理想点方向距端点连线的最大垂距；无凸膝点时标记并采用预定折中代表。图中“最好”是各组最好seed，平均值才回答整体表现。双目标前沿最小WSMSE仅作选解诊断。','',
           '400次当前框架与双目标运行核查300代历史种群，其中398次在指定数值容差内一致。Case2双目标2334、2365与历史分叉；两seed未剔除，复跑要求与首次尝试300代参数逐位一致，之后延长到1000代。原失败记录、8次回调复现探针和协议修订全部保留。未将延长后的阶段冒充新独立样本。传统GA和频段实验为新对照。另做16次有界最小二乘，仅诊断可达性，额外预算单独记录，不混入公平主实验。','']
    for case in ['case1','case2']:
        lines += [f'## {case.upper()}：目标与最终拟合','',f'![{case}1000代最终曲线](figures/{case}_full_best_1000.png)','',
                  '下表在训练频点计算；最大误差的Case1使用目标及候选模态加密曲线，Case2使用全部真实测点。幅相RMSE和最大误差都需看，不能用一个WSMSE分数代替曲线。','',
                  '| 代数 | 组别 | 最佳seed | 最佳WSMSE | 幅值RMSE / dB | 相位RMSE / ° | 曲线最大幅值偏差 / dB | 曲线最大相位偏差 / ° |','|---:|---|---:|---:|---:|---:|---:|---:|']
        for g in [300,1000]:
            for method in METHODS:
                b=find('full',case,method,g)['best'];v=b['curve']
                lines.append(f"|{g}|{LABEL[method]}|{b['seed']}|{b['wsmse']:.6g}|{np.sqrt(b[METRICS[0]]):.3f}|{np.rad2deg(np.sqrt(b[METRICS[1]])):.3f}|{v['max_amplitude_db']:.3f}|{v['max_phase_deg']:.3f}|")
        lines += ['',f'[300代曲线](figures/{case}_full_best_300.png) · [1000代逐点曲线]({case}_full_curves_1000.csv) · [1000代参数]({case}_full_parameters_1000.csv)','',
                  '### 260–1000代：全部100次平均WSMSE','',
                  '| 组别 | 260代 | 280代 | 300代 | 600代 | 1000代 |','|---|---:|---:|---:|---:|---:|']
        for method in METHODS:
            lines.append('|'+LABEL[method]+'|'+'|'.join(f"{find('full',case,method,g)['mean_metrics'][2]:.6g}" for g in [260,280,300,600,1000])+'|')
        lines += ['|双目标 / 前沿最小WSMSE（诊断）|'+'|'.join(f"{find('full',case,'dual_nsga',g,'minimum')['mean_metrics'][2]:.6g}" for g in [260,280,300,600,1000])+'|','',
                  f'![{case}代数统计](figures/{case}_evolution.png)','',
                  f'![{case}固定seed曲线](figures/{case}_fixed_seeds.png)','',
                  '### 选解影响','',
                  '| 代数 | 膝点平均WSMSE | 同前沿最小WSMSE平均 | 膝点最好值 | 同前沿最小WSMSE最好值 | 无凸膝点次数 / 100 |','|---:|---:|---:|---:|---:|---:|']
        for g in [300,1000]:
            k=find('full',case,'dual_nsga',g);w=find('full',case,'dual_nsga',g,'minimum')
            lines.append(f"|{g}|{k['mean_metrics'][2]:.6g}|{w['mean_metrics'][2]:.6g}|{k['best']['wsmse']:.6g}|{w['best']['wsmse']:.6g}|{k['knee_fallbacks']}|")
        lines += ['','膝点只在已有前沿选择折中解，不能创造未搜索到的参数。其WSMSE高于同前沿最小值是取舍，不能直接称为选错或程序错误。','']
    lines += ['## Case2：是否只是实验条件不一致','',
              '论文Table V列Case2为96频点；Fig.18图示范围约10kHz–30MHz，并比较WSMSE基线140代与原多目标方法72代的等时间结果。当前201点40Hz–110MHz、双幅相主目标和相同代数比较都与其不同。这里固定取原CSV范围内108点，不插值、不平滑，不声称复现原96点或原全部辅助目标方法。','',
              '| 代数 | 组别 | 全频训练，统一108点评价：平均 | 频段训练，同108点评价：平均 | 频段训练最好WSMSE |','|---:|---|---:|---:|---:|']
    for g in [300,1000]:
        for method in ['scalar_ga','dual_nsga']:
            full=find('full','case2',method,g);band=find('paper_band','case2',method,g)
            lines.append(f"|{g}|{LABEL[method]}|{full['mean_paper_band_wsmse']:.6g}|{band['mean_metrics'][2]:.6g}|{band['best']['wsmse']:.6g}|")
    lines += ['', '全频与频段训练的主目标不同；表中统一108点消除了评价频率不同造成的不可比。改频段只用于条件敏感性验证，不能拿频段内好成绩声称全部201点拟合成功。','',
              '![Case2频段训练曲线](figures/case2_paper_band_best_1000.png)','',
              '## 可达性和局部搜索诊断','',
              'Case1真参数在当前模型和180训练点上WSMSE为零；因此不能说幅相损失在数学上无法拟合这些已知结构。下面从每组最佳进化参数启动有界最小二乘，残差为[dB差,√α×rad差]/√N，max_nfev=2000，边界不变。有限差分模型查询数计入独立成本；停止或达到上限都不证明全局最优。','',
              '| 条件 | Case | 组别 | 起点代数 | 初始WSMSE | 精修后WSMSE | 模型查询数 | 求解状态 |','|---|---|---|---:|---:|---:|---:|---:|']
    for r in POLISH:
        lines.append(f"|{r['condition']}|{r['case']}|{LABEL[r['method']]}|{r['generation']}|{r['initial_wsmse']:.6g}|{r['final_metrics'][2]:.6g}|{r['actual_model_queries']}|{r['status']}|")
    lines += ['', '![Case1精修诊断](figures/case1_polish.png)','', '![Case2精修诊断](figures/case2_polish.png)','',
              '训练分数降低若仍伴随加密曲线偏差，说明采样指标对窄峰的约束不足。精修只验证从该起点还可下降，不验证随机进化一定应在300代找到好解。Case2的16MHz测点相位107°，正值无源RLC无法精确表示；它单独贡献的WSMSE理论下限仅0.007612，不能解释数值上远大的全部残差。其余残差不能据此全部归因于噪声，尚无真参数或重复测量佐证。','',
              '## 程序与结果核验','']
    v=json.loads((OUT/'VALIDATION.json').read_text());p=json.loads((OUT/'PREFLIGHT.json').read_text())
    lines += [f"- 最终800个seed结果完整，没有剔除seed；首次两次复现断言中止保留后重新跑完。独立重算{v['individuals_recomputed']:,}个体/阶段记录的幅相损失，目标维数、参数边界、初始化和正式评价预算均通过。",
              f"- 400次300代核查有398次匹配，匹配组最大log参数差{v['maximum_matched_replay_parameter_difference']:.3g}，最大指标差{v['maximum_matched_replay_metric_difference']:.3g}。",
              '- Case2双目标2334、2365两seed与历史不匹配。原回调与新回调在当前服务器均得到相同轨迹，早期历史参数差约1e-15–1e-14后发生选择分叉，符合浮点敏感性特征；具体触发仍未完全定位。不能写成全量历史复现通过，不能凭本轮证明程序绝对没有问题。',
              '- 主实验135,000,000次评价；首次两次中止额外90,000次，8次300代复现探针额外360,000次，有界精修成本见表。补跑保留同初始化和同种子，且要求300代与首次尝试逐位一致，不按拟合成绩挑选重跑结果。',
              '- 独立节点导纳法与原串并联公式一致。广边界Case2节点矩阵病态，普通双精度独立求解器曾产生偏差；重建60位高精度导纳后通过。此问题属于新检查器的数值精度，未修改正式模型或优化。',
              '- 独立损失重算、Case1真值零误差、传统GA对WSMSE与负倒数的排序/短运行一致检查通过。以上检查范围内未发现实现错误，不能推广为所有程序绝对没有bug。','',
              '| Case | 独立电路最大相对差 | 独立损失 | GA标量变换排序运行 |','|---|---:|---|---|']
    for x in p['checks']:lines.append(f"|{x['case']}|{x['nodal_max_relative_error']:.3g}|通过|一致|")
    lines += ['', '## 数据与复核入口','',
              '[全seed与阶段误差表](all_seed_stage_metrics.csv) · [配对差与bootstrap区间](PAIRED.json) · [最佳代表及参数](SUMMARY.json) · [核验结果](VALIDATION.json) · [独立预检](PREFLIGHT.json) · [服务器来源哈希](SERVER_PROVENANCE.json) · [精修状态及成本](POLISH_SUMMARY.json) · [历史复现分叉探针](REPLAY_PROBE.json)','',
              '完整末代种群、初始化、每20代参数和1000代历史保留在本地实验目录server_run/retrieved及服务器对应输出目录。原始服务器回收压缩包与逐文件哈希均核验，未筛选隐藏失败结果。旧程序、原96频点和7月结果当前缺失，历史效果是否退化仍缺直接证据。','',
              f'[实验方案与复现异常后的流程修订](PROTOCOL.md) · [论文来源]({ROOT.as_posix()}/Feature-Engineering_Enabled_Multi-Objective_Evolutionary_Impedance_Fitting_Technique.pdf)','']
    text='\n'.join(lines)
    absolute=re.sub(r'\]\((?!https?://|[A-Za-z]:/)([^)]+)\)',lambda m:']('+str(OUT/m.group(1)).replace('\\','/')+')',text)
    (OUT/'REPORT.md').write_text(absolute,encoding='utf8')
    html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>老师疑点实验核验</title><style>body{max-width:1140px;margin:40px auto;padding:0 28px;color:#172033;font:16px/1.7 "Microsoft YaHei",sans-serif}h1,h2,h3{line-height:1.4}img{max-width:100%}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:7px;border:1px solid #d7dfeb;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#edf2f8}a{color:#2563eb}</style>'+markdown.markdown(text,extensions=['tables'])+'</html>'
    (OUT/'index.html').write_text(html,encoding='utf8')
    for link in re.findall(r'\]\(([^)]+)\)',absolute):
        if not link.startswith('http'):assert Path(link).is_file(),link
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ['DELIVERY.json','老师疑点_实验核验.zip']]
    record=dict(passed=True,figures=len(list((OUT/'figures').glob('*.png'))),figures_visually_reviewed=11,
                output_sha256={p.relative_to(OUT).as_posix():sha(p) for p in files},source_sha256={p.name:sha(p) for p in list(HERE.glob('*.py'))+list(HERE.glob('*.md'))})
    dump(OUT/'DELIVERY.json',record)
    with zipfile.ZipFile(OUT/'老师疑点_实验核验.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[OUT/'DELIVERY.json']:z.write(p,p.relative_to(OUT).as_posix())
    print(json.dumps(dict(delivery='passed',figures=record['figures'],files=len(files))))

if __name__=='__main__':main()

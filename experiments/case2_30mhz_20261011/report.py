from core import *
import shutil,re,zipfile
sys.path.append(str(ROOT/'tmp/curve_refinement_deps'))
import markdown
SUMMARY=json.loads((OUT/'SUMMARY.json').read_text())
LABEL={'scalar_ga':'WSMSE传统GA','scalar_nsga':'WSMSE当前框架','dual_nsga':'双目标膝点'}

def main():
    shutil.copy2(HERE/'PROTOCOL.md',OUT/'PROTOCOL.md')
    lines=['# Case2正式拟合范围修订：40 Hz–30 MHz','',
           '**此前Case2确实使用了40 Hz–110 MHz的全部201点。按老师的新要求，正式拟合上限已改为30 MHz，保留183个原始测点，实际末点为29 MHz。30 MHz以上18点不参与目标、评估或展示。** Case1本来最高1 MHz，不需重跑。','',
           '此前10kHz–30MHz的108点实验仅是诊断，不是本轮设置。本轮保留全部原低频测点，不平滑、不插值，也没有删去范围内的相位107°测点。之前针对30MHz以上拟合不好的判断不适用于当前任务范围。','',
           '服务器完成三方法各100seed（2300–2399），同seed同初始化、相同边界、alpha=17.38和SBX/PM，种群150、1000代，共300次、45,000,000次评价。300代结果独立展示，不能与1000代混合比较。没有加精修或辅助目标。','']
    for g in [300,1000]:
        group=[r for r in SUMMARY if r['generation']==g]
        lines += [f'## {g}代：目标与最终拟合','',f'![{g}代最终拟合](figures/case2_30mhz_best_{g}.png)','',
                  '| 方法 | 平均WSMSE / 100seed | 最好WSMSE | 最好seed | 幅值RMSE / dB | 相位RMSE / ° | 最大幅值偏差 / dB | 最大相位偏差 / ° |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in group:
            b=r['best'];lines.append(f"|{LABEL[r['method']]}|{r['mean_metrics'][2]:.6g}|{b['wsmse']:.6g}|{b['seed']}|{np.sqrt(b[METRICS[0]]):.3f}|{np.rad2deg(np.sqrt(b[METRICS[1]])):.3f}|{b['max_amplitude_db']:.3f}|{b['max_phase_deg']:.3f}|")
        dual=next(r for r in group if r['method']=='dual_nsga')
        lines += ['',f"双目标每seed先选归一化前沿的最大凸垂距膝点，再在100个代表中按WSMSE选最佳。最佳seed的选解状态为`{dual['best']['reason']}`；本阶段{dual['knee_fallbacks']}/100前沿缺少凸膝点，按预定规则取退化折中并明确记录。主图不是平均seed，不能代替总体统计。",'',
                  f'[逐点曲线](curves_{g}.csv) · [电路参数](parameters_{g}.csv)','']
    lines += ['## 改上限的影响：统一在183点上评价','',
              '上轮40Hz–110MHz训练的每seed最终代表重新在本轮同一183点计算WSMSE。双目标仍使用各自末代前沿的膝点。下表是同seed配对比较，避免直接比较201点与183点上的原始训练分数。','',
              '| 代数 | 方法 | 原全频训练，在183点评价：平均WSMSE | 本轮30MHz内训练：平均WSMSE | 配对均值差（本轮−原全频）95% bootstrap区间 | 本轮较好seed / 100 |','|---:|---|---:|---:|---|---:|']
    for r in SUMMARY:
        ci=r['paired_ci95'];lines.append(f"|{r['generation']}|{LABEL[r['method']]}|{r['old_full_common183_mean']:.6g}|{r['mean_metrics'][2]:.6g}|[{ci[0]:.5g}, {ci[1]:.5g}]|{r['cap_better_seeds']}|")
    lines += ['', '![代数统计](figures/evolution.png)','',
              '## 判断','', (HERE/'VERDICT.md').read_text(encoding='utf8'),'','## 核验与数据','',
              '全部300次完整完成，无失败seed。120,000条末代个体/阶段代表的幅相损失独立重算通过；目标维数、边界、预算、原CSV哈希、保留频点、同seed初始化及膝点选解检查通过。双目标退化前沿没有删除；所有seed均保留。','',
              '[全部seed指标](all_seed_metrics.csv) · [全部阶段指标](all_stage_metrics.csv) · [汇总与最好参数](SUMMARY.json) · [核验结果](VALIDATION.json) · [实验方案](PROTOCOL.md)','',
              '本轮完整原始种群、初始化与历史在experiments/case2_30mhz_20261011/server_run/retrieved/case2_30mhz_20261011；服务器输出目录为/root/autodl-tmp/wsmse_teacher_check_20261011_workspace/output/case2_30mhz_20261011。压缩包SHA256和逐文件哈希核验后回收。','']
    text='\n'.join(lines);absolute=re.sub(r'\]\(([^)]+)\)',lambda m:']('+str(OUT/m.group(1)).replace('\\','/')+')',text)
    (OUT/'REPORT.md').write_text(absolute,encoding='utf8')
    (OUT/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Case2 30MHz对照</title><style>body{max-width:1140px;margin:36px auto;padding:0 26px;font:16px/1.7 "Microsoft YaHei",sans-serif;color:#172033}img{max-width:100%}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #d7dfeb;padding:7px;text-align:right}th{background:#edf2f8}a{color:#2563eb}</style>'+markdown.markdown(text,extensions=['tables'])+'</html>',encoding='utf8')
    for link in re.findall(r'\]\(([^)]+)\)',absolute):assert Path(link).is_file(),link
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ['DELIVERY.json','Case2_30MHz_对照结果.zip']]
    dump(OUT/'DELIVERY.json',dict(passed=True,source_sha256={p.name:sha(p) for p in HERE.glob('*') if p.suffix in ['.py','.md']},output_sha256={p.relative_to(OUT).as_posix():sha(p) for p in files},figures_visually_reviewed=3))
    with zipfile.ZipFile(OUT/'Case2_30MHz_对照结果.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[OUT/'DELIVERY.json']:z.write(p,p.relative_to(OUT).as_posix())
    print(json.dumps(dict(passed=True,files=len(files))))

if __name__=='__main__':main()

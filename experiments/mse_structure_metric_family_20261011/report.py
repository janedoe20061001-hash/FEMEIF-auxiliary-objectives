from core import *
import csv
from statistics import fmean

NAMES={'case1':'Case1','ladder8':'梯形网络8','mixed0p25':'混合支路0.25','three0p25':'三级网络0.25','control0p25':'简单RLC对照'}
METHODS={'MAG_PHA':'基础MAG/PHA','MAG_PHA_INTEGRAL':'原C：顺序累计','MAG_PHA_RADIAL':'R：径向极值','MAG_PHA_OCCUPANCY':'F：占据进程','MAG_PHA_ANGULAR':'T：角度进程','MAG_PHA_BRIDGE':'B：整段连接','MAG_PHA_EXCESS':'X：额外往返'}
def readcsv(name):
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    rows=readcsv('analysis/REPRESENTATIVES.csv');primary=[r for r in rows if r['rule']=='primary']
    integrity=json.loads((OUT/'INTEGRITY.json').read_text());audit=json.loads((OUT/'analysis/AUDIT.json').read_text());cpu=json.loads((OUT/'CPU_SUMMARY.json').read_text())
    sums=json.loads((OUT/'batch/summaries.json').read_text());assert len(primary)==140 and integrity['evaluations']==8400000
    paired=[];phase_rows=[]
    for ident in CFG['targets']:
        for seed in CFG['seeds']:
            a=next(r for r in primary if r['target']==ident and r['seed']==str(seed) and r['method']=='MAG_PHA')
            for method in CFG['methods']:
                b=next(r for r in primary if r['target']==ident and r['seed']==str(seed) and r['method']==method)
                both=float(b['mag'])<=float(a['mag']) and float(b['pha'])<=float(a['pha'])
                repaired=a['focus_joint_ok']=='False' and b['focus_joint_ok']=='True';broken=a['focus_joint_ok']=='True' and b['focus_joint_ok']=='False'
                paired.append(dict(target=ident,method=method,seed=seed,repaired=repaired,broken=broken,both_global_improve=both,repair_without_global_harm=repaired and both,mag_delta=float(b['mag'])-float(a['mag']),pha_delta=float(b['pha'])-float(a['pha']),local_phase_delta_deg=float(b['local_phase_rmse_deg'])-float(a['local_phase_rmse_deg']),focus_joint_ok=b['focus_joint_ok']=='True',all_amp_ok=b['all_amp_ok']=='True'))
                d=json.loads((OUT/f'analysis/details/{ident}_{method}_{seed}_primary.json').read_text(encoding='utf-8'))
                phase=d['diagnostic'].get('phase_events',[])
                phase_rows.append(dict(target=ident,method=method,seed=seed,phase_events=len(phase),phase_complete=sum(e.get('complete',False) for e in phase),other_amp_incomplete=sum(e['event_id'] not in CFG['focus_events'][ident] and e['width_evaluable'] and not e['complete'] for e in d['diagnostic']['events']),extra_amp=len(d['diagnostic']['extra_events'])))
    mod=load('family_report_csv',HERE/'analyze.py');mod.csvout(OUT/'analysis/PAIRED_COMPARISONS.csv',paired);mod.csvout(OUT/'analysis/OTHER_STRUCTURES.csv',phase_rows)
    note='# 五类新指标全部试过以后：配对实验结果\n\n本轮140/140任务完成：五个理论电路、七组方法、四个新seed。没有上传网站。\n\n'
    note+=f'建议先读[通俗结论]({(OUT/"FINDINGS.md").as_posix()})；这里保留完整数值和检查办法。\n\n'
    note+='## 先弄明白这次在比较什么\n\n把阻抗实部、虚部看作一个向量：它的长度是幅值，朝向是相位，频率告诉我们它在曲线中的位置。MAG/PHA检查整条曲线的幅值、相位平均平方误差。五个新指标分别补看峰谷的径向高度、频率占据分布、角度推进过程、峰谷之间连接段、额外往返。原C作为已经测试过的顺序累计对照。\n\n'
    note+='每个新指标单独作为第三目标加入MAG/PHA；没有一次堆上五项，也没有按结果调权。R/F/T/X是从三维曲线提取特征，B/C直接比较固定归一化的三维轨迹；不能把它们全叫欧氏距离。本轮B指连接段扩展，不同于上一轮叫B的弦方向候选。\n\n'
    note+=f'元件结构、目标参数与搜索范围见[电路说明]({(OUT/"TARGETS.md").as_posix()})；精确设置及公式见[开跑前的约定]({(OUT/"PROTOCOL.md").as_posix()})。\n\n'
    note+='## 1. 每组最后选出的曲线怎么样\n\n每个seed各从最后保留的200个候选中选MAG＋25.12×PHA最小者，各组使用同一个规则。基础目标在搜索中仍独立；这个加权式只决定如何展示代表。下面均值是四条曲线各自的分数平均，曲线本身没有平均。\n\n'
    note+='重点结构同时合格，表示选定峰谷的位置、高度、宽度都在既定容差内：位置≤0.02个log10单位、高度≤3 dB、宽度为目标0.5—2倍。它不等于整条曲线完全吻合，需同时查看幅相与其他结构。局部相位RMSE按固定完整连接跨度独立积分，单位为度。\n\n'
    note+='|目标|方法|重点结构合格|MAG均值 / dB²|PHA均值 / rad²|局部相位RMSE / °|\n|---|---|---:|---:|---:|---:|\n'
    for ident in CFG['targets']:
        for method in CFG['methods']:
            rr=[r for r in primary if r['target']==ident and r['method']==method];assert len(rr)==4
            note+=f'|{NAMES[ident]}|{METHODS[method]}|{sum(r["focus_joint_ok"]=="True" for r in rr)}/4|{fmean(float(r["mag"]) for r in rr):.6g}|{fmean(float(r["pha"]) for r in rr):.6g}|{fmean(float(r["local_phase_rmse_deg"]) for r in rr):.5g}|\n'
    note+='\n## 2. 局部修好是否付出了整体代价\n\n|目标|方法|修好失败结构 / 4|原合格变失败 / 4|幅相都不高于基线 / 4|修好且幅相均不退步 / 4|\n|---|---|---:|---:|---:|---:|\n'
    for ident in CFG['targets']:
        for method in CFG['methods'][1:]:
            rr=[r for r in paired if r['target']==ident and r['method']==method]
            note+=f'|{NAMES[ident]}|{METHODS[method]}|{sum(r["repaired"] for r in rr)}/4|{sum(r["broken"] for r in rr)}/4|{sum(r["both_global_improve"] for r in rr)}/4|{sum(r["repair_without_global_harm"] for r in rr)}/4|\n'
    note+='\n最后一列要求改善发生在同一个seed，避免把不同列的好处拼在一起。简单对照基础组若已合格，第三目标仍可能拖慢搜索；这也是筛选指标的代价。\n\n'
    note+='## 3. 换一种共同选解规则，结论是否一样\n\n补充检查允许每项MAG/PHA不超过配对基线主代表的1.10倍，再加固定0.01 dB²和(0.5°)²对应rad²。在这个范围内，各组都用原瓶颈最大值最小者选解，而非自己的辅助分数。代表另独立复算，有候选但复算超范围也不算合格；没候选计入分母。\n\n'
    note+='|目标|方法|范围内有代表 / 4|复算仍在范围 / 4|重点结构合格且仍在范围 / 4|\n|---|---|---:|---:|---:|\n'
    for ident in CFG['targets']:
        for method in CFG['methods']:
            rr=[r for r in rows if r['target']==ident and r['method']==method and r['rule']=='limited_tradeoff'];valid=[r for r in rr if r['secondary_reference_eligible']=='True']
            note+=f'|{NAMES[ident]}|{METHODS[method]}|{len(rr)}/4|{len(valid)}/4|{sum(r["focus_joint_ok"]=="True" for r in valid)}/4|\n'
    note+='\n此规则是搜索结束后的检查，搜索时并没有约束MAG/PHA退步；不能声称第三目标天然保护基础误差。\n\n'
    note+='## 4. 重点之外的曲线\n\n|目标|方法|全部可评价幅值事件合格 / 4|重点外幅值缺陷均数|额外幅值事件均数|相位事件通过数 / 检查数|\n|---|---|---:|---:|---:|---:|\n'
    for ident in CFG['targets']:
        for method in CFG['methods']:
            rr=[r for r in phase_rows if r['target']==ident and r['method']==method];pr=[r for r in primary if r['target']==ident and r['method']==method]
            note+=f'|{NAMES[ident]}|{METHODS[method]}|{sum(r["all_amp_ok"]=="True" for r in pr)}/4|{fmean(r["other_amp_incomplete"] for r in rr):.4g}|{fmean(r["extra_amp"] for r in rr):.4g}|{sum(r["phase_complete"] for r in rr)}/{sum(r["phase_events"] for r in rr)}|\n'
    note+='\n相位事件是相位变化率的转折检查，敏感于位置与形状，不等同于某段实际转角大小；局部RMSE和正反转角轨迹也必须一起看。幅值事件数只是缺陷数量，不代表严重程度。\n\n'
    note+='## 5. 固定seed的曲线例子\n\n以下固定使用预先编号最小的43000，所有组按主规则选代表。末列显示整个连接跨度，前列放大重点峰谷。图中R/F/T/B/X对应前文五候选，C为原对照。只画典型展示，不从140条中挑成绩最好的seed。\n\n'
    for ident in CFG['targets']:
        note+=f'### {NAMES[ident]}\n\n![幅相曲线]({(OUT/f"figures/{ident}_local.png").as_posix()})\n\n'
        if ident in ('case1','ladder8'):note+=f'![三维曲线]({(OUT/f"figures/{ident}_3d.png").as_posix()})\n\n三维图用原始f、Re Z、Im Z展示，轴比例会影响直观距离；优化评分使用目标固定尺度与log频率。\n\n'
    note+='## 6. 计算代价和CPU\n\n'
    note+=f'分配{cpu["allocated_cpus"]:.0f}核，32个单线程工作进程，优化阶段CPU平均{cpu["optimization_mean_percent"]:.2f}%，90%分位{cpu["optimization_p90_percent"]:.2f}%，内存峰值{cpu["peak_memory_bytes"]/2**30:.2f} GiB。每2秒记录真实CPU使用，平均包括尾部任务不足32个的时段。较长预测任务优先排队，没有运行无用计算抬高比例。\n\n'
    note+='|目标|方法|单seed均秒数|基础频率查询均数|辅助查询均数|\n|---|---|---:|---:|---:|\n'
    for ident in CFG['targets']:
        for method in CFG['methods']:
            rr=[r for r in sums if r['target']==ident and r['method']==method]
            note+=f'|{NAMES[ident]}|{METHODS[method]}|{fmean(r["seconds"] for r in rr):.1f}|{fmean(r["base_frequency_queries"] for r in rr):.0f}|{fmean(r["geometry_frequency_queries"] for r in rr):.0f}|\n'
    note+='\n每组60,000次候选评价相同，不等于耗时相同。频率查询不是浮点运算计数；并行时延也不作为孤立算法速度。\n\n'
    note+='## 7. 数值核查与结论边界\n\n'
    note+=f'140个任务共840万次评价，每个任务15个保存阶段；同seed初始种群、源输入SHA256、全部阶段文件SHA256已核验。独立复核{audit["curves"]}条主/补充代表，基础分数或结构稳定性异常{audit["structure_and_base_score_failures"]}条，辅助采样异常{audit["geometry_sampling_failures"]}条。原失败标记保留，见问题记录。两种规则可能选中同一条曲线。\n\n'
    note+='训练网格有限，代表评分加密不代表搜索中的所有候选都可靠。每个辅助还可能有自己的盲点：R不识别宽度和位置，F可能在结构远离窗口后饱和，T需要可靠相位，B对固定尺度与覆盖有依赖，X为平线打零分。因此分数降低不能单独充当成功。\n\n'
    note+='基础误差使用冻结的自适应频率积分与原测点权重，局部加密不直接按点数增加权重。独立局部相位RMSE则按完整局部跨度的log频率积分，所以它与全频PHA看不同范围，不能直接互相替代。\n\n'
    note+='四个seed属于初筛，所有五电路均参与过开发，尚未做未见电路或噪声数据验证。本轮两目标变三目标也改变种群保留行为，没有无关第三目标对照，不能把所有差异唯一归于新指标的信息。所有理论目标存在零误差参数，不宣称MSE理论无法拟合。未组合指标、未扩大种子、未上传网站。\n\n'
    for name,label in [('FINDINGS.md','通俗结论与四个主线问题'),('ISSUES.md','问题、处理与影响'),('analysis/REPRESENTATIVES.csv','逐seed独立分数'),('analysis/EVENTS.csv','逐事件连续偏差'),('analysis/PAIRED_COMPARISONS.csv','逐seed配对代价'),('analysis/OTHER_STRUCTURES.csv','其他幅相结构'),('analysis/RETAINED_SCORES.csv','最后种群摘要'),('analysis/AUDIT.json','原数值核查'),('CPU_SAMPLES.csv','CPU记录')]:note+=f'- [{label}]({(OUT/name).as_posix()})\n'
    (OUT/'REPORT.md').write_text(note,encoding='utf-8');dump(OUT/'REPORT_SUMMARY.json',dict(primary=primary,paired=paired,other=phase_rows))
    print('Numerical report and paired tables ready; interpretation remains to be reviewed',flush=True)

if __name__=='__main__':main()

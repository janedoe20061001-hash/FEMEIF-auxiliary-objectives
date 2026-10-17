"""Readable evidence tables, without classifying auxiliary loss as success."""
from core import *
import csv
from statistics import fmean

NAMES={'case1':'Case1','ladder8':'梯形网络倍率8','mixed0p25':'混合支路倍率0.25','control0p25':'单并联RLC对照'}
METHODS={'MAG_PHA':'基础MAG/PHA','MAG_PHA_MAX':'原版：瓶颈最大值','MAG_PHA_MEAN':'A：瓶颈等份平均','MAG_PHA_CHORD':'B：A＋三维走向/展开量','MAG_PHA_INTEGRAL':'C：顺序累计偏差'}

def main():
    with (OUT/'analysis/REPRESENTATIVES.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    integrity=json.loads((OUT/'INTEGRITY.json').read_text());audit=json.loads((OUT/'analysis/AUDIT.json').read_text());cpu=json.loads((OUT/'CPU_SUMMARY.json').read_text())
    sums=json.loads((OUT/'batch/summaries.json').read_text())
    assert integrity['checked_jobs']==80 and integrity['evaluations']==4800000
    primary=[r for r in rows if r['rule']=='primary'];assert len(primary)==80
    note='# 修订三维指标：新一轮实验结果\n\n2026年10月11日。本轮五种方法、四个目标、各四个新seed，80/80正式任务完成。没有上传网站。\n\n'
    note+='## 先看这次最有用的结果\n\n**候选C（顺序累计三维偏差）目前最值得继续验证，但还不能当成通用指标。** 梯形网络重点谷峰由基础组0/4合格变为C组4/4，四个配对整体幅值、相位误差也都下降。混合支路由2/4变为4/4，原先失败的两个seed同时改善结构和整体幅相；原先已经合格的两个seed整体幅相反而退步，有限取舍检查中C也只有2/4。\n\n'
    note+='Case1五种方法的重点峰谷仍都是0/4，局部变好还没有变成组合与全曲线同时变好。A只改汇总，在梯形网络有部分帮助；B加走向和长度后，没有稳定超过A。只有四个seed，合格是既定门槛下的判断，不表示曲线完全重合。\n\n'
    note+=f'若第一次接触这些实验，建议先读[结果怎么理解]({(OUT/"FINDINGS.md").as_posix()})，再查下面的完整表格。辅助评分复核有8条不一致记录，均在单并联RLC对照中；基础分数和结构检查没有异常，原失败标记保留。\n\n'
    note+='## 从零开始，先说这次在比较什么\n\nMAG/PHA分别是整条曲线的幅值、相位平均平方误差，分数越低越好。辅助指标另外关心局部三维轨迹有没有伸出、转向、折回。每个seed是一组可重复的随机起跑；五种方法从同样的200组参数开始，各搜索300代。全部元件参数同时未知。\n\n'
    note+=f'候选设计、公式及要求见[通俗设计稿]({(ROOT/"docs/三维曲线指标_新一轮候选与实验约定_20261011.md").as_posix()})。目标电路图、参数及范围见[电路说明]({(OUT/"TARGETS.md").as_posix()})。\n\n'
    note+='|方法|想检验的改动|\n|---|---|\n|基础MAG/PHA|不加辅助目标|\n|原版|峰谷瓶颈分数取最大值，作为新seed上的重复对照|\n|A|只将峰谷分数改为等份平均|\n|B|在A上加三档三维箭头方向和长度误差，系数固定1|\n|C|顺序匹配整段累计偏差，各窗口等份汇总|\n\n'
    note+='原始实部、虚部只按目标固定尺度归一化，频率为log10(f)；没有对候选曲线自行缩放、旋转或平移。各分区窗口仍未完全覆盖峰谷之间的连接段，局部相位另查。候选下降不自动表示曲线修好。\n\n'
    note+='## 1. 按统一的幅相规则选代表\n\n五组都从末代保留种群选MAG＋25.12×PHA最小者，不查看辅助分数。优化时基础两目标仍独立，此加权式只用于选解。结构合格要求重点峰谷同时通过原来的位置、高度、宽度门槛：中心偏差≤0.02个log10单位，高度≤3 dB，宽度为目标0.5—2倍。不是每点完全重合，连续偏差仍保留。\n\n'
    note+='|目标|方法|重点结构同时合格|MAG均值 / dB²|PHA均值 / rad²|局部相位RMSE均值 / °|\n|---|---|---:|---:|---:|---:|\n'
    for ident in CFG['targets']:
        for method in CFG['methods']:
            rr=[r for r in primary if r['target']==ident and r['method']==method];assert len(rr)==4
            note+=f'|{NAMES[ident]}|{METHODS[method]}|{sum(r["focus_joint_ok"]=="True" for r in rr)}/4|{fmean(float(r["mag"]) for r in rr):.6g}|{fmean(float(r["pha"]) for r in rr):.6g}|{fmean(float(r["local_phase_rmse_deg"]) for r in rr):.5g}|\n'
    note+='\n每条代表独立复算，再取四个seed均值；曲线本身不平均。各电路局部相位窗口不同，数值不作为跨电路难度排行。旧seed结果不混入此表。\n\n'
    note+='## 2. 每个配对有没有整体代价？\n\n|目标|新增方法|结构从失败变合格 / 4|从合格变失败 / 4|MAG和PHA都不高于基线的配对 / 4|\n|---|---|---:|---:|---:|\n'
    for ident in CFG['targets']:
        for method in CFG['methods'][1:]:
            pairs=[]
            for seed in CFG['seeds']:
                a=next(r for r in primary if r['target']==ident and r['method']=='MAG_PHA' and r['seed']==str(seed))
                b=next(r for r in primary if r['target']==ident and r['method']==method and r['seed']==str(seed))
                pairs.append((a,b))
            repaired=sum(a['focus_joint_ok']=='False' and b['focus_joint_ok']=='True' for a,b in pairs)
            broken=sum(a['focus_joint_ok']=='True' and b['focus_joint_ok']=='False' for a,b in pairs)
            both=sum(float(b['mag'])<=float(a['mag']) and float(b['pha'])<=float(a['pha']) for a,b in pairs)
            note+=f'|{NAMES[ident]}|{METHODS[method]}|{repaired}/4|{broken}/4|{both}/4|\n'
    note+='\n第三列改善与最后一列整体改善可能发生在不同seed，不能将两列拼成同一条曲线的成功；逐seed记录可以核对。\n\n'
    note+='## 3. 有限幅相取舍下的共同选法\n\n各组候选的MAG/PHA分别不超过配对基线主代表的1.10倍，再加固定下限0.01 dB²和(0.5°)²对应rad²。在此范围内统一选原瓶颈最大值最小者，而不是各自选自己的辅助分数。然后独立复算是否仍达标。没有候选仍计入分母。\n\n'
    note+='|目标|方法|训练范围内有代表 / 4|独立复算仍在范围内 / 4|结构合格且仍在范围内 / 4|\n|---|---|---:|---:|---:|\n'
    for ident in CFG['targets']:
        for method in CFG['methods']:
            rr=[r for r in rows if r['target']==ident and r['method']==method and r['rule']=='limited_tradeoff']
            valid=[r for r in rr if r['secondary_reference_eligible']=='True']
            note+=f'|{NAMES[ident]}|{METHODS[method]}|{len(rr)}/4|{len(valid)}/4|{sum(r["focus_joint_ok"]=="True" for r in valid)}/4|\n'
    note+='\n这是一项明确的取舍检查，不是搜索过程中已限制幅相退步；不能声称辅助目标天然保护整条曲线。\n\n'
    note+='## 4. 其他结构有没有受损？\n\n|目标|方法|重点之外不合格幅值事件均数|多余幅值事件均数|全部可评价目标幅值事件合格 / 4|\n|---|---|---:|---:|---:|\n'
    for ident in CFG['targets']:
        for method in CFG['methods']:
            details=[json.loads((OUT/f'analysis/details/{ident}_{method}_{s}_primary.json').read_text()) for s in CFG['seeds']]
            incomplete=[sum(e['event_id'] not in CFG['focus_events'][ident] and e['width_evaluable'] and not e['complete'] for e in d['diagnostic']['events']) for d in details]
            note+=f'|{NAMES[ident]}|{METHODS[method]}|{fmean(incomplete):.4g}|{fmean(len(d["diagnostic"]["extra_events"]) for d in details):.4g}|{sum(d["summary"]["all_amp_ok"] for d in details)}/4|\n'
    note+='\n事件数量不是严重程度；全部目标事件合格也不表示没有其他起伏或相位完全正确。\n\n'
    note+='## 5. 固定seed的曲线例子\n\n固定展示新seed中编号最小的42000，所有方法都按主选解规则，不平均曲线、不依据成绩挑seed。图中的A、B、C对应前文定义；完整判断仍依据四个配对。\n\n'
    for ident in CFG['targets']:
        note+=f'### {NAMES[ident]}\n\n![局部幅相]({(OUT/f"figures/{ident}_local.png").as_posix()})\n\n'
        if ident in ('case1','ladder8'):note+=f'![三维轨迹]({(OUT/f"figures/{ident}_3d.png").as_posix()})\n\n三维展示为原始f、Re Z、Im Z，透视和轴比例影响视觉距离，分数使用固定归一化坐标。\n\n'
    note+='## 6. 开销和CPU利用率\n\n'
    note+=f'服务器分配{cpu["allocated_cpus"]:.0f}核，优化阶段每2秒采样，包含任务末尾的CPU平均利用率约{cpu["optimization_mean_percent"]:.2f}%，90%分位约{cpu["optimization_p90_percent"]:.2f}%。内存峰值约{cpu["peak_memory_bytes"]/2**30:.2f} GiB。没有运行无用计算来抬高利用率。\n\n'
    note+='采用32个单线程优化进程，队列自动补位。主体计算时实测接近100%；最后剩余任务少于32个，空出来的核没有新的正式任务可接，平均值因此低于100%。正式优化约13.2分钟，连同预检、冒烟与独立核验到打包约14.0分钟。后续批次可用已有耗时预估优先安排较长任务，减少尾部等待；本轮没有为抬高利用率追加seed或更改预算。\n\n'
    note+='|目标|方法|单seed平均秒数|基础频点查询均数|辅助频点查询均数|\n|---|---|---:|---:|---:|\n'
    for ident in CFG['targets']:
        for method in CFG['methods']:
            rr=[r for r in sums if r['target']==ident and r['method']==method]
            note+=f'|{NAMES[ident]}|{METHODS[method]}|{fmean(r["seconds"] for r in rr):.1f}|{fmean(r["base_frequency_queries"] for r in rr):.0f}|{fmean(r["geometry_frequency_queries"] for r in rr):.0f}|\n'
    note+='\n同为60,000次评价，不等于计算成本相同。频点查询数不是浮点运算数；耗时受32任务并行负载与不同参数的自适应采样影响，不当成孤立算法速度。\n\n'
    note+='## 7. 核查与结论范围\n\n'
    note+=f'80任务、4,800,000次正式评价、每任务15个保存阶段、五组同seed初始化和源输入/阶段哈希已核验。共{audit["curves"]}条主/补充代表记录，基础分数或结构稳定性异常{audit["structure_and_base_score_failures"]}条，辅助评分采样异常{audit["geometry_sampling_failures"]}条。两条规则可能选中同一个候选。异常原记录保留，不能把程序完成当成全部核验通过。\n\n'
    note+='辅助评分复核用257、513、1025点及候选模态锚点，尺度与容差维持绝对0.01＋相对1%。两目标变三目标也会改变候选保留方式；本轮没有无关第三目标对照，不能唯一归因于某种几何信息。单峰对照的A与原最大值公式相同，另检查四个seed最终种群是否完全一致。\n\n'
    note+='8条辅助采样异常对应5条不同参数曲线；补到2049、4097级后在原容差下稳定，训练分数均偏高。补充计算只用于核查，不重跑优化、不替换原AUDIT.json中的失败标记。三种开发难例的代表没有出现这类异常，但未证明搜索中每个候选的辅助采样都可靠。\n\n'
    note+='只有四个seed，全部目标都参与开发，尚不能宣称普遍有效或证明MSE理论无法拟合。不同模型的参数维数与边界不同。主结果、有限取舍、数值问题、成本分别解读。\n\n'
    for name,label in [('FINDINGS.md','新结论与四个问题'),('ISSUES.md','问题与影响'),('analysis/REPRESENTATIVES.csv','逐seed独立分数'),('analysis/EVENTS.csv','逐事件连续偏差'),('analysis/PAIRED_COMPARISONS.csv','相同seed的逐对比较'),('analysis/AUDIT.json','核验原记录'),('analysis/SUPPLEMENTAL_GEOMETRY_AUDIT.json','异常分数的补充复核'),('CPU_SAMPLES.csv','CPU逐次采样')]:
        note+=f'- [{label}]({(OUT/name).as_posix()})\n'
    (OUT/'REPORT.md').write_text(note,encoding='utf-8')
    print('Readable evidence report generated; interpretation remains an independent step',flush=True)

if __name__=='__main__':main()

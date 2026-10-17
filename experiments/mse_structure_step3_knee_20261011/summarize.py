"""Shared pairing tables for the replacement, with no scalar-loss selection."""
from reselect import *
from collections import Counter

LABELS={'MAG_PHA':'基础MAG/PHA','MAG_PHA_GEOMETRY':'G：原瓶颈','MAG_PHA_MAX':'G：原瓶颈','MAG_PHA_MEAN':'A：平均瓶颈','MAG_PHA_CHORD':'D：方向/伸展','MAG_PHA_INTEGRAL':'C：顺序累计','MAG_PHA_RADIAL':'R：径向极值','MAG_PHA_OCCUPANCY':'F：频率占据','MAG_PHA_ANGULAR':'T：角度进程','MAG_PHA_BRIDGE':'B：跨窗连接','MAG_PHA_EXCESS':'X：额外往返'}
TITLE={'pilot':'第一轮：原局部瓶颈指标','revision':'第二轮：汇总、方向和顺序累计修订','family':'第三轮：五类结构候选初筛'}

def readcsv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def truth(r,k):return str(r.get(k)).lower()=='true'

def main():
    groups=[];pairs=[];rows=[]
    for batch in BATCHES:
        rr=readcsv(OUT/batch/'REPRESENTATIVES.csv');rows.extend(rr)
        for target in sorted({r['target'] for r in rr}):
            for method in sorted({r['method'] for r in rr}):
                subset=[r for r in rr if r['target']==target and r['method']==method]
                if not subset:continue
                pp=[]
                for r in subset:
                    b=next(b for b in rr if b['target']==target and b['seed']==r['seed'] and b['method']=='MAG_PHA')
                    dm=float(r['mag'])-float(b['mag']);dp=float(r['pha'])-float(b['pha'])
                    pp.append(dict(batch=batch,target=target,method=method,seed=int(r['seed']),mag_delta=dm,pha_delta=dp,local_phase_delta=float(r['local_phase_rmse_deg'])-float(b['local_phase_rmse_deg']),focus_repaired=truth(r,'focus_joint_ok') and not truth(b,'focus_joint_ok'),both_mag_pha_nonworse=dm<=1e-12 and dp<=1e-12,focus_ok_and_both_nonworse=truth(r,'focus_joint_ok') and dm<=1e-12 and dp<=1e-12,extra_amp=int(r['extra_amp']),extra_amp_delta=int(r['extra_amp'])-int(b['extra_amp'])))
                pairs.extend(pp)
                selections=[json.loads((OUT/batch/f"selections/{target}_{method}_{r['seed']}.json").read_text()) for r in subset]
                groups.append(dict(batch=batch,target=target,method=method,n=len(subset),focus_pass=sum(truth(r,'focus_joint_ok') for r in subset),all_amp_pass=sum(truth(r,'all_amp_ok') for r in subset),all_phase_pass=sum(truth(r,'all_phase_ok') for r in subset),both_nonworse=sum(p['both_mag_pha_nonworse'] for p in pp),focus_and_both=sum(p['focus_ok_and_both_nonworse'] for p in pp),repaired=sum(p['focus_repaired'] for p in pp),mean_mag=float(np.mean([float(r['mag']) for r in subset])),mean_pha=float(np.mean([float(r['pha']) for r in subset])),mean_local_phase_deg=float(np.mean([float(r['local_phase_rmse_deg']) for r in subset])),extra_amp_total=sum(int(r['extra_amp']) for r in subset),extra_phase_total=sum(int(r['extra_phase']) for r in subset),knee_count=sum(s['reason']=='convex_knee' for s in selections)))
    csvout(OUT/'GROUPS.csv',groups);csvout(OUT/'PAIRED.csv',pairs);csvout(OUT/'REPRESENTATIVES.csv',rows)
    dump(OUT/'SUMMARY.json',dict(jobs=len(rows),original_evaluations=len(rows)*60000,new_optimization_evaluations=0,stages=len(rows)*15,groups=groups,knees={b:json.loads((OUT/b/'STATUS.json').read_text())['knee_counts'] for b in BATCHES}))
    lines=['# 第三步：统一按MAG/PHA膝点重新选解的完整结果','',
      '2026年10月11日。本版替换第三步三轮旧选解结果；每seed最终只使用第300代MAG/PHA二维前沿膝点，无明确膝点时使用已冻结的退化折中代表。辅助指标不参与最终代表的挑选。','',
      '252个任务来自三轮已完成的完整优化种群，每任务60000次评价，合计15120000次；本次没有新增优化评价。全部3780个阶段重选、50400个末代候选训练分数重算，252条最终曲线重新独立加密复核。','',
      '## 怎么读下面的表','',
      '“重点过关”同时要求指定峰谷的位置、高度、宽度在原门槛内，不等于精确重合。“重点＋幅相不退步”还要求与同seed基础组相比，两项全频MSE都不增加。只改善一个目标时不会计入这个数。表中MSE是先计算每seed代表误差，再等权平均；没有平均曲线。','',
      '“额外幅值事件”是目标没有的可检测峰谷总数；“全相位过关”是全频相位事件都在门槛内的seed数。“真膝点”表示找到有正凸出的前沿，不包含退化折中。每方法前沿分别归一化，归一化坐标不可当作跨组质量分数。','',
      '指标代号：G原瓶颈，A平均瓶颈，D旧方向/伸展（原修订稿称B，避免与新跨窗B混淆），C顺序累计，R径向极值，F频率占据，T角度进程，B跨窗连接，X额外往返。','']
    for batch in BATCHES:
        lines+=['## '+TITLE[batch],'',f"seed固定为{ {'pilot':'41000–41003','revision':'42000–42003','family':'43000–43003'}[batch]}，不同轮不按编号伪配对。",'',
                '|电路|指标|重点过关|重点＋幅相不退步|MAG / dB²|PHA / rad²|局部相位RMSE / °|全相位过关|额外幅值事件|真膝点|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for g in (g for g in groups if g['batch']==batch):
            lines.append(f"|{g['target']}|{LABELS[g['method']]}|{g['focus_pass']}/{g['n']}|{g['focus_and_both']}/{g['n']}|{g['mean_mag']:.6g}|{g['mean_pha']:.6g}|{g['mean_local_phase_deg']:.3f}|{g['all_phase_pass']}/{g['n']}|{g['extra_amp_total']}|{g['knee_count']}/{g['n']}|")
        lines+=['',f"[逐seed误差]({batch}/REPRESENTATIVES.csv) · [逐峰谷及目标宽度单位位置偏差]({batch}/EVENTS.csv) · [阶段代表]({batch}/STAGES.csv) · [数值复核]({batch}/AUDIT.json)",'']
    lines+=['## 固定seed曲线','',
      '每轮固定使用冻结配置中的第一个seed；不挑冠军、不平均曲线。第三轮各方法同图，虚线为目标。局部放大看峰谷与相位，三维图保留物理Re Z/Im Z。图例中C是原顺序累计，B是新跨窗；可在逐seed记录中查到该曲线是真膝点还是退化代表。','']
    for t in ['case1','ladder8','mixed0p25','three0p25','control0p25']:
        lines += [f'### {t}：seed 43000','',f'![{t}局部曲线](family/figures/{t}_local.png)','']
    for t in ['case1','ladder8']:lines += [f'![{t}三维曲线](family/figures/{t}_3d.png)','']
    lines+=['### 膝点怎样被选中','', '红星是新代表；虚线连接两个归一化极端点。此图按方法分别归一化，只展示选法，不能比较红星坐标判断哪个方法更好。','', '![固定seed二维前沿](family/figures/fronts.png)','',
      '## 评价条件与边界','',
      '全部是模型正确、真参数在搜索范围内的理论开发目标，每条件只有4seed。没有未见电路、实测数据、噪声、更多seed或组合指标检验。结果描述规定预算下的搜索和选解，不证明MSE理论无法拟合。辅助组搜索实际为MAG/PHA加第三目标；最终用其保留种群的MAG/PHA投影前沿选解，可能舍弃辅助目标优异但基础目标被支配的解。','',
      '膝点取决于当次保留前沿和极端点，离群极端值、前沿稀疏、不同目标跨度会影响最终折中。它不是唯一正确的定义，也不是全局最优保证。原始前沿和归一化记录全部可复算。','',
      '固定位置门槛0.02decade，对极窄峰谷可能太宽松。高度门槛3dB、宽度比0.5–2均沿用；连续位置偏差另以目标3dB宽度单位提供，没有根据本轮成绩改变合格门槛。','',
      '[通俗解释](FINDINGS.md) · [问题及影响](ISSUES.md) · [逐seed配对](PAIRED.csv) · [膝点选法与核验约定](../../experiments/mse_structure_step3_knee_20261011/PROTOCOL.md)','']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(groups=len(groups),representatives=len(rows),knees=Counter(json.loads(p.read_text())['reason'] for p in OUT.glob('*/selections/*.json'))),ensure_ascii=False))

if __name__=='__main__':main()

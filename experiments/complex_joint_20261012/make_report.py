"""Beginner report, exhaustive paired tables, uncertainty, figures and issue ledger."""
from pathlib import Path
import json,csv,hashlib,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'output/complex_joint_20261012'
# Read recovered frozen experiment, not potentially edited active models.
RECOVERED=OUT/'retrieved';DATA=RECOVERED/'output/complex_joint_20261012';CODE=RECOVERED/'experiments/complex_joint_20261012'
import numpy as np

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def writecsv(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def link(p,label):return f'[{label}]({p.resolve().as_posix()})'
def bootstrap_mean(v):
    v=np.asarray(v);rng=np.random.default_rng(20261012)
    means=np.mean(rng.choice(v,size=(10000,len(v)),replace=True),axis=1)
    return np.quantile(means,[.025,.975]).tolist()

def plots(ident,cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=np.random.default_rng(20261012).choice(cfg['seeds'],3,replace=False).tolist()
    methods=['MAG_PHA','H','A','D_H_A','PTC_ref','C_ref']
    fig,ax=plt.subplots(len(methods),3,figsize=(14,3.3*len(methods)),squeeze=False)
    fig.suptitle(f'{ident} | fixed random seeds {selected} | final MAG/PHA knee',fontsize=13)
    for row,method in enumerate(methods):
        for j,seed in enumerate(selected):
            p=DATA/'jobs'/ident/method/f'seed_{seed}'/'audited_curve.npz'
            if not p.exists():continue
            with np.load(p) as a:x=a['x'];z=a['z'];t=a['target']
            if j==0:
                ax[row,0].plot(x,t.real,'k--',lw=1.8,label='Target');ax[row,1].plot(x,t.imag,'k--',lw=1.8);ax[row,2].plot(t.real,t.imag,'k--',lw=1.8)
            label=f'seed {seed}'
            ax[row,0].plot(x,z.real,lw=1,label=label);ax[row,1].plot(x,z.imag,lw=1);ax[row,2].plot(z.real,z.imag,lw=1)
        ax[row,0].set_ylabel(method+'\nRe Z / ohm');ax[row,1].set_ylabel('Im Z / ohm')
        ax[row,2].set_ylabel('Im Z / ohm')
        for k in range(3):ax[row,k].grid(alpha=.2)
        ax[row,0].legend(fontsize=7)
    ax[-1,0].set_xlabel('log10 frequency / Hz');ax[-1,1].set_xlabel('log10 frequency / Hz');ax[-1,2].set_xlabel('Re Z / ohm')
    fig.tight_layout(rect=(0,0,1,.98));p=OUT/'figures'/f'{ident.replace("/","_")}_joint.png';p.parent.mkdir(exist_ok=True)
    fig.savefig(p,dpi=140);plt.close(fig)
    # A 3D example, three independent seed curves, frequency retains correspondence.
    fig=plt.figure(figsize=(13,9))
    for i,method in enumerate(['MAG_PHA','H','A','D_H_A']):
        aa=fig.add_subplot(2,2,i+1,projection='3d')
        for j,seed in enumerate(selected):
            pp=DATA/'jobs'/ident/method/f'seed_{seed}'/'audited_curve.npz'
            if not pp.exists():continue
            with np.load(pp) as a:x=a['x'];z=a['z'];t=a['target']
            if j==0:aa.plot(x,t.real,t.imag,'k--',lw=2,label='Target')
            aa.plot(x,z.real,z.imag,lw=1,label=f'{seed}')
        aa.set_title(method);aa.set_xlabel('log10 f');aa.set_ylabel('Re Z / ohm');aa.set_zlabel('Im Z / ohm');aa.legend(fontsize=6)
    fig.tight_layout();p3=OUT/'figures'/f'{ident.replace("/","_")}_3d.png';fig.savefig(p3,dpi=140);plt.close(fig)
    return p,p3

def main():
    assert (OUT/'RECOVERY.json').exists(),'No verified recovery yet'
    recovery=read(OUT/'RECOVERY.json');assert recovery['verified'];cfg=read(CODE/'protocol.json');status=read(DATA/'STATUS.json')
    manifest=read(DATA/'MANIFEST.json');inputs=read(CODE/'inputs/manifest.json')
    records=[];issues=[];expected=cfg['planned_jobs'];initial={};sources_ok=True
    for name,digest in manifest['hashes'].items():
        p=RECOVERED/name
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            sources_ok=False;issues.append(f'输入/源代码哈希不一致：{name}')
    for ident in inputs['targets']:
        for method in cfg['methods']:
            for seed in cfg['seeds']:
                d=DATA/'jobs'/ident/method/f'seed_{seed}'
                if not (d/'summary.json').exists():issues.append(f'未完成：{ident}/{method}/{seed}');continue
                s=read(d/'summary.json')
                assert s['evaluations']==60000
                for name,digest in s['files_sha256'].items():assert hashlib.sha256((d/name).read_bytes()).hexdigest()==digest,(d,name)
                key=(ident,seed);first=initial.setdefault(key,s['initial_sha256']);assert first==s['initial_sha256'],'Unpaired initialization'
                if not (d/'audit.json').exists():issues.append(f'缺少独立复核：{ident}/{method}/{seed}');continue
                a=read(d/'audit.json');assert a['nodal_passed']
                if not a['accepted']:issues.append(f'复核门槛未全部通过：{ident}/{method}/{seed}（加密收敛={a["converged"]}、基础分数={a["base_training_agrees"]}、辅助分数={a["aux_training_agrees"]}、结构稳定={a["diagnostic_stable"]}）')
                events=[e for e in a['amplitude_events'] if e['width_evaluable']]
                row=dict(target=ident,method=method,seed=seed,accepted=a['accepted'],mag_mse=a['objective'][0],pha_mse=a['objective'][1],
                    amplitude_failed=sum(not e['complete'] for e in events),amplitude_total=len(events),
                    amplitude_missing=sum(not e['matched'] for e in a['amplitude_events']),amplitude_extra=len(a['extra_amplitude_events']),
                    phase_failed=sum(not e['complete'] for e in a['phase_events']),phase_total=len(a['phase_events']),phase_extra=len(a['extra_phase_events']),
                    D=a['anchored_geometry']['D'],H=a['anchored_geometry']['H'],A=a['anchored_geometry']['A'],
                    seconds=s['seconds'],cpu_seconds=s['cpu_seconds'],base_frequency_queries=s['base_frequency_queries'],auxiliary_frequency_queries=s['auxiliary_frequency_queries'],
                    knee_reason=a['selection']['reason'],nodal_mode=a['nodal_check']['mode'])
                records.append(row)
    assert records,'No auditable records'
    writecsv(OUT/'PER_SEED.csv',records);by={(r['target'],r['method'],r['seed']):r for r in records};paired=[];overview=[]
    for ident in inputs['targets']:
        for method in cfg['methods']:
            rr=[by[ident,method,s] for s in cfg['seeds'] if (ident,method,s) in by]
            if not rr:continue
            mean={k:float(np.mean([r[k] for r in rr])) for k in ('mag_mse','pha_mse','amplitude_failed','phase_failed','amplitude_extra','phase_extra','seconds','cpu_seconds','D','H','A')}
            overview.append(dict(target=ident,method=method,n=len(rr),accepted=sum(r['accepted'] for r in rr),**mean))
            if method=='MAG_PHA':continue
            for metric in ('mag_mse','pha_mse','amplitude_failed','phase_failed','amplitude_extra','phase_extra','D','H','A','seconds'):
                pp=[(r,by[ident,'MAG_PHA',r['seed']]) for r in rr if (ident,'MAG_PHA',r['seed']) in by]
                if not pp:continue
                delta=[r[metric]-b[metric] for r,b in pp];ci=bootstrap_mean(delta)
                paired.append(dict(target=ident,method=method,metric=metric,n=len(pp),both_accepted=sum(r['accepted'] and b['accepted'] for r,b in pp),
                    mean_delta=float(np.mean(delta)),ci_low=ci[0],ci_high=ci[1],better=sum(v<0 for v in delta),equal=sum(v==0 for v in delta),worse=sum(v>0 for v in delta)))
    writecsv(OUT/'SUMMARY.csv',overview);writecsv(OUT/'PAIRED.csv',paired)
    basic=['# 复阻抗联合轨迹指标实验：通俗报告','',
        '我们要回答的是：在保留幅值和相位两个基础目标的情况下，增加描述“复阻抗曲线怎样随频率行走”的指标，能否帮助优化器修好局部结构？',
        '',f'本批已回收{status.get("completed",0)}/{expected}个优化任务；{len(records)}个任务有可读取的独立复核，其中{sum(r["accepted"] for r in records)}个通过全部数值及结构稳定门槛。',
        '如果没有全部通过，下面的结果是带复核标记的开发观察，不能当作已经验证的改进。每个任务200个体、300代、60000次候选评价；每个条件每组8个seed，初始化逐组核对一致。',
        '', '## 1. 三种新指标分别在看什么', '',
        '把频率想成播放进度，Re Z和Im Z想成画面里点的两个坐标。D看这一小段往哪里走、走了多少；H同时看前后两段如何接在一起，转向和变化量是否相符；A看从窗口起点走到当前位置时，三个投影积累的有向面积怎样变化。A比较的是一整段随频率展开的过程，面积不能当作储能或Q。',
        '', '它们仍可能看漏东西。例如H、A对整体平移存在盲区，H对整体旋转也有盲区。因此基础MAG/PHA继续约束阻抗的实际水平，各组均未对候选曲线做自由对齐。',
        '', '十组为基础MAG/PHA、D、H、A、D+H、D+A、H+A、D+H+A、PTC参考、统一窗口有序路径C参考。组合等权合成一个辅助目标，各组最后均从MAG/PHA二维前沿选膝点。膝点不是两个MSE都最小的唯一解；退化回退原因完整保存。',
        '', '## 2. 怎么比较才公平', '',
        '每个条件使用同一批seed、同一初始化、边界、算法和候选评价数。最终比较同seed的结果，负的差值代表减少误差或缺陷。八个seed仍偏少，置信区间只是开发阶段描述，不代表完成了多重检验或独立推广验证。',
        '', '理论数据经过模型辅助自适应积分，独立检查另用加密网格与候选结构锚点；实测Case2只有原201点，没有补造目标点，模型、测量噪声和稀疏采样的影响需单列。辅助指标窗口都来自目标并覆盖全频带，这种局部等权也会改变优化强调的区域，不能把收益完全归于几何公式。',
        '', '## 3. 每个电路的结果', '']
    for ident in inputs['targets']:
        rec=read(CODE/'inputs'/ident/'target.json')
        basic+=[f'### {ident}','',rec['circuit']['expression'],'',link(CODE/'inputs'/ident/'circuit.svg','查看常规电路图'),'']
        if rec['kind']=='measured':basic+=['以下参数仅为程序适配器的名义值，不是真实测得的电路参数。','']
        basic+=['|元件|目标值/适配名义值|单位|','|---|---:|---|']
        basic += [f'|{n}|{v:.7g}|{u}|' for n,v,u in zip(rec['circuit']['names'],rec['circuit']['parameters'],rec['circuit']['units'])]
        basic+=['','|组别|复核通过|幅值MSE dB²|相位MSE rad²|未完整恢复幅值事件|未完整恢复相位过程|额外幅值/相位事件|平均秒数|','|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in overview:
            if r['target']==ident:basic.append(f'|{r["method"]}|{r["accepted"]}/{r["n"]}|{r["mag_mse"]:.6g}|{r["pha_mse"]:.6g}|{r["amplitude_failed"]:.3g}|{r["phase_failed"]:.3g}|{r["amplitude_extra"]:.3g}/{r["phase_extra"]:.3g}|{r["seconds"]:.1f}|')
        basic+=['','表中的缺陷数量是逐seed计数后取平均；曲线展示不取平均。Case2的相位结构计数不适用，表中0不意味着连续相位过程完整恢复。','']
    basic+=['## 4. 固定随机seed的典型曲线','','以下展示固定随机选出的3个seed，不以好坏挑seed。三维图保留频率轴，Re/Im投影辅助看形态；数值判断以逐频率复核为准。','']
    for ident in ('case1','user_nested/base','rlc_ladder/lc_reciprocal_8p0','mixed_resonators/lc_reciprocal_0p25','three_level/base','case2'):
        p,p3=plots(ident,cfg);basic += [f'### {ident}','',f'![联合轨迹]({p.as_posix()})','',f'![三维轨迹]({p3.as_posix()})','']
    basic+=['## 5. 回到四个研究问题','',
        '1. 哪类结构反复难恢复？先看各电路基础组的峰谷高度、位置、宽度及相位过程失败计数，并核对轨迹图。不能用辅助目标本身下降替代结构恢复。',
        '2. 哪个条件使它更难？比较同电路三档LC条件；这会同时改变尺度、阻尼和参数联动，没有单独隔离峰宽。',
        '3. 针对性指标能否改善？按同seed比较基础误差和结构缺陷，连同额外结构、复核通过率与成本一起看。',
        '4. 能否推广、代价是什么？本批覆盖全部既有电路，但这些目标都参与过开发；还需要全新电路、噪声和实测验证。相同候选数不等于相同CPU时间。',
        '',link(OUT/'PAIRED.csv','完整配对差值及95%自助区间')+'；'+link(OUT/'PER_SEED.csv','逐seed数据')+'；'+link(CODE/'PROTOCOL.md','正式可复算方案')+'；'+link(OUT/'ISSUES.md','问题与故障记录')]
    (OUT/'REPORT.md').write_text('\n'.join(basic)+'\n',encoding='utf-8')
    findings=['# 本批配对结果：证据与边界','',
        '仅逐seed双方均通过复核的配对，才适合支持“该配置改善”的数值结论。以下筛选只描述本批开发结果，并非选择性删掉失败；全量见CSV。','',
        '|目标|方法|指标|平均差值|95%自助区间|改善seed|双方复核通过|','|---|---|---|---:|---|---:|---:|']
    for r in paired:
        if r['metric'] in ('amplitude_failed','phase_failed'):
            findings.append(f'|{r["target"]}|{r["method"]}|{r["metric"]}|{r["mean_delta"]:.5g}|[{r["ci_low"]:.5g}, {r["ci_high"]:.5g}]|{r["better"]}/{r["n"]}|{r["both_accepted"]}/{r["n"]}|')
    findings+=['','结构事件使用既有冻结门槛。幅值事件“完整”包含位置、高度及3dB宽度，相位过程包含位置、局部斜率与宽度；阈值附近的微小变化可改变计数，需同时看连续数值、图形和局部贡献。',
        '', '本实验不能证明MSE理论上无法拟合。最终选解仍受前沿形状与归一化尺度影响；保存的全种群与阶段膝点可用于区分搜索与选解。']
    (OUT/'FINDINGS.md').write_text('\n'.join(findings)+'\n',encoding='utf-8')
    pre=read(DATA/'PREFLIGHT.json')
    notes=['# 问题、处理与结论影响','',
        '准备阶段发现过SSH Python路径不在默认PATH、SSH依赖未装在科研环境；分别采用服务器Python绝对路径和独立传输依赖目录解决，未进入优化结果。',
        '', 'PTC预检最初使用不同密度网格的目标事件与候选事件，真参数出现离散峰位置差。正式运行前统一到目标事件目录的4级网格，保持目标/候选检测规则一致；独立加密漂移仍标记。',
        '', 'C参考真参数处的约1e-14距离来自log参数转换的浮点舍入；零解单元检查按距离量纲使用1e-10浮点容差，平方损失仍用1e-15。未改变结果复核门槛。',
        '',f'Case2宽范围边界预检出现双精度节点电流相消，采用70位节点解按原门槛复核；预检高精度回退{len(pre.get("nodal_fallbacks",[]))}例。正式结果中回退{sum(r["nodal_mode"]!="double" for r in records)}例。双精度原偏差保留，不改搜索分数。',
        '',f'回收SHA256和ZIP CRC已通过；输入及源代码哈希全部一致={sources_ok}。预算、阶段和配对初始化已复核。',
        '', '固定辅助网格可能漏掉更窄候选结构；理论D/H/A另经候选锚点积分，漂移失败记录保留。Case2不能用加密检查创造不存在的目标数据。',
        '',f'优化任务{status.get("completed",0)}/{expected}，审计记录{len(records)}/{expected}，全部门槛通过{sum(r["accepted"] for r in records)}/{len(records)}。',
        '', '## 逐项记录','']+['- '+s for s in issues]
    (OUT/'ISSUES.md').write_text('\n'.join(notes)+'\n',encoding='utf-8')
    (OUT/'REPORT_STATUS.json').write_text(json.dumps(dict(complete=len(records)==expected and sources_ok,records=len(records),accepted=sum(r['accepted'] for r in records),issues=len(issues)),indent=2),encoding='utf-8')
    print(f'Report written: {OUT / "REPORT.md"}; audited records {len(records)}/{expected}',flush=True)

if __name__=='__main__':main()

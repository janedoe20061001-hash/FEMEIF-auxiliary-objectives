"""Post-hoc diagnostic of repeated amplitude/phase ratios; no new optimization."""
from pathlib import Path
import json,csv,hashlib,sys
import numpy as np
from scipy.stats import pearsonr,spearmanr
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'output/log_radius_distance_20261007/ratio_analysis'
sys.path.insert(0,str(HERE));import analyze as ana
LABEL={'wsmse':'WSMSE','mean_distance':'原Ω距离','logradius_distance':'老师式距离'}
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csvout(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8-sig',newline='') as h:w=csv.DictWriter(h,fieldnames=keys);w.writeheader();w.writerows(rows)
def n(x):return f'{float(x):.6g}'
def table(h,r):return '\n'.join(['| '+' | '.join(h)+' |','| '+' | '.join(['---']*len(h))+' |',*['| '+' | '.join(map(str,x))+' |' for x in r]])
def link(p):return p.resolve().as_posix()
def main():
    OUT.mkdir(exist_ok=True);rng=np.random.default_rng(2026100719);draw=rng.integers(0,100,(10000,100));datasets=[];sources=[]
    for batch,folder in [('旧种子',ROOT/'output/primary_loss_comparison_20261007/selected'),('新种子',ROOT/'output/log_radius_distance_20261007/selected')]:
        arrays=[]
        for c in ana.CASES:
            path=folder/f'{c}.npz';sources.append(path);d=np.load(path);m=d['metrics']
            np.testing.assert_allclose(((20*np.log10(abs(d['curves'])/abs(d['target'])))**2).mean(axis=-1),m[:,:,0],rtol=1e-9,atol=1e-9)
            np.testing.assert_allclose(np.square(np.angle(d['curves'])-np.angle(d['target'])).mean(axis=-1),m[:,:,1],rtol=1e-9,atol=1e-9)
            assert m.shape[0]==100;arrays.append(m[:,:,:2])
            for gi,g in enumerate(d['groups']):
                a=m[:,gi,0];p=m[:,gi,1];ratio=a/p;means=m[:,gi,:2].mean(axis=0)
                boot=m[draw,gi,:2].mean(axis=1);bk=boot[:,0]/boot[:,1]
                datasets.append(dict(batch=batch,circuit=c,group=str(g),magnitude_mse=float(means[0]),phase_mse=float(means[1]),ratio_of_means=float(means[0]/means[1]),ratio_ci95=np.quantile(bk,[.025,.975]).tolist(),seed_ratio_p10_p90=np.quantile(ratio,[.1,.9]).tolist(),mean_of_seed_ratios=float(ratio.mean()),pearson=float(pearsonr(a,p).statistic),spearman=float(spearmanr(a,p).statistic),log_component_energy_ratio=float(means[0]/means[1]/(20/np.log(10))**2)))
        if batch=='新种子':data=np.stack(arrays,axis=1)
    bs=data[draw].mean(axis=1);mu=data.mean(axis=0);cross=[]
    for gi,g in enumerate(ana.GROUPS):
        ra=mu[0,gi,0]/mu[1,gi,0];rp=mu[0,gi,1]/mu[1,gi,1];q=ra/rp;bq=(bs[:,0,gi,0]/bs[:,1,gi,0])/(bs[:,0,gi,1]/bs[:,1,gi,1]);ci=np.quantile(bq,[.025,.975])
        cross.append(dict(group=g,case1_over_case2_magnitude=float(ra),case1_over_case2_phase=float(rp),ratio_of_ratios=float(q),ci95=ci.tolist(),within_5percent_entire_ci=bool(ci[0]>=.95 and ci[1]<=1.05)))
    br=[]
    for c in ana.CASES:
        d=np.load(ROOT/'output/log_radius_distance_20261007/selected'/f'{c}.npz');f=d['frequency'];x=np.log10(f);edges=np.linspace(x[0],x[-1],9)
        da=20*np.log10(abs(d['curves'])/abs(d['target']));dp=np.angle(d['curves'])-np.angle(d['target'])
        for gi,g in enumerate(ana.GROUPS):
            totals=[(da[:,gi]**2).sum()/100,(dp[:,gi]**2).sum()/100]
            for bi in range(8):
                mask=(x>=edges[bi])&((x<edges[bi+1]) if bi<7 else (x<=edges[bi+1]));a=(da[:,gi,mask]**2).sum()/100;p=(dp[:,gi,mask]**2).sum()/100
                br.append(dict(circuit=c,group=g,band=bi+1,lo_hz=float(10**edges[bi]),hi_hz=float(10**edges[bi+1]),points=int(mask.sum()),amplitude_fraction=float(a/totals[0]),phase_fraction=float(p/totals[1]),ratio_within_band=float(a/p) if p>1e-14 else None))
    # Existing self-designed circuits only: different targets, no new server runs.
    selfrows=[];folder=ROOT/'output/primary_loss_comparison_20261007/selected'
    for path in sorted(folder.glob('*.npz')):
        if path.stem in ana.CASES:continue
        sources.append(path);d=np.load(path);family=path.stem.rsplit('_c',1)[0];mm=d['metrics'].mean(axis=0)
        for gi,g in enumerate(d['groups']):
            a,p=mm[gi,:2];selfrows.append(dict(circuit=path.stem,family=family,group=str(g),seeds=len(d['seeds']),magnitude_mse=float(a),phase_mse=float(p),ratio_of_means=float(a/p) if p>0 else None,above_numerical_floor=bool(a>1e-8 and p>1e-10)))
    finite=[r for r in selfrows if r['group']=='mean_distance' and r['above_numerical_floor']];assert finite
    selfsummary=dict(circuits=len(selfrows)//2,descriptive_floor_amplitude=1e-8,descriptive_floor_phase=1e-10,meaningful_distance_circuits=len(finite),ratio_range=[min(r['ratio_of_means'] for r in finite),max(r['ratio_of_means'] for r in finite)],ratio_percentiles=np.quantile([r['ratio_of_means'] for r in finite],[.1,.5,.9]).tolist(),near_90_count=sum(85<=r['ratio_of_means']<=95 for r in finite))
    # Passive circuit counterexample: scale every impedance by a common k.
    m=ana.Model('case1');r=m.record;p=np.asarray(r['target_parameters']);k=1.1;scaled=np.asarray([v/k if name[0]=='C' else v*k for name,v in zip(r['parameter_names'],p)])
    z0=ana.ind.independent_spectra('case1',m.f,p)[0];z=ana.ind.independent_spectra('case1',m.f,scaled)[0];np.testing.assert_allclose(z,k*z0,rtol=1e-10,atol=1e-8)
    err=ana.ind.independent_metrics(np.atleast_2d(z),z0,m.alpha)[0]
    np.testing.assert_allclose(err[0],(20*np.log10(k))**2,rtol=1e-10);assert err[1]<1e-24
    counter=dict(k=k,magnitude_mse_db2=float(err[0]),phase_mse_rad2=float(err[1]),inside_original_bounds=bool(np.all(np.log10(scaled)>=m.lower) and np.all(np.log10(scaled)<=m.upper)),passive_positive_parameters=True)
    sources.extend([Path(__file__),HERE/'analyze.py',ROOT/'experiments/seed_replication_20261007/analyze.py'])
    result=dict(ratio_summary=datasets,crosscase_new=cross,existing_selfdesigned_summary=selfsummary,positive_gain_counterexample=counter,db_conversion_squared=float((20/np.log(10))**2),bootstrap_replicates=10000,bootstrap_rng=2026100719,posthoc=True,new_optimizations=0,source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    dump(OUT/'SUMMARY.json',result);csvout(OUT/'band_contributions.csv',br);csvout(OUT/'selfdesigned_ratios.csv',selfrows);csvout(OUT/'ratios.csv',[dict(r,ratio_ci95=str(r['ratio_ci95']),seed_ratio_p10_p90=str(r['seed_ratio_p10_p90'])) for r in datasets])
    keyrows=[r for r in datasets if r['batch']=='新种子'];oldrows=[r for r in datasets if r['batch']=='旧种子' and r['group']=='mean_distance'];examples=sorted(finite,key=lambda r:r['ratio_of_means']);picked=examples[:3]+examples[-3:]
    worst=[]
    for c in ana.CASES:
        for g in ana.GROUPS:
            rr=[b for b in br if b['circuit']==c and b['group']==g];q=max(rr,key=lambda b:b['amplitude_fraction']+b['phase_fraction']);worst.append([c,LABEL[g],f"{n(q['lo_hz'])}–{n(q['hi_hz'])}",f"{100*q['amplitude_fraction']:.1f}%",f"{100*q['phase_fraction']:.1f}%",n(q['ratio_within_band'])])
    txt=['# 幅值MSE与相位MSE比例接近：追加核查',
      '**原Ω距离组的跨Case比例接近已在两批seed中复现；新坐标距离也出现相近的总体比值，但还不能确认两比例稳定落在±5%范围内。目前没有证据支持通用固定比例，也尚未识别导致约90这一数值的机制。**',
      '本次仅分析已有结果并计算一个确定性的电路反例，没有再次启动优化，没有更新网站。新的比值区间与频带分析为事后诊断，未按区间结果调换目标或删seed。',
      '## 1. 先核对观察是什么',
      'K＝幅值MSE的seed均值／相位MSE的seed均值，采用dB²/rad²的数值比。红框两组K接近，黄框两组K也接近；不是幅值MSE与相位MSE本身数值相等。K与“逐seed比值后再平均”是不同统计量。',
      table(['Case','组','K','K的seed bootstrap 95%区间','逐seed K：10%–90%','幅相MSE Pearson相关'],[[r['circuit'],LABEL[r['group']],n(r['ratio_of_means']),f"[{n(r['ratio_ci95'][0])}, {n(r['ratio_ci95'][1])}]",f"[{n(r['seed_ratio_p10_p90'][0])}, {n(r['seed_ratio_p10_p90'][1])}]",n(r['pearson'])] for r in keyrows]),
      '原Ω距离旧种子批的Case1/Case2 K分别为'+ '、'.join(n(r['ratio_of_means']) for r in oldrows)+'。它在换seed后仍接近，因此“只由原100个seed碰巧凑出来”的解释缺乏支持。但第三次看到的红框数据复用了第二批结果，不是又一次独立复现；黄框是新优化目标的结果，仍共享两个目标电路、相同采样和seed。',
      '## 2. 两个Case的比例是否足够接近',
      'Q＝K_case1/K_case2，也等于(跨Case幅值MSE比)/(跨Case相位MSE比)。Q=1表示两个K相等；共同单位换算会抵消。下面用100个配对seed为块抽样10,000次，RNG=2026100719。',
      table(['组','跨Case幅值比','跨Case相位比','Q','Q的95%区间','区间完全在[0.95,1.05]？'],[[LABEL[r['group']],n(r['case1_over_case2_magnitude']),n(r['case1_over_case2_phase']),n(r['ratio_of_ratios']),f"[{n(r['ci95'][0])}, {n(r['ci95'][1])}]",'是' if r['within_5percent_entire_ci'] else '否'] for r in cross]),
      '±5%沿用先前种子复现实验的操作范围，非物理定律。区间包含1不能用来证明两比例相等；WSMSE对照也没有同样的跨Case关系。',
      '## 3. 为什么可能一起变化：数学基础与尚未证实的解释',
      r'写 u=ln(|Z|/|Z*|)、v=θ−θ*，则 M_A=(20/ln10)² mean(u²)，M_θ=mean(v²)，所以 K=75.4447×mean(u²)/mean(v²)。这只是精确换算公式，不要求实虚部分的平方误差相等。',
      '例如红框K约88，表示两部分在自然对数/弧度单位下的误差能量比约1.16–1.17。它解释了数值尺度，却没有解释为何能量比接近、为何各目标不同。相位改用degree²时，K会整体除以(180/π)²，约90会变为约0.027；数值90不是单位无关的物理常数。',
      '同一阻抗模型的幅值与相位由共同的元件参数、极点和零点决定。满足相应最小相位条件时，还存在对数幅值与相位的Hilbert/Bode关系；这一理论说明二者可能耦合，不能推出当前有限频段、log f采样、平方均值中的K固定。[Stanford原始教材说明](https://sepwww.stanford.edu/sep/prof/pvi/spec/paper_html/node13.html)，[Caltech课程讲义](https://www.cds.caltech.edu/~murray/courses/cds101/fa04/lectures/L8.1_pid.pdf)。本次没有验证全部测量目标是否满足相关理论条件，不将理论当作比值90的证明。',
      '可检验的机制假设：距离目标容许某些峰谷、相位平台被同时漏掉，或参数落入相近的错误模式；不同seed更多地改变偏差严重程度，而幅相偏差模式相似，因而误差高度相关并呈近似比例。这是推断，Pearson相关与重复的总体比值都不能单独证明它。',
      table(['Case','组','幅相贡献之和最高的频带/Hz','幅值平方误差贡献','相位平方误差贡献','该频带K'],worst),
      '上述频带通过各原频点平方误差直接相加，贡献是跨seed平方误差总和的比例；按幅相两项归一化贡献之和取最大，不代表二者各自最大频带相同。不是用两组独立的RMSE平均来求比。完整八频带均保留，因此总体比例可能掩盖频带之间的差异。',
      '尤其原Ω距离组：Case1主要误差带中的K=106.17，Case2主要误差带中的K=72.78，总体却都约88。各频带混合后接近，并不说明逐频带也成同一比例，或两个Case有完全相同的偏差机制。不能把总体比值直接升级为点对点或频带级的定律。',
      '## 4. 已有自设计电路与严格反例',
      f"读取原单目标比较的{selfsummary['circuits']}个自设计电路（每电路20个seed、原两种目标）。原Ω距离组中，{len(finite)}个电路均值超过幅值1e−8dB²且相位1e−10rad²的数值下限，其K范围为[{n(selfsummary['ratio_range'][0])}, {n(selfsummary['ratio_range'][1])}]，10%/50%/90%分位为"+str([round(x,4) for x in selfsummary['ratio_percentiles']])+f"；只有{selfsummary['near_90_count']}个落在[85,95]。这些是旧实验的描述性复核，条件不同、近零误差的比值很不稳定；原始值和过滤标志都保留，不把它当作与两个论文Case同预算的确认实验。",
      table(['自设计电路','幅值MSE','相位MSE','K'],[[r['circuit'],n(r['magnitude_mse']),n(r['phase_mse']),n(r['ratio_of_means'])] for r in picked]),
      '更直接的反例是在Case1中令所有R、L乘1.1，所有C除1.1。这样每个元件阻抗和总阻抗都精确乘1.1，仍为被动正参数电路且在原参数边界内。幅值曲线整体增加20log10(1.1)dB，相位完全不变。',
      f"独立电路公式验证幅值MSE={n(counter['magnitude_mse_db2'])}dB²，相位MSE={n(counter['phase_mse_rad2'])}rad²（数值舍入量，理论为0）。因此不存在适用于任意电路参数偏差的有限固定幅相MSE比例。这个反例不否定当前优化选解的条件性比例。",
      '## 5. 当前判断及最能解释机制的下一步',
      '当前支持：原Ω距离组在两批seed上关系可重复；新老师式目标也呈接近的总体比值与幅相共同变化。当前不支持：每个seed固定比例、所有电路固定比例、单凭三次出现就能确定机制。截图里原Ω距离红框两行是复用数据，不能重复计作独立证据。',
      '比继续只换seed更有辨别力的是：对已有选解按“高频结构漏失/峰位偏移/整体幅值偏移”分组，逐频带检查K；再在同一电路上改变观察频段、加密Case1漏峰处，或设置只改变幅值尺度的目标，在同算法预算下复验。若K随偏差模式和观察频带改变，就更符合条件性结构关系；若多个不同电路与采样条件仍保持相同K，才有依据追究更广泛机制。',
      '\n'.join(f'- [{label}]({link(path)})' for label,path in [('完整比值、区间与来源',OUT/'SUMMARY.json'),('逐频带平方误差贡献',OUT/'band_contributions.csv'),('全部自设计电路比值',OUT/'selfdesigned_ratios.csv'),('原实验报告',OUT.parent/'REPORT.md'),('独立seed复现实验',ROOT/'output/seed_replication_20261007/REPORT.md')])]
    (OUT.parent/'RATIO_ANALYSIS.md').write_text('\n\n'.join(txt)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

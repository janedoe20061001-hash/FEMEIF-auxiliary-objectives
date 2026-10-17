"""Readable circuit-by-circuit reporting from immutable optimization results."""
from pathlib import Path
import csv
import hashlib
import html
import json
import shutil
import sys
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'output/primary_loss_comparison_20261007'
from model import module
COLS=['magnitude_mse_db2','phase_mse_rad2','wsmse','mean_distance_ohm']
METRICS=['幅值 MSE（dB²）','相位 MSE（rad²）','WSMSE','平均距离（Ω）']
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fmt(x):return f'{x:.6g}'
def engineering(name,x):
    units=[(1e3,'kΩ'),(1,'Ω'),(1e-3,'mΩ')] if name.startswith('R') else ([(1,'H'),(1e-3,'mH'),(1e-6,'µH'),(1e-9,'nH')] if name.startswith('L') else [(1e-3,'mF'),(1e-6,'µF'),(1e-9,'nF'),(1e-12,'pF')])
    for scale,unit in units:
        if x>=scale:return f'{x/scale:.5g} {unit}'
    scale,unit=units[-1];return f'{x/scale:.5g} {unit}'
def metric_table(rows):
    text=['| 优化目标与选解规则 | 幅值 MSE（dB²） | 相位 MSE（rad²） | WSMSE | 平均距离（Ω） |','|---|---:|---:|---:|---:|']
    for row in rows:
        title='最小 WSMSE' if row['group']=='wsmse' else '最小平均距离'
        text.append('| '+title+' | '+' | '.join(fmt(row[k]) for k in COLS)+' |')
    return text
def main():
    summary=json.loads((OUT/'SUMMARY.json').read_text(encoding='utf-8'));records=json.loads((HERE/'circuits.json').read_text(encoding='utf-8'))
    archive=HERE/'server_run/retrieved';manifest=json.loads((archive/'runs/manifest.json').read_text())
    for name,digest in manifest['source_sha256'].items():assert sha(HERE/name)==digest
    assert sha(HERE/'analyze.py')==sha(archive/'analyze.py')
    (OUT/'circuits').mkdir(exist_ok=True);(OUT/'schematics').mkdir(exist_ok=True)
    viz=HERE/'visualization_source.py'
    if not viz.exists():shutil.copy2(ROOT/'experiments/parameter_curve_population_20261006/gallery_site.py',viz)
    schem=module('scalar_report_schematic',viz).Schematic
    lookup={(m['circuit'],m['group']):m for m in summary['means']}
    cmp={(c['circuit'],c['metric']):c for c in summary['paired_comparisons']}
    details=[];circuit_index=['# 电路与逐电路结果','',
        '每页先列结构、参数与 alpha，再列两组 seed 均值和配对区间。图中示例固定为 seed 2300，不挑跨 seed 最优。','',
        '| 电路 | 条件 | α | seeds/组 | 结果页 |','|---|---|---:|---:|---|']
    for r in records:
        ident=r['id'];drawing=schem();svg=drawing.render(r['tree'],r['title']);(OUT/'schematics'/f'{ident}.svg').write_text(svg,encoding='utf-8')
        with np.load(OUT/'selected'/f'{ident}.npz') as selected:
            example=10**selected['log_parameters'][0]
            example_metrics=selected['metrics'][0]
        means=[lookup[(ident,g)] for g in ('wsmse','mean_distance')]
        page=[f'# {r["title"]} · {r["condition"]}','',f'电路编号：{ident}；每组 {r["seed_count"]} 个 seed；alpha={r["alpha"]:.8g}。','',
              f'![电路结构](../schematics/{ident}.svg)','',
              '## 参数','',
              '目标参数与示例拟合参数分列。Case2 实测目标没有已知元件真值；示例是两组各自在 seed 2300 内按自身目标取的最小值解。','',
              '| 元件 | 目标参数 | WSMSE 组示例 | 距离组示例 | 固定搜索下界 | 固定搜索上界 |','|---|---:|---:|---:|---:|---:|']
        for i,name in enumerate(r['parameter_names']):
            target='未知（实测）' if r['target_parameters'] is None else engineering(name,r['target_parameters'][i])
            page.append(f'| {name} | {target} | {engineering(name,example[0,i])} | {engineering(name,example[1,i])} | {engineering(name,10**r["lower_log10"][i])} | {engineering(name,10**r["upper_log10"][i])} |')
        page+=['','## 跨 seed 均值','',*metric_table(means),'','## 距离组减去 WSMSE 组的配对差','',
               '| 量 | 均值差 | 95% bootstrap 区间 | 距离组更低的 seed 数 |','|---|---:|---|---:|']
        for metric,label in zip(COLS,METRICS):
            c=cmp[(ident,metric)];lo,hi=c['bootstrap95'];page.append(f'| {label} | {c["distance_minus_wsmse_mean"]:+.6g} | [{lo:+.6g}, {hi:+.6g}] | {c["distance_lower_seeds"]}/{r["seed_count"]} |')
        page+=['',f'![配对分布](../figures/{ident}_paired.png)','',
               '## 图中 seed 2300 自己的误差','',
               *metric_table([dict(group=g,**dict(zip(COLS,example_metrics[j]))) for j,g in enumerate(('wsmse','mean_distance'))]),'',
               '本表对应下面这一条seed曲线；上面的跨seed均值是另一个统计层次。','',
               f'![固定示例的阻抗曲线](../figures/{ident}_curves.png)','',
               '[返回总报告](../REPORT.md) · [电路目录](../CIRCUITS.md)']
        (OUT/'circuits'/f'{ident}.md').write_text('\n'.join(page)+'\n',encoding='utf-8')
        circuit_index.append(f'| {r["title"]} | {r["condition"]} | {r["alpha"]:.5g} | {r["seed_count"]} | [{ident}](circuits/{ident}.md) |')
        details.append(dict(id=ident,family=r['family'],title=r['title'],condition=r['condition'],alpha=r['alpha'],seeds=r['seed_count'],
                            means=means,comparisons=[cmp[(ident,k)] for k in COLS],parameters=r['target_parameters'],names=r['parameter_names'],
                            examples=example.tolist(),example_metrics=example_metrics.tolist(),svg=f'schematics/{ident}.svg'))
    (OUT/'CIRCUITS.md').write_text('\n'.join(circuit_index)+'\n',encoding='utf-8')
    report=['# WSMSE 与原始阻抗平均距离：单目标优化对照','',
        '**2,320 次运行全部完成。两组分别按自身目标在每个 seed 内选最小值解，再对所有 seeds 等权取平均。** 两项 MSE 始终来自该同一个解。','',
        '[老师三项问题的独立核查：距离定义、比例接近、seed2300分带贡献](AUDIT.md)。全部运行与末代个体已独立复算，原数值保留。','',
        '**主要结果：在 Case1 和 Case2，距离组的平均欧姆距离更小，但幅值 MSE、相位 MSE 均明显更大；48 个自设计电路中存在反例，所以两种主目标有实质差异，不能直接认作等价或宣布其中一个普遍更优。**','',
        'Case1、Case2 各 100 个配对 seed；自设计为 16 种拓扑×3 组参数=48 个电路，各 20 个配对 seed。每次 300 代，总计 90,120,000 次评价。','',
        '## Case1 与 Case2','',
        '表中四项均为跨 seed 均值。每组自身目标的最小值单独列出，同时计算另一目标，方便交叉比较；WSMSE 与 Ω 量纲不同，不能直接比两列数值大小。','']
    for ident in ('case1','case2'):
        report += [f'### {ident}（alpha={lookup[(ident,"wsmse")]["alpha"]}）','',*metric_table([lookup[(ident,g)] for g in ('wsmse','mean_distance')]),'',
            '| 距离组相对 WSMSE 组 | 均值变化 | 配对均值差的 95% 区间 | 距离组更低的种子 |','|---|---:|---|---:|']
        for metric,label in zip(COLS,METRICS):
            c=cmp[(ident,metric)];lo,hi=c['bootstrap95'];report.append(f'| {label} | {c["relative_change_pct"]:+.1f}% | [{lo:+.6g}, {hi:+.6g}] | {c["distance_lower_seeds"]}/100 |')
        report += ['',f'[结构、参数、逐 seed 分布和示例曲线](circuits/{ident}.md)','',f'![进化历史](figures/{ident}_history.png)','']
    report += ['## 48 个自设计电路','',
        '以下按每个电路先平均，再数均值更低的电路数。计数未设实用误差门槛，近零误差也按严格均值大小计入，所以不能把数量直接当成实用优势。48 个目标属于 16 种拓扑，不能把 960 个 seeds 当作 960 个独立电路，也不混合不同尺度的原始误差计算“总冠军”。','',
        '| 比较量 | 距离组均值更低的电路数 | WSMSE 组均值更低的电路数 |','|---|---:|---:|']
    for metric,label in zip(COLS,METRICS):
        tally=summary['synthetic_tally'][metric];report.append(f'| {label} | {tally["distance_lower_mean"]}/48 | {tally["wsmse_lower_mean"]}/48 |')
    report += ['',f'两项 MSE 均值同时更低：距离组 {summary["synthetic_distance_both_components_lower"]}/48，WSMSE 组 {summary["synthetic_wsmse_both_components_lower"]}/48；其余是分量折衷或相等。这里只是样本均值方向，不能当成所有电路都显著。','',
        '### 各拓扑的分量差异','',
        '比值=距离组误差均值/WSMSE 组误差均值，拓扑行取其 3 个电路比值的几何平均。小于 1 表示距离组较低。近零误差时比值可能夸大实际差距，原始均值见逐电路表。','',
        '| 拓扑 | 幅值误差比 | 距离组幅值更低 | 相位误差比 | 距离组相位更低 |','|---|---:|---:|---:|---:|']
    for family in dict.fromkeys(r['family'] for r in records[2:]):
        ff=[next(c for c in summary['topology_summary'] if c['family']==family and c['metric']==metric) for metric in COLS[:2]]
        title=next(r['title'] for r in records if r['family']==family)
        report.append(f'| {title} | {ff[0]["geometric_mean_error_ratio"]:.4g} | {ff[0]["distance_lower_circuits"]}/3 | {ff[1]["geometric_mean_error_ratio"]:.4g} | {ff[1]["distance_lower_circuits"]}/3 |')
    report += ['', '[查看全部 50 个电路的结构、参数和结果](CIRCUITS.md) · [简洁图册](index.html)','',
        '## 两种主目标为什么可能不同','',
        'WSMSE 使用 dB 幅值误差与 rad 相位误差的加权平方和；平均距离使用原始欧姆残差。令 rho=|Z|，delta=phi−phi*，逐点恒等式为：','',
        'D_i²=(rho−rho*)²＋2 rho rho* (1−cos(delta))。','',
        '所以欧姆距离的相位惩罚随阻抗尺度变化：相同相位差，在大阻抗处通常产生更大的绝对距离。dB 幅值误差则更接近相对幅值差。平均距离逐点开根号，也改变了大残差点的惩罚增长方式。因此这次既改变了坐标尺度，也改变了误差聚合，不能把结果全部归因于“平方或开根号”这一点。','',
        '## 方法与限制','',
        '- WSMSE=M_A＋alpha M_phi。Case1/Case2 沿用论文表 V 的 25.12/17.38；自设计 alpha=(目标 dB 幅值跨度/目标 rad 相位跨度)²，在运行前固定。此规则不是最优权重证明。',
        '- D=mean sqrt[(Re残差)²＋(Im残差)²]，同频点的频率残差为零。不是整体 RMS，也不是归一化三维距离。',
        '- 两组只有一个进化目标，使用相同 NSGA-II 单目标模式、初始化、算子、边界与评价预算；没有辅助指标，没有额外局部优化。',
        '- 自设计为无噪声、已知正确拓扑，元件范围真值±2 decades。Case2 是实测目标，不删除原测量点；全部 201 点与论文的 96 点不同。本轮不宣称复现论文原数值。',
        '- 自设计使用固定 240 个 log 等距频点，高 Q 尖峰可能未充分解析；效果只限该采样和 300 代预算。重复运行与跨拓扑验证仍不证明对任意电路普遍有效。',
        '- 配对区间以 seed 为单位，每电路独立 bootstrap 10,000 次；大量电路/指标比较未做多重比较校正，区间作探索性证据，不据此作普适显著性声明。','',
        '核验：全部 300,400 个末代个体重算四项误差，核对标量目标、各组自身最小值选解、配对初始化、种子复现、参数边界、每代评价计数与代码哈希；全部通过。','',
        'WSMSE 来源：[英文论文](D:/develop/FEMEIF/Feature-Engineering_Enabled_Multi-Objective_Evolutionary_Impedance_Fitting_Technique.pdf)第 2 页式(2)、第 6 页表 V；图中幅值单位 dB、相位 rad。','',
        '[逐 seed 四项误差](selected_metrics.csv) · [组均值](group_means.csv) · [配对统计](paired_comparisons.csv) · [完整统计](SUMMARY.json) · [核验](VALIDATION.json)']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    dump(OUT/'REPORT_INPUTS.json',dict(report_source_sha256=sha(__file__),schematic_source_sha256=sha(viz),source_match=True,
        summary_sha256=sha(OUT/'SUMMARY.json'),circuits_sha256=sha(HERE/'circuits.json')))
    # The compact HTML presents one selected circuit; generated below by build_gallery.
    build_gallery(details,summary,OUT)
    dump(OUT/'INTEGRITY.json',dict(archive=json.loads((HERE/'server_run/ARCHIVES.json').read_text()),local_source_matches_remote=True,
        files_sha256={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='INTEGRITY.json'}))
    print(json.dumps(dict(report_complete=True,circuit_pages=50,schematics=50)))

def build_gallery(details,summary,out):
    options=''.join(f'<option value="{html.escape(d["id"])}">{html.escape(d["title"])} · {html.escape(d["condition"])}</option>' for d in details)
    payload=json.dumps(details,ensure_ascii=False).replace('<','\\u003c')
    headline=[]
    for ident in ('case1','case2'):
        for group in ('wsmse','mean_distance'):
            m=next(m for m in summary['means'] if m['circuit']==ident and m['group']==group)
            headline.append('<tr><th>'+ident+'</th><td>'+('WSMSE' if group=='wsmse' else '平均距离')+'</td>'+''.join(f'<td>{fmt(m[k])}</td>' for k in COLS)+'</tr>')
    counts=[]
    for metric,label in zip(COLS[:2],METRICS[:2]):
        t=summary['synthetic_tally'][metric];counts.append(f'<tr><th>{label}</th><td>{t["distance_lower_mean"]}/48</td><td>{t["wsmse_lower_mean"]}/48</td></tr>')
    counts.append(f'<tr><th>两项 MSE 同时更低</th><td>{summary["synthetic_distance_both_components_lower"]}/48</td><td>{summary["synthetic_wsmse_both_components_lower"]}/48</td></tr>')
    markup='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>WSMSE 与平均距离 · 优化对照</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f5f7fa;color:#25364a;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1180px;margin:auto;padding:30px 24px 60px}h1{font-size:27px;margin:0 0 12px}h2{font-size:21px;margin:0 0 14px}h3{font-size:17px;margin:8px 0}.muted{color:#607084;font-size:13px}p{margin:10px 0}section{background:white;border:1px solid #e0e6ec;border-radius:12px;margin:20px 0;padding:22px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px 10px;text-align:right;border-bottom:1px solid #e7ebef;white-space:nowrap}th:first-child,td:first-child{text-align:left}thead th{background:#f2f5f8}select{font:inherit;padding:10px;border:1px solid #b7c5d4;border-radius:7px;width:100%;margin-bottom:12px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}.circuit{width:100%;height:220px;object-fit:contain}.plot{display:block;width:100%;height:auto;margin-top:12px}a{color:#1769aa}.formula{background:#edf4fb;padding:12px 16px;border-radius:8px}.blue{color:#0284c7}.orange{color:#bd581b}@media(max-width:760px){main{padding:18px 12px}section{padding:16px}.grid{grid-template-columns:1fr}h1{font-size:23px}td,th{padding:7px}.circuit{height:190px}}</style>
<main><h1>WSMSE 与原始阻抗平均距离</h1><p>两种标量主目标的直接优化对照。2,320 次运行 · 两个 Case＋48 个自设计电路 · 300 代 · 9,012 万次评价。</p>
<p class="formula">每个 seed：本组末代种群 → 取本组目标最小的一个解 → 同一个解计算四项误差 → 跨 seeds 取平均。</p>
<p class="formula"><a href="AUDIT.md">老师问题的独立核查</a>：距离定义、两个 Case 的比例接近，以及 seed 2300 的逐频点/频带欧姆距离贡献。原优化结果不变。</p>
<section><h2>Case1 与 Case2：跨 100 个 seed 的均值</h2><div class="scroll"><table><thead><tr><th>电路</th><th>优化目标</th><th>幅值 MSE · dB²</th><th>相位 MSE · rad²</th><th>WSMSE</th><th>平均距离 · Ω</th></tr></thead><tbody>__HEADLINE__</tbody></table></div><p class="muted">四列均来自每次按本组目标选出的同一个体。WSMSE 与 Ω 量纲不同，不直接比较两列的数值大小。</p></section>
<section><h2>自设计电路：先逐电路平均，再比较方向</h2><div class="scroll"><table><thead><tr><th>比较量</th><th>距离组均值更低</th><th>WSMSE 组均值更低</th></tr></thead><tbody>__SYNTH__</tbody></table></div><p class="muted">16 种拓扑×3 组参数，每电路每组 20 seeds。计数未设实用误差门槛，近零差异也计入；数量不等于实际优势或普适显著性。</p></section>
<section><h2>逐电路查看：结构、参数与结果</h2><label for="circuit-select">选择电路</label><select id="circuit-select">__OPTIONS__</select><p id="meta" class="muted"></p>
<div class="grid"><div><h3>电路结构</h3><img id="schematic" class="circuit" alt="所选电路结构"><a id="svg-link">查看完整电路图</a></div><div><h3>电路参数</h3><p class="muted">拟合示例固定为 seed 2300，分别按各组目标选解。</p><div class="scroll"><table><thead><tr><th>元件</th><th>目标</th><th class="blue">WSMSE 组示例</th><th class="orange">距离组示例</th></tr></thead><tbody id="parameters"></tbody></table></div></div></div></section>
<section><h2>两组跨 seed 均值</h2><div class="scroll"><table><thead><tr><th>优化与选解目标</th><th>幅值 MSE · dB²</th><th>相位 MSE · rad²</th><th>WSMSE</th><th>平均距离 · Ω</th></tr></thead><tbody id="means"></tbody></table></div>
<h3 style="margin-top:22px">距离组减去 WSMSE 组：配对统计</h3><p class="muted">负数表示距离组更低。区间跨零时，不能只凭均值方向断定改进。每电路独立 bootstrap，未作多重比较校正。</p><div class="scroll"><table><thead><tr><th>比较量</th><th>均值差</th><th>95% 区间</th><th>距离组更低的 seeds</th></tr></thead><tbody id="paired"></tbody></table></div>
<img id="paired-plot" class="plot" alt="两组配对误差分布"></section>
<section><h2>同一 seed 的曲线对照</h2><p class="muted">固定展示 seed 2300。下面误差表与图对应同一个 seed；前面的均值表使用全部 seeds。</p><div class="scroll"><table><thead><tr><th>seed 2300选解</th><th>幅值 MSE · dB²</th><th>相位 MSE · rad²</th><th>WSMSE</th><th>同频平均距离 · Ω</th></tr></thead><tbody id="example-metrics"></tbody></table></div><img id="curves-plot" class="plot" alt="目标与两组拟合的幅值相位曲线"><p class="muted">幅值纵轴为 dB，相位纵轴为度数。平均距离在同频实部/虚部的原始 Ω 坐标中计算。</p></section>
<section><h2>公式与范围</h2><p>WSMSE＝幅值 MSE＋α×相位 MSE；幅值使用 dB，相位使用 rad 主值。平均距离＝mean|Z−Z*|，使用原始实部、虚部的欧姆距离；同频点的频率坐标残差为零。</p><p>Case1/Case2 的 α＝25.12/17.38；自设计 α＝（目标幅值跨度／目标相位跨度）²，运行前固定。自设计无噪声、正确结构、真值中心 ±2 decades 边界；高 Q 尖峰可能受 240 点采样限制。Case2 保留全部 201 个测量点。</p><p><a href="REPORT.md">完整报告</a> · <a href="CIRCUITS.md">50 个电路目录</a> · <a href="selected_metrics.csv">逐 seed 数据</a> · <a href="group_means.csv">所有组均值</a> · <a href="VALIDATION.json">核验记录</a></p></section></main>
<script type="application/json" id="data">__PAYLOAD__</script><script>
const data=JSON.parse(document.getElementById('data').textContent), cols=['magnitude_mse_db2','phase_mse_rad2','wsmse','mean_distance_ohm'];
const labels=['幅值 MSE','相位 MSE','WSMSE','平均距离'];
const fmt=x=>String(Number(Number(x).toPrecision(6)));
function unit(n,x){let u=n[0]==='R'?[[1e3,'kΩ'],[1,'Ω'],[1e-3,'mΩ']]:n[0]==='L'?[[1,'H'],[1e-3,'mH'],[1e-6,'µH'],[1e-9,'nH']]:[[1e-3,'mF'],[1e-6,'µF'],[1e-9,'nF'],[1e-12,'pF']];let s=u.find(a=>x>=a[0])||u.at(-1);return fmt(x/s[0])+' '+s[1];}
function row(target,values){const tr=document.createElement('tr');for(const v of values){let td=document.createElement('td');td.textContent=v;tr.append(td);}target.append(tr);}
function show(id){const d=data.find(d=>d.id===id);document.getElementById('meta').textContent=d.id+' · '+d.condition+' · α='+fmt(d.alpha)+' · 每组 '+d.seeds+' seeds';
let image=document.getElementById('schematic');image.src=d.svg;image.alt=d.title+'结构';const link=document.getElementById('svg-link');link.href=d.svg;
const par=document.getElementById('parameters');par.replaceChildren();d.names.forEach((n,i)=>row(par,[n,d.parameters?unit(n,d.parameters[i]):'实测·未知',unit(n,d.examples[0][i]),unit(n,d.examples[1][i])]));
const means=document.getElementById('means');means.replaceChildren();d.means.forEach(m=>row(means,[m.group==='wsmse'?'最小 WSMSE':'最小平均距离',...cols.map(k=>fmt(m[k]))]));
const examples=document.getElementById('example-metrics');examples.replaceChildren();d.example_metrics.forEach((m,i)=>row(examples,[i===0?'WSMSE组选解':'距离组选解',...m.map(fmt)]));
const paired=document.getElementById('paired');paired.replaceChildren();d.comparisons.forEach((c,i)=>row(paired,[labels[i],fmt(c.distance_minus_wsmse_mean),'['+c.bootstrap95.map(fmt).join(', ')+']',c.distance_lower_seeds+'/'+d.seeds]));
document.getElementById('paired-plot').src='figures/'+d.id+'_paired.png';document.getElementById('curves-plot').src='figures/'+d.id+'_curves.png';}
document.getElementById('circuit-select').addEventListener('change',e=>show(e.target.value));show('case1');
</script></html>'''
    markup=markup.replace('__HEADLINE__',''.join(headline)).replace('__SYNTH__',''.join(counts)).replace('__OPTIONS__',options).replace('__PAYLOAD__',payload)
    (out/'index.html').write_text(markup,encoding='utf-8')

if __name__=='__main__':main()

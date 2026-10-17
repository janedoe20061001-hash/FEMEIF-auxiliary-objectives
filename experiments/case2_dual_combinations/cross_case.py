"""Descriptive replication table: compare each case with its OWN baseline."""
import csv
import hashlib
import json
from pathlib import Path
from objectives import CATALOG

HERE=Path(__file__).resolve().parent


def compare(output,case2_rows):
    spec=json.loads((HERE/'SOURCE.json').read_text())
    case2={frozenset(r['group'].split('+')):r for r in case2_rows if '+' in r['group']}
    labels={name:ident for ident,name,_ in CATALOG}
    rows=[]
    for source in spec['case1_comparisons']:
        path=HERE/source['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=source['sha256']:
            raise ValueError('Case1 comparison snapshot changed')
        with path.open(encoding='utf-8-sig',newline='') as f:
            for a in csv.DictReader(f):
                key=frozenset(a['group'].split('+'))
                if (a['front']!='wsmse_selected' or key not in case2 or
                        not key<=set(spec['sets'][source['set_key']])):continue
                b=case2[key]
                am,ap=float(a['magnitude_change_pct']),float(a['phase_change_pct'])
                bm,bp=b['magnitude_change_pct'],b['phase_change_pct']
                rows.append(dict(group=b['group'],case1_batch=source['label'],
                    case1_seed_start=source['seed_start'],case1_magnitude_change_pct=am,case1_phase_change_pct=ap,
                    case2_magnitude_change_pct=bm,case2_phase_change_pct=bp,
                    case1_both_means_lower=am<0 and ap<0,case2_both_means_lower=bm<0 and bp<0,
                    both_mean_conclusion_agrees=(am<0 and ap<0)==(bm<0 and bp<0)))
    if not rows:raise ValueError('No matched Case1 combinations')
    out=Path(output)
    with (out/'cross_case.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    text=['# Case1结论在Case2上的适用性','',
          '只比较各Case相对自身baseline的变化，不直接比较两种电路的绝对MSE。Case1不同seed批次分别列出，保留此前成功和未成功的结果。',
          '“一致”仅指两项均值是否同时下降这一描述性判断相同；不等于统计上已证明可迁移。当前只覆盖两套既定三指标集合，不能代表Case1所有指标组合。',
          '电路、频段、数据噪声、种群规模及选解权重均存在差别，结果变化不能仅归因于噪声。Case2区间见STABILITY.md；组合与组成项比较见CONSTITUENTS.md。','',
          '|组合|Case1批次|Case1幅值变化%|Case1相位变化%|Case2幅值变化%|Case2相位变化%|双均值改善判断一致|',
          '|---|---|---:|---:|---:|---:|---|']
    for r in rows:
        label='+'.join(labels[a] for a in r['group'].split('+'))
        text.append(f"|{label}|{r['case1_batch']}|{r['case1_magnitude_change_pct']:.2f}|{r['case1_phase_change_pct']:.2f}|{r['case2_magnitude_change_pct']:.2f}|{r['case2_phase_change_pct']:.2f}|{'是' if r['both_mean_conclusion_agrees'] else '否'}|")
    for name,members in spec['sets'].items():
        relevant=[r for key,r in case2.items() if key<=set(members)]
        success=sum(r['both_means_lower'] for r in relevant)
        text.extend(['',f"集合{name}在Case2的四个组合中，{success}/4个组合的两项均值均低于baseline。"])
    (out/'CROSS_CASE.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return rows

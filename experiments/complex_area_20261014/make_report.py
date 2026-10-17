"""Generate reproducible tables and draft; final scientific interpretation is manual."""
from core import *
from run import verify,path_for
import csv,time

def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    recovered=OUT/'retrieved';base=recovered/'output/complex_area_20261014';dest=OUT/'analysis';dest.mkdir(exist_ok=True)
    recovery=json.loads((OUT/'RECOVERY.json').read_text());assert recovery['verified']
    manifest=json.loads((base/'MANIFEST.json').read_text())
    for name,h in manifest['hashes'].items():assert sha(recovered/name)==h,('source_or_input_hash',name)
    rows=[];events=[];failed=[];pair_init={}
    for ident in target_ids():
        for method in CFG['methods']:
            for seed in CFG['seeds']:
                d=path_for(base,ident,method,seed)
                if not (d/'summary.json').exists() or not (d/'audit.json').exists():
                    failed.append(dict(target=ident,method=method,seed=seed,status='execution_or_audit_incomplete'));continue
                s=verify(d);a=json.loads((d/'audit.json').read_text());assert sha(d/'audited_curve.npz')==a['curve_sha256']
                key=(ident,seed);old=pair_init.setdefault(key,s['initial_sha256']);assert old==s['initial_sha256']
                for e in a['amplitude_events']:events.append(dict(target=ident,method=method,seed=seed,accepted=a['accepted'],**e))
                focused=[e for e in a['amplitude_events'] if e['event_id'] in ('P2','V3')] if ident!='case1' else a['amplitude_events']
                with np.load(d/'audited_curve.npz') as z:
                    x=z['x'];v=z['z'];ref=z['target']
                    band=(45000,58000) if ident.endswith('/base') else (38500,45000) if ident!='case1' else (10**x[0],10**x[-1])
                    inside=(x>=np.log10(band[0]))&(x<=np.log10(band[1]));xx=x[inside]
                    local=float(np.trapezoid(abs(v[inside]-ref[inside])**2,x=xx)/(xx[-1]-xx[0])/max(abs(ref[inside]))**2)
                rows.append(dict(target=ident,method=method,seed=seed,accepted=a['accepted'],mag=a['objective'][0],pha=a['objective'][1],
                    focused_complete=sum(e['complete'] for e in focused),focused_total=len(focused),
                    all_complete=sum(e['complete'] for e in a['amplitude_events']),all_total=len(a['amplitude_events']),
                    extra_amplitude=len(a['extra_amplitude_events']),phase_complete=sum(e['complete'] for e in a['phase_events']),
                    extra_phase=len(a['extra_phase_events']),local_complex_error=local,cpu_seconds=s['cpu_seconds'],
                    frequency_queries=s['base_frequency_queries']+s['auxiliary_frequency_queries'],quadrature_refined=s['quadrature_refined'],
                    selection_reason=a['selection']['reason']))
    write_csv(dest/'SEEDS.csv',rows);write_csv(dest/'EVENTS.csv',events);write_csv(dest/'INCOMPLETE.csv',failed)
    paired=[]
    # Report all paired differences, not only favorable or accepted rows.
    lookup={(r['target'],r['method'],r['seed']):r for r in rows}
    for ident in target_ids():
        for method in CFG['methods'][1:]:
            for seed in CFG['seeds']:
                a=lookup.get((ident,method,seed));b=lookup.get((ident,'MAG_PHA',seed))
                if a is None or b is None:continue
                paired.append(dict(target=ident,method=method,seed=seed,both_accepted=a['accepted'] and b['accepted'],
                    mag_delta=a['mag']-b['mag'],pha_delta=a['pha']-b['pha'],local_error_delta=a['local_complex_error']-b['local_complex_error'],
                    complete_delta=a['focused_complete']-b['focused_complete'],extra_delta=a['extra_amplitude']-b['extra_amplitude'],
                    cpu_ratio=a['cpu_seconds']/b['cpu_seconds']))
    write_csv(dest/'PAIRED.csv',paired)
    checks=dict(expected=CFG['planned_jobs'],completed_audited=len(rows),accepted=sum(r['accepted'] for r in rows),incomplete=failed,
        recovered_sha256=recovery['sha256'],source_hashes_verified=True,budget_stage_file_hashes_verified=True,paired_initializations_verified=True,
        automated_tables_complete=len(rows)==CFG['planned_jobs'],manual_review_complete=False,updated_unix=time.time())
    dump(OUT/'REVIEW_STATUS.json',checks)
    text=['# 联合面积实验：自动统计初稿','',f"已回收并逐文件核对 {len(rows)}/{CFG['planned_jobs']} 个任务；数值及结构复核接受 {checks['accepted']} 个。",'',
        '这份表不是最终结论。尚需人工检查局部连续误差、全种群机会、三维轨迹和反例。失败样本不能悄悄排除；通过数值检查也不代表结构拟合成功。','',
        '|条件|方法|已审计|通过复核|幅值MSE中位数|相位MSE中位数|关注结构完整数中位数|CPU核时|','|---|---|---:|---:|---:|---:|---:|---:|']
    for ident in target_ids():
        for method in CFG['methods']:
            rr=[r for r in rows if r['target']==ident and r['method']==method]
            if rr:text.append(f"|{ident}|{method}|{len(rr)}|{sum(r['accepted'] for r in rr)}|{np.median([r['mag'] for r in rr]):.5g}|{np.median([r['pha'] for r in rr]):.5g}|{np.median([r['focused_complete'] for r in rr]):g}|{sum(r['cpu_seconds'] for r in rr)/3600:.3f}|")
    text+=['','方案见 experiments/complex_area_20261014/PROTOCOL.md；原始逐seed、事件连续量及配对差在 analysis/。所有条件都参与过指标开发，8个seed不足以宣称独立推广。']
    (OUT/'RAW_REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(checks,ensure_ascii=False),flush=True)

if __name__=='__main__':main()

"""Descriptive pre-optimization diagnostics; no outcome-based metric tuning."""
from core import *
import csv

def main():
    rows=[json.loads(p.read_text()) for p in sorted((OUT/'preflight_rows').glob('*.json'))]
    rows=[r for r in rows if r['source']=='20261014 official knee final']
    dest=OUT/'development';dest.mkdir(exist_ok=True)
    details=[]
    for r in rows:
        values=np.asarray(r['window_scores']);old=values[:,2];new=values[:,3]
        for k,label in enumerate(r['window_labels']):
            details.append(dict(target=r['target'],source_method=r['method'],seed=r['seed'],window=label,
                original_ARX=float(old[k]),scaled_ARX=float(new[k]),old_share=float(old[k]/old.sum()) if old.sum() else 0,
                new_share=float(new[k]/new.sum()) if new.sum() else 0,
                J0=float(values[k,0]),J0_weighted=float(values[k,1])))
    with (dest/'WINDOW_ALLOCATION.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
    pre=json.loads((OUT/'PREFLIGHT.json').read_text());text=['# ARX轻量实现：开发集合数值与计算成本核验','',
        '本文件仅报告预检和固定历史候选的评分分配，不报告新一轮优化效果。归一化权重改变不等于结构改善。','',
        f"固定开发核验共{pre['cases']}条，数值失败{pre['failed']}条。",'',
        '|开发条件|辅助求值中位加速倍数|','|---|---:|']
    text.extend(f'|{t}|{v:.2f}|' for t,v in pre['median_speedup_by_target'].items())
    text.extend(['','加速比较为同机同候选、预热并交替计时，不等于完整优化任务的加速倍数。',
        '原ARX分数与编译ARX_L具有相同定义；新尺度ARX_S_L改变跨窗口取舍。全部间隙与末端窗口保留，局部窗口内排序只发生正比例缩放。',
        '原Python源码、两版工程试算记录、逐候选数值核验及完整窗口贡献均保留。后续效果需在正式216任务中判断。'])
    (OUT/'DEVELOPMENT_CHECK.md').write_text('\n'.join(text)+'\n',encoding='utf-8')

if __name__=='__main__':main()

"""Mechanical reuse of the audited runner/report; executed before freezing."""
from pathlib import Path
root=Path(__file__).resolve().parents[1];src=root/'experiments/curve3d_refinement_20261003';dst=root/'experiments/curve3d_directional_20261003'
if (dst/'FROZEN.json').exists():raise RuntimeError('Already frozen')
text=(src/'finish_report.py').read_text(encoding='utf-8')
text=text.replace('from global_metrics import PARENTS,CATALOG','from global_metrics import COMPARISONS,CATALOG')
text=text.replace('for name,parent in PARENTS.items():','for name,parent in COMPARISONS:')
text=text.replace('range(100,200)','range(200,300)')
text=text.replace('40-test','44-test').replace('len(p)==40','len(p)==44')
text=text.replace('p_40_primary','p_44_primary').replace('所有 40 项','所有 44 项')
text=text.replace('第三轮：指标改进与新随机种子正式复核','第四轮：方向、端点对称与双辅助正式实验')
text=text.replace('（100–199）','（200–299）')
text=text.replace('所有 44 项主目标比较统一 Holm 修正；','所有 24 项对基线和 20 项对父组的主目标比较统一 Holm 修正；')
text=text.replace('累计弧长不看方向／平移，累计面积不唯一决定形状，标准差不是抗异常点方法。',
    '三个变体不消除平移、噪声与不可达目标的问题；方向分解有部分主误差信息重合，双辅助也可能变差。')
(dst/'finish_report.py').write_text(text,encoding='utf-8')
print(dst/'finish_report.py')

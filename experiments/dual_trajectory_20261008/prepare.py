from pathlib import Path
import json,hashlib,shutil,tarfile
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
source=ROOT/'experiments/optimized_curve_structure_20261007'
dst=HERE/'frozen_dual';dst.mkdir(exist_ok=False)
for name in ['model.py','run_dual.py','circuits.json']:shutil.copyfile(source/name,dst/name)
shutil.copytree(source/'frozen',dst/'frozen',ignore=shutil.ignore_patterns('__pycache__'))
seeds=sorted(np.random.default_rng(20261008).choice(np.arange(2300,2400),3,replace=False).tolist())
config=dict(cases=['case1','case2'],seed_pool=[2300,2399],analysis_seeds=list(range(2300,2400)),rng_seed=20261008,seeds=seeds,stages=list(range(20,301,20)),objectives=['magnitude_mse_db2','phase_mse_rad2'],selection='nondominated front min WSMSE; alpha is selection only',sensitivity='front min max(MAG,alpha*PHA)',sampling='original frozen target points; this is a replay for trajectory, not a sampling comparison',planned_runs=200,planned_evaluations=10500000)
(HERE/'CONFIG.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8')
(HERE/'PROTOCOL.md').write_text(f'''# MAG/PHA双目标每20代过程查看

优化只使用两个目标：平均平方dB幅值误差、平均平方rad相位误差。沿用原始频点、边界、200/150种群、300代、初始化与算子。

从2300–2399等概率无放回抽3个seed；RNG=20261008；两Case共用{seeds}。在重新运行和读取这些seed的曲线前冻结；不按效果挑选。

每20代保存环境选择后的完整种群、非支配前沿和参数，每seed在同代前沿取最小WSMSE代表解。α沿用论文值，仅用于选解，不参与双目标优化。没有唯一的双目标“最好”。同时保存min max(MAG,αPHA)均衡选解。

每个阶段把目标与3条独立seed曲线画在同图，幅值/相位分图，不平均曲线，不跨seed选冠军。不同阶段的代表解可以是不同个体，不声称连续跟踪同一个个体，也不要求误差单调。

Case1解析目标/拟合使用候选模态加密网格检查，Case2只在测点比较。每阶段做网格敏感性检查；统计局部峰谷/相位缺陷。这里只随机3seeds，不能替代100seeds总体结论。

后台重放全部100seeds/Case，共200次运行；随机3seeds只用于画图。末代种群参数与旧实验相同seed逐项核对；独立电路及损失核验所有3000阶段种群。全部交付本地Markdown，不发布网站。
''',encoding='utf8')
manifest={p.relative_to(HERE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.rglob('*') if p.is_file() and p.suffix!='.gz' and '__pycache__' not in str(p)}
(HERE/'SOURCES.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
with tarfile.open(HERE/'bundle.tar.gz','w:gz') as tar:
    for name in ['run.py','CONFIG.json','PROTOCOL.md','SOURCES.json','frozen_dual']:tar.add(HERE/name,arcname=name)
print(json.dumps(config,ensure_ascii=False))

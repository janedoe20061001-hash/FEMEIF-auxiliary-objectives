from common import *
from concurrent.futures import ProcessPoolExecutor,as_completed
import time,platform
def job(arg):
    s,seed=arg;m=Target(s);cat=target_catalog(m);rows=[]
    with np.load(HERE/'batch'/label(s)/f'seed_{seed}'/'trajectory.npz') as d:
        for g,xs in zip(d['generations'],d['log_parameters']):
            for index,x in enumerate(xs):
                r2=regional(m,10**x,cat,2);r4=regional(m,10**x,cat,4)
                for a,b in zip(r2,r4):
                    stable=all(abs(a[k]-b[k])<=max(floor,.01*abs(b[k])) for k,floor in [('mag_contribution',.0001),('phase_contribution',np.deg2rad(.05)**2),('global_mag',.01),('global_phase',np.deg2rad(.5)**2)])
                    rows.append(dict(scale=s,seed=seed,generation=int(g),rule=['min_wsmse','balanced_minmax'][index],window=b['window'],weighting=b['weighting'],stable=bool(stable)))
    return rows
def main():
    cfg=json.loads((HERE/'CONFIG.json').read_text());assert json.loads((HERE/'batch/status.json').read_text())['status']=='complete'
    assert json.loads((OUT/'analysis/VALIDATION.json').read_text())['quadrature_failures']==0
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(job,(s,seed)) for s in cfg['run_scales'] for seed in cfg['seeds']]
        for f in as_completed(futures):rows.extend(f.result())
    csvout(OUT/'analysis/REGIONAL_CONVERGENCE.csv',rows);assert all(r['stable'] for r in rows)
    # Numerical source and target inputs remain unchanged throughout optimization.
    hashes=json.loads((HERE/'INPUT_HASHES.json').read_text())
    for name,h in hashes.items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==h,name
    for seed in cfg['seeds']:
        xx=[np.load(HERE/'batch'/label(s)/f'seed_{seed}'/'initial_log_parameters.npy') for s in cfg['run_scales']]
        for x in xx[1:]:np.testing.assert_array_equal(xx[0],x)
    manifests={}
    for p in HERE.rglob('*'):
        if not p.is_file() or any(x in p.parts for x in ['.venv','__pycache__','smoke','timing']):continue
        if p.name=='SOURCE_MANIFEST.json':continue
        manifests[p.relative_to(HERE).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    dump(OUT/'SOURCE_MANIFEST.json',manifests)
    source_files=[OLD/'server_run/retrieved/analysis'/name for name in ['front_diagnostics.csv','front_events.csv']]
    for arm in ['original','adaptive']:
        for seed in range(2300,2400):
            source_files.extend([OLD/f'server_run/retrieved/runs/{arm}/seed_{seed}/trajectory.npz',OLD/f'server_run/retrieved/runs/{arm}/seed_{seed}/generation_1000.npz'])
    dump(OUT/'OFFLINE_INPUT_HASHES.json',{p.relative_to(HERE.parents[1]).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
    dump(OUT/'AUDIT.json',dict(passed=True,new_regional_rows=len(rows),new_unstable_regional_rows=0,input_sources_unchanged=True,paired_initializations_exact=True,offline_input_files=len(source_files)))
    text='''# 第二步核验记录

本轮首批探索为三档×8个seed×300代；全部24次正式运行完成，不漏报或替换失败seed。短运行和耗时预检使用独立seed，不计入结果。

## 已核验

- 预定目标、种子、窗口、候选对条件、结构门槛、选解规则、绘图种子在新结果之前固定，见PROTOCOL.json。
- 六档目标真参数均在范围内，原网络实现与独立闭式并联RLC求和响应一致，真值幅相误差接近数值零。
- 3档×51条预检候选：批量/单条自适应评分一致，评分与2/4档独立密积分在允许误差内。
- 24次初始化精确等于对应seed生成的200×18矩阵；相同seed三档矩阵逐元素相同。
- 4800个末代候选评分重新计算；非支配前沿、主/均衡选解、参数边界核验。
- 4800个末代候选全部独立密网格积分复查；网格加密稳定，训练MAG/PHA偏差在预定允许范围内。
- 720条阶段代表的保存成绩重新计算；全部代表与末代前沿曲线的事件诊断通过加密稳定检查。
- 新阶段代表的6480条区域积分另作2/4档加密复核，通过预定绝对/相对允许误差；已有1600条代表的14400条区域积分同样通过。
- 数值源文件及目标输入的散列值在运行后保持一致；旧数据输入散列另存，不将旧服务器运行拼进新配对基线。

## 统计与解释限制

- seed是统计单位；大量前沿候选对不当作独立重复。精确克隆参数候选去重；相同曲线的参数排列不一定被完全去重，故候选对数量仅为附录信息。
- 所有贡献比例逐曲线计算后取平均；总误差为零时比例无定义。training_discrete、training_cell_reference、common_logf三种口径分别保存。
- 新目标的23—26 kHz固定窗口不会包含加宽条件的V2；跨条件局部相位主要看共同的10—31.62 kHz窗口，事件检查始终针对各目标真实P3/V2。核心窗口随目标宽度变化，跨条件需同时检查权重。
- P3宽度变化伴随V2位置、谷峰间距、相位、背景、真参数位置变化，不是纯峰宽因果对照。
- Case1各RLC单元串联求和可交换顺序。固定参数编号的真值距离不等于唯一物理解距离；目标参数位置信息是搜索范围描述，不能单独说明恢复难度原因。
- 峰谷匹配限制为0.25 decade，合格位置限制0.02 decade；极窄峰的工程定位要求可能需要相对峰宽的更严格标准。本轮沿用原标准，并报告连续误差/门槛敏感性，不临时改主判据。
- 8个seed/300代属于探索。bootstrap零宽区间可能由样本退化造成，不证明等效；边际Wilson区间另报。
- 诊断例从旧结果按预定的最近成绩且显著形状差异筛选，展示图按新实验前固定的3个seed生成；不平均曲线。
- 本轮未加入辅助目标，没有回答辅助指标是否有效，也未用Case2或不同拓扑验证推广。

## 文件

analysis/VALIDATION.json、analysis/QUADRATURE.csv、analysis/REGIONAL_CONVERGENCE.csv、AUDIT.json记录复算。
SOURCE_MANIFEST.json记录新源代码、目标输入和原始结果散列；OFFLINE_INPUT_HASHES.json记录旧输入散列。
全部原始初始参数、每20代完整种群、两种代表轨迹保存在本地experiments/mse_structure_step2_20261009/batch。
'''
    (OUT/'AUDIT.md').write_text(text,encoding='utf8')
    (OUT/'REPRODUCE.md').write_text('''# 本地复现

在D:/develop/FEMEIF下使用独立Python环境。运行所用版本保存在experiments/mse_structure_step2_20261009/batch/manifest.json；依赖采用experiments/primary_loss_comparison_20261007/frozen/case1/requirements-lock.txt。

## 从已有原始结果重新生成分析

1. 使用该实验目录.venv/Scripts/python.exe运行offline.py（需要第一步本地原始结果和分析CSV）。
2. 运行analyze.py --workers 2；已完成的chunks会复用。若要完整重算，先把自己的旧分析输出备份到另外目录，不覆盖提供的数据。
3. 依次运行summarize.py、plots.py、audit.py、report.py。

## 重新优化

setup_inputs.py复制第一步数值源和固定规则；prepare.py生成六档目标；preflight.py核验；run.py smoke --workers 2与run.py timing --workers 2检查环境；run.py batch --workers 4运行正式24组。

run.py不会覆盖已经存在的smoke/timing/batch目录。独立复现实验应复制源码/目标到另一个实验目录并调整输出路径，保留已交付原始数据。不要删除本次结果来腾出复现目录。

源码和输入散列见SOURCE_MANIFEST.json；旧输入散列见OFFLINE_INPUT_HASHES.json；环境和运行摘要分别保存在batch/manifest.json、batch/status.json、batch/summaries.json。

本地全部使用Python评分；不加载第一步服务器Linux native.so，不连接服务器。
''',encoding='utf8')
    print('Audit passed')
if __name__=='__main__':main()

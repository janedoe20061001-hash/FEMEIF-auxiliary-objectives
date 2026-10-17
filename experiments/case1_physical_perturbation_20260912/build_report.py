"""Generate a readable report directly from saved experimental results."""
from pathlib import Path
import csv
import json
import math

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
rows=[json.loads(s) for s in (OUT/'all_runs.jsonl').read_text(encoding='utf-8').splitlines()]
summary=list(csv.DictReader((OUT/'summary.csv').open(encoding='utf-8-sig')))
checks=json.loads((OUT/'checks.json').read_text())
target=json.loads((OUT/'target_features.json').read_text())
run=json.loads((OUT/'run_summary.json').read_text())


def num(x):
    x=float(x)
    if abs(x)<1e-12:return '≈0'
    return f'{x:.5g}'


def table(headers, data):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in data])


def random_row(fam,level):
    return next(r for r in summary if r['kind']=='random' and r['family']==fam and float(r['level_pct'])==level)


def image_link(name):
    return (OUT/name).as_posix()


group_table=table(['扰动等级 p','幅值 MSE / dB²','相位 MSE / rad²','H_res','H_cancel','H_bw'],[
    [f'{level:g}%']+[num(random_row('all',level)[key+'_median']) for key in
     ['magnitude_mse_db2','phase_mse_rad2','H_resonance','H_antiresonance','H_bandwidth']]
    for level in [.1,.3,1,3,10,30]])
family_table=table(['扰动对象（p=1%）','幅值 MSE / dB²','相位 MSE / rad²','H_res','H_cancel','H_bw'],[
    [fam]+[num(random_row(fam,1)[key+'_median']) for key in
     ['magnitude_mse_db2','phase_mse_rad2','H_resonance','H_antiresonance','H_bandwidth']]
    for fam in ['R','L','C','all']])
control_table=table(['整体变换（a=1.1）','幅值 MSE','相位 MSE','H_res','H_cancel','H_bw'],[
    [label]+[num(next(r for r in rows if r['kind']=='structured' and r['family']==fam and r['sign']==1 and r['level_pct']==10)[k])
             for k in ['magnitude_mse_db2','phase_mse_rad2','H_resonance','H_antiresonance','H_bandwidth']]
    for fam,label in [('global_frequency','L、C 同乘 a'),('global_R','R 同乘 a'),
                      ('global_LC_ratio','L 乘 a，C 除 a'),('global_impedance','R、L 乘 a，C 除 a')]])
target_table=table(['高阻谐振频率 / Hz','+45° 频率 / Hz','−45° 频率 / Hz','相对带宽 b','1/b（等效 Q）'],[
    [num(f),num(edges[0]),num(edges[1]),num(b),num(q)] for f,edges,b,q in zip(
        target['resonance_hz'],target['phase_band_edges_hz'],target['relative_phase_bandwidth'],target['effective_Q'])])
corr=list(csv.DictReader((OUT/'correlations_fixed_level.csv').open(encoding='utf-8-sig')))
corr_table=table(['指标（all，p=1%，64 个方向）','与幅值 MSE 的 Spearman','与相位 MSE 的 Spearman'],[
    [label]+[num(next(r['spearman'] for r in corr if r['level_pct']=='1.0' and r['a']==key and r['b']==b))
             for b in ['magnitude_mse_db2','phase_mse_rad2']]
    for key,label in [('H_resonance','H_res'),('H_antiresonance','H_cancel'),('H_bandwidth','H_bw')]])
count_zero=sum(r['crossing_count_deficit_10']==0 for r in rows if r['kind']=='random' and r['level_pct']<=10)
count_total=sum(r['kind']=='random' and r['level_pct']<=10 for r in rows)

text=rf'''# Case I：物理特征指标的参数扰动实验

日期：2026-09-12。已完成 **{len(rows):,} 组阻抗谱**，包括未扰动目标。

本轮只回答“参数改变时，指标怎样响应，以及两个 MSE 怎样变化”。没有运行 NSGA-II，没有验证辅助目标组合的优化收益，也没有进行噪声试验。

## 主要观察

1. **高阻谐振位置与支路间抵消位置主要响应 L、C 变化；带宽对 R 变化提供了不同的响应。** 在 1% 的 R 类随机扰动中，H_bw 中位数约 0.00598，而两个位置指标约为 10⁻⁶。不同指标的数值不能直接比较“谁更好”，它们所测特征及尺度不同。
2. 固定方向增加扰动时，三个指标在本轮 64 个全部参数随机方向上都随等级单调不减；但**固定同一扰动等级比较不同方向时，带宽误差与两个 MSE 未必同向**。p=1% 时相关系数约 −0.094。因此不能把“响应幅度不同”写成“已证明互补协同”。
3. 存在明确盲区：整体阻抗放大 1.1 倍时，三个特征指标均不变，幅值 MSE 为 0.68534 dB²。它们有意忽略整体阻抗尺度，不能替代两个主目标。这种不敏感是否有利于保留潜力解，尚未验证。
4. 大扰动会碰到事件匹配的边界。本轮有 {run['rows_with_missing']} 组在 30% 等级出现未匹配事件或缺少带宽端点。未匹配不一定表示物理事件消失，也可能是移出固定匹配窗口或测量频带。

## 1. 电路与数据来源

电路为六个并联 RLC 单元串联，采用英文论文 Table III、Fig.10 的参数：

| 单元 | R / Ω | L / μH | C / μF |
| --- | --- | --- | --- |
| 1 | 2000 | 1585 | 10 |
| 2 | 3000 | 10 | 3.98 |
| 3 | 500 | 100 | 1 |
| 4 | 1200 | 2.512 | 0.631 |
| 5 | 800 | 1.259 | 0.501 |
| 6 | 50 | 0.251 | 0.158 |

$$Z(\omega)=\sum_{{k=1}}^6\frac{{1}}{{1/R_k+j(\omega C_k-1/(\omega L_k))}}.$$

工作目录只有两份 PDF，缺少中文文档引用的正式程序及 401 点自适应频率 CSV。**本次重新生成 100 Hz–1 MHz 的 65,537 点等对数间隔频率网格，所有样本共用。它不是原 401 点实验的复现，数值不得直接与原筛选结果混比。**

源文件：

- [英文论文]({(HERE.parents[1]/'Feature-Engineering_Enabled_Multi-Objective_Evolutionary_Impedance_Fitting_Technique.pdf').as_posix()})：Table III 与 Fig.10。
- [主目标公式说明]({(HERE.parents[1]/'双主目标实验_第一第二层级指标公式详解.pdf').as_posix()})：第 2 节，dB 幅值 MSE 与直接相减的弧度相位 MSE。

## 2. 三个候选指标及物理含义

目标指标均为零；所有损失越小越好。指标只比较端口响应特征，不把目标元件值或候选元件相对真值的误差放入指标。它们与已有事件位置类指标有关联，不声称是已经证实的新颖方法。

### A. 高阻谐振位置偏差 H_res

在正电阻的被动 Case I 中，从端口电抗 X 的正到负过零处识别并联型高阻谐振，得到频率 $f_{{r,k}}$。

单并联 RLC 的自然谐振频率是 $1/(2\pi\sqrt{{LC}})$。本指标比较的却是**完整电路的实际端口过零频率**，不是直接比较各单元的 $1/\sqrt{{LC}}$；其他单元的背景响应也会影响位置。

它希望识别“主要谐振过程已经出现在正确频段、绝对幅值仍有偏差”的候选。

### B. 支路间电抗抵消位置偏差 H_cancel

在 X 从负到正的过零处识别相邻高阻谐振之间的低阻抵消频率，记为 $f_{{c,k}}$。程序历史字段名为 `H_antiresonance`，报告统一称“支路间抵消”，避免不同领域对谐振/反谐振命名的歧义。

对 Case I，$\operatorname{{Im}}Z=\sum_k\operatorname{{Im}}Z_k$，因此它检查多个单元的容感贡献在什么频率相互抵消。它不仅受单个 LC 谐振位置影响，也受各单元相对响应强度影响。

它希望区别“峰的位置相近，但各单元之间的相对配合仍不同”的候选。

### C. 相对谐振带宽偏差 H_bw

对每个高阻谐振，在其相邻两个零点之间寻找最近的 +45° 左端点和 −45° 右端点：

$$b_k=\frac{{f_{{-,k}}-f_{{+,k}}}}{{f_{{r,k}}}}.$$

对单个理想并联 RLC，这两个端点是半功率点，$b=1/Q$，且

$$Q=R\sqrt{{C/L}},\qquad b=\frac{{1}}{{R}}\sqrt{{L/C}}.$$

因此，小扰动近似满足

$$\Delta\ln f_0\approx-\tfrac12(\Delta\ln L+\Delta\ln C),$$
$$\Delta\ln b\approx-\Delta\ln R+\tfrac12\Delta\ln L-\tfrac12\Delta\ln C.$$

前者描述频率移动，后者描述阻尼及 LC 配比造成的宽窄变化。**对多个单元共同构成的端口响应，b 只是等效相位过渡带宽；不将它等同于某个内部单元的真实 Q，也不将其解释为一般复杂电路的总储能。**

它希望保留“谐振宽窄已正确，但谐振位置或整体阻抗尺度仍有偏差”的候选。

### 统一损失与匹配

分别按目标同类型事件的对数频率中点划分互不重叠的窗口。在每个窗口内选择最近的同类型候选事件；不能重复匹配一个事件。带宽使用同一个已匹配高阻谐振的端点。

$$H_{{res}}=\frac{{1}}{{m_r}}\sum_k\ell(\ln(\widehat f_{{r,k}}/f_{{r,k}}^*)),$$
$$H_{{cancel}}=\frac{{1}}{{m_c}}\sum_k\ell(\ln(\widehat f_{{c,k}}/f_{{c,k}}^*)),$$
$$H_{{bw}}=\frac{{1}}{{m_r}}\sum_k\ell(\ln(\widehat b_k/b_k^*)),$$

其中 $\ell(t)=\min(|t|,1)$；事件未匹配或带宽端点缺失时，该事件罚值为 1。多余事件没有独立罚项。截断与缺失罚值是本轮固定设计选择，不是物理常数。

H 是无量纲的平均绝对自然对数偏差；只有偏差很小时，100H 才近似“平均相对偏差百分数”。不能把 H 与单位为 dB²/rad² 的 MSE 直接按数值大小比较。

### 计算复杂度与本次数值实现

扫描过零和 ±45° 端点为 O(n)，事件排序由频率顺序天然给出。一般的窗口匹配直接实现为 O(mk)，其中 m 为目标事件数、k 为候选事件数；Case I 分别为 6 和 5，数量很小。可以用有序扫描进一步实现线性匹配。

本轮为了避免将插值误差当成特征变化，先用密集阻抗谱括住过零，再对**同一电路响应**做 28 步向量化二分细化。这是合成数据的数值参考版本，没有利用目标元件真值来评判候选，没有运行参数优化。但它比未来“仅在已有频点上插值”的轻量版本多了响应求值。全部三项特征含细化的单次中位时间约 {run['feature_median_ms']:.2f} ms（本机，不含整条阻抗谱的预先计算），不能据此声称已证明比旧实现快。

本轮目标识别出 6 个高阻谐振、5 个支路间抵消点：

{target_table}

5 个抵消频率 / Hz：{', '.join(num(v) for v in target['antiresonance_hz'])}。

## 3. 参数扰动协议

等级 p = 0.1%、0.3%、1%、3%、10%、30%。采用乘性对数扰动：

$$\theta_i=\theta_i^*\exp[\ln(1+p/100)d_i],\qquad \max_i|d_i|=1.$$

因此参数因子位于 $[1/(1+p/100),1+p/100]$。例如 p=10% 时为 0.90909–1.1 倍，**并不是对称的 ±10% 加性扰动，也不是每个参数都变化 10%**。

- 单元件扰动：18 个元件 × 正反两个方向 × 6 等级 = 216 组。
- 物理结构对照：4 种全局变换 × 正反两个方向 × 6 等级 = 48 组。
- 联合随机扰动：仅 R、仅 L、仅 C、全部参数，4 类 × 64 个方向 × 6 等级 = 1536 组。
- 未扰动目标：1 组。合计 1801 组。

固定随机种子 20260912。每类随机方向先从 [-1,1] 均匀采样，再按最大绝对值归一化。同一方向在所有等级复用，避免把方向差异混成幅度效应。不同类别的随机方向分别抽样，跨类别中位数比较是描述性比较。

主目标保持原文定义：幅值先转 dB，再计算逐点等权 MSE；相位使用 `np.angle` 的弧度值直接相减，再计算 MSE。由于目标自比较为零，这里报告绝对变化，不能报告“相对零基线下降百分比”。

## 4. 不同扰动程度的结果

下表为全部 R、L、C 联合随机扰动的 64 个方向中位数：

{group_table}

![扰动响应]({image_link('perturbation_response.png')})

所有指标在本轮 all 类的 64 条固定方向轨迹上都随等级单调不减。这个结论限于当前目标、当前范围与所抽方向，不是任意参数邻域的数学保证。缺失罚值也可能贡献大扰动时的增长。

## 5. 相同扰动等级，不同物理参数

下表均取 p=1%、每类 64 个方向的中位数：

{family_table}

![单参数敏感性]({image_link('parameter_sensitivity.png')})

该图中每一行分别归一化。为避免把 MSE 的平方增长与辅助指标的一阶增长混为一谈，图中两个主目标使用 RMSE；原始 MSE 仍保存在所有结果表中。

可以看到：带宽指标对 R 提供明显响应，而两个位置指标对 R 很弱；谐振频率指标对于不同 LC 单元的相对参数变化比较均匀，主目标则对不同元件赋予明显不同的响应强度。这是可观察的选择偏好差别，尚未证明哪种偏好更有利于演化搜索。

作为旧计数型指标的参照，本轮 p≤10% 的 {count_total} 个随机样本中，有 {count_zero} 个在“至少 10 次过零”指标上仍为零；位置和带宽指标在数量已满足时继续变化。这里只实现了中文文档给出的过零缺额公式，没有调用缺失的原程序，也未复现原先全部事件指标。

## 6. 结构对照：检查指标看到了什么，也检查它漏掉什么

{control_table}

四种操作均作用于全部单元：

- **L、C 同乘 a**：严格满足 $Z_{{new}}(\omega)=Z_{{target}}(a\omega)$，主要表现为频率轴整体移动；相对带宽保持不变（事件仍正确匹配时）。
- **R 同乘 a**：主要改变阻尼与谐振宽度，高 Q 情形下位置变化较弱。
- **L 乘 a、C 除 a**：每个单元的 LC 乘积不变，但 LC 配比改变；单元自然频率不变，端口位置仍可能因叠加影响略变。
- **R、L 同乘 a，C 除 a**：严格满足 $Z_{{new}}=aZ_{{target}}$。所有频率特征与相对带宽不变，但绝对幅值偏离目标。这是可复核的盲区，而非程序故障。

对于最后一种 a=1.1 的变换，理论幅值 MSE 为 $[20\log_{{10}}(1.1)]^2=0.68534175$ dB²，与计算一致；相位 MSE 为数值零。

## 7. 与两个 MSE 的关系：固定幅度后再看

不能把 0.1% 到 30% 的样本全部混在一起算相关，然后把“大家随扰动变大”解释为协同。下表固定 p=1%，只比较同一幅度下的不同随机方向：

{corr_table}

带宽与主目标的相关弱且略为负，表明某些方向的带宽更接近目标，并不意味着两个 MSE 也更小；这只是本轮样本的描述性结果，没有据此做显著性判断。两项位置指标之间约为 0.458，位置与带宽之间也不是完全一致。

这同时给出“有不同信息偏好”的证据和“不能保证组合协同”的提醒。要判断是否值得进入后续优化实验，应保留这一结果，不能只报告随扰动等级增长的漂亮曲线。

## 8. 阻抗谱示例

下图固定随机方向 all_0，展示 1%、10%、30% 扰动，避免按结果好坏挑例子。完整复阻抗数组保存在 `example_spectra.npz`，其余谱可由每行保存的参数及电路公式精确重建。

![阻抗谱示例]({image_link('example_spectra.png')})

## 9. 数值核验与适用范围

- 目标自比较：两个 MSE 与三个辅助指标为零；目标为 6 个高阻谐振和 5 个抵消点。
- 交换串联单元次序：相对响应误差 {checks['permutation_relative_error']:.3g}。
- 整体阻抗缩放恒等式：相对响应误差 {checks['impedance_scale_relative_error']:.3g}。
- 整体频率缩放恒等式：相对响应误差 {checks['frequency_scale_relative_error']:.3g}。
- 单并联 RLC 的 ±45° 相对带宽等于 1/Q：相对误差 {checks['single_RLC_relative_bandwidth_error']:.3g}。
- 对 {checks['convergence_cases']} 个覆盖不同扰动类型、等级的样本，频率点数从 65,537 加密到 131,073：幅值 MSE 最大相对变化约 {checks['max_magnitude_mse_grid_change']*100:.5g}%，相位 MSE 最大相对变化约 {checks['max_phase_mse_grid_change']*100:.5g}%；细化后的特征损失最大绝对变化 {checks['max_refined_feature_grid_change']:.3g}。
- 若不做二分细化，只在 65,537 点谱上直接线性插值，这 32 个样本的特征损失最大绝对误差仍可达 {checks['max_interpolation_only_feature_error']:.5g}。**窄带特征的采样问题必须单独处理；本轮结果不代表 401 点实测谱也能达到同样精度。**

没有噪声试验，没有 Case II 试验，没有证明端口特征能唯一恢复内部元件。特别是带宽端点在噪声、重叠模式和未充分采样时可能不稳定。本轮先建立干净模型上的参数响应事实，不将其扩展成噪声鲁棒性或组合优化有效性的结论。

## 10. 文件与复现

- `run_study.py`：电路、指标、扰动、统计、数值核验与绘图。
- `build_report.py`：从结果自动生成本报告。
- `results/config.json`：完整实验配置与目标参数。
- `results/all_runs.csv`：逐样本的参数、两个 MSE、三个指标及匹配状态。
- `results/all_runs.jsonl`：同上，另外保存每个事件的有符号对数偏差，未定义项为 null。
- `results/summary.csv`：每类/等级的中位数及 10%、90% 分位数。
- `results/correlations_fixed_level.csv`：固定扰动等级的相关性。
- `results/monotonicity.csv`：每个方向上的单调性检查。
- `results/numerical_convergence.csv`、`checks.json`：数值验证。
- `results/example_spectra.npz`：目标及三个固定示例的完整复阻抗谱。

使用 Python 3.12、numpy、matplotlib。本次依赖安装在工作区 `tmp/perturbation_deps`，脚本会在存在时自动使用；没有该目录时使用当前 Python 环境中的包。

```powershell
& 'C:\Users\Admin（无密码）\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'D:\develop\FEMEIF\experiments\case1_physical_perturbation_20260912\run_study.py'
& 'C:\Users\Admin（无密码）\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'D:\develop\FEMEIF\experiments\case1_physical_perturbation_20260912\build_report.py'
```
'''
(HERE/'实验报告.md').write_text(text,encoding='utf-8')
print('Report written:',HERE/'实验报告.md')

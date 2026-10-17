"""Draw fixed analytical illustrations, NOT fitting or metric benchmarking.

No measured spectra, random perturbations, objective scoring, or optimizers.
The only interval calculation is the exact analytical mean used as a figure.
"""
from pathlib import Path
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tmp' / 'perturbation_deps'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm

OUT = ROOT / 'output' / 'trend_metric_explanation_20261004'
OUT.mkdir(parents=True, exist_ok=True)
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({
    'font.family': 'Microsoft YaHei', 'font.size': 11,
    'axes.unicode_minus': False, 'axes.spines.top': False,
    'axes.spines.right': False, 'svg.fonttype': 'path',
    'figure.facecolor': 'white', 'axes.facecolor': '#fbfcfe',
    'text.color': '#17324d', 'axes.labelcolor': '#17324d',
    'xtick.color': '#52667b', 'ytick.color': '#52667b',
})

BLUE, ORANGE, RED = '#087f8c', '#e08127', '#c65760'
erf = np.vectorize(math.erf, otypes=[float])

def peak(u, center=.5, width=.14):
    return np.exp(-((u-center)/width)**2)

def primitive(u, center=.5, width=.14):
    return width*math.sqrt(math.pi)/2*erf((u-center)/width)

EXAMPLES = [
    ('01_offset', '整体偏高', lambda u: peak(u)+.2,
     lambda u: .2*u, '候选在全频带都高 0.2。',
     '每个区间平均都为 +0.2；弧长却完全不变。'),
    ('02_shift', '峰沿频率移动', lambda u: peak(u, .57),
     lambda u: primitive(u, .57)-primitive(u),
     '峰向右移 0.07，宽度和高度不变。',
     '左边偏低、右边偏高；跨过两侧的区间会抵消。'),
    ('03_width', '峰位不变、峰变宽', lambda u: peak(u, width=.24),
     lambda u: primitive(u, width=.24)-primitive(u),
     '峰位置和高度不变，两侧变宽。',
     '两侧持续偏高；峰位指标不足以描述这种差异。'),
    ('04_ripple', '快速正负起伏', lambda u: peak(u)+.2*np.cos(16*np.pi*u),
     lambda u: .2*np.sin(16*np.pi*u)/(16*np.pi),
     '叠加 8 周期余弦；不预设它是噪声。',
     '短区间仍有偏差，长区间中的正负起伏容易抵消。'),
]

u = np.linspace(0, 1, 1201)  # Drawing resolution, not a metric setting.
ends = np.linspace(0, 1, 241)
aa, bb = np.meshgrid(ends, ends)
valid = bb > aa
norm = TwoSlopeNorm(vmin=-.5, vcenter=0, vmax=.5)

def draw_row(axes, example, compact=False):
    slug, title, candidate, integral, detail, takeaway = example
    y0, y1 = peak(u), candidate(u)
    r = y1-y0
    a0, a1, a2 = axes
    a0.plot(u, y0, color=BLUE, lw=2.2, label='目标')
    a0.plot(u, y1, color=ORANGE, lw=2, ls='--', label='示例候选')
    a0.set(xlim=(0,1), ylim=(-.27,1.3), xlabel='归一化 log 频率 u', ylabel='示意坐标 y')
    a0.set_title(title + '：曲线', loc='left', fontweight='bold', pad=12)
    a0.legend(loc='upper left', frameon=False, fontsize=9, ncol=2)
    a1.axhline(0, color='#9eacb9', lw=.8)
    a1.fill_between(u, r, 0, where=r>=0, color=RED, alpha=.3)
    a1.fill_between(u, r, 0, where=r<0, color=BLUE, alpha=.25)
    a1.plot(u, r, color='#52667b', lw=1.5)
    a1.set(xlim=(0,1), ylim=(-.53,.53), xlabel='归一化 log 频率 u', ylabel='残差 r = 候选 − 目标')
    a1.set_title('逐点偏差：正负不能混为一谈', loc='left', fontsize=11, pad=12)
    values = np.full(aa.shape, np.nan)
    values[valid] = (integral(bb[valid])-integral(aa[valid]))/(bb[valid]-aa[valid])
    mesh = a2.pcolormesh(ends, ends, np.ma.masked_invalid(values),
                         cmap='RdBu_r', norm=norm, shading='nearest', rasterized=True)
    a2.plot([0,1], [0,1], color='#8495a4', lw=.7, ls=':')
    a2.text(.64, .24, 'b ≤ a\n不参与', ha='center', va='center', color='#91a0af', fontsize=10)
    a2.set(xlim=(0,1), ylim=(0,1), xlabel='区间左端 a', ylabel='区间右端 b')
    a2.set_title('每个区间的平均偏差 m(a,b)', loc='left', fontsize=11, pad=12)
    for ax in axes:
        ax.set_xticks([0,.5,1])
        ax.grid(alpha=.15)
    cb = axes[0].figure.colorbar(mesh, ax=a2, fraction=.045, pad=.035)
    cb.set_label('红：平均偏高｜蓝：平均偏低', fontsize=9)
    cb.set_ticks([-.5,0,.5])
    if not compact:
        axes[0].figure.text(.06, .025, detail+'  '+takeaway, fontsize=11)

for example in EXAMPLES:
    fig, axs = plt.subplots(1, 3, figsize=(14.7,4.25), gridspec_kw={'width_ratios':[1.1,1,1.05]})
    fig.subplots_adjust(left=.06, right=.975, top=.76, bottom=.20, wspace=.36)
    fig.suptitle('数学示例 · '+example[1], x=.06, ha='left', y=.96, fontsize=18, fontweight='bold')
    fig.text(.06,.85,'不是 Case1 / Case2 数据，不是优化结果；色标四图一致，颜色不是最终指标得分。',fontsize=10,color='#687b8e')
    draw_row(axs,example)
    fig.savefig(OUT / (example[0]+'.png'), dpi=160, bbox_inches='tight', pad_inches=.18)
    fig.savefig(OUT / (example[0]+'.svg'), bbox_inches='tight', pad_inches=.18)
    plt.close(fig)

fig, axs = plt.subplots(4,3,figsize=(15,14.8),gridspec_kw={'width_ratios':[1.1,1,1.05]})
fig.subplots_adjust(left=.06,right=.975,top=.90,bottom=.085,hspace=.62,wspace=.36)
fig.suptitle('全区间平均残差：它在看什么，又会忽略什么？',x=.06,ha='left',y=.975,fontsize=22,fontweight='bold')
fig.text(.06,.941,'固定数学示例，不使用实测数据，不评估优化收益。左看曲线，中看逐点偏差，右看各区间的平均偏差。',fontsize=12)
for row, example in zip(axs,EXAMPLES):
    draw_row(row,example,True)
fig.text(.06,.02,'结论：持续同号偏差会保留，正负起伏可能抵消；这不是只有新指标才具备的作用，也不能替代原来的两个主目标。',fontsize=12)
fig.savefig(OUT/'overview.png',dpi=140,bbox_inches='tight',pad_inches=.18)
plt.close(fig)
print('Created four analytical figure pairs (PNG/SVG) and overview.png in', OUT)

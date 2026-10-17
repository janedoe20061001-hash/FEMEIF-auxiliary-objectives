"""Original scientific diagrams following the paper's worked-example method.

All examples are synthetic. Scores call the existing Case II implementation;
no optimization result or fitted component parameter is inferred from a diagram.
"""
from pathlib import Path
import importlib.util
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "figures" / "ii01_explanation"
OUT.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location(
    "ii01_objectives", ROOT / "experiments/case2_dual_pdf18/objectives.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
NAME = "characteristic_frequency_shift_error_decades"
font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
plt.rcParams.update({
    "font.family": "Microsoft YaHei", "font.size": 11,
    "axes.unicode_minus": False, "axes.spines.top": False,
    "axes.spines.right": False, "axes.labelcolor": "#173247",
    "text.color": "#173247", "axes.titleweight": "bold",
    "svg.fonttype": "none", "savefig.facecolor": "white",
})
TARGET, LEFT, RIGHT, NORMAL = "#203b50", "#168c8b", "#df7d2e", "#8b60a9"


def response(x):
    # One parallel RLC. Analytic model is evaluated beyond the display range,
    # so shifts need neither endpoint padding nor interpolation.
    f = 10 ** np.asarray(x)
    r, l, c = 120.0, 1e-3, 1 / ((2*np.pi*1e4)**2 * 1e-3)
    w = 2*np.pi*f
    return 1 / (1/r + 1j*(w*c - 1/(w*l)))


def amp(z):
    return 20*np.log10(np.maximum(abs(z), 1e-300))


def style(ax):
    ax.grid(alpha=.16)
    ax.tick_params(labelsize=10)


def save(fig, stem):
    fig.savefig(OUT / (stem + ".png"), dpi=210, bbox_inches="tight")
    fig.savefig(OUT / (stem + ".svg"), bbox_inches="tight")
    plt.close(fig)


x = np.linspace(2, 6, 501)
z = response(x)
candidates = [response(x+.03), response(x-.08)]
obj = module.Objectives(10**x, aux=(NAME,), target_z=z)
scores = obj.evaluate_spectra(np.array(candidates))[:, 2]
shifts = [np.real((a-z)*np.conj(obj.g))/obj.den for a in candidates]

fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), layout="constrained")
ax = axes[0]
ax.plot(x, amp(z), c=TARGET, lw=2.5, label="目标曲线")
for a, color, label in zip(candidates, [LEFT, RIGHT], ["候选 A：左移 0.03 decade", "候选 B：右移 0.08 decade"]):
    ax.plot(x, amp(a), c=color, lw=2, label=label)
for peak, color in [(4, TARGET), (3.97, LEFT), (4.08, RIGHT)]:
    ax.axvline(peak, color=color, ls=":", lw=1.1)
ax.annotate("峰位置向右错开", xy=(4.08, 40), xytext=(4.32, 36),
            arrowprops={"arrowstyle":"->", "color":RIGHT}, color=RIGHT)
ax.set(xlim=(3.35,4.65), xlabel=r"对数频率 $x = \log_{10}(f / \mathrm{Hz})$", ylabel="幅值 / dB",
       title="(a) 先看同一特征在频率轴上错开多少")
ax.legend(loc="lower center", fontsize=9.5)
style(ax)
ax = axes[1]
ax.axhline(0, c="#aebfc6", lw=1)
for s, color, label in zip(shifts, [LEFT, RIGHT], ["A 的局部投影估计", "B 的局部投影估计"]):
    ax.plot(x, s, c=color, lw=2, label=label)
ax.axhline(.03, c=LEFT, ls=":", label="A 的线性参考：+0.03")
ax.axhline(-.08, c=RIGHT, ls=":", label="B 的线性参考：−0.08")
ax.set(xlim=(2,6), xlabel="对数频率 x", ylabel=r"$\delta x_i$ / decade",
       title="(b) 不找峰，逐点用残差和切线估计")
ax.legend(loc="lower right", fontsize=9)
ax.text(.025,.97, f"本例 II-01：A = {scores[0]:.4f}，B = {scores[1]:.4f}",
        transform=ax.transAxes, va="top", bbox={"facecolor":"white","edgecolor":"#d7e2e2"})
style(ax)
save(fig, "01_frequency_shift")

fig, ax = plt.subplots(figsize=(8.5, 5.8), layout="constrained")
q = np.array([1.0,1.0])
g = np.array([2.4,1.5])
r = np.array([1.5,2.7])
projection = g*np.dot(r,g)/np.dot(g,g)
end = q+r
foot = q+projection
t = np.linspace(-.32,1.12,200)
curve = q[None,:] + t[:,None]*g[None,:] + .30*(t*t)[:,None]*np.array([-g[1],g[0]])
ax.plot(curve[:,0],curve[:,1], c=TARGET, lw=2.5, label="目标复阻抗路径（局部示意）")
ax.plot([q[0]-.9*g[0],q[0]+1.25*g[0]], [q[1]-.9*g[1],q[1]+1.25*g[1]],
        c="#91a2ac",ls="--",lw=1.4, label="目标点的切线")
for start, finish, color, width in [(q,end,RIGHT,2.5),(q,foot,LEFT,3),(foot,end,NORMAL,2),(q,q+.65*g,TARGET,1.5)]:
    ax.annotate("", xy=finish, xytext=start, arrowprops={"arrowstyle":"->","color":color,"lw":width})
ax.scatter(*q,c=TARGET,s=55,zorder=5)
ax.scatter(*end,c=RIGHT,s=65,zorder=5)
ax.scatter(*foot,c=LEFT,s=40,zorder=5)
ax.text(q[0]-.7,q[1]-.43,r"目标点 $Z_i^\star$",color=TARGET)
ax.text(end[0]+.13,end[1]-.04,r"同一频点的候选 $\widehat Z_i$",color=RIGHT)
ax.text(1.02,2.85,r"残差 $r_i$",color=RIGHT)
ax.text(2.16,1.35,"沿切线的分量\n"+r"$\approx g_i\,\delta x_i$",color=LEFT)
ax.text(3.20,3.02,"垂直分量\nII-01 不直接计入",color=NORMAL)
ax.text(2.76,1.77,r"方向和速度 $g_i$",color=TARGET)
ax.set(xlim=(-.05,4.9),ylim=(.1,4.45),aspect="equal",
       xlabel="阻抗实部 Re Z（示意单位）",ylabel="阻抗虚部 Im Z（示意单位）",
       title="把残差拆成：沿切线错位 + 垂直于切线的形变")
ax.legend(loc="upper left",fontsize=9.5)
style(ax)
save(fig,"02_complex_projection")

s = shifts[1]
w = obj.weight
contribution = w*abs(s)/max(w.sum(),1e-12)
fig, axes = plt.subplots(3,1,figsize=(10.8,7.6),sharex=True,layout="constrained")
axes[0].plot(x,abs(s),c=RIGHT,lw=2)
axes[0].set(ylabel=r"$|\delta x_i|$ / decade",title="(a) 先取绝对值：左移和右移都计为错位")
axes[1].plot(x,w,c=LEFT,lw=2)
axes[1].fill_between(x,0,w,color=LEFT,alpha=.10)
axes[1].set(ylabel=r"权重 $w_i$",ylim=(0,1.06),title="(b) 目标变化快，权重高；变化慢，权重低")
axes[2].plot(x,np.cumsum(contribution),c=TARGET,lw=2.5)
axes[2].set(ylabel="累计损失 / decade",xlabel=r"对数频率 $x = \log_{10}(f / \mathrm{Hz})$",
            title=r"(c) 将每点的 $w_i |\delta x_i| / \sum_j w_j$ 累加")
axes[2].scatter([x[-1]],[scores[1]],c=TARGET,zorder=5)
axes[2].text(5.1,scores[1]*.78,f"最终 II-01 = {scores[1]:.4f}",bbox={"facecolor":"white","edgecolor":"#d7e2e2"})
for ax in axes: style(ax)
save(fig,"03_weighted_accumulation")

# Equal residual energy at every sample, but different residual directions.
# These are explicit linearized constructions, not two optimization solutions.
a = .035
zt = z+a*obj.g
zn = z+1j*a*obj.g
construction_scores = obj.evaluate_spectra(np.array([zt,zn]))[:,2]
complex_mse = [float(np.mean(abs(v-z)**2)) for v in [zt,zn]]
np.testing.assert_allclose(complex_mse[0],complex_mse[1],rtol=1e-12)
np.testing.assert_allclose(construction_scores[1],0,atol=1e-12)
fig, axes = plt.subplots(1,2,figsize=(12.6,4.8),layout="constrained",gridspec_kw={"width_ratios":[1.35,1]})
ax = axes[0]
ax.plot(x,amp(z),c=TARGET,lw=2.5,label="目标")
ax.plot(x,amp(zt),c=LEFT,lw=2,label="候选 T：r = a g（沿切线）")
ax.plot(x,amp(zn),c=NORMAL,lw=2,label="候选 N：r = j a g（垂直切线）")
ax.set(xlim=(3.4,4.6),xlabel="对数频率 x",ylabel="幅值 / dB",title="(a) 同样大的复数残差，方向不同")
ax.legend(fontsize=9.5,loc="lower center")
style(ax)
ax=axes[1]
ax.bar([0,1],construction_scores,color=[LEFT,NORMAL],width=.55)
for i,v in enumerate(construction_scores):ax.text(i,v+.0012,f"{v:.5f}",ha="center")
ax.set(xticks=[0,1],xticklabels=["沿切线 T","垂直切线 N"],ylim=(0,.05),ylabel="II-01 / decade",
       title="(b) N 的 II-01 = 0，仍有形变")
ax.text(.04,.95,f"两者复阻抗 MSE 相同：{complex_mse[0]:.3f} Ω²\n这里不是幅值 MSE 或相位 MSE相同。",
        transform=ax.transAxes,va="top",fontsize=10,bbox={"facecolor":"#f3f6f7","edgecolor":"none"})
style(ax)
save(fig,"04_direction_counterexample")

result = {
    "example_kind":"synthetic_parallel_RLC_and_linearized_residuals",
    "frequency_points":len(x),"frequency_Hz":[float(10**x[0]),float(10**x[-1])],
    "right_shift_decades":.08,"left_shift_decades":.03,
    "left_II01":float(scores[0]),"right_II01":float(scores[1]),
    "right_frequency_factor":float(10**.08),
    "tangential_II01":float(construction_scores[0]),
    "normal_II01":float(construction_scores[1]),"equal_complex_mse_ohm2":complex_mse[0],
    "tangent_magnitude_mse_db2":float(obj.evaluate_spectra(zt)[0,0]),
    "normal_magnitude_mse_db2":float(obj.evaluate_spectra(zn)[0,0]),
    "epsilon_v":float(obj.ev),
    "score_source":"experiments/case2_dual_pdf18/objectives.py",
    "weight_sum":float(w.sum()),"weighted_sum_check":float(contribution.sum()),
}
np.testing.assert_allclose(contribution.sum(),scores[1],rtol=1e-12)
(OUT/"example_values.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
print(json.dumps(result,ensure_ascii=False,indent=2))

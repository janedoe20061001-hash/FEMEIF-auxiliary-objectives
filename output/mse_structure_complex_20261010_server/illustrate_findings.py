"""Few fixed-seed local views; no optimization or new score is performed."""
import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[k]="1"
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/"experiments/mse_structure_complex_20261010"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from protocol import targets,source_hashes
from sampling import Quadrature

base = ROOT/"retrieved/other"
manifest = json.loads((base/"batch_other/manifest.json").read_text(encoding="utf-8"))
cfg = manifest["config"]
assert manifest["hashes"]==source_hashes(cfg)
ts = {t["id"]:t for t in targets(cfg,"core",["rlc_ladder","three_level"])}
out = ROOT/"figures"
out.mkdir(exist_ok=True)

def curves(ident,lo,hi):
    t = ts[ident]
    q = Quadrature(t["circuit"],cfg)
    params=[]
    chunks=[np.linspace(lo,hi,4097)]
    for seed in cfg["plot_seeds"]:
        with np.load(base/"batch_other"/ident/f"seed_{seed}/generation_0300.npz") as a:
            p=a["log_parameters"][int(a["selected_index"])].copy()
        params.append((seed,p))
        f=10**q.reference_grid(10**p,2)
        chunks.append(f[(f>=lo)&(f<=hi)])
    f=np.unique(np.concatenate(chunks))
    zz=[("Target",*t["circuit"].response(f))]
    for seed,p in params:
        zz.append((f"seed {seed}",*t["circuit"].response(f,10**p)))
    return f,zz

for ident,lo,hi,name in [("rlc_ladder/base",25000,29000,"ladder_base_local"),("rlc_ladder/lc_reciprocal_8p0",46000,51500,"ladder_8_local")]:
    f,zz=curves(ident,lo,hi)
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    for label,z,dz in zz:
        style=dict(color="black",ls="--",lw=2) if label=="Target" else dict(lw=1.3)
        axes[0].plot(f/1000,20*np.log10(abs(z)),label=label,**style)
        axes[1].plot(f/1000,np.rad2deg(np.angle(z)),label=label,**style)
    axes[0].set(title=f"{ident}: fixed three seeds, generation 300",ylabel="Magnitude (dB ohm)")
    axes[1].set(xlabel="Frequency (kHz)",ylabel="Phase (degree)")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out/f"{name}.png",dpi=160)
    plt.close(fig)
    if name=="ladder_8_local":
        fig=plt.figure(figsize=(10,7))
        ax=fig.add_subplot(projection="3d")
        for label,z,dz in zz:
            style=dict(color="black",ls="--",lw=2) if label=="Target" else dict(lw=1.3)
            ax.plot(f/1000,z.real,z.imag,label=label,**style)
        ax.set(xlabel="Frequency (kHz)",ylabel="Re Z (ohm)",zlabel="Im Z (ohm)",title="Local impedance trajectories: same fixed seeds, generation 300")
        ax.view_init(elev=24,azim=-62)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out/"ladder_8_3d.png",dpi=160)
        plt.close(fig)

f,zz=curves("three_level/base",38000,75000)
fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
for label,z,dz in zz:
    style=dict(color="black",ls="--",lw=2) if label=="Target" else dict(lw=1.3)
    axes[0].plot(f/1000,np.rad2deg(np.angle(z)),label=label,**style)
    axes[1].plot(f/1000,np.rad2deg(np.log(10)*np.imag(dz/z)),label=label,**style)
axes[0].set(title="three_level/base: distinguish phase from phase slope",ylabel="Phase (degree)")
axes[1].set(xlabel="Frequency (kHz)",ylabel="Phase slope (degree / decade)")
for ax in axes:
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(out/"three_level_phase_shoulder.png",dpi=160)
plt.close(fig)
print("Generated four explanatory views from the existing fixed seeds; no optimization.")

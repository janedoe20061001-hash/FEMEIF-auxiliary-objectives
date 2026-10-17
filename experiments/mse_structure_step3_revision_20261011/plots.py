from core import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    out=OUT/"figures";out.mkdir(exist_ok=True)
    selected_seeds=[CFG["seeds"][0],CFG["seeds"][-1]]
    dump(OUT/"PLOT_SELECTION.json",dict(rule="first_and_last_seed_by_id_not_by_result",seeds=selected_seeds))
    for ident in CFG["targets"]:
        data=target(ident);geo=Geometry(data);q=data["quadrature"];c=data["circuit"]
        curves=[]
        for seed in selected_seeds:
            for method in CFG["methods"]:
                with np.load(OUT/f"batch/{ident}/{method}/seed_{seed}/generation_0300.npz") as a:
                    p=a["log_parameters"][int(a["primary"])].copy()
                curves.append((seed,method,p))
        fig,axes=plt.subplots(2,len(geo.windows),figsize=(6*len(geo.windows),6),squeeze=False)
        for k,w in enumerate(geo.windows):
            x=np.unique(np.r_[np.linspace(w["lo"],w["hi"],2049),q.reference_grid(c.p,2)])
            x=x[(x>=w["lo"])&(x<=w["hi"])]
            for _,_,p in curves:
                a=q.reference_grid(10**p,2);x=np.unique(np.r_[x,a[(a>=w["lo"])&(a<=w["hi"])]])
            f=10**x
            z,_=c.response(f)
            for ax,y in ((axes[0,k],20*np.log10(abs(z))),(axes[1,k],np.rad2deg(np.angle(z)))):
                ax.plot(f/1000,y,"k--",lw=2,label="Target")
            for i,(seed,method,p) in enumerate(curves):
                z,_=c.response(f,10**p);label=f'{"MAG/PHA" if method=="MAG_PHA" else "+ Geometry"}, {seed}'
                axes[0,k].plot(f/1000,20*np.log10(abs(z)),label=label,lw=1.2)
                axes[1,k].plot(f/1000,np.rad2deg(np.angle(z)),label=label,lw=1.2)
            axes[0,k].set_title(f'{ident}, {w["event"]}: common primary selector, generation 300')
            axes[0,k].set_ylabel("Magnitude (dB ohm)")
            axes[1,k].set(xlabel="Frequency (kHz)",ylabel="Phase (degree)")
            for row in axes:
                row[k].grid(alpha=.2);row[k].legend(fontsize=7)
                row[k].ticklabel_format(axis="x",style="plain",useOffset=False)
        fig.tight_layout();fig.savefig(out/f"{ident}_local.png",dpi=150);plt.close(fig)
        if ident in ("case1","ladder8"):
            lo,hi=min(w["lo"] for w in geo.windows),max(w["hi"] for w in geo.windows)
            pp=[(m,p) for seed,m,p in curves if seed==selected_seeds[0]]
            x=np.unique(np.r_[np.linspace(lo,hi,2049),*[q.reference_grid(10**p,2) for _,p in pp],q.reference_grid(c.p,2)])
            f=10**x[(x>=lo)&(x<=hi)]
            fig=plt.figure(figsize=(10,7));ax=fig.add_subplot(projection="3d")
            z,_=c.response(f);ax.plot(f/1000,z.real,z.imag,"k--",lw=2,label="Target")
            for m,p in pp:
                z,_=c.response(f,10**p);ax.plot(f/1000,z.real,z.imag,lw=1.4,label="MAG/PHA" if m=="MAG_PHA" else "+ Geometry")
            ax.set(xlabel="Frequency (kHz)",ylabel="Re Z (ohm)",zlabel="Im Z (ohm)",
                   title=f"{ident}: same primary selection, seed {selected_seeds[0]}")
            ax.view_init(elev=25,azim=-60);ax.legend();fig.tight_layout()
            # Reserve room for the projected z-axis label outside the axes.
            ax.set_box_aspect((1,1,1),zoom=.8)
            ax.tick_params(labelsize=9)
            for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
                axis.label.set_size(10)
                axis.labelpad=6
            fig.subplots_adjust(left=.05,right=.87,bottom=.14,top=.90)
            fig.savefig(out/f"{ident}_3d.png",dpi=150);plt.close(fig)
    # Explicitly identified positive example, not a substitute for all seeds.
    # This is selected by the observed diagnostic rule, and labeled as such.
    unique=[]
    for seed in CFG["seeds"]:
        d=json.loads((OUT/f"analysis/details/case1_MAG_PHA_GEOMETRY_{seed}_primary.json").read_text(encoding="utf-8"))
        if any(e["event_id"]=="P3" and e["complete"] for e in d["focus"]) and not d["summary"]["focus_joint_ok"]:
            unique.append(seed)
    if unique:
        seed=min(unique)
        data=target("case1");geo=Geometry(data);q=data["quadrature"];c=data["circuit"]
        w=next(w for w in geo.windows if w["event"]=="P3")
        samples=[]
        for method in CFG["methods"]:
            d=json.loads((OUT/f"analysis/details/case1_{method}_{seed}_primary.json").read_text(encoding="utf-8"))
            samples.append((method,np.array(d["log_parameters"])))
        x=np.unique(np.r_[np.linspace(w["lo"],w["hi"],2049),*[q.reference_grid(10**p,2) for _,p in samples],q.reference_grid(c.p,2)])
        f=10**x[(x>=w["lo"])&(x<=w["hi"])]
        fig=plt.figure(figsize=(12,6));grid=fig.add_gridspec(2,2)
        axes=[fig.add_subplot(grid[i,0]) for i in range(2)];ax3=fig.add_subplot(grid[:,1],projection="3d")
        for name,p in [("Target",c.p),*[("MAG/PHA" if m=="MAG_PHA" else "+ Geometry",10**p) for m,p in samples]]:
            z,_=c.response(f,p);style=dict(color="black",ls="--",lw=2) if name=="Target" else dict(lw=1.3)
            axes[0].plot(f/1000,20*np.log10(abs(z)),label=name,**style)
            axes[1].plot(f/1000,np.rad2deg(np.angle(z)),label=name,**style)
            ax3.plot(f/1000,z.real,z.imag,label=name,**style)
        axes[0].set(ylabel="Magnitude (dB ohm)",title=f"Case1 P3, seed {seed}: peak recovers, joint structure still fails")
        axes[1].set(xlabel="Frequency (kHz)",ylabel="Phase (degree)")
        for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8);ax.ticklabel_format(axis="x",style="plain",useOffset=False)
        ax3.set(xlabel="Frequency (kHz)",ylabel="Re Z (ohm)",zlabel="Im Z (ohm)");ax3.view_init(25,-60);ax3.legend(fontsize=8)
        fig.tight_layout();fig.savefig(out/"case1_partial_peak_example.png",dpi=150);plt.close(fig)
        selection=json.loads((OUT/"PLOT_SELECTION.json").read_text())
        selection["additional_example"]=dict(target="case1",seed=seed,rule="first_by_seed_id_new_primary_P3_complete_but_focus_joint_incomplete",candidates=unique)
        dump(OUT/"PLOT_SELECTION.json",selection)
    print("Fixed-rule figures and explicitly labeled diagnostic example generated",flush=True)

if __name__=="__main__":
    main()

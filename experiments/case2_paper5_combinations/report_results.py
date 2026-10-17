"""Audit every final population and produce generic Case2 combination reports."""
from pathlib import Path
from collections import defaultdict
import argparse,csv,hashlib,json
import numpy as np
import experiment as e
from objectives import CATALOG,Objectives
from case2 import PARAMETER_NAMES,LOWER,UPPER,load_target,impedance
from final_selection import select_wsmse
from summarize import summarize
from analysis import analyze


def main():
    ap=argparse.ArgumentParser();ap.add_argument("directory",type=Path);args=ap.parse_args();root=args.directory
    manifest=json.loads((root/"manifest.json").read_text());cfg=manifest["config"]
    for name,digest in manifest["code_sha256"].items():
        if hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()!=digest:raise ValueError(f"code changed: {name}")
    freq,target=load_target(cfg["target_data"]);alpha=cfg["final_selection"]["phase_weight"]
    expected_groups=[e.group_name(g) for g in e.groups(cfg)]
    seeds=range(cfg["seeds"]["start"],cfg["seeds"]["start"]+cfg["seeds"]["count"])
    expected={(g,s) for g in expected_groups for s in seeds}
    records=[];selected=[];initials=defaultdict(set);seen=set()
    for path in sorted(root.glob("*/seed_*/summary.json")):
        r=json.loads(path.read_text());key=(r["group"],r["seed"])
        if key in seen or r["fingerprint"]!=manifest["fingerprint"] or r["status"]!="complete":raise ValueError(f"invalid record {path}")
        seen.add(key);initial=np.load(path.parent/"initial_log_parameters.npy")
        digest=hashlib.sha256(initial.tobytes()).hexdigest();
        if digest!=r["initial_population_sha256"]:raise ValueError("initial population hash mismatch")
        initials[r["seed"]].add(digest)
        with np.load(path.parent/"final_population.npz") as pop:
            x=pop["log_parameters"];f=pop["objectives"]
            if x.shape!=(cfg["population"],15) or f.shape!=(cfg["population"],len(r["objective_names"])):raise ValueError("population shape mismatch")
            if np.any(x<LOWER) or np.any(x>UPPER) or not np.isfinite(f).all():raise ValueError("invalid population values")
            verified=Objectives(freq,cfg["features"],target_z=target).evaluate_spectra(impedance(freq,10**x))
            np.testing.assert_allclose(verified,f[:,:2],rtol=1e-11,atol=1e-12)
            check=select_wsmse(f,alpha);idx=check["wsmse_selected_index"]
            if idx!=int(pop["selected_index"]):raise ValueError("selected index mismatch")
            row=dict(group=r["group"],seed=r["seed"],**check)
            row.update({n:float(v) for n,v in zip(PARAMETER_NAMES,pop["selected_parameters"])})
            selected.append(row)
        records.append(r)
    if seen!=expected or any(len(v)!=1 for v in initials.values()):raise ValueError("missing runs or unpaired initial populations")
    rows=summarize(root);main_rows={r["group"]:r for r in rows if r["front"]=="wsmse_selected"}
    with (root/"selected_individuals.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(selected[0]));w.writeheader();w.writerows(selected)
    labels={"baseline":("BASE","双主目标基线")};labels.update({name:(ident,title) for ident,name,title in CATALOG})
    def lab(g):
        if g=="baseline":return labels[g]
        parts=g.split("+");return "+".join(labels[p][0] for p in parts),"＋".join(labels[p][1] for p in parts)
    improved=[];text=[f"# {cfg['experiment_name']}","",f"完成 {len(records)} 次运行；{len(expected_groups)} 组，每组 {cfg['seeds']['count']} seeds，种群 {cfg['population']}，{cfg['generations']} 代。",
      f"每个 seed 从最终种群选择 magnitude MSE + {alpha} × phase MSE 最小的同一个体；WSMSE 不参与进化。","",
      "|组别|辅助目标|幅值均值|相位均值|幅值变化%|相位变化%|两项同降seed|",
      "|---|---|---:|---:|---:|---:|---:|"]
    for g in expected_groups:
        r=main_rows[g];ident,title=lab(g)
        if g!="baseline" and r["both_means_lower"]:improved.append(g)
        text.append(f"|{ident}|{title}|{r['magnitude_mean']:.6g}|{r['phase_mean']:.6g}|{r['magnitude_change_pct']:.3f}|{r['phase_change_pct']:.3f}|{r['seeds_with_both_lower']}/{cfg['seeds']['count']}|")
    combos=[g for g in expected_groups if "+" in g]
    text[2]+=f" 两项均值均低于基线：全部非基线 {len(improved)}/{len(expected_groups)-1}；组合 {sum(g in improved for g in combos)}/{len(combos)}。"
    (root/"RESULTS.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    audit=dict(status="passed",runs=len(records),expected_runs=len(expected),same_seed_initial_populations_match=True,
               final_populations_recomputed=True,selection_recomputed=True,fingerprint=manifest["fingerprint"])
    (root/"completion_audit.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    analyze(records,cfg,root)
    print(json.dumps(dict(status="complete",runs=len(records),improved=len(improved))))


if __name__=="__main__":main()

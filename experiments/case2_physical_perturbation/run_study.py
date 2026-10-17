"""Deterministic Case2 perturbation study around a fitted reference individual."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
from scipy.stats import spearmanr
from case2 import PARAMETER_NAMES,impedance,load_target
import pdf18_objectives as pdf
import paper5_objectives as paper

HERE=Path(__file__).resolve().parent
LEVELS=(.1,.3,1.,3.,10.,30.)
FAMILIES={"R":np.arange(0,7),"L":np.arange(7,11),"C":np.arange(11,15),"all":np.arange(15)}
TARGET_SPEC={"file":"data/targetImpedance_case2.csv","sha256":"ec50906f6cd3e5fdc5792734f7339f8368d2a3280b0a0d14451b0420acc7b725","magnitude_unit":"ohm","phase_unit":"degree","sampling":"all_rows","points":201}


def read_reference(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as f:rows=[r for r in csv.DictReader(f) if r["group"]=="baseline"]
    if len(rows)!=100:raise ValueError("expected 100 completed baseline selections")
    row=min(rows,key=lambda r:float(r["wsmse_selected_value"]))
    params=np.asarray([float(row[n]) for n in PARAMETER_NAMES])
    return row,params


def factor(level,direction):return np.exp(np.log1p(level/100)*direction)


def plan(reference):
    rows=[dict(kind="reference",family="reference",level_pct=0.,direction_id="reference",sign=0,params=reference.copy())]
    for i,name in enumerate(PARAMETER_NAMES):
        for sign in (-1,1):
            direction=np.zeros(15);direction[i]=sign
            for level in LEVELS:rows.append(dict(kind="single",family=name[0],level_pct=level,direction_id=f"{name}_{sign:+d}",sign=sign,params=reference*factor(level,direction)))
    for family,ids in FAMILIES.items():
        for sign in (-1,1):
            direction=np.zeros(15);direction[ids]=sign
            for level in LEVELS:rows.append(dict(kind="structured",family=family,level_pct=level,direction_id=f"global_{family}_{sign:+d}",sign=sign,params=reference*factor(level,direction)))
    rng=np.random.default_rng(20260921)
    for family,ids in FAMILIES.items():
        for j in range(64):
            direction=np.zeros(15);raw=rng.uniform(-1,1,len(ids));raw/=np.max(np.abs(raw));direction[ids]=raw
            for level in LEVELS:rows.append(dict(kind="random",family=family,level_pct=level,direction_id=f"{family}_{j:02d}",sign=0,params=reference*factor(level,direction)))
    if len(rows)!=1765:raise AssertionError(len(rows))
    return rows


def write_csv(path,rows):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--baseline-selected",type=Path,required=True);ap.add_argument("--output",type=Path,default=HERE/"results");args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    source,reference=read_reference(args.baseline_selected);freq,target=load_target(TARGET_SPEC)
    rows=plan(reference);params=np.asarray([r.pop("params") for r in rows]);z=impedance(freq,params)
    pobj=pdf.Objectives(freq,pdf.DEFAULTS,pdf.AUX,target_z=target);paperobj=paper.Objectives(freq,paper.DEFAULTS,paper.AUX,target_z=target)
    pv=pobj.evaluate_spectra(z);qv=paperobj.evaluate_spectra(z)
    names=["magnitude_mse_db2","phase_mse_rad2"]+["pdf_"+a for a in pdf.AUX]+["paper_"+a for a in paper.AUX]
    values=np.column_stack([pv,qv[:,2:]])
    base=values[0]
    output=[]
    for i,row in enumerate(rows):
        out=dict(row,sample_id=i)
        out.update({n:float(v) for n,v in zip(PARAMETER_NAMES,params[i])})
        out.update({n:float(v) for n,v in zip(names,values[i])})
        out.update({"delta_"+n:float(v-b) for n,v,b in zip(names,values[i],base)})
        output.append(out)
    write_csv(args.output/"all_runs.csv",output)
    summary=[]
    for kind in ("single","structured","random"):
      for family in sorted({r["family"] for r in output if r["kind"]==kind}):
       for level in LEVELS:
        subset=[r for r in output if r["kind"]==kind and r["family"]==family and r["level_pct"]==level]
        for name in names:
            a=np.asarray([r[name] for r in subset]);summary.append(dict(kind=kind,family=family,level_pct=level,metric=name,n=len(a),median=float(np.median(a)),q10=float(np.quantile(a,.1)),q90=float(np.quantile(a,.9)),reference=float(base[names.index(name)])))
    write_csv(args.output/"summary.csv",summary)
    correlations=[]
    fixed=[r for r in output if r["kind"]=="random" and r["family"]=="all" and r["level_pct"]==1.]
    for name in names[2:]:
        for primary in names[:2]:
            correlations.append(dict(auxiliary=name,primary=primary,spearman=float(spearmanr([r[name] for r in fixed],[r[primary] for r in fixed]).statistic)))
    write_csv(args.output/"correlations_1pct_all.csv",correlations)
    monotonic=[]
    trajectories={(r["kind"],r["direction_id"]):[] for r in output if r["kind"] in ("single","structured","random")}
    for r in output:
        if r["kind"] in ("single","structured","random"):trajectories[r["kind"],r["direction_id"]].append(r)
    for (kind,direction),items in trajectories.items():
        items=sorted(items,key=lambda r:r["level_pct"])
        for name in names:
            a=np.asarray([r[name] for r in items]);monotonic.append(dict(kind=kind,direction_id=direction,metric=name,nondecreasing=bool(np.all(np.diff(a)>=-1e-12)),decreases=int(np.sum(np.diff(a)<-1e-12))))
    write_csv(args.output/"monotonicity.csv",monotonic)
    config=dict(seed=20260921,levels_pct=LEVELS,random_directions_per_family=64,total_spectra=len(output),reference_selection="minimum WSMSE among 100 Case2 baseline selected individuals",reference_seed=int(source["seed"]),reference_wsmse=float(source["wsmse_selected_value"]),source_file=str(args.baseline_selected),source_sha256=hashlib.sha256(args.baseline_selected.read_bytes()).hexdigest(),target=TARGET_SPEC)
    (args.output/"config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    text=["# Case2：拟合参考解周围的参数扰动与指标敏感性","",f"完成 {len(output)} 条谱。参考点固定为100个正式baseline选解中WSMSE最低者（seed {source['seed']}），它是拟合参考解，不是真实元件参数。",
          "扰动包含15个单参数正反方向、R/L/C/全参数整体缩放正反方向，以及R/L/C/全参数各64个固定随机方向；档位为0.1%、0.3%、1%、3%、10%、30%。",
          "所有误差都相对原始201点实测目标计算；参数不裁剪到优化边界。该实验描述局部响应，不能证明辅助目标加入进化后有效。","",
          "## 1%全参数随机方向的秩相关","","|辅助指标|幅值MSE|相位MSE|","|---|---:|---:|"]
    by={(r["auxiliary"],r["primary"]):r["spearman"] for r in correlations}
    for name in names[2:]:text.append(f"|{name}|{by[name,names[0]]:.3f}|{by[name,names[1]]:.3f}|")
    text += ["","完整逐样本结果见 all_runs.csv；summary.csv、monotonicity.csv 和 correlations_1pct_all.csv 分别保存分组分位数、轨迹单调性及固定档位相关。"]
    (args.output/"REPORT.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    print(json.dumps(dict(status="complete",spectra=len(output),reference_seed=int(source["seed"]))))


if __name__=="__main__":main()

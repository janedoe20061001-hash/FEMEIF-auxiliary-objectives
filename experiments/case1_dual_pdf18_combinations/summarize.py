"""Paired-seed summaries. Full-objective fronts and primary-only fronts stay separate."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def summarize(root):
    root=Path(root)
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    expected=manifest["config"]["seeds"]["count"]
    data={}
    reference=root.parent/"reference"
    sources=[root]+([reference] if reference.exists() else [])
    for source in sources:
      source_manifest=json.loads((source/"manifest.json").read_text(encoding="utf-8"))
      for path in source.glob("*/seed_*/summary.json"):
        r=json.loads(path.read_text(encoding="utf-8"))
        if r["fingerprint"]!=source_manifest["fingerprint"] or r["status"]!="complete":
            raise ValueError(f"Incompatible record {path}")
        if r["seed"] in data.get(r["group"],{}):
            raise ValueError(f"Duplicate record {path}")
        data.setdefault(r["group"],{})[r["seed"]]=r
    baseline=data.get("baseline",{})
    output=[]
    for group,records in sorted(data.items()):
        seeds=sorted(set(records)&set(baseline))
        if not seeds:
            continue
        for front in ("wsmse_selected","primary_front","full_front"):
            prefix="wsmse_selected_" if front=="wsmse_selected" else f"{front}_mean_"
            ka=f"{prefix}magnitude_mse";kp=f"{prefix}phase_mse"
            a=np.array([[records[s][ka],records[s][kp]] for s in seeds])
            b=np.array([[baseline[s][ka],baseline[s][kp]] for s in seeds])
            av,bv=a.mean(axis=0),b.mean(axis=0)
            percent=[float(100*(x/y-1)) if y>0 else None for x,y in zip(av,bv)]
            alpha=manifest["config"]["final_selection"]["phase_weight"]
            wa,wb=float(av[0]+alpha*av[1]),float(bv[0]+alpha*bv[1])
            output.append(dict(group=group,front=front,paired_seeds=len(seeds),
                expected_seeds=expected,complete=len(seeds)==expected,
                magnitude_mean=float(av[0]),phase_mean=float(av[1]),
                baseline_magnitude_mean=float(bv[0]),baseline_phase_mean=float(bv[1]),
                magnitude_change_pct=percent[0],phase_change_pct=percent[1],
                both_means_lower=bool(np.all(av<bv)),
                seeds_with_both_lower=int(np.all(a<b,axis=1).sum()),
                wsmse_alpha=alpha,wsmse_mean=wa,baseline_wsmse_mean=wb,
                wsmse_change_pct=float(100*(wa/wb-1)) if wb>0 else None,
                seeds_with_wsmse_lower=int(np.sum(a[:,0]+alpha*a[:,1]<b[:,0]+alpha*b[:,1])),
                seeds_compared=",".join(map(str,seeds))))
    if not output:
        raise ValueError("No completed paired baseline/group runs")
    with (root/"paired_summary.csv").open("w",encoding="utf-8-sig",newline="") as fh:
        writer=csv.DictWriter(fh,fieldnames=list(output[0]))
        writer.writeheader();writer.writerows(output)
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    print(json.dumps(summarize(args.directory),indent=2))

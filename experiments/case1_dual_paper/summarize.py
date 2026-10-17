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
    for path in root.glob("*/seed_*/summary.json"):
        r=json.loads(path.read_text(encoding="utf-8"))
        if r["fingerprint"]!=manifest["fingerprint"] or r["status"]!="complete":
            raise ValueError(f"Incompatible record {path}")
        data.setdefault(r["group"],{})[r["seed"]]=r
    baseline=data.get("baseline",{})
    output=[]
    for group,records in sorted(data.items()):
        seeds=sorted(set(records)&set(baseline))
        if not seeds:
            continue
        for front in ("full_front","primary_front"):
            ka=f"{front}_mean_magnitude_mse";kp=f"{front}_mean_phase_mse"
            a=np.array([[records[s][ka],records[s][kp]] for s in seeds])
            b=np.array([[baseline[s][ka],baseline[s][kp]] for s in seeds])
            av,bv=a.mean(axis=0),b.mean(axis=0)
            percent=[float(100*(x/y-1)) if y>0 else None for x,y in zip(av,bv)]
            output.append(dict(group=group,front=front,paired_seeds=len(seeds),
                expected_seeds=expected,complete=len(seeds)==expected,
                magnitude_mean=float(av[0]),phase_mean=float(av[1]),
                baseline_magnitude_mean=float(bv[0]),baseline_phase_mean=float(bv[1]),
                magnitude_change_pct=percent[0],phase_change_pct=percent[1],
                both_means_lower=bool(np.all(av<bv)),
                seeds_with_both_lower=int(np.all(a<b,axis=1).sum()),
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

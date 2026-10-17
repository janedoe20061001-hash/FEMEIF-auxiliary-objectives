"""Read independent source manifests without rewriting historical fingerprints."""
import json
from pathlib import Path
import numpy as np


def sources(root):
    root=Path(root)
    paths=[root]+sorted(p.parent for p in (root.parent/"reference").glob("*/manifest.json"))
    manifests=[json.loads((p/"manifest.json").read_text(encoding="utf-8")) for p in paths]
    current=manifests[0]
    for other in manifests[1:]:
        if {k:v for k,v in current["config"].items() if k!="groups"}!={k:v for k,v in other["config"].items() if k!="groups"}:
            raise ValueError("Reference configuration mismatch")
        a,b=np.asarray(current["frequency_hz"]),np.asarray(other["frequency_hz"])
        # Windows/Linux geomspace can differ at the final binary digit.
        # Grid settings must still match exactly above; no sampling change is allowed.
        if a.shape!=b.shape or not np.allclose(a,b,rtol=4*np.finfo(float).eps,atol=0):
            raise ValueError("Reference frequencies mismatch")
        for name in ("objectives.py","final_selection.py","requirements-lock.txt"):
            if current["code_sha256"][name]!=other["code_sha256"][name]:
                raise ValueError("Reference definitions mismatch")
        for name in ("numpy","pymoo","scipy","moocore"):
            if current["versions"][name]!=other["versions"][name]:
                raise ValueError("Reference dependencies mismatch")
    return list(zip(paths,manifests))


def validate_references(root,plan):
    all_sources=sources(root)
    if len(all_sources)!=3:
        raise ValueError("Expected both historical reference sources before launching")
    cfg=all_sources[0][1]["config"]
    if cfg["groups"]!=plan["new_pairs"]:
        raise ValueError("Run configuration differs from planned new pairs")
    groups=["baseline"]+plan["candidates"]+["+".join(g) for g in plan["reused_pairs"]]
    seeds=range(cfg["seeds"]["start"],cfg["seeds"]["start"]+cfg["seeds"]["count"])
    expected={(g,s) for g in groups for s in seeds}
    seen=set()
    for path,mf in all_sources[1:]:
        for f in path.glob("*/seed_*/summary.json"):
            r=json.loads(f.read_text(encoding="utf-8"));key=(r["group"],r["seed"])
            if key in seen or r["status"]!="complete" or r["fingerprint"]!=mf["fingerprint"]:
                raise ValueError("Invalid or duplicate historical record")
            if any(not (f.parent/n).is_file() for n in ("initial_log_parameters.npy","final_population.npz","history.csv")):
                raise ValueError("Incomplete historical run files")
            seen.add(key)
    if seen!=expected:
        raise ValueError("Historical seed coverage mismatch")
    a=np.asarray(all_sources[0][1]["frequency_hz"])
    return dict(status="passed",reference_runs=len(seen),fingerprints=[m["fingerprint"] for p,m in all_sources[1:]],
        max_frequency_relative_roundoff=max(float(np.max(np.abs(a-np.asarray(m["frequency_hz"]))/a)) for p,m in all_sources[1:]))

"""Audit completed primary curves without touching live optimization directories."""
from core import *
audit_curve=load("step3_analysis_tool",HERE/"analyze.py").audit_curve
from concurrent.futures import ProcessPoolExecutor,as_completed

def main():
    manifest=json.loads((OUT/"batch/manifest.json").read_text(encoding="utf-8"))
    assert manifest["hashes"]==hashes()
    jobs=[]
    for ident in CFG["targets"]:
        for seed in CFG["seeds"]:
            baseline=OUT/f"batch/{ident}/MAG_PHA/seed_{seed}"
            if not(baseline/"summary.json").exists():
                continue
            with np.load(baseline/"generation_0300.npz") as a:
                ref=a["objectives"][int(a["primary"]),:2].copy()
            for method in CFG["methods"]:
                d=OUT/f"batch/{ident}/{method}/seed_{seed}"
                if not(d/"summary.json").exists():
                    continue
                detail=OUT/f"analysis/details/{ident}_{method}_{seed}_primary.json"
                if detail.exists():
                    cached=json.loads(detail.read_text(encoding="utf-8"))
                    if cached.get("analyzer_sha256")==hashlib.sha256((HERE/"analyze.py").read_bytes()).hexdigest():
                        continue
                with np.load(d/"generation_0300.npz") as a:
                    ix=int(a["primary"]);p=a["log_parameters"][ix].copy();f=a["objectives"][ix,:2].copy()
                g=float(Geometry(target(ident)).score(p)[0])
                eligible=bool(np.all(f<=1.10*ref+np.array(CFG["selection"]["absolute_floors"])))
                jobs.append((ident,method,seed,"primary",p,f,g,eligible))
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(audit_curve,j) for j in jobs]):
            r,e=future.result()
            print(json.dumps({k:r[k] for k in ("target","method","seed","focus_joint_ok","stable","base_score_ok","geometry_stable","geometry_training_ok","mag","pha")}),flush=True)

if __name__=="__main__":
    main()

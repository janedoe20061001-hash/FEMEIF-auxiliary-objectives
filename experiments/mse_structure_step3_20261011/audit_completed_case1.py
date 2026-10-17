"""Audit completed Case1 primary representatives once; final analysis reuses exact-input cache."""
from core import *
audit_curve=load("step3_analysis_tool",HERE/"analyze.py").audit_curve
from concurrent.futures import ProcessPoolExecutor,as_completed

def main():
    jobs=[]
    for seed in CFG["seeds"]:
        with np.load(OUT/f"batch/case1/MAG_PHA/seed_{seed}/generation_0300.npz") as a:
            baseline=a["objectives"][int(a["primary"]),:2].copy()
        for method in CFG["methods"]:
            directory=OUT/f"batch/case1/{method}/seed_{seed}"
            assert json.loads((directory/"summary.json").read_text())["status"]=="complete"
            with np.load(directory/"generation_0300.npz") as a:
                ix=int(a["primary"]);p=a["log_parameters"][ix].copy();f=a["objectives"][ix,:2].copy()
            g=float(Geometry(target("case1")).score(p)[0])
            eligible=bool(np.all(f<=1.10*baseline+np.array(CFG["selection"]["absolute_floors"])))
            jobs.append(("case1",method,seed,"primary",p,f,g,eligible))
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(audit_curve,j) for j in jobs]):
            r,e=future.result()
            print(json.dumps(r,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()

"""Explicit plan/run/smoke subcommands. Running without arguments only shows help."""
from pathlib import Path
import argparse
import copy
import csv
import hashlib
import itertools
import json
import platform
import time
from importlib.metadata import version
import numpy as np
import pymoo
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from objectives import AUX,TARGET,LOWER,UPPER,Objectives,DEFAULTS
from final_selection import select_wsmse

HERE=Path(__file__).resolve().parent


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def config(path):
    path=Path(path).resolve()
    cfg=json.loads(path.read_text(encoding="utf-8"))
    for key in ("population","generations"):
        if type(cfg[key]) is not int or cfg[key]<2:
            raise ValueError(f"{key} must be an integer >=2")
    if cfg["algorithm"]!="NSGA-II" or cfg["encoding"]!="log10":
        raise ValueError("This runner implements only NSGA-II/log10")
    if type(cfg["seeds"]["count"]) is not int or cfg["seeds"]["count"]<1 or type(cfg["seeds"]["start"]) is not int or cfg["seeds"]["start"]<0:
        raise ValueError("Invalid seeds")
    for k in ("crossover_probability","mutation_individual_probability","mutation_variable_probability"):
        if not 0<=cfg[k]<=1:
            raise ValueError(f"{k} must be in [0,1]")
    for k in ("crossover_eta","mutation_eta"):
        if cfg[k]<0:
            raise ValueError(f"{k} must be nonnegative")
    fs=cfg["features"]
    if fs != DEFAULTS:
        raise ValueError("Frozen 3D definitions have no tunable feature constants")
    if cfg.get("selection_objectives") != "all" or cfg.get("report_front") != "wsmse_selected":
        raise ValueError("All objectives guide evolution; WSMSE selection is for reporting only")
    fspec=cfg.get("final_selection",{})
    if fspec.get("method")!="min_wsmse_final_population" or fspec.get("tie_break")!="first_index":
        raise ValueError("Invalid final selection protocol")
    select_wsmse(np.zeros((1,2)),fspec["phase_weight"])
    if cfg["auxiliary_transform"] != "loss":
        raise ValueError("PDF18 protocol uses raw losses")
    fcfg=cfg["frequency"]
    if fcfg["file"]:
        fp=Path(fcfg["file"])
        if not fp.is_absolute():
            fp=path.parent/fp
        freq=np.loadtxt(fp,delimiter=",",ndmin=1)
    else:
        if type(fcfg["points"]) is not int or fcfg["points"]<3:
            raise ValueError("Invalid frequency point count")
        if not 0<fcfg["min_hz"]<fcfg["max_hz"]:
            raise ValueError("Invalid frequency range")
        freq=np.geomspace(fcfg["min_hz"],fcfg["max_hz"],fcfg["points"])
    Objectives(freq,fs,transform=cfg["auxiliary_transform"],epsilon=cfg["reciprocal_epsilon"])
    groups(cfg)
    return cfg,freq


def groups(cfg):
    if cfg["groups"]=="single_auxiliary":
        return [()] + [(a,) for a in AUX]
    if not isinstance(cfg["groups"],list):
        raise ValueError("groups must be single_auxiliary or explicit singleton lists")
    out=[]
    for g in cfg["groups"]:
        if not isinstance(g,list) or len(g)>1 or any(a not in AUX for a in g):
            raise ValueError("This protocol allows baseline or exactly one auxiliary")
        out.append(tuple(g))
    if len(out)!=len(set(out)) or not out:
        raise ValueError("Empty/repeated groups")
    return out


def group_name(aux):
    return "+".join(aux) if aux else "baseline"


def initial_population(cfg,seed):
    return np.random.default_rng(seed).uniform(LOWER,UPPER,(cfg["population"],18))


def nd_indices(f):
    return NonDominatedSorting().do(f,only_non_dominated_front=True)


def stats(f,alpha=25.12):
    full=nd_indices(f)
    primary=nd_indices(f[:,:2])
    return dict(full_front_size=len(full),primary_front_size=len(primary),
        full_front_mean_magnitude_mse=float(f[full,0].mean()),
        full_front_mean_phase_mse=float(f[full,1].mean()),
        primary_front_mean_magnitude_mse=float(f[primary,0].mean()),
        primary_front_mean_phase_mse=float(f[primary,1].mean()),
        minimum_magnitude_mse=float(f[:,0].min()),minimum_phase_mse=float(f[:,1].min()),
        **select_wsmse(f,alpha))


class Case1(Problem):
    def __init__(self,cfg,freq,aux):
        super().__init__(n_var=18,n_obj=2+len(aux),xl=LOWER,xu=UPPER)
        self.objectives=Objectives(freq,cfg["features"],aux,cfg["auxiliary_transform"],cfg["reciprocal_epsilon"])

    def _evaluate(self,x,out,*args,**kwargs):
        out["F"]=self.objectives.evaluate(x)


class Trace(Callback):
    def __init__(self,alpha=25.12):
        super().__init__();self.rows=[];self.alpha=alpha

    def notify(self,algorithm):
        self.rows.append(dict(generation=int(algorithm.n_gen),
                             evaluations=int(algorithm.evaluator.n_eval),**stats(algorithm.pop.get("F"),self.alpha)))


def manifest(cfg,freq):
    code={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
          for name in ("experiment.py","objectives.py","global_metrics.py","legacy_objectives.py","summarize.py","final_selection.py","report_results.py","launch_batch.py","requirements.txt","requirements-lock.txt")}
    record=dict(config=cfg,frequency_hz=freq.tolist(),code_sha256=code,
                target_features=Objectives(freq,cfg["features"]).target_summary(),
                versions=dict(python=platform.python_version(),numpy=np.__version__,pymoo=pymoo.__version__,
                              scipy=version("scipy"),moocore=version("moocore")))
    record["fingerprint"]=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
    return record


def run_one(cfg,freq,aux,seed,directory,fingerprint):
    directory=Path(directory)
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite {directory}")
    directory.mkdir(parents=True)
    x0=initial_population(cfg,seed)
    np.save(directory/"initial_log_parameters.npy",x0)
    problem=Case1(cfg,freq,aux)
    alpha=cfg["final_selection"]["phase_weight"]
    trace=Trace(alpha)
    alg=NSGA2(pop_size=cfg["population"],n_offsprings=cfg["population"],
        sampling=x0.copy(),crossover=SBX(prob=cfg["crossover_probability"],eta=cfg["crossover_eta"]),
        mutation=PM(prob=cfg["mutation_individual_probability"],
                    prob_var=cfg["mutation_variable_probability"],eta=cfg["mutation_eta"]),
        eliminate_duplicates=False)
    alg.tournament_type="comp_by_rank_and_crowding"
    t=time.perf_counter()
    result=minimize(problem,alg,("n_gen",cfg["generations"]),seed=seed,
                    callback=trace,verbose=False,save_history=False)
    x,f=result.pop.get("X"),result.pop.get("F")
    selected=select_wsmse(f,alpha)
    selected_index=selected["wsmse_selected_index"]
    np.savez_compressed(directory/"final_population.npz",log_parameters=x,
        parameters_R_L_C=10**x.reshape(-1,6,3),objectives=f,
        full_front_indices=nd_indices(f),primary_front_indices=nd_indices(f[:,:2]),
        selected_index=selected_index,selected_parameters_R_L_C=10**x[selected_index].reshape(6,3),
        selected_primary_mse=f[selected_index,:2],selection_alpha=alpha)
    with (directory/"history.csv").open("w",encoding="utf-8-sig",newline="") as fh:
        writer=csv.DictWriter(fh,fieldnames=list(trace.rows[0]))
        writer.writeheader();writer.writerows(trace.rows)
    record=dict(status="complete",fingerprint=fingerprint,seed=seed,group=group_name(aux),
        objective_names=["magnitude_mse_db2","phase_mse_rad2"]+list(aux),
        initial_population_sha256=hashlib.sha256(x0.tobytes()).hexdigest(),
        evaluations=int(result.algorithm.evaluator.n_eval),generations=len(trace.rows),
        elapsed_seconds=time.perf_counter()-t,**stats(f,alpha))
    assert record["evaluations"]==cfg["population"]*cfg["generations"]
    assert record["generations"]==cfg["generations"]
    dump(directory/"summary.json",record) # Completion marker written last.
    return record


def plan(cfg):
    gs=groups(cfg)
    return dict(status="plan_only",seed_start=cfg["seeds"]["start"],seed_count=cfg["seeds"]["count"],
        population=cfg["population"],generations=cfg["generations"],group_count=len(gs),
        groups=[dict(name=group_name(g),auxiliaries=list(g),objective_count=2+len(g)) for g in gs],
        planned_runs=len(gs)*cfg["seeds"]["count"],
        planned_evaluations=len(gs)*cfg["seeds"]["count"]*cfg["population"]*cfg["generations"])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command",choices=["plan","run","smoke"])
    ap.add_argument("--config",type=Path,default=HERE/"config.json")
    ap.add_argument("--output",type=Path)
    ap.add_argument("--group",help="Run one configured group, e.g. baseline or SS+PTC")
    ap.add_argument("--seed",type=int,help="Run one configured seed")
    ap.add_argument("--resume",action="store_true",help="Skip verified complete runs only")
    args=ap.parse_args()
    cfg,freq=config(args.config)
    if args.command=="plan":
        info=plan(cfg)
        if args.output:
            args.output.mkdir(parents=True,exist_ok=True)
            dump(args.output/"plan.json",info)
            dump(args.output/"manifest.json",manifest(cfg,freq))
        print(json.dumps(info,indent=2));return
    if args.command=="smoke":
        cfg=copy.deepcopy(cfg)
        cfg.update(population=12,generations=3,seeds=dict(start=0,count=1))
        default=HERE/"smoke"
    else:
        default=HERE/"runs"
    out=(args.output or default).resolve()
    gs=groups(cfg)
    if args.group:
        gs=[g for g in gs if group_name(g)==args.group]
        if not gs:
            ap.error("--group does not match a configured group")
    seeds=list(range(cfg["seeds"]["start"],cfg["seeds"]["start"]+cfg["seeds"]["count"]))
    if args.seed is not None:
        if args.seed not in seeds:
            ap.error("--seed not in configured seed set")
        seeds=[args.seed]
    mf=manifest(cfg,freq)
    out.mkdir(parents=True,exist_ok=True)
    existing=out/"manifest.json"
    if existing.exists():
        if json.loads(existing.read_text(encoding="utf-8"))["fingerprint"]!=mf["fingerprint"]:
            raise ValueError("Configuration, code, frequency, or runtime differs from existing output")
    else:
        dump(existing,mf)
    for seed in seeds:
        for aux in gs:
            rd=out/group_name(aux)/f"seed_{seed:03d}"
            if args.resume and (rd/"summary.json").exists():
                old=json.loads((rd/"summary.json").read_text(encoding="utf-8"))
                if (old.get("status")!="complete" or old.get("fingerprint")!=mf["fingerprint"]
                    or old.get("seed")!=seed or old.get("group")!=group_name(aux)
                    or not (rd/"final_population.npz").exists() or not (rd/"history.csv").exists()):
                    raise ValueError(f"Invalid completion record: {rd}")
                print(f"skip {seed} {group_name(aux)}",flush=True);continue
            print(f"{args.command}: seed={seed} group={group_name(aux)}",flush=True)
            run_one(cfg,freq,aux,seed,rd,mf["fingerprint"])
    print("Complete. Smoke results are software checks, not scientific experiment results.")


if __name__=="__main__":
    main()

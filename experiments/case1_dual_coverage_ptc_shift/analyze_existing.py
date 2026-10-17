"""Read audited historical populations; never runs an optimizer or changes metrics."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import numpy as np
from final_selection import select_wsmse
from constituents import analyze_constituents

HERE=Path(__file__).resolve().parent


def load_archive(archive,config):
    expected_digest=archive.with_name("raw_results.sha256").read_text().split()[0]
    digest=hashlib.file_digest(archive.open("rb"),"sha256").hexdigest()
    if digest!=expected_digest:raise ValueError("Archive checksum mismatch")
    groups={"+".join(g) if g else "baseline" for g in config["groups"]}
    wanted={f"seed_{s:03d}" for s in range(config["seeds"]["start"],config["seeds"]["start"]+config["seeds"]["count"])}
    manifests={};records={};populations={};initials={}
    with tarfile.open(archive,"r|gz") as tar:
        for m in tar:
            if not m.isfile():continue
            parts=Path(m.name).parts
            if parts[-1]=="manifest.json":
                manifests["/".join(parts[:-1])]=json.load(tar.extractfile(m));continue
            if len(parts)<4 or parts[-3] not in groups or parts[-2] not in wanted:continue
            name="/".join(parts[:-1])
            if parts[-1]=="summary.json":
                records[name]=json.load(tar.extractfile(m))
            elif parts[-1]=="initial_log_parameters.npy":
                x=np.load(io.BytesIO(tar.extractfile(m).read()),allow_pickle=False)
                initials[name]=hashlib.sha256(x.tobytes()).hexdigest()
            elif parts[-1]=="final_population.npz":
                with np.load(io.BytesIO(tar.extractfile(m).read()),allow_pickle=False) as z:
                    check=select_wsmse(z["objectives"],config["final_selection"]["phase_weight"])
                    idx=check["wsmse_selected_index"]
                    if idx!=int(z["selected_index"]):raise ValueError("Stored selected index differs")
                    np.testing.assert_array_equal(z["selected_primary_mse"],z["objectives"][idx,:2])
                    populations[name]=check
    if not records or records.keys()!=populations.keys() or records.keys()!=initials.keys():
        raise ValueError("Incomplete archived records/populations/initials")
    used=set();hashes={}
    for path,r in records.items():
        source="/".join(Path(path).parts[:-2]);mf=manifests[source];used.add(source)
        if r["status"]!="complete" or r["fingerprint"]!=mf["fingerprint"]:raise ValueError("Provenance mismatch")
        if r["evaluations"]!=config["population"]*config["generations"]:raise ValueError("Budget mismatch")
        if initials[path]!=r["initial_population_sha256"]:raise ValueError("Initial population hash mismatch")
        hashes.setdefault(r["seed"],set()).add(initials[path])
        if r["group"]!=Path(path).parts[-2] or f"seed_{r['seed']:03d}"!=Path(path).parts[-1]:
            raise ValueError("Record path mismatch")
        for k,v in populations[path].items():
            if not np.isclose(r[k],v,rtol=1e-12,atol=1e-14):raise ValueError("Selected metric mismatch")
    if any(len(h)!=1 for h in hashes.values()):raise ValueError("Different paired initial populations")
    for source in used:
        mf=manifests[source]
        if {k:v for k,v in mf["config"].items() if k!="groups"}!={k:v for k,v in config.items() if k!="groups"}:
            raise ValueError("Reference configuration mismatch")
        for n in ("objectives.py","final_selection.py","requirements-lock.txt"):
            if mf["code_sha256"][n]!=hashlib.sha256((HERE/n).read_bytes()).hexdigest():
                raise ValueError("Objective or selection definition mismatch")
    return list(records.values()),dict(archive=str(archive.resolve()),sha256=digest,
        audited_populations=len(populations),paired_initials_verified=True,source_fingerprints={s:manifests[s]["fingerprint"] for s in sorted(used)})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads((HERE/"config.json").read_text())
    cfg["seeds"]=dict(start=0,count=100)
    cfg["groups"]=[g for g in cfg["groups"] if len(g)<3]
    records,source=load_archive(args.archive,cfg)
    result=analyze_constituents(records,cfg,json.loads((HERE/"analysis_config.json").read_text()),args.output,"discovery / 旧seed探索")
    (args.output/"source_audit.json").write_text(json.dumps(source,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result["findings"],indent=2))


if __name__=="__main__":main()

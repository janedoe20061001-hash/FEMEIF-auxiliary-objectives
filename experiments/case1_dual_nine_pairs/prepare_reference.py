"""Extract only eligible completed references; this never runs an optimizer."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--singles",type=Path,required=True)
    parser.add_argument("--combinations",type=Path,required=True)
    args=parser.parse_args()
    plan=json.loads((HERE/"pair_plan.json").read_text())
    dest=HERE/"reference"
    if dest.exists(): raise FileExistsError("Reference already exists; inspect it before rebuilding")
    for archive in (args.singles,args.combinations):
        expected=archive.with_name("raw_results.sha256").read_text().split()[0]
        if hashlib.file_digest(archive.open("rb"),"sha256").hexdigest()!=expected:
            raise ValueError("Source archive checksum mismatch")
    provenance={}
    for tag,archive,groups in (
        ("singles",args.singles,["baseline"]+plan["candidates"]),
        ("pairs",args.combinations,["+".join(g) for g in plan["reused_pairs"]])):
        records=[]
        with tarfile.open(archive,"r|gz") as tar:
            for member in tar:
                parts=Path(member.name).parts
                if not member.isfile() or parts[0]!="runs": continue
                if parts==("runs","manifest.json") or (len(parts)==4 and parts[1] in groups and parts[3] in ("summary.json","final_population.npz","initial_log_parameters.npy","history.csv")):
                    target=dest/tag/Path(*parts[1:]);target.parent.mkdir(parents=True,exist_ok=True)
                    content=tar.extractfile(member).read();target.write_bytes(content)
                    if parts[-1]=="summary.json": records.append(json.loads(content))
        mf=json.loads((dest/tag/"manifest.json").read_text())
        cfg=mf["config"];seeds=range(cfg["seeds"]["start"],cfg["seeds"]["start"]+cfg["seeds"]["count"])
        assert {(r["group"],r["seed"]) for r in records}=={(g,s) for g in groups for s in seeds}
        assert len(records)==len(groups)*len(seeds)
        assert all(r["status"]=="complete" and r["fingerprint"]==mf["fingerprint"] for r in records)
        provenance[tag]=dict(archive=str(archive.resolve()),sha256=archive.with_name("raw_results.sha256").read_text().split()[0],records=len(records),fingerprint=mf["fingerprint"])
    (dest/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    with tarfile.open(HERE/"reference.tar.gz","w:gz",compresslevel=1) as tar:tar.add(dest,arcname="reference")
    digest=hashlib.file_digest((HERE/"reference.tar.gz").open("rb"),"sha256").hexdigest()
    (HERE/"reference.sha256").write_text(digest+"  reference.tar.gz\n")
    print(json.dumps(provenance,indent=2))


if __name__=="__main__":
    main()

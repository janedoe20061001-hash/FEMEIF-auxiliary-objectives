"""Check delivered conclusions and artifact references without rerunning optimization."""
from core import *
import csv, re, datetime

def main():
    status=json.loads((OUT/"batch/status.json").read_text())
    manifest=json.loads((OUT/"batch/manifest.json").read_text())
    integrity=json.loads((OUT/"INTEGRITY.json").read_text())
    audit=json.loads((OUT/"analysis/AUDIT.json").read_text())
    assert status["status"]=="complete" and status["completed"]==32 and status["failed"]==0
    assert manifest["hashes"]==hashes()
    assert integrity["checked_jobs"]==32 and integrity["evaluations"]==1920000
    with (OUT/"analysis/REPRESENTATIVES.csv").open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==58 and audit["curves"]==58
    assert audit["structure_and_base_score_failures"]==0 and audit["geometry_sampling_failures"]==1
    counts={}
    expected={"case1":[0,0],"ladder8":[0,3],"mixed0p25":[2,1],"control0p25":[4,4]}
    for ident in CFG["targets"]:
        counts[ident]=[]
        for method in CFG["methods"]:
            rr=[r for r in rows if r["target"]==ident and r["method"]==method and r["rule"]=="primary"]
            assert len(rr)==4
            counts[ident].append(sum(r["focus_joint_ok"]=="True" for r in rr))
        assert counts[ident]==expected[ident]
    for seed in CFG["seeds"]:
        a,b=[next(r for r in rows if r["target"]=="case1" and r["seed"]==str(seed) and r["method"]==m and r["rule"]=="primary") for m in CFG["methods"]]
        assert float(b["mag"])>float(a["mag"]) and float(b["pha"])>float(a["pha"])
        assert float(b["geometry_reference"])<float(a["geometry_reference"])
    notes=[OUT/n for n in ("REPORT.md","FINDINGS.md","ISSUES.md","TARGETS.md","PREREGISTRATION.md")]
    notes+=[ROOT/"docs/三维阻抗曲线_第三步更新设计_20261011.md"]
    refs=[]
    for path in notes:
        for url in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding="utf-8")):
            if re.match(r"^[A-Za-z]:/",url):
                assert Path(url).exists(), (path,url)
                refs.append(url)
    figs=sorted((OUT/"figures").glob("*.png"))
    assert len(figs)==7 and all(p.stat().st_size>10000 for p in figs)
    dump(OUT/"DELIVERY_VERIFICATION.json",dict(
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        checked="optimizer source/input provenance; report primary counts and Case1 tradeoffs; local links and plot files",
        formal_tasks=32,evaluations=1920000,primary_success_counts=counts,
        representative_records=58,base_and_structure_failures=0,geometry_sampling_failures=1,
        numerical_audit_passed=False,report_references_checked=len(refs),
        helper_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob("*.py")},
        delivered_file_hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [*notes,*figs]}))
    print(json.dumps(dict(tasks=32,records=58,counts=counts,geometry_failures=1,local_links=len(refs),figures=7),ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()

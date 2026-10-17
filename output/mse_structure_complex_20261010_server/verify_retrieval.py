"""Read-only integrity checks for recovered results, independent of watcher state."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
CODE = ROOT.parents[1]/"experiments/mse_structure_complex_20261010"
sys.path.insert(0,str(CODE))
from protocol import source_hashes, verify_job, initial, targets

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def verify():
    checked, groups, failed_candidates = [], [], []
    seen = set()
    for group in ("first","other"):
        base = ROOT/"retrieved"/group
        if not base.exists():
            continue
        manifest = read(base/f"batch_{group}/manifest.json")
        cfg = manifest["config"]
        assert source_hashes(cfg)==manifest["hashes"], "Local frozen source/config differs"
        prepared = base/"inputs_core"
        assert read(prepared/"manifest.json")==manifest["prepared_manifest"], "Prepared manifest differs"
        for name,digest in manifest["prepared_manifest"]["target_hashes"].items():
            assert hashlib.sha256((prepared/name).read_bytes()).hexdigest()==digest, f"Target changed: {name}"
        ids = sorted({t["circuit"]["id"] for t in manifest["plan"]["targets"]})
        planned = targets(cfg,manifest["plan"]["profile"],ids)
        assert len(planned)*len(cfg["seeds"])==manifest["plan"]["jobs"]
        audit_path = base/f"analysis_{group}/AUDIT.json"
        audit = read(audit_path)
        assert audit["completed_jobs"]==audit["planned_jobs"]==manifest["plan"]["jobs"]
        proof_path = base/"PARALLEL_ANALYSIS.json"
        if proof_path.exists():
            proof = read(proof_path)
            assert proof["original_hashes"]==manifest["hashes"] and proof["crosschecks_exact"]
            assert proof["helper_sha256"]==hashlib.sha256((base/"parallel_analysis.py").read_bytes()).hexdigest()
            serial = base/"analysis_other_serial_partial/details.jsonl"
            data = serial.read_bytes()
            prefix = b"\n".join(data.split(b"\n")[:proof["serial_prefix_records"]])+b"\n"
            assert hashlib.sha256(prefix).hexdigest()==proof["serial_prefix_sha256"]
        refinement_path = base/f"AUDIT_REFINEMENT_{group}.json"
        if audit["passed"] and refinement_path.exists():
            refinement = read(refinement_path)
            assert refinement["passed"] and refinement["final_audit_passed"]
            assert refinement["tolerances_unchanged"] and refinement["optimization_unchanged"] and refinement["refinements"]==[32,64]
            assert refinement["source_hashes"]==manifest["hashes"]
            assert refinement["helper_sha256"]==hashlib.sha256((base/"refine_parallel_analysis.py").read_bytes()).hexdigest()
            old_path = base/refinement["original_analysis"]/"AUDIT.json"
            old_audit = read(old_path)
            assert refinement["original_audit_sha256"]==hashlib.sha256(old_path.read_bytes()).hexdigest()
            assert [r["identity"] for r in refinement["records"]]==old_audit["failures"]
            assert all(r["accepted"] and r["stable_refined"] and r["objective_ok"] and r["refinements_used"] in ([32,64],[128,256]) for r in refinement["records"])
            affected = {(r["identity"]["target"],r["identity"]["seed"],r["identity"]["generation"],r["identity"]["index"]):r for r in refinement["records"]}
            found = set()
            with (base/f"analysis_{group}/details.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    r = json.loads(line)
                    k = (r["target"],r["seed"],r["generation"],r["index"])
                    if k in affected:
                        assert r["training_objective"]==affected[k]["original_record"]["training_objective"]
                        assert r["audit_passed"] and r["stable"]
                        assert all(r[n]==v for n,v in affected[k]["refined_record"].items())
                        found.add(k)
            assert found==set(affected)
        excluded = audit["failures"]
        if not audit["passed"]:
            supplement = read(base/f"AUDIT_REFINEMENT_{group}.json")
            assert supplement["passed"] and supplement["tolerances_unchanged"]
            assert supplement["original_audit_sha256"]==hashlib.sha256(audit_path.read_bytes()).hexdigest()
            assert supplement["original_hashes"]==manifest["hashes"]
            assert supplement["helper_sha256"]==hashlib.sha256((base/"inspect_audit.py").read_bytes()).hexdigest()
            assert len(supplement["records"])==len(excluded)
            assert all(r["accepted"] and r["high_converged"] and r["high_objective_ok"] and not r["representative"] for r in supplement["records"])
            assert [r["identity"] for r in supplement["records"]]==excluded
        originals = {}
        for t in planned:
            for seed in cfg["seeds"]:
                key = (t["id"],seed)
                assert key not in seen, "Duplicate job across archives"
                seen.add(key)
                d = base/f"batch_{group}"/t["id"]/f"seed_{seed}"
                summary = verify_job(d,cfg)
                x0 = np.load(d/"initial_log_parameters.npy")
                np.testing.assert_array_equal(x0,initial(t,cfg,seed))
                assert hashlib.sha256(x0.tobytes()).hexdigest()==summary["initial_sha256"]
                paired_key = (t["base"].ident,seed)
                if paired_key in originals:
                    np.testing.assert_array_equal(x0,originals[paired_key])
                else:
                    originals[paired_key]=x0
                generations = []
                for file in sorted(d.glob("generation_*.npz")):
                    with np.load(file) as a:
                        X,F = a["log_parameters"],a["optimization_objective"]
                        assert X.shape==(cfg["population"],len(t["circuit"].names)) and F.shape==(cfg["population"],2)
                        assert np.isfinite(X).all() and np.isfinite(F).all() and (F>=0).all()
                        assert (X>=t["lower"]-1e-12).all() and (X<=t["upper"]+1e-12).all()
                        assert int(a["selected_index"]) in a["front_indices"] and int(a["balanced_index"]) in a["front_indices"]
                    generations.append(int(file.stem.split("_")[-1]))
                assert generations==sorted(set(range(cfg["save_every"],cfg["generations"]+1,cfg["save_every"]))|{cfg["generations"]})
                assert summary["evaluations"]==60000
                checked.append(dict(target=t["id"],seed=seed,evaluations=summary["evaluations"],saved_stages=len(generations)))
        with (base/f"analysis_{group}/representatives.csv").open(encoding="utf-8-sig",newline="") as stream:
            representatives = list(csv.DictReader(stream))
        assert all(r["stable"]=="True" and r["audit_passed"]=="True" for r in representatives), "Representative failed independent check"
        assert len(representatives)==manifest["plan"]["jobs"]*len(generations)*2
        failed_candidates.extend(dict(group=group,**f) for f in excluded)
        groups.append(group)
    result = dict(passed_for_retrieved_groups=True,fully_complete=len(checked)==168,checked_jobs=len(checked),expected_jobs=168,
                  verified_groups=groups,total_evaluations=sum(r["evaluations"] for r in checked),saved_stages_per_job=15,
                  checks=["original_source_config_hashes","target_hashes","prepared_manifest","saved_checkpoint_hashes","seed_initialization","paired_initialization","parameter_bounds","finite_scores","representative_front_membership","complete_stage_counts","per_seed_budget","independent_representative_audits","supplement_provenance"],
                  excluded_original_candidates=failed_candidates,jobs=checked)
    (ROOT/"INTEGRITY.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return {k:v for k,v in result.items() if k not in ("jobs","checks","excluded_original_candidates")}

if __name__=="__main__":
    print(json.dumps(verify(),ensure_ascii=False))

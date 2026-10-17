import ast
import copy
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import experiment as e
from stability import paired_stats,analyze
from constituents import analyze_constituents


class ValidationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg,cls.freq=e.config(e.HERE/"config.json")
        cls.settings=json.loads((e.HERE/"analysis_config.json").read_text())

    def test_frozen_plan(self):
        p=e.plan(self.cfg)
        self.assertEqual((p["planned_runs"],p["planned_evaluations"]),(800,48000000))
        self.assertEqual((p["seed_start"],p["seed_count"]),(200,100))
        self.assertEqual(sorted(map(len,e.groups(self.cfg))),[0,1,1,1,2,2,2,3])
        self.assertEqual({a for g in e.groups(self.cfg) for a in g},
                         {"turning_event_coverage_5_of_6","paper_ptc","characteristic_frequency_shift_error_decades"})
        source=json.loads((e.HERE/"SOURCE.json").read_text(encoding="utf-8"))
        old=source["config"]
        self.assertEqual({k:v for k,v in old.items() if k not in ("seeds","groups")},
                         {k:v for k,v in self.cfg.items() if k not in ("seeds","groups")})
        def functions(path):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        b=functions(e.HERE/"experiment.py")
        for name in ("run_one","Case1","initial_population","Trace","stats"):
            self.assertEqual(source["search_function_ast_sha256"][name],hashlib.sha256(b[name].encode()).hexdigest())
        for name,digest in source["files_sha256"].items():
            self.assertEqual(hashlib.sha256((e.HERE/name).read_bytes()).hexdigest(),digest)

    def test_bootstrap_preserves_pairing_and_metrics(self):
        b=np.column_stack((np.arange(1,101)**2,np.arange(1,101)/100))
        a=b+np.array([-1,-.005])
        rows=paired_stats(a,b,self.settings)
        for r,delta in zip(rows,[-1,-.005]):
            for k in ("mean_difference","ci95_low","ci95_high","family_ci_low","family_ci_high"):
                self.assertAlmostEqual(r[k],delta,places=12)
            self.assertEqual(r["both_lower_seeds"],100)

    def test_tail_sensitivity_and_ci_width(self):
        b=np.ones((100,2))*1000
        d=np.ones((100,2));d[0]=-200
        for r in paired_stats(b+d,b,self.settings):
            self.assertLess(r["mean_difference"],0)
            self.assertEqual(r["remove_best_mean_difference"],1)
            self.assertGreater(r["leave_one_out_max"],0)
            self.assertGreaterEqual(r["family_ci_high"],r["ci95_high"])
            self.assertLessEqual(r["family_ci_low"],r["ci95_low"])
        self.assertEqual(paired_stats(b+d,b,self.settings),paired_stats(b+d,b,self.settings))

    def test_missing_duplicate_seed_rejection_and_order(self):
        cfg=copy.deepcopy(self.cfg);cfg["seeds"]=dict(start=200,count=3)
        records=[]
        for g in e.groups(cfg):
            for s in range(200,203):
                records.append(dict(group=e.group_name(g),seed=s,wsmse_selected_alpha=25.12,
                    wsmse_selected_magnitude_mse=1+s+(1 if not g else 0),
                    wsmse_selected_phase_mse=.1+(1 if not g else 0)))
        with tempfile.TemporaryDirectory() as td:
            first=analyze(records,cfg,self.settings,td,"test")
            second=analyze(records[::-1],cfg,self.settings,td,"test")
            self.assertEqual(first,second)
            for bad in (records[:-1],records+[records[0]]):
                with self.assertRaises(ValueError):
                    analyze(bad,cfg,self.settings,td,"test")

    def test_small_end_to_end_and_no_formal_runs(self):
        cfg=copy.deepcopy(self.cfg)
        cfg.update(population=8,generations=3,seeds=dict(start=200,count=3))
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/"runs";root.mkdir();mf=e.manifest(cfg,self.freq)
            e.dump(root/"manifest.json",mf)
            for g in e.groups(cfg):
                for seed in range(200,203):
                    e.run_one(cfg,self.freq,g,seed,root/e.group_name(g)/f"seed_{seed:03d}",mf["fingerprint"])
            result=subprocess.run([sys.executable,str(e.HERE/"report_results.py"),str(root),"--no-archive"],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            audit=json.loads((root/"completion_audit.json").read_text())
            self.assertEqual(audit["runs"],24)
            st=json.loads((root/"stability.json").read_text(encoding="utf-8"))
            self.assertIn("smoke",st["stage"])
            self.assertEqual(st["records"],24)
            self.assertTrue((root/"STABILITY.md").exists())
            comp=json.loads((root/"constituent_stability.json").read_text(encoding="utf-8"))
            self.assertEqual((comp["comparisons"],comp["component_intervals"]),(12,24))
        self.assertFalse((e.HERE/"runs").exists())

    def test_existing_pairs_analysis_family_and_missing_seed(self):
        cfg=copy.deepcopy(self.cfg);cfg["groups"]=[g for g in cfg["groups"] if len(g)<3]
        cfg["seeds"]=dict(start=0,count=3)
        records=[dict(group=e.group_name(g),seed=s,wsmse_selected_alpha=25.12,
                 wsmse_selected_magnitude_mse=10-len(g)+s,
                 wsmse_selected_phase_mse=1-len(g)/10+s/100)
                 for g in cfg["groups"] for s in range(3)]
        with tempfile.TemporaryDirectory() as td:
            r=analyze_constituents(records,cfg,self.settings,td,"test")
            self.assertEqual((r["comparisons"],r["component_intervals"]),(6,12))
            self.assertTrue(all(x["both_adjusted_ci_below_zero"] for x in r["findings"]))
            with self.assertRaises(ValueError):
                analyze_constituents(records[:-1],cfg,self.settings,td,"test")


if __name__=="__main__":
    unittest.main()

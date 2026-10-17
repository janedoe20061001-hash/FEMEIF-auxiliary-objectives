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


class ValidationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg,cls.freq=e.config(e.HERE/"config.json")
        cls.settings=json.loads((e.HERE/"analysis_config.json").read_text())

    def test_frozen_plan(self):
        p=e.plan(self.cfg)
        self.assertEqual((p["planned_runs"],p["planned_evaluations"]),(800,48000000))
        self.assertEqual((p["seed_start"],p["seed_count"]),(100,100))
        self.assertEqual(sorted(map(len,e.groups(self.cfg))),[0,1,1,1,2,2,2,3])
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
        cfg=copy.deepcopy(self.cfg);cfg["seeds"]=dict(start=100,count=3)
        records=[]
        for g in e.groups(cfg):
            for s in range(100,103):
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
        cfg.update(population=8,generations=3,seeds=dict(start=100,count=3))
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/"runs";root.mkdir();mf=e.manifest(cfg,self.freq)
            e.dump(root/"manifest.json",mf)
            for g in e.groups(cfg):
                for seed in range(100,103):
                    e.run_one(cfg,self.freq,g,seed,root/e.group_name(g)/f"seed_{seed:03d}",mf["fingerprint"])
            result=subprocess.run([sys.executable,str(e.HERE/"report_results.py"),str(root),"--no-archive"],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            audit=json.loads((root/"completion_audit.json").read_text())
            self.assertEqual(audit["runs"],24)
            st=json.loads((root/"stability.json").read_text(encoding="utf-8"))
            self.assertIn("smoke",st["stage"])
            self.assertEqual(st["records"],24)
            self.assertTrue((root/"STABILITY.md").exists())
        self.assertFalse((e.HERE/"runs").exists())


if __name__=="__main__":
    unittest.main()

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import experiment as e
from objectives import Objectives


class CombinationChecks(unittest.TestCase):
    def test_plan_and_duplicate_groups(self):
        cfg,f=e.config(e.HERE/"config.json")
        self.assertEqual(e.plan(cfg)["planned_runs"],400)
        self.assertEqual(sorted(map(len,e.groups(cfg))),[2,2,2,3])
        for gs in ([[["bad"]]], [[cfg["groups"][0][0]]*2], [cfg["groups"][0],cfg["groups"][0][::-1]]):
            with self.assertRaises((ValueError,TypeError)):
                e.groups(dict(cfg,groups=gs))

    def test_combo_columns_equal_single_evaluations(self):
        cfg,f=e.config(e.HERE/"config.json")
        x=e.initial_population(cfg,7)[:20]
        for g in e.groups(cfg):
            combined=Objectives(f,cfg["features"],g).evaluate(x)
            for j,a in enumerate(g):
                single=Objectives(f,cfg["features"],(a,)).evaluate(x)
                np.testing.assert_array_equal(combined[:,:2],single[:,:2])
                np.testing.assert_array_equal(combined[:,2+j],single[:,2])

    def test_unchanged_search_engine(self):
        import ast
        old=e.HERE.parent/"case1_dual_pdf18"/"experiment.py"
        if not old.exists():
            self.skipTest("Original source checked locally")
        def functions(path):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        a,b=functions(old),functions(e.HERE/"experiment.py")
        for name in ("run_one","Case1","initial_population","Trace","stats"):
            self.assertEqual(a[name],b[name])

    def test_smoke_and_reference_report(self):
        cfg,f=e.config(e.HERE/"config.json")
        cfg=copy.deepcopy(cfg);cfg.update(population=8,generations=3,seeds=dict(start=0,count=1))
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);ref=p/"reference";root=p/"runs";ref.mkdir();root.mkdir()
            singles=list(dict.fromkeys(a for g in cfg["groups"] for a in g))
            rc=dict(cfg,groups=[[]]+[[a] for a in singles])
            for dest,c in ((ref,rc),(root,cfg)):
                mf=e.manifest(c,f);e.dump(dest/"manifest.json",mf)
                for g in e.groups(c):
                    e.run_one(c,f,g,0,dest/e.group_name(g)/"seed_000",mf["fingerprint"])
            subprocess.run([sys.executable,str(e.HERE/"report_results.py"),str(root),"--no-archive"],check=True,capture_output=True)
            audit=json.loads((root/"completion_audit.json").read_text())
            self.assertEqual((audit["new_runs"],audit["reused_runs"]),(4,4))
            self.assertTrue((root/"constituent_comparisons.csv").exists())


if __name__=="__main__":
    unittest.main()

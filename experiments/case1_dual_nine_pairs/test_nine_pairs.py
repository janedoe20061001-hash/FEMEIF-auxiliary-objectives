import ast
import copy
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import experiment as e
from sources import sources,validate_references
from objectives import Objectives


class PairChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg,cls.freq=e.config(e.HERE/"config.json")
        cls.plan=json.loads((e.HERE/"pair_plan.json").read_text())

    def test_complete_pairs_no_repeat_and_unchanged_search(self):
        allpairs=set(map(frozenset,self.plan["all_pairs"]))
        new=set(map(frozenset,e.groups(self.cfg)))
        reused=set(map(frozenset,self.plan["reused_pairs"]))
        self.assertEqual(allpairs,set(map(frozenset,itertools.combinations(self.plan["candidates"],2))))
        self.assertEqual((len(allpairs),len(new),len(reused)),(36,33,3))
        self.assertFalse(new & reused);self.assertEqual(new|reused,allpairs)
        self.assertEqual(e.plan(self.cfg)["planned_runs"],3300)
        self.assertEqual(e.plan(self.cfg)["planned_evaluations"],198000000)
        source=json.loads((e.HERE/"SOURCE.json").read_text())
        self.assertEqual({k:v for k,v in source["config"].items() if k!="groups"},
                         {k:v for k,v in self.cfg.items() if k!="groups"})
        for node in ast.parse((e.HERE/"experiment.py").read_text()).body:
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in source["search_ast"]:
                digest=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
                self.assertEqual(digest,source["search_ast"][node.name])

    def test_all_pair_objectives_match_single_columns(self):
        x=e.initial_population(self.cfg,17)[:12]
        single={a:Objectives(self.freq,self.cfg["features"],(a,)).evaluate(x) for a in self.plan["candidates"]}
        for g in self.plan["all_pairs"]:
            pair=Objectives(self.freq,self.cfg["features"],g).evaluate(x)
            for j,a in enumerate(g):
                np.testing.assert_array_equal(pair[:,:2],single[a][:,:2])
                np.testing.assert_array_equal(pair[:,j+2],single[a][:,2])

    def test_small_complete_report_and_source_mismatch(self):
        cfg=copy.deepcopy(self.cfg);cfg.update(population=8,generations=3,seeds=dict(start=0,count=1))
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/"runs"
            splits=((root,cfg["groups"]),
                    (Path(td)/"reference/singles",[[]]+[[a] for a in self.plan["candidates"]]),
                    (Path(td)/"reference/pairs",self.plan["reused_pairs"]))
            for dest,groups in splits:
                dest.mkdir(parents=True);c=dict(cfg,groups=groups);mf=e.manifest(c,self.freq);e.dump(dest/"manifest.json",mf)
                for g in e.groups(c):
                    e.run_one(c,self.freq,g,0,dest/e.group_name(g)/"seed_000",mf["fingerprint"])
            result=subprocess.run([sys.executable,str(e.HERE/"report_results.py"),str(root),"--no-archive"],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            audit=json.loads((root/"completion_audit.json").read_text())
            self.assertEqual((audit["runs"],audit["new_runs"],audit["reused_runs"]),(46,33,13))
            self.assertTrue((root/"constituent_comparisons.csv").exists())
            self.assertEqual(validate_references(root,self.plan)["reference_runs"],13)
            ref=Path(td)/"reference/pairs/manifest.json"
            bad=json.loads(ref.read_text());bad["config"]["seeds"]["start"]=100;e.dump(ref,bad)
            with self.assertRaises(ValueError): sources(root)
        self.assertFalse((e.HERE/"runs").exists())


if __name__=="__main__":
    unittest.main()

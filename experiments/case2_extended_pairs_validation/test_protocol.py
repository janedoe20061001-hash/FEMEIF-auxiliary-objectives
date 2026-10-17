import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import experiment as e
from analysis import paired


class ProtocolChecks(unittest.TestCase):
    def test_pair_screen_shape(self):
        cfg,f=e.config(e.HERE/"config_pair_screen.json")
        groups=e.groups(cfg)
        self.assertEqual(len(groups),46)
        self.assertEqual(sum(len(g)==2 for g in groups),36)
        self.assertEqual(e.plan(cfg)["planned_runs"],4600)

    def test_validation_shape_and_new_seeds(self):
        cfg,f=e.config(e.HERE/"config_validation.json")
        groups=e.groups(cfg)
        self.assertEqual(cfg["seeds"],{"start":100,"count":100})
        self.assertEqual(len(groups),14)
        self.assertEqual(sum(len(g)==2 for g in groups),6)
        self.assertEqual(sum(len(g)==3 for g in groups),2)

    def test_paired_bootstrap_direction(self):
        baseline=np.ones((20,2))
        result=paired(baseline-.1,baseline,resamples=1000,components=4)
        self.assertTrue(all(r["adjusted_ci_high"]<0 for r in result))

    def test_smoke_run(self):
        cfg,f=e.config(e.HERE/"config_validation.json")
        cfg=copy.deepcopy(cfg);cfg.update(population=8,generations=3,seeds={"start":100,"count":1})
        group=e.groups(cfg)[-1]
        mf=e.manifest(cfg,f)
        with tempfile.TemporaryDirectory(prefix="case2_extended_") as td:
            r=e.run_one(cfg,f,group,100,Path(td)/"run",mf["fingerprint"])
            self.assertEqual(r["evaluations"],24)
            self.assertEqual(r["group"],e.group_name(group))


if __name__=="__main__":unittest.main()

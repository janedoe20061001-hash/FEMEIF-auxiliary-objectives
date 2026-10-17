import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np

import experiment as e
from objectives import Objectives
from case2 import load_target, impedance
from analysis import paired


class ChampionProtocolChecks(unittest.TestCase):
    def test_formal_plan(self):
        cfg, _ = e.config(e.HERE / "config.json")
        plan = e.plan(cfg)
        self.assertEqual(cfg["seeds"], {"start": 200, "count": 200})
        self.assertEqual(plan["group_count"], 11)
        self.assertEqual(plan["planned_runs"], 2200)
        self.assertEqual(plan["planned_evaluations"], 99_000_000)

    def test_unified_order_and_primary_identity(self):
        cfg, frequency = e.config(e.HERE / "config.json")
        _, target = load_target(cfg["target_data"])
        group = e.groups(cfg)[-1]
        objective = Objectives(frequency, cfg["features"], group, target_z=target)
        out = objective.evaluate_spectra(impedance(frequency, 10 ** np.zeros((2, 15))))
        self.assertEqual(out.shape, (2, 2 + len(group)))

    def test_paired_bootstrap_direction(self):
        baseline = np.ones((20, 2))
        rows = paired(baseline - .1, baseline, resamples=1000, components=20)
        self.assertTrue(all(row["adjusted_ci_high"] < 0 for row in rows))

    def test_cross_family_smoke(self):
        cfg, frequency = e.config(e.HERE / "config.json")
        cfg = copy.deepcopy(cfg)
        cfg.update(population=8, generations=3, seeds={"start": 200, "count": 1})
        group = e.groups(cfg)[-1]
        mf = e.manifest(cfg, frequency)
        with tempfile.TemporaryDirectory(prefix="case2_champion_") as tmp:
            record = e.run_one(cfg, frequency, group, 200, Path(tmp) / "run", mf["fingerprint"])
            self.assertEqual(record["evaluations"], 24)
            self.assertEqual(record["objective_names"][2:], list(group))


if __name__ == "__main__":
    unittest.main()


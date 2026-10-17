import tempfile
import unittest
from pathlib import Path
import numpy as np
import experiment as e


class RobustnessProtocolChecks(unittest.TestCase):
    def test_plan(self):
        cfg, _, _ = e.config(e.HERE / "config.json")
        plan = e.plan(cfg)
        self.assertEqual(plan["planned_runs"], 1200)
        self.assertEqual(plan["planned_evaluations"], 54_000_000)

    def test_observation_is_deterministic_and_group_independent(self):
        cfg, frequency, target = e.config(e.HERE / "config.json")
        for condition in cfg["conditions"]:
            a = e.observation(frequency, target, condition, 400)
            b = e.observation(frequency, target, condition, 400)
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])
            self.assertEqual(a[2], b[2])

    def test_cross_condition_smoke(self):
        cfg, frequency, target = e.config(e.HERE / "config.json")
        cfg = dict(cfg); cfg.update(population=8, generations=3)
        mf = e.manifest(cfg, frequency, target)
        with tempfile.TemporaryDirectory(prefix="case2_robustness_") as tmp:
            for condition in cfg["conditions"]:
                for group in e.GROUPS:
                    path = Path(tmp) / condition["name"] / e.group_name(group)
                    record = e.run_one(cfg, frequency, target, condition, group, 400,
                                       path, mf["fingerprint"])
                    self.assertEqual(record["evaluations"], 24)
                    self.assertGreaterEqual(record["observed_points"], 5)


if __name__ == "__main__":
    unittest.main()


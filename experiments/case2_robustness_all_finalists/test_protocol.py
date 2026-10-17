import tempfile
import unittest
from pathlib import Path
import experiment as e


class AllFinalistsChecks(unittest.TestCase):
    def test_plan(self):
        cfg, _, _ = e.config(e.HERE / "config.json")
        plan = e.plan(cfg)
        self.assertEqual(plan["planned_runs"], 5400)
        self.assertEqual(plan["planned_evaluations"], 243_000_000)
        self.assertEqual(len(plan["groups"]), 9)

    def test_representative_smoke(self):
        cfg, frequency, target = e.config(e.HERE / "config.json")
        cfg = dict(cfg); cfg.update(population=8, generations=3)
        mf = e.manifest(cfg, frequency, target)
        with tempfile.TemporaryDirectory(prefix="case2_all_robust_") as tmp:
            for group in (e.FINALIST_GROUPS[0], e.FINALIST_GROUPS[-1]):
                record = e.run_one(cfg, frequency, target, cfg["conditions"][-1],
                                   group, 400, Path(tmp) / e.group_name(group),
                                   mf["fingerprint"])
                self.assertEqual(record["evaluations"], 24)
                self.assertEqual(record["group"], e.group_name(group))


if __name__ == "__main__":
    unittest.main()


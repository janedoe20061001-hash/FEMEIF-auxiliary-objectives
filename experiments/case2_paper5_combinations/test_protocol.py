import copy,tempfile,unittest
from pathlib import Path
import experiment as e

class Checks(unittest.TestCase):
    def test_plan(self):
        cfg,f=e.config(e.HERE/"config.json")
        self.assertEqual(len(e.groups(cfg)),32)
        self.assertEqual(e.plan(cfg)["planned_runs"],3200)
    def test_smoke(self):
        cfg,f=e.config(e.HERE/"config.json");cfg=copy.deepcopy(cfg)
        cfg.update(population=8,generations=3,seeds={"start":0,"count":1})
        mf=e.manifest(cfg,f)
        with tempfile.TemporaryDirectory(prefix="case2_paper5_") as td:
            r=e.run_one(cfg,f,e.groups(cfg)[-1],0,Path(td)/"run",mf["fingerprint"])
            self.assertEqual(r["evaluations"],24)
            self.assertEqual(len(r["objective_names"]),7)

if __name__=="__main__":unittest.main()

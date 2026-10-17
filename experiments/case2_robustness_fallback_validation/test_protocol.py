import tempfile,unittest
from pathlib import Path
import experiment as e
class Checks(unittest.TestCase):
 def test_plan(self):
  cfg,_,_=e.config(e.HERE/'config.json');p=e.plan(cfg);self.assertEqual(p['planned_runs'],1000);self.assertEqual(p['planned_evaluations'],45_000_000)
 def test_smoke(self):
  cfg,f,z=e.config(e.HERE/'config.json');cfg=dict(cfg);cfg.update(population=8,generations=3);mf=e.manifest(cfg,f,z)
  with tempfile.TemporaryDirectory(prefix='fallback_') as d:
   for g in e.GROUPS:
    r=e.run_one(cfg,f,z,cfg['conditions'][0],g,500,Path(d)/e.group_name(g),mf['fingerprint']);self.assertEqual(r['evaluations'],24)
if __name__=='__main__':unittest.main()


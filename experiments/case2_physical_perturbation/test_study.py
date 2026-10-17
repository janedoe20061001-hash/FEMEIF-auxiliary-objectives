import unittest
import numpy as np
from run_study import plan

class Checks(unittest.TestCase):
    def test_plan_count_and_positive_parameters(self):
        rows=plan(np.geomspace(.01,1e4,15))
        self.assertEqual(len(rows),1765)
        self.assertTrue(all(np.all(r["params"]>0) for r in rows))

if __name__=="__main__":unittest.main()

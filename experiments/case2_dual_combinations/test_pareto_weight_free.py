import unittest
import numpy as np
from pareto_weight_free import primary_front, coverage, multiplicative_epsilon


class ParetoChecks(unittest.TestCase):
    def test_front_removes_dominated_and_duplicates(self):
        points=np.array([[1,4],[2,2],[4,1],[3,3],[2,2]])
        np.testing.assert_array_equal(primary_front(points),[[1,4],[2,2],[4,1]])

    def test_identical_fronts_are_symmetric(self):
        a=np.array([[1.,4.],[2.,2.],[4.,1.]])
        self.assertEqual(coverage(a,a),1.)
        self.assertEqual(multiplicative_epsilon(a,a),1.)

    def test_uniformly_better_front(self):
        a=np.array([[1.,2.],[2.,1.]])
        b=2*a
        self.assertEqual(coverage(a,b),1.)
        self.assertEqual(coverage(b,a),0.)
        self.assertEqual(multiplicative_epsilon(a,b),.5)
        self.assertEqual(multiplicative_epsilon(b,a),2.)

    def test_crossing_fronts(self):
        a=np.array([[1.,5.],[5.,1.]])
        b=np.array([[2.,3.],[3.,2.]])
        self.assertEqual(coverage(a,b),0.)
        self.assertEqual(coverage(b,a),0.)
        self.assertGreater(multiplicative_epsilon(a,b),1.)
        self.assertGreater(multiplicative_epsilon(b,a),1.)


if __name__=='__main__':unittest.main(verbosity=2)

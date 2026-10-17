"""Software/formula verification only; no reduced-budget optimization experiment."""
import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid

case=Path(sys.argv.pop(1)).resolve()
sys.path.insert(0,str(case))
import experiment as e
import legacy_objectives as old
from objectives import Objectives
from global_metrics import GlobalObjectives,AUX

class FormulaTests(unittest.TestCase):
    def test_identity_and_constant_residual(self):
        f=np.geomspace(10,1e6,1001);target=np.ones(len(f),complex)*(2+3j)
        obj=GlobalObjectives(f,aux=AUX,target_z=target)
        np.testing.assert_allclose(obj.evaluate_spectra(target),0,atol=1e-28)
        a=.3+.2j
        scores=obj.evaluate_spectra((1+a)*target)[0,2:]
        np.testing.assert_allclose(scores,[abs(a)**2/3,abs(a),abs(a)**2],rtol=1e-6)

    def test_independent_grid_definition(self):
        rng=np.random.default_rng(17);u=np.r_[0,np.sort(rng.uniform(0,1,21)),1]
        target=(2+1j)*np.ones(len(u));r=rng.normal(size=len(u))+1j*rng.normal(size=len(u))
        obj=GlobalObjectives(10**u,aux=AUX,target_z=target)
        values=obj.evaluate_spectra((1+r)*target)[0,2:]
        c=cumulative_trapezoid(r,u,initial=0)
        expected=[.5*trapezoid(abs(c)**2+abs(c[-1]-c)**2,u),
                  max(abs(c[j]-c[i]) for i in range(len(c)) for j in range(i,len(c))),
                  trapezoid(abs(r)**2,u)]
        np.testing.assert_allclose(values,expected,rtol=2e-14,atol=1e-14)

    def test_frequency_reversal_and_common_scale(self):
        rng=np.random.default_rng(21);u=np.r_[0,np.sort(rng.uniform(0,1,31)),1]
        z=2+rng.random(len(u))+1j*rng.random(len(u));zh=z*(1+.1*rng.normal(size=len(u)))
        a=GlobalObjectives(10**u,aux=AUX,target_z=z).evaluate_spectra(zh)[:,2:]
        b=GlobalObjectives(10**(1-u[::-1]),aux=AUX,target_z=z[::-1]).evaluate_spectra(zh[::-1])[:,2:]
        c=GlobalObjectives(10**u,aux=AUX,target_z=z*1e6).evaluate_spectra(zh*1e6)[:,2:]
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-15)
        np.testing.assert_allclose(a,c,rtol=1e-12,atol=1e-15)

    def test_original_primary_definitions_and_initials(self):
        cfg,f=e.config(case/'config.json');x=e.initial_population(cfg,7)
        if case.name=='case1':
            target=old.impedance(f,old.TARGET)[0];z=old.impedance(f,10**x)
        else:
            from case2 import impedance,load_target
            _,target=load_target(cfg['target_data']);z=impedance(f,10**x)
        previous=old.Objectives(f,old.DEFAULTS,target_z=target).evaluate_spectra(z)[:,:2]
        for aux in [(),*[(n,) for n in AUX]]:
            current=GlobalObjectives(f,aux=aux,target_z=target).evaluate_spectra(z)
            np.testing.assert_array_equal(current[:,:2],previous)
            self.assertTrue(np.isfinite(current).all())
        self.assertEqual(cfg['seeds'],{'start':0,'count':100})
        self.assertEqual(cfg['generations'],300)
        self.assertEqual(cfg['population'],200 if case.name=='case1' else 150)

    def test_invalid_target_rejected(self):
        with self.assertRaises(ValueError):
            GlobalObjectives([1,10],aux=AUX,target_z=[0,1])

if __name__=='__main__':
    unittest.main()

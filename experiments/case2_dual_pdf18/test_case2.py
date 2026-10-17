"""Circuit, units, objective transfer, and end-to-end checks; no formal runs."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import experiment as e
from case2 import impedance, load_target, PARAMETER_NAMES, LOWER, UPPER
from objectives import Objectives, AUX


class Case2Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.f = e.config(e.HERE / "config.json")
        _, cls.z = load_target(cls.cfg["target_data"])

    def test_protocol_and_units(self):
        self.assertEqual(e.plan(self.cfg)["planned_evaluations"],85500000)
        self.assertEqual((self.cfg["population"],self.cfg["generations"],self.cfg["seeds"]["count"]),(150,300,100))
        self.assertEqual(self.cfg["mutation_variable_probability"],.08)
        self.assertEqual(self.cfg["final_selection"]["phase_weight"],17.38)
        self.assertEqual(len(self.f),201)
        self.assertEqual((self.f[0],self.f[-1]),(40.,110e6))
        self.assertAlmostEqual(abs(self.z[0]),1.46e6)
        self.assertAlmostEqual(np.angle(self.z[0]),np.deg2rad(-87.1))
        bad=copy.deepcopy(self.cfg["target_data"]);bad["sha256"]="changed"
        with self.assertRaises(ValueError):load_target(bad)

    def test_circuit_against_nodal_analysis(self):
        # Independent nine-node KCL solve, 1 A injected at the input.
        for frequency in [40.,1e4,1e6,110e6]:
            w=2*np.pi*frequency; matrix=np.zeros((9,9),complex)
            # Keep branch impedances balanced so the reference KCL system is
            # well-conditioned, including at 40 Hz. Bounds are checked separately.
            p=np.r_[np.arange(1.,8.),np.arange(1.,5.)/w,1/(w*np.arange(1.,5.))]
            def add(a,b,y):
                if a:matrix[a-1,a-1]+=y
                if b:matrix[b-1,b-1]+=y
                if a and b:matrix[a-1,b-1]-=y;matrix[b-1,a-1]-=y
            for k in range(3):
                a,b,mid=k+1,k+2,k+7
                add(a,b,1j*w*p[11+k]);add(a,b,1/p[2*k+1])
                add(a,mid,1/(1j*w*p[7+k]));add(mid,b,1/p[2*k])
            add(4,5,1j*w*p[14]);add(5,6,1/p[6]);add(6,0,1/(1j*w*p[10]))
            current=np.zeros(9);current[0]=1
            expected=np.linalg.solve(matrix,current)[0]
            np.testing.assert_allclose(impedance([frequency],p)[0,0],expected,rtol=1e-9)
        rng=np.random.default_rng(77);params=10**rng.uniform(LOWER,UPPER,(32,15))
        z=impedance(self.f,params)
        self.assertTrue(np.isfinite(z).all())
        self.assertTrue(np.all(z.real>=params[:,6,None]*(1-1e-12)))

    def test_objective_transfer_and_primary_units(self):
        obj=Objectives(self.f,self.cfg["features"],AUX,target_z=self.z)
        spectrum=np.vstack([self.z,self.z*2])
        result=obj.evaluate_spectra(spectrum)
        np.testing.assert_allclose(result[:,0],[0,(20*np.log10(2))**2],atol=1e-12)
        np.testing.assert_allclose(result[:,1],0,atol=1e-28)
        self.assertTrue(np.isfinite(result).all())
        # Target has two detected turns. Original capped thresholds imply these aliases.
        sample=impedance(self.f,10**e.initial_population(self.cfg,7)[:12])
        scores=obj.evaluate_spectra(sample)
        np.testing.assert_array_equal(scores[:,2+AUX.index('turning_event_global_count_5')],scores[:,2+AUX.index('turning_event_global_count_4')])
        np.testing.assert_array_equal(scores[:,2+AUX.index('turning_event_coverage_5_of_6')],scores[:,2+AUX.index('turning_event_coverage_4_of_6')])

    def test_target_grid_mismatch_rejected(self):
        f=self.f.copy();f[0]+=1
        with self.assertRaises(ValueError):e.Case2(self.cfg,f,())
        with self.assertRaises(ValueError):e.manifest(self.cfg,f)

    def test_launcher_requires_explicit_run(self):
        with tempfile.TemporaryDirectory(prefix="case2_no_launch_") as temp:
            output=Path(temp)/'must_not_exist'
            result=subprocess.run([sys.executable,str(e.HERE/'launch_batch.py'),'--output',str(output)],capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn(b'No batch started',result.stderr)
            self.assertFalse(output.exists())

    def test_paired_smoke_and_report(self):
        cfg=copy.deepcopy(self.cfg);cfg.update(population=8,generations=3,seeds=dict(start=0,count=2))
        mf=e.manifest(cfg,self.f)
        with tempfile.TemporaryDirectory(prefix="case2_smoke_") as temp:
            root=Path(temp);e.dump(root/'manifest.json',mf)
            for seed in range(2):
                for aux in e.groups(cfg):
                    r=e.run_one(cfg,self.f,aux,seed,root/e.group_name(aux)/f'seed_{seed:03d}',mf['fingerprint'])
                    self.assertEqual(r['evaluations'],24)
                    self.assertEqual(r['initial_population_sha256'],hashlib.sha256(e.initial_population(cfg,seed).tobytes()).hexdigest())
            subprocess.run([sys.executable,str(e.HERE/'report_results.py'),str(root),'--no-archive'],check=True,capture_output=True)
            audit=json.loads((root/'completion_audit.json').read_text())
            self.assertEqual((audit['status'],audit['runs']),('passed',38))
            stability=json.loads((root/'stability.json').read_text(encoding='utf-8'))
            self.assertEqual(len(stability['findings']),18)
            self.assertEqual(stability['settings']['familywise_components'],36)


if __name__=='__main__':
    unittest.main(verbosity=2)

"""Numerical/experimental-integrity checks; only 8-individual, 3-generation runs."""
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import experiment as e
from objectives import *
from summarize import summarize


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg,cls.freq=e.config(e.HERE/"config.json")

    def test_paper_budget_and_groups(self):
        p=e.plan(self.cfg)
        self.assertEqual((p["seed_count"],p["population"],p["generations"]),(100,200,300))
        self.assertEqual(p["planned_runs"],3200)
        self.assertEqual(p["planned_evaluations"],192000000)
        self.assertEqual([sum(len(g)==n for g in e.groups(self.cfg)) for n in range(6)],[1,5,10,10,5,1])

    def test_physics_and_all_objective_identity(self):
        f=np.geomspace(100,1e6,180)
        ref=impedance(f,TARGET)
        np.testing.assert_allclose(ref,impedance(f,TARGET[::-1]),rtol=1e-12)
        p=TARGET.copy();p[:,:2]*=1.1;p[:,2]/=1.1
        np.testing.assert_allclose(impedance(f,p),1.1*ref,rtol=1e-11)
        obj=Objectives(f,self.cfg["features"],AUX)
        values=obj.evaluate(np.log10(TARGET).reshape(1,18))
        np.testing.assert_allclose(values,0,atol=1e-7)
        self.assertEqual(len(obj.events),11)

    def test_primary_independence_and_scalar_batch_agreement(self):
        x=e.initial_population(dict(self.cfg,population=5),9)
        obj=Objectives(self.freq,self.cfg["features"],AUX)
        batch=obj.evaluate(x)
        np.testing.assert_allclose(batch,np.vstack([obj.evaluate(v) for v in x]),rtol=1e-13)
        for group in e.groups(self.cfg):
            actual=Objectives(self.freq,self.cfg["features"],group).evaluate(x)
            cols=[0,1]+[2+AUX.index(v) for v in group]
            np.testing.assert_allclose(actual,batch[:,cols],rtol=1e-13)
        z=impedance(self.freq,10**x.reshape(-1,6,3))
        np.testing.assert_allclose(batch[:,0],((20*np.log10(abs(z))-obj.a)**2).mean(axis=1))
        np.testing.assert_allclose(batch[:,1],((np.angle(z)-obj.phase)**2).mean(axis=1))

    def test_peak_matching_all_matches_and_missing(self):
        args=(np.array([3.]),np.array([1]),np.array([2.,6.]),.41)
        # Two same-type matches accumulate; opposite-type event is ignored.
        actual=ptc_cost(np.array([2.8,3.1,3.2]),np.array([1,-1,1]),*args)
        self.assertAlmostEqual(actual,2*np.expm1(.41*.2/4))
        self.assertEqual(ptc_cost(np.array([3.]),np.array([-1]),*args),.41)

    def test_optional_reciprocal_and_invalid_inputs(self):
        x=e.initial_population(dict(self.cfg,population=4),3)
        raw=Objectives(self.freq,self.cfg["features"],AUX).evaluate(x)
        inv=Objectives(self.freq,self.cfg["features"],AUX,"negative_reciprocal").evaluate(x)
        np.testing.assert_array_equal(raw[:,:2],inv[:,:2])
        for j,name in enumerate(AUX):
            expected=-(1-raw[:,2+j])*11 if name=="LFP" else -1/(raw[:,2+j]+1e-12)
            np.testing.assert_allclose(inv[:,2+j],expected,atol=1e-12)
        for bad in ([100,100,1000],[100,50,1000],[0,100,1000],[100,np.nan,1000]):
            with self.assertRaises(ValueError):
                Objectives(bad,self.cfg["features"])

    def test_distinguish_primary_and_full_fronts(self):
        f=np.array([[1.,1.,10.],[2.,2.,0.],[3.,3.,20.]])
        result=e.stats(f)
        self.assertEqual(result["full_front_size"],2)
        self.assertEqual(result["primary_front_size"],1)
        self.assertEqual(result["full_front_mean_magnitude_mse"],1.5)
        self.assertEqual(result["primary_front_mean_magnitude_mse"],1.)

    def test_summary_uses_ratio_of_seed_means(self):
        with tempfile.TemporaryDirectory(prefix="case1_summary_check_") as tmp:
            root=Path(tmp)
            e.dump(root/"manifest.json",dict(fingerprint="test",config=dict(seeds=dict(count=2))))
            for group,values in [("baseline",[10.,100.]),("SS",[5.,120.])]:
                for seed,value in enumerate(values):
                    path=root/group/f"seed_{seed:03d}";path.mkdir(parents=True)
                    r=dict(status="complete",fingerprint="test",group=group,seed=seed)
                    for front in ("full_front","primary_front"):
                        r[f"{front}_mean_magnitude_mse"]=value
                        r[f"{front}_mean_phase_mse"]=value/10
                    e.dump(path/"summary.json",r)
            ss=[r for r in summarize(root) if r["group"]=="SS"]
            self.assertTrue(all(r["complete"] for r in ss))
            self.assertAlmostEqual(ss[0]["magnitude_change_pct"],100*(62.5/55-1))
            self.assertFalse(ss[0]["both_means_lower"])

    def test_ss_dtw_and_lfp_rules(self):
        fs=self.cfg["features"];x=np.arange(3.)
        self.assertEqual(patterns(np.array([0.,20.,40.]),x,fs)[0],1.)
        self.assertTrue(1.4<patterns(np.array([0.,-20.,0.]),x,fs)[0]<1.6)
        self.assertEqual(dtw(np.array([[0.,0.,1.]]),np.array([0.,1.,1.]))[0],0.)
        self.assertEqual(dtw(np.array([[3.]]),np.array([1.]))[0],2.)
        # Whole-spectrum dB offset: all LFP zones pass below threshold, fail above.
        obj=Objectives(self.freq,fs,("SS","PTC","LFP","LMEP"))
        for scale,expected in [(1.1,0.),(2.,1.)]:
            p=TARGET.copy();p[:,:2]*=scale;p[:,2]/=scale
            v=obj.evaluate(np.log10(p).reshape(1,18))[0]
            self.assertAlmostEqual(v[4],expected)
            self.assertAlmostEqual(v[5],20*np.log10(scale),places=8)
            self.assertLess(abs(v[2]),1e-8);self.assertEqual(v[3],0)

    def test_against_previous_comparison_at_180_points(self):
        previous=e.HERE.parent/"case1_physical_perturbation_20260912/paper_comparison/all_runs.jsonl"
        if not previous.exists():
            self.skipTest("Optional prior comparison data absent")
        rows=[json.loads(line) for line in previous.read_text(encoding="utf-8").splitlines()][::100]
        obj=Objectives(self.freq,self.cfg["features"],AUX)
        v=obj.evaluate(np.log10(np.array([r["params"] for r in rows])).reshape(-1,18))
        for j,key in enumerate(["SS_180","PTC_180","DTW","LFP_loss_180","LMEP_180"]):
            np.testing.assert_allclose(v[:,2+j],[r[key] for r in rows],rtol=1e-7,atol=1e-8)

    def test_all_groups_short_runs_reproducibility_and_summaries(self):
        cfg=copy.deepcopy(self.cfg)
        cfg.update(population=8,generations=3,seeds=dict(start=0,count=1))
        with tempfile.TemporaryDirectory(prefix="case1_code_check_") as tmp:
            root=Path(tmp);mf=e.manifest(cfg,self.freq);e.dump(root/"manifest.json",mf)
            hashes=set()
            for group in e.groups(cfg):
                folder=root/e.group_name(group)/"seed_000"
                r=e.run_one(cfg,self.freq,group,0,folder,mf["fingerprint"])
                hashes.add(r["initial_population_sha256"])
                self.assertEqual(r["evaluations"],24)
                with np.load(folder/"final_population.npz") as data:
                    self.assertEqual(data["objectives"].shape,(8,2+len(group)))
                    self.assertTrue(np.all(data["log_parameters"]>=LOWER))
                    self.assertTrue(np.all(data["log_parameters"]<=UPPER))
                with (folder/"history.csv").open(encoding="utf-8-sig") as fh:
                    history=list(csv.DictReader(fh))
                self.assertEqual([int(h["evaluations"]) for h in history],[8,16,24])
            self.assertEqual(len(hashes),1)
            repeat=root/"repeat"
            e.run_one(cfg,self.freq,(),0,repeat,mf["fingerprint"])
            with np.load(repeat/"final_population.npz") as a,np.load(root/"baseline/seed_000/final_population.npz") as b:
                np.testing.assert_array_equal(a["objectives"],b["objectives"])
                np.testing.assert_array_equal(a["log_parameters"],b["log_parameters"])
            with self.assertRaises(FileExistsError):
                e.run_one(cfg,self.freq,(),0,repeat,mf["fingerprint"])
            result=summarize(root)
            self.assertEqual(len(result),64)
            self.assertTrue(all(r["paired_seeds"]==1 for r in result))
            baseline=[r for r in result if r["group"]=="baseline"]
            self.assertTrue(all(r["magnitude_change_pct"]==r["phase_change_pct"]==0 for r in baseline))


if __name__=="__main__":
    unittest.main(verbosity=2)

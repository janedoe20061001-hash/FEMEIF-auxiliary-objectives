"""Small software checks only; no 100-seed formal experiment is run here."""
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import experiment as e
import objectives as o
from summarize import summarize


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg,cls.freq=e.config(e.HERE/"config.json")
        cls.model=o.Objectives(cls.freq,o.DEFAULTS,o.AUX)

    def test_configuration_and_stage_separation(self):
        plan=e.plan(self.cfg)
        self.assertEqual((plan["seed_count"],plan["population"],plan["generations"]),(100,200,300))
        self.assertEqual(plan["group_count"],19)
        self.assertEqual(plan["planned_runs"],1900)
        self.assertEqual(plan["planned_evaluations"],114000000)
        self.assertEqual(self.cfg["selection_objectives"],"all")
        self.assertEqual(self.cfg["report_front"],"wsmse_selected")
        self.assertTrue(all(len(g)<=1 for g in e.groups(self.cfg)))
        with self.assertRaises(ValueError):
            e.groups(dict(self.cfg,groups=[[o.AUX[0],o.AUX[1]]]))
        s=e.stats(np.array([[1.,1.,10.],[2.,2.,0.],[3.,3.,20.]]))
        self.assertEqual(s["full_front_size"],2)
        self.assertEqual(s["primary_front_size"],1)

    def test_auxiliary_enters_survival_ranking_but_reporting_is_read_only(self):
        from pymoo.core.population import Population
        from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
        x=np.tile(np.log10(o.TARGET).reshape(1,18),(3,1))
        # Both primary errors get worse, while the auxiliary improves.
        f=np.array([[1.,1.,10.],[2.,2.,5.],[3.,3.,0.]])
        pop=Population.new("X",x,"F",f)
        problem=e.Case1(self.cfg,self.freq,(o.AUX[0],))
        ranked=RankAndCrowding().do(problem,pop,n_survive=3,random_state=np.random.default_rng(0))
        np.testing.assert_array_equal(ranked.get("rank"),[0,0,0])
        baseline=Population.new("X",x,"F",f[:,:2])
        baseline=RankAndCrowding().do(e.Case1(self.cfg,self.freq,()),baseline,
                                     n_survive=3,random_state=np.random.default_rng(0))
        np.testing.assert_array_equal(baseline.get("rank"),[0,1,2])
        before=ranked.get("F").copy()
        self.assertEqual(e.stats(ranked.get("F"))["primary_front_size"],1)
        np.testing.assert_array_equal(ranked.get("F"),before)

    def test_identity_and_target_diagnostics(self):
        actual=self.model.evaluate_spectra(self.model.z)
        np.testing.assert_allclose(actual,0,atol=1e-10)
        diag=self.model.target_summary()
        self.assertEqual((diag["target_turns"],diag["target_crossings"],diag["target_amplitude_extrema"]),(6,11,11))

    def test_nonuniform_derivative_quadratic_exactness(self):
        x=np.array([0.,.1,.4,.9,1.7,3.])
        z=(2+3j)*x**2+(5-2j)*x+7
        expected=2*(2+3j)*x+(5-2j)
        np.testing.assert_allclose(o.derivative(z,x),expected,atol=2e-13)

    def test_turning_local_peaks_greedy_gap_and_ties(self):
        # Equal peaks 3 indices apart: the earlier one wins. Four apart both survive.
        theta=np.array([0.,3.,0.,0.,3.,0.,0.,0.,3.,0.])
        np.testing.assert_array_equal(o.select_turns(theta,2.,4),[2,9])
        # Angle index 1 maps back to original sample index 2.
        np.testing.assert_array_equal(o.select_turns(np.array([0.,2.,2.,0.]),1.),[2])
        straight=np.column_stack([np.arange(6.),np.zeros((6,2))])
        np.testing.assert_allclose(o.turning_angles(straight),0)

    def test_regions_crossings_and_signed_zero(self):
        mask=o.region_matches(np.array([.5]),np.array([0.,.5,1.]))
        self.assertTrue(mask[0][0] and mask[1][0])
        imag=np.array([-1.,-0.,+0.,1.,-1.]);u=np.linspace(0,1,5)
        locations,direction=o.crossings(imag,u)
        np.testing.assert_allclose(locations,[.375,.875])
        np.testing.assert_array_equal(direction,[-1,-1]) # -0 < +0 is False, as specified.
        np.testing.assert_array_equal(o.crossings(np.ones(5),u)[0],[])

    def test_pdf_ptc_is_nearest_average_with_neighbor_threshold(self):
        target=np.array([3.,5.]);types=np.array([1,-1]);edges=np.array([2.,4.,6.])
        cost=o.nearest_ptc(np.array([3.1,3.2,5.]),np.array([1,1,1]),target,types,edges,4.)
        self.assertAlmostEqual(cost,(np.expm1(.41*.1/4)+.41)/2)
        self.assertEqual(len(o.peaks_troughs(np.array([0.,.19,0.]),np.arange(3.))[0]),0)
        self.assertEqual(len(o.peaks_troughs(np.array([0.,.2,0.]),np.arange(3.))[0]),1)
        self.assertEqual(o.nearest_ptc(np.array([]),np.array([]),np.array([]),
                                     np.array([]),np.array([0.,1.]),1.),0)

    def test_dtw_norm_path_and_tie_rules(self):
        p=np.zeros((1,1,3));q=np.array([[[0.,3.,4.]]])
        np.testing.assert_allclose(o.band_dtw(p,q),[5.])
        p=np.zeros((1,3,3))
        value,total,steps=o.band_dtw(p,p,return_details=True)
        self.assertEqual(steps[0],3);self.assertEqual(value[0],0)
        p=np.array([[[0.,0.,0.],[0.,0.,0.],[0.,1.,0.]]])
        q=np.array([[[0.,0.,0.],[0.,1.,0.],[0.,1.,0.]]])
        value,total,steps=o.band_dtw(p,q,return_details=True)
        self.assertEqual(total[0],0);self.assertEqual(steps[0],4)
        np.testing.assert_allclose(o.band_dtw(p,q,window=0),[1/3])

    def test_soft_dtw_against_enumerated_paths(self):
        p=np.array([[0.,0.,0.],[.5,.2,0.],[1.,.3,.1]])
        q=np.array([[0.,.1,0.],[.5,.25,0.],[1.,.4,.1]])
        window=1;gamma=.1;n=len(p)
        costs=np.sum((p[:,None]-q[None,:])**2,axis=-1)
        def walk(i,j,total):
            total+=costs[i,j]
            if i==n-1 and j==n-1:
                return [total]
            values=[]
            for di,dj in [(1,1),(1,0),(0,1)]:
                a,b=i+di,j+dj
                if a<n and b<n and abs(a-b)<=window:values+=walk(a,b,total)
            return values
        totals=np.array(walk(0,0,0.))
        best=totals.min()
        expected=best-gamma*np.log(np.exp(-(totals-best)/gamma).sum())
        self.assertAlmostEqual(o.band_soft_dtw(p[None],q[None],window,gamma)[0],expected)
        self.assertLess(o.band_soft_dtw(np.zeros((1,3,3)),np.zeros((1,3,3)))[0],0)
        np.testing.assert_allclose(o.soft_divergence(p[None],p[None]),0,atol=1e-10)
        np.testing.assert_allclose(o.soft_divergence(p[None],q[None]),o.soft_divergence(q[None],p[None]),atol=1e-10)

    def test_process_projection_and_speed_examples(self):
        f=np.geomspace(100,1e6,11);x=np.log10(f)
        target=(3+4j)*x+20
        names=["characteristic_frequency_shift_error_decades","transition_speed_relative_error","normal_shape_error_relative"]
        model=o.Objectives(f,o.DEFAULTS,names,target_z=target)
        delta=.1;candidate=target+(3+4j)*delta
        v=model.evaluate_spectra(candidate)[0]
        estimate=25*delta/(25+.25**2)
        self.assertAlmostEqual(v[2],estimate)
        self.assertAlmostEqual(v[3],0,places=12)
        expected=5*(delta-estimate)/np.median(abs(target))
        self.assertAlmostEqual(v[4],expected)
        normal=model.evaluate_spectra(target+(-.4+.3j))[0]
        self.assertAlmostEqual(normal[2],0,places=12)
        self.assertAlmostEqual(normal[4],.5/model.sz)
        reversed_speed=model.evaluate_spectra(-target)[0,3]
        self.assertAlmostEqual(reversed_speed,0,places=12)

    def test_translation_and_h1_closed_form(self):
        f=np.geomspace(100,1e6,11);u=np.linspace(0,1,11)
        target=(1+u)+1j*(1+2*u)
        names=["translation_aligned_spatial_shape_error","sobolev_h1_error","hausdorff95_error"]
        model=o.Objectives(f,o.DEFAULTS,names,target_z=target)
        values=model.evaluate_spectra(target+(1+2j))[0]
        self.assertAlmostEqual(values[2],0,places=12)
        # Normalized residual is (1,1); derivatives cancel: sqrt(E0)=sqrt(2).
        self.assertAlmostEqual(values[3],np.sqrt(2))
        p=model.p;q=model.path((target+1+2j)[None])[0]
        distances=np.sqrt(np.sum((p[:,None]-q[None,:])**2,axis=-1))
        expected=max(np.quantile(distances.min(axis=0),.95),np.quantile(distances.min(axis=1),.95))
        self.assertAlmostEqual(values[4],expected)

    def test_singletons_primary_agreement_and_paper_baseline(self):
        x=e.initial_population(dict(self.cfg,population=3),17)
        all_values=self.model.evaluate(x)
        for i,name in enumerate(o.AUX):
            model=o.Objectives(self.freq,o.DEFAULTS,[name])
            np.testing.assert_allclose(model.evaluate(x),all_values[:,[0,1,2+i]],rtol=2e-12,atol=1e-10)
        path=e.HERE.parent/"case1_dual_paper/objectives.py"
        if path.exists():
            spec=importlib.util.spec_from_file_location("paper_objectives_reference",path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            oldcfg=json.loads((path.parent/"config.json").read_text())
            previous=module.Objectives(self.freq,oldcfg["features"]).evaluate(x)
            np.testing.assert_array_equal(previous,all_values[:,:2])

    def test_nineteen_small_runs_and_repeatability(self):
        cfg=copy.deepcopy(self.cfg);cfg.update(population=8,generations=3,seeds=dict(start=0,count=1))
        with tempfile.TemporaryDirectory(prefix="pdf18_code_checks_") as tmp:
            root=Path(tmp);mf=e.manifest(cfg,self.freq);e.dump(root/"manifest.json",mf)
            hashes=set()
            for group in e.groups(cfg):
                record=e.run_one(cfg,self.freq,group,0,root/e.group_name(group)/"seed_000",mf["fingerprint"])
                hashes.add(record["initial_population_sha256"])
                self.assertEqual(record["evaluations"],24)
                self.assertEqual(len(record["objective_names"]),2+len(group))
            self.assertEqual(len(hashes),1)
            e.run_one(cfg,self.freq,(),0,root/"repeat",mf["fingerprint"])
            with np.load(root/"baseline/seed_000/final_population.npz") as a,np.load(root/"repeat/final_population.npz") as b:
                np.testing.assert_array_equal(a["objectives"],b["objectives"])
                np.testing.assert_array_equal(a["log_parameters"],b["log_parameters"])
            results=summarize(root)
            self.assertEqual(len(results),57)
            self.assertEqual(results[0]["front"],"wsmse_selected")
            # Changing final-report weight does not alter any evolved candidate.
            other=copy.deepcopy(cfg);other["final_selection"]["phase_weight"]=1.
            e.run_one(other,self.freq,(),0,root/"other_weight",mf["fingerprint"])
            with np.load(root/"baseline/seed_000/final_population.npz") as a,np.load(root/"other_weight/final_population.npz") as b:
                np.testing.assert_array_equal(a["objectives"],b["objectives"])
                np.testing.assert_array_equal(a["log_parameters"],b["log_parameters"])

    def test_selected_metrics_belong_to_one_minimum_wsmse_individual(self):
        from final_selection import select_wsmse
        f=np.array([[1.,10.,0.],[4.,2.,100.],[20.,.1,0.]])
        result=select_wsmse(f,2.)
        self.assertEqual(result["wsmse_selected_index"],1)
        self.assertEqual(result["wsmse_selected_magnitude_mse"],4.)
        self.assertEqual(result["wsmse_selected_phase_mse"],2.)
        self.assertEqual(result["wsmse_selected_value"],8.)
        self.assertEqual(select_wsmse(np.array([[2.,3.],[4.,2.]]),2.)["wsmse_selected_index"],0)
        for weight in [0.,-1.,float("nan")]:
            with self.assertRaises(ValueError):
                select_wsmse(f,weight)


if __name__=="__main__":
    unittest.main(verbosity=2)

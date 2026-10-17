"""Numerical and mocked lifecycle checks; NEVER runs an optimization search."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from numpy.polynomial import polynomial as poly
from circuits import library, Circuit, P, S, WREF
from protocol import HERE, read_config, targets, initial, source_hashes, dump, plan
from sampling import Quadrature
from diagnostics import events, match, diagnostic
from prepare import prepare

CFG = read_config()


class NumericalChecks(unittest.TestCase):
    def test_01_topology_and_analytic_formula(self):
        lib = library()
        self.assertEqual(len(lib), 7)
        c = lib["user_nested"]
        self.assertEqual(c.names, ["R1","L1","R2","R3","C1","L2"])
        f = np.geomspace(100, 1e6, 211)
        v, w = c.values, 2j*np.pi*f
        inner = 1/(1/(v["R3"]+1/(w*v["C1"]))+1/(w*v["L2"]))
        expected = v["R1"] + 1/(1/(w*v["L1"])+1/(v["R2"]+inner))
        np.testing.assert_allclose(c.response(f)[0], expected, rtol=1e-13)
        with self.assertRaises(ValueError):
            Circuit("bad","bad",S("R1","R1","L1","C1"),dict(R1=1,L1=1,C1=1),("L1","C1"),"R1","R1")

    def test_02_nodal_polynomial_and_derivative(self):
        rng = np.random.default_rng(2026101001)
        f = np.geomspace(*CFG["band_hz"], 43)
        for c in library().values():
            for multiplier in [np.ones(len(c.names)), *10**rng.uniform(-1.5,1.5,(5,len(c.names)))]:
                p = c.p * multiplier
                z,dz = c.response(f,p)
                np.testing.assert_allclose(z,c.nodal_response(f,p),rtol=2e-8,atol=1e-9,err_msg=c.ident)
                num,den = c.rational(p)
                rr = poly.polyval(2j*np.pi*f/WREF,num)/poly.polyval(2j*np.pi*f/WREF,den)
                np.testing.assert_allclose(z,rr,rtol=2e-8,atol=1e-9,err_msg=c.ident)
                step = 1e-6
                numerical = (c.response(f*np.exp(step),p)[0]-c.response(f*np.exp(-step),p)[0])/(2*step)
                np.testing.assert_allclose(dz,numerical,rtol=2e-5,atol=1e-5,err_msg=c.ident)
                self.assertTrue(c.modes(p))
                # Batched evaluation equals individual circuit evaluation.
                batched = c.response(f,np.array([p,c.p]))[0]
                np.testing.assert_allclose(batched[0],z,rtol=1e-13)
                np.testing.assert_allclose(batched[1],c.response(f)[0],rtol=1e-13)

    def test_03_interventions_and_pairing(self):
        ts = targets(CFG,"extended")
        expected = len(library())*(1+sum(sum(f != 1 for f in CFG["factors"][k]) for k in CFG["profiles"]["extended"]))
        self.assertEqual(len(ts),expected)
        for t in ts:
            c,b = t["circuit"],t["base"]
            self.assertTrue(np.all(np.log10(c.p)>=t["lower"]))
            self.assertTrue(np.all(np.log10(c.p)<=t["upper"]))
            if t["intervention"] == "lc_reciprocal":
                l,k = c.lc
                np.testing.assert_allclose(c.values[l]*c.values[k],b.values[l]*b.values[k],rtol=1e-14,atol=0)
            base = next(a for a in ts if a["id"] == b.ident+"/base")
            np.testing.assert_array_equal(initial(t,CFG,33000),initial(base,CFG,33000))
        self.assertEqual(plan(CFG,"core",None)["planned_evaluations"],len(targets(CFG,"core"))*len(CFG["seeds"])*CFG["population"]*CFG["generations"])

    def test_04_objective_zero_and_independent_quadrature(self):
        rng = np.random.default_rng(2026101002)
        # All 49 analytic targets and two random candidates per circuit.
        for t in targets(CFG,"extended"):
            c = t["circuit"]
            q = Quadrature(c,CFG)
            np.testing.assert_allclose(q.score(np.log10(c.p)),0,atol=1e-20)
        for c in library().values():
            q = Quadrature(c,CFG)
            pp = np.log10(c.p) + rng.uniform(-1.5,1.5,(2,len(c.names)))
            scores = q.score(pp)
            np.testing.assert_array_equal(scores,np.array([q.score_one(p) for p in pp]))
            for p,score in zip(pp,scores):
                r1 = q.reference(p,2)[0]
                r2 = q.reference(p,4)[0]
                tol = np.array(CFG["audit"]["objective_atol"])+CFG["audit"]["objective_rtol"]*abs(r2)
                self.assertTrue(np.all(abs(r1-r2)<=tol),c.ident)
                self.assertTrue(np.all(abs(score-r2)<=tol),(c.ident,score,r2))
            small = np.log10(c.p)+.001
            r = q.reference(small,4)[0]
            s = q.score(small)[0]
            self.assertTrue(np.all(abs(r-s)<=np.array(CFG["audit"]["objective_atol"])+CFG["audit"]["objective_rtol"]*abs(r)),c.ident)
        # Extreme high-Q control and a strongly mismatched narrow candidate.
        c = library()["control_rlc"]
        p = c.p.copy()
        p[0] *= 30
        p[1] *= 30
        p[2] /= 30
        q = Quadrature(c,CFG)
        score = q.score(np.log10(p))[0]
        ref = q.reference(np.log10(p),4)[0]
        self.assertTrue(np.all(abs(score-ref)<=np.array(CFG["audit"]["objective_atol"])+CFG["audit"]["objective_rtol"]*abs(ref)))

    def test_05_events_missing_and_null_widths(self):
        c = library()["control_rlc"]
        q = Quadrature(c,CFG)
        x = q.reference_grid(refinement=4)
        z,dz = c.response(10**x)
        target = events(x,z,CFG)
        self.assertEqual(len(target),1)
        self.assertAlmostEqual(target[0]["frequency_hz"],1e4,delta=.1)
        rows,extra = match(target,events(x,z,CFG),CFG)
        self.assertTrue(rows[0]["complete"])
        rows,extra = match(target,[],CFG)
        self.assertFalse(rows[0]["matched"])
        self.assertIsNone(rows[0]["height_delta_db"])
        shallow = [dict(target[0],width_3db_decades=None,width_3db_hz=None,left_3db_x=None,right_3db_x=None)]
        rows,_ = match(shallow,shallow,CFG)
        self.assertFalse(rows[0]["width_evaluable"])
        self.assertIsNone(rows[0]["width_ratio"])
        from diagnostics import phase_events
        rec,_ = diagnostic(q,np.log10(c.p),target,phase_events(x,z,dz,CFG))
        self.assertTrue(rec["stable"])
        self.assertTrue(all(r["complete"] for r in rec["phase_events"]))
        np.testing.assert_allclose(rec["objective"],0,atol=1e-20)

    def test_06_resource_exhaustion_is_failure(self):
        cfg = json.loads(json.dumps(CFG))
        cfg["adaptive"]["max_points"] = 2
        c = library()["control_rlc"]
        with self.assertRaises(RuntimeError):
            Quadrature(c,cfg).score(np.log10(c.p))

    def test_07_prepare_and_plan_no_optimizer(self):
        from pymoo.optimize import minimize
        with patch("pymoo.optimize.minimize",side_effect=AssertionError("Optimizer called during preparation")):
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)/"targets"
                self.assertEqual(prepare(CFG,"extended",None,out),len(targets(CFG,"extended")))
                manifest = json.loads((out/"manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["status"],"prepared")
                self.assertEqual(len(manifest["target_hashes"]),len(targets(CFG,"extended")))
                with self.assertRaises(FileExistsError):
                    prepare(CFG,"extended",None,out)
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp)/"test_config.json"
            dump(cfg_path,CFG)
            proc = subprocess.run([sys.executable,str(HERE/"run.py"),"plan","--config",str(cfg_path)],capture_output=True,text=True,check=True)
            self.assertFalse(json.loads(proc.stdout)["optimization_started"])

    def test_08_mocked_checkpoint_analysis_roundtrip(self):
        """Static fake population, NO evolutionary steps or real minimize call."""
        from run import one
        from analyze import analyze
        cfg = json.loads(json.dumps(CFG))
        cfg.update(population=4,generations=20,seeds=[33000],plot_seeds=[33000])
        cfg["factors"]["lc_reciprocal"] = [1.]
        t = targets(cfg,ids=["control_rlc"])[0]

        def fake_minimize(problem, algorithm, termination, seed, callback, **kwargs):
            x = np.tile(np.log10(t["circuit"].p),(4,1))
            scores = problem.quadrature.score(x)
            # Exercise actual callback on a static known population only.
            alg = SimpleNamespace(n_gen=20,pop={"X":x,"F":scores},evaluator=SimpleNamespace(n_eval=80),problem=problem)
            callback.notify(alg)
            return SimpleNamespace(algorithm=alg)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prep, batch, out = root/"prepared",root/"MOCK_ONLY",root/"analysis"
            prepare(cfg,"core",["control_rlc"],prep)
            batch.mkdir()
            manifest = dict(config=cfg,plan=plan(cfg,"core",["control_rlc"]),hashes=source_hashes(cfg),
                            prepared_manifest=json.loads((prep/"manifest.json").read_text(encoding="utf-8")))
            dump(batch/"manifest.json",manifest)
            with patch("pymoo.optimize.minimize",new=fake_minimize):
                rec = one((t,33000,cfg,str(batch)))
            self.assertEqual(rec["status"],"complete",rec)
            result = analyze(batch,prep,out,"all",True,True)
            self.assertTrue(result["passed"],result)
            rows = (out/"details.jsonl").read_text().splitlines()
            self.assertEqual(len(rows),4)
            self.assertTrue((out/"REPORT.md").exists())
            with np.load(out/t["id"]/"stage_0020_3d_data.npz") as xyz:
                np.testing.assert_allclose(xyz["curves"][0],xyz["target"],rtol=1e-10,atol=1e-9)
            from protocol import verify_job
            job = batch/t["id"]/'seed_33000'
            verify_job(job,cfg)
            (job/'generation_0020.npz').write_bytes(b"corrupt static fixture")
            with self.assertRaises(RuntimeError):
                verify_job(job,cfg)

    def test_09_similar_scores_different_shapes(self):
        from analyze import population_check
        t = targets(CFG,ids=["control_rlc"])[0]
        def rec(height,width):
            return dict(stable=True,objective=[1.,.01],phase_events=[],events=[dict(event_id="P1",matched=True,complete=False,
                        width_evaluable=True,height_delta_db=height,width_ratio=width)])
        out,pairs = population_check(t,33000,300,{0:(rec(-8.,.5),True),1:(rec(0.,1.),True)},CFG)
        self.assertFalse(out["any_joint_amp_complete"])
        self.assertTrue(all(p["has_meaningful_shape_difference"] for p in pairs))
        self.assertEqual(out["valid_candidates"],2)
        _,none = population_check(t,33000,300,{0:(rec(-8.,.5),False),1:(rec(0.,1.),True)},CFG)
        self.assertTrue(all(not p["has_meaningful_shape_difference"] for p in none))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--config", type=Path)
    opt = parser.parse_args()
    CFG = read_config(opt.config)
    import time
    start = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NumericalChecks))
    from importlib.metadata import version
    if opt.report:
        opt.report.parent.mkdir(parents=True,exist_ok=True)
        dump(opt.report,dict(passed=result.wasSuccessful(),checks=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                            seconds=time.perf_counter()-start,hashes=source_hashes(CFG),
                            versions={n:version(n) for n in ("numpy","scipy","pymoo","matplotlib","moocore")},
                            python=sys.version,optimization_run=False,note="Only numerical evaluations and mocked optimizer lifecycle; no optimization search."))
    sys.exit(0 if result.wasSuccessful() else 1)

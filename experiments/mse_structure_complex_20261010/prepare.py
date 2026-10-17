"""Prepare analytic targets only. Does not import or invoke an optimizer."""
import argparse
import numpy as np
from protocol import read_config, targets, dump, source_hashes, describe
from sampling import Quadrature
from diagnostics import events, match, phase_events, phase_match
from schematic import schematic


def prepare(cfg, profile, ids, root):
    from pathlib import Path
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="preparing", config=cfg, profile=profile, ids=ids, hashes=source_hashes(cfg))
    dump(root / "manifest.json", manifest)
    ts = targets(cfg, profile, ids)
    for t in ts:
        c = t["circuit"]
        q = Quadrature(c, cfg)
        previous = None
        stable = False
        for level in cfg["audit"]["refinements"]:
            x = q.reference_grid(refinement=level)
            z, dz = c.response(10**x)
            ev = events(x, z, cfg)
            pe = phase_events(x, z, dz, cfg)
            if previous is not None:
                old_ev, old_pe = previous
                rows, extra = match(old_ev, ev, cfg)
                prows, pextra = phase_match(old_pe, pe, cfg)
                stable = (not extra and len(rows) == len(ev) and not pextra and len(prows) == len(pe)
                          and all(r["matched"] and abs(r["shift_decades"]) < .001 and abs(r["height_delta_db"]) < .1
                                  and (r["width_ratio"] is not None and abs(r["width_ratio"]-1) < .01 if r["width_evaluable"] else True) for r in rows)
                          and all(r["matched"] and abs(r["shift_decades"]) < .001 and abs(r["width_ratio"]-1) < .01 and abs(r["slope_ratio"]-1) < .01 for r in prows))
                if stable:
                    break
            previous = (ev, pe)
        if not stable:
            raise RuntimeError(f'Target event catalog did not converge: {t["id"]}')
        np.testing.assert_allclose(q.score(np.log10(c.p)), 0, atol=1e-20)
        sample_f = np.geomspace(*cfg["band_hz"], 31)
        direct, _ = c.response(sample_f)
        np.testing.assert_allclose(direct, c.nodal_response(sample_f), rtol=1e-8, atol=1e-9)
        d = root / t["id"]
        d.mkdir(parents=True)
        dump(d / "target.json", dict(**describe(t), events=ev, phase_events=pe, catalog_stable=stable, refinement=level,
                                     note="Event labels are local to this target, not causal identities across interventions."))
        np.savez_compressed(d / "target_curve.npz", frequency_hz=10**x, real_ohm=z.real, imag_ohm=z.imag,
                            magnitude_db=20*np.log10(abs(z)), phase_rad=np.angle(z))
        (d / "circuit.svg").write_text(schematic(c), encoding="utf-8")
    # Record base-to-intervention event correspondences; heuristic matching may fail.
    import json
    for t in ts:
        if t["intervention"] == "base":
            continue
        base = json.loads((root / t["base"].ident / "base" / "target.json").read_text(encoding="utf-8"))["events"]
        ev = json.loads((root / t["id"] / "target.json").read_text(encoding="utf-8"))["events"]
        correspondence, extra = match(base, ev, cfg)
        dump(root / t["id"] / "base_event_correspondence.json", dict(matches=correspondence, unmatched_changed_events=extra,
              caution="Frequency-based one-to-one matching only; disappearance/merge/creation not causally distinguished."))
    manifest["status"] = "prepared"
    manifest["target_hashes"] = {p.relative_to(root).as_posix(): __import__("hashlib").sha256(p.read_bytes()).hexdigest() for p in root.rglob("target.json")}
    dump(root / "manifest.json", manifest)
    return len(ts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--profile", choices=["core", "extended"], default="core")
    parser.add_argument("--circuits", nargs="+")
    parser.add_argument("--out", required=True)
    opt = parser.parse_args()
    print(f"Prepared {prepare(read_config(opt.config), opt.profile, opt.circuits, opt.out)} targets; optimization not run.")

"""Target-specific events; never assigns Case1's P3/V2 to other topologies."""
import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import linear_sum_assignment


def events(x, z, config):
    y = 20 * np.log10(abs(z))
    result = []
    for kind, sign, letter in [("peak", 1, "P"), ("valley", -1, "V")]:
        ix, pr = find_peaks(sign * y, prominence=config["events"]["prominence_db"])
        for j, i in enumerate(ix):
            width, left, right = None, None, None
            if pr["prominences"][j] >= 3:
                wp = peak_widths(sign * y, np.array([i]), rel_height=3 / pr["prominences"][j],
                                prominence_data=tuple(pr[k][j:j+1] for k in ["prominences", "left_bases", "right_bases"]))
                left, right = [float(np.interp(v, np.arange(len(x)), x)) for v in (wp[2][0], wp[3][0])]
                width = right - left
            result.append(dict(event_id=f"{letter}{j+1}", kind=kind, x=float(x[i]), frequency_hz=float(10**x[i]),
                               height_db=float(y[i]), prominence_db=float(pr["prominences"][j]),
                               width_3db_decades=width, width_3db_hz=float(10**right - 10**left) if width else None,
                               left_3db_x=left, right_3db_x=right))
    return result


def match(target, fit, config):
    rule = config["events"]
    rows, used = [], set()
    for kind in ["peak", "valley"]:
        tt = [a for a in target if a["kind"] == kind]
        ff = [b for b in fit if b["kind"] == kind]
        cost = np.full((len(tt), len(ff) + len(tt)), rule["match_decades"] + 1e-6)
        for i, a in enumerate(tt):
            for j, b in enumerate(ff):
                shift = abs(a["x"] - b["x"])
                cost[i, j] = shift if shift <= rule["match_decades"] else 1e6
        ir, jr = linear_sum_assignment(cost)
        pairs = {i: j for i, j in zip(ir, jr) if j < len(ff) and cost[i, j] <= rule["match_decades"]}
        for i, a in enumerate(tt):
            b = ff[pairs[i]] if i in pairs else None
            if b:
                used.add(b["event_id"])
            shift = b["x"] - a["x"] if b else None
            height = b["height_db"] - a["height_db"] if b else None
            ratio = b["width_3db_decades"] / a["width_3db_decades"] if b and b["width_3db_decades"] and a["width_3db_decades"] else None
            width_ok = ratio is not None and 1 / rule["width_ratio"] <= ratio <= rule["width_ratio"]
            complete = b is not None and abs(shift) <= rule["position_decades"] and abs(height) <= rule["height_db"] and width_ok
            rows.append(dict(event_id=a["event_id"], kind=kind, matched=b is not None, complete=bool(complete),
                             fit_event_id=b["event_id"] if b else None, shift_decades=shift, height_delta_db=height,
                             width_ratio=ratio, width_evaluable=a["width_3db_decades"] is not None,
                             target_frequency_hz=a["frequency_hz"], target_width_hz=a["width_3db_hz"]))
    return rows, [b["event_id"] for b in fit if b["event_id"] not in used]


def event_signature(rows, extras):
    return tuple((r["event_id"], r["matched"], r["complete"]) for r in rows), tuple(extras)


def phase_events(x, z, dz, config):
    """Analytic phase slope peaks; widths are half-height slope widths, not 3 dB."""
    slope = np.rad2deg(np.log(10) * np.imag(dz / z))
    cfg = config["phase_events"]
    result = []
    for sign in (1, -1):
        ix, props = find_peaks(sign * slope, height=cfg["min_slope_deg_per_decade"], prominence=cfg["prominence_deg_per_decade"])
        for j, i in enumerate(ix):
            wp = peak_widths(sign * slope, np.array([i]), rel_height=.5,
                            prominence_data=tuple(props[k][j:j+1] for k in ["prominences", "left_bases", "right_bases"]))
            left, right = [float(np.interp(v, np.arange(len(x)), x)) for v in (wp[2][0], wp[3][0])]
            result.append(dict(event_id=f'{"U" if sign == 1 else "D"}{j+1}', sign=sign, x=float(x[i]), frequency_hz=float(10**x[i]),
                               slope_deg_per_decade=float(slope[i]), width_decades=right-left))
    return result


def phase_match(target, fit, config):
    rows, used = [], set()
    rule = config["events"]
    for sign in (1, -1):
        tt, ff = [[v for v in vs if v["sign"] == sign] for vs in (target, fit)]
        cost = np.full((len(tt), len(ff)+len(tt)), rule["match_decades"] + 1e-6)
        for i, a in enumerate(tt):
            for j, b in enumerate(ff):
                dist = abs(a["x"] - b["x"])
                cost[i,j] = dist if dist <= rule["match_decades"] else 1e6
        ir, jr = linear_sum_assignment(cost)
        mapping = {i: j for i,j in zip(ir,jr) if j < len(ff) and cost[i,j] <= rule["match_decades"]}
        for i, a in enumerate(tt):
            b = ff[mapping[i]] if i in mapping else None
            if b:
                used.add(b["event_id"])
            shift = b["x"] - a["x"] if b else None
            width = b["width_decades"] / a["width_decades"] if b and a["width_decades"] > 0 else None
            slope = abs(b["slope_deg_per_decade"] / a["slope_deg_per_decade"]) if b else None
            complete = b is not None and abs(shift) <= rule["position_decades"] and width is not None and .5 <= width <= 2 and .5 <= slope <= 2
            rows.append(dict(event_id=a["event_id"], matched=b is not None, complete=bool(complete), shift_decades=shift,
                             width_ratio=width, slope_ratio=slope, target_frequency_hz=a["frequency_hz"]))
    return rows, [b["event_id"] for b in fit if b["event_id"] not in used]


def diagnostic(quadrature, logp, target, target_phase=None):
    """Refine independent grids; instability remains visible and blocks acceptance."""
    cfg, previous, stable = quadrature.config, None, False
    for level in cfg["audit"]["refinements"]:
        score, x, z, t = quadrature.reference(logp, level)
        fit = events(x, z, cfg)
        rows, extra = match(target, fit, cfg)
        _, dz = quadrature.c.response(10**x, 10**logp)
        phase_rows, phase_extra = phase_match(target_phase or [], phase_events(x, z, dz, cfg), cfg)
        signature = (event_signature(rows, extra), tuple((r["event_id"],r["matched"],r["complete"]) for r in phase_rows), tuple(phase_extra))
        if previous is not None:
            old_score, old_signature = previous
            stable = signature == old_signature and np.all(abs(score - old_score) <= np.array(cfg["audit"]["objective_atol"]) + cfg["audit"]["objective_rtol"] * abs(score))
            if stable:
                break
        previous = (score, signature)
    uniform_mass = np.diff(x) / (x[-1] - x[0])
    amp = (20 * np.log10(abs(z) / abs(t)))**2
    pha = (np.angle(z) - np.angle(t))**2
    uniform = np.array([((v[:-1] + v[1:]) / 2) @ uniform_mass for v in (amp, pha)])
    # Core windows use actual left/right 3 dB crossings; absent widths stay null.
    local = []
    for a in target:
        if a["left_3db_x"] is None:
            local.append(dict(event_id=a["event_id"], mass=None, amp_contribution=None, pha_contribution=None,
                              amp_fraction=None, pha_fraction=None, phase_rmse_deg=None))
            continue
        lo, hi = a["left_3db_x"], a["right_3db_x"]
        xx = np.unique(np.r_[lo, x[(x > lo) & (x < hi)], hi])
        zz, _ = quadrature.c.response(10**xx, 10**logp)
        tt, _ = quadrature.c.response(10**xx)
        cell = np.clip(np.searchsorted(quadrature.edges, (xx[:-1] + xx[1:]) / 2, side="right") - 1, 0, len(quadrature.edges) - 2)
        mass = np.diff(xx) / np.diff(quadrature.edges)[cell] / (len(quadrature.edges) - 1)
        av, pv = (20 * np.log10(abs(zz) / abs(tt)))**2, (np.angle(zz) - np.angle(tt))**2
        ac, pc = [float(((v[:-1] + v[1:]) / 2) @ mass) for v in (av, pv)]
        local.append(dict(event_id=a["event_id"], mass=float(mass.sum()), amp_contribution=ac, pha_contribution=pc,
                          amp_fraction=ac / score[0] if score[0] > 1e-12 else None,
                          pha_fraction=pc / score[1] if score[1] > 1e-12 else None,
                          phase_rmse_deg=float(np.rad2deg(np.sqrt(pc / mass.sum())))))
    bands = []
    for lo, hi in zip(np.linspace(x[0], x[-1], 9)[:-1], np.linspace(x[0], x[-1], 9)[1:]):
        xx = np.unique(np.r_[lo, x[(x > lo) & (x < hi)], hi])
        zz, _ = quadrature.c.response(10**xx, 10**logp)
        tt, _ = quadrature.c.response(10**xx)
        mass = np.diff(xx) / (hi - lo)
        av, pv = (20*np.log10(abs(zz)/abs(tt)))**2, (np.angle(zz)-np.angle(tt))**2
        mse = [float(((v[:-1]+v[1:])/2) @ mass) for v in (av,pv)]
        bands.append(dict(low_hz=float(10**lo), high_hz=float(10**hi), amplitude_rmse_db=float(np.sqrt(mse[0])),
                          phase_rmse_deg=float(np.rad2deg(np.sqrt(mse[1])))))
    return dict(stable=bool(stable), refinement=level, objective=score.tolist(), uniform_log_objective=uniform.tolist(),
                amplitude_rmse_db=float(np.sqrt(uniform[0])), phase_rmse_deg=float(np.rad2deg(np.sqrt(uniform[1]))),
                events=rows, extra_events=extra, phase_events=phase_rows, extra_phase_events=phase_extra, local=local, bands=bands), (x, z, t)

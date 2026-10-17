"""General analytic-circuit quadrature, preserving the old 180 cell masses."""
import numpy as np


def anchors(circuit, parameters, lo, hi, density=1):
    chunks = []
    for fc, width in circuit.modes(parameters):
        if fc <= 0 or not np.isfinite(fc):
            raise FloatingPointError("Invalid root anchor")
        center = np.log10(fc)
        if center + .5 < lo or center - .5 > hi:
            continue
        local = np.linspace(-width, width, 17 * density)
        tails = np.geomspace(width, max(width, .5), 17 * density)
        chunks.append(np.clip(center + np.r_[local, -tails, tails], lo, hi))
    return np.concatenate(chunks) if chunks else np.empty(0)


class Quadrature:
    def __init__(self, circuit, config, tolerance_factor=1):
        self.c = circuit
        self.config = config
        self.tolerance_factor = tolerance_factor
        original = np.linspace(*np.log10(config["band_hz"]), config["original_points"])
        self.edges = np.r_[original[0], (original[:-1] + original[1:]) / 2, original[-1]]
        self.base = np.unique(np.r_[original, self.edges, anchors(circuit, circuit.p, *original[[0, -1]])])
        self.queries = 0
        self.evaluations = 0
        self.max_points = 0
        self.max_layers = 0

    def mass(self, x):
        cell = np.clip(np.searchsorted(self.edges, (x[1:] + x[:-1]) / 2, side="right") - 1, 0, len(self.edges) - 2)
        m = np.diff(x) / np.diff(self.edges)[cell] / (len(self.edges) - 1)
        if abs(m.sum() - 1) > 1e-10:
            raise AssertionError("Frequency weights do not sum to one")
        return m

    def query(self, x, p):
        z, _ = self.c.response(10**x, p)
        t, _ = self.c.response(10**x)
        self.queries += len(x)
        return np.array([20 * np.log10(abs(z)), np.angle(z), 20 * np.log10(abs(t)), np.angle(t)])

    def score_one(self, logp):
        p = 10**np.asarray(logp)
        limits = self.config["adaptive"]
        x = np.unique(np.r_[self.base, anchors(self.c, p, self.edges[0], self.edges[-1])])
        y = self.query(x, p)
        allx, ally = [x], [y]
        a, b, ya, yb = x[:-1], x[1:], y[:, :-1], y[:, 1:]
        count, depth = len(x), 0
        while len(a):
            depth += 1
            if depth > limits["max_depth"]:
                raise RuntimeError("Adaptive depth exhausted; no score accepted")
            mid = (a + b) / 2
            resolvable = (mid > a) & (mid < b)
            if not np.all(resolvable):
                # Near-identical pole/zero anchors may be one float apart.
                # Existing endpoint quadrature retains their mass; only stop
                # when an unresolved interval has a significant value jump.
                jump = abs(yb[:, ~resolvable] - ya[:, ~resolvable])
                if np.any(jump[[0,2]] > limits["amplitude_db"] * self.tolerance_factor) or np.any(jump[[1,3]] > np.deg2rad(limits["phase_deg"]) * self.tolerance_factor):
                    raise RuntimeError("Frequency resolution exhausted with significant response jump")
                a,b,ya,yb,mid = a[resolvable],b[resolvable],ya[:,resolvable],yb[:,resolvable],mid[resolvable]
                if not len(mid):
                    break
            ym = self.query(mid, p)
            count += len(mid)
            if count > limits["max_points"]:
                raise RuntimeError("Adaptive point budget exhausted; no score accepted")
            allx.append(mid)
            ally.append(ym)
            error = abs(ym - (ya + yb) / 2)
            amp = limits["amplitude_db"] * self.tolerance_factor
            phase = np.deg2rad(limits["phase_deg"]) * self.tolerance_factor
            fail = (error[0] > amp) | (error[2] > amp) | (error[1] > phase) | (error[3] > phase)
            a, b, ya, yb = (np.r_[a[fail], mid[fail]], np.r_[mid[fail], b[fail]],
                            np.column_stack([ya[:, fail], ym[:, fail]]), np.column_stack([ym[:, fail], yb[:, fail]]))
        x, y = np.concatenate(allx), np.concatenate(ally, axis=1)
        order = np.argsort(x)
        x, y = x[order], y[:, order]
        loss = np.array([(y[0] - y[2])**2, (y[1] - y[3])**2])
        self.evaluations += 1
        self.max_points = max(count, self.max_points)
        self.max_layers = max(depth, self.max_layers)
        return ((loss[:, :-1] + loss[:, 1:]) / 2) @ self.mass(x)

    def score(self, log_parameters):
        return np.array([self.score_one(p) for p in np.atleast_2d(log_parameters)])

    def reference_grid(self, p=None, refinement=1):
        lo, hi = self.edges[[0, -1]]
        values = [self.c.p] if p is None else [self.c.p, p]
        return np.unique(np.r_[np.linspace(lo, hi, 2048 * refinement + 1), self.edges,
                               *[anchors(self.c, v, lo, hi, 24 * refinement) for v in values]])

    def reference(self, logp, refinement=1):
        p = 10**np.asarray(logp)
        x = self.reference_grid(p, refinement)
        z, _ = self.c.response(10**x, p)
        t, _ = self.c.response(10**x)
        errors = np.array([(20 * np.log10(abs(z) / abs(t)))**2, (np.angle(z) - np.angle(t))**2])
        score = ((errors[:, :-1] + errors[:, 1:]) / 2) @ self.mass(x)
        return score, x, z, t

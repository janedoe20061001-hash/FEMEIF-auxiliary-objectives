"""Frozen refinements, with the two unchanged LP metrics as replication controls."""
import numpy as np
from curve_math import area, arc_knots, arc_distance

CATALOG = [('D-03-LP', 'area_logpolar', '原版：终点有向面积'),
           ('D-05-LP', 'arclength_logpolar', '原版：归一化弧长对应形状'),
           ('R-03-C', 'cumulative_area_logpolar', '改进：全过程累计有向面积'),
           ('R-05-C', 'cumulative_length_logpolar', '改进：保留总长度的累计弧长'),
           ('R-03-S', 'area_stdpolar', '尺度对照：标准差坐标的有向面积'),
           ('R-05-S', 'arclength_stdpolar', '尺度对照：标准差坐标的弧长形状')]
AUX = tuple(row[1] for row in CATALOG)
PARENTS = {AUX[2]: AUX[0], AUX[3]: AUX[1], AUX[4]: AUX[0], AUX[5]: AUX[1]}
DEFAULTS = {}

def squared_linear_integral(e, h):
    a, b = e[..., :-1, :], e[..., 1:, :]
    return np.sum(h / 3 * np.sum(a*a + a*b + b*b, axis=-1), axis=-1)

def cumulative_area(p):
    r = p - p[..., :1, :]
    increments = .5 * np.cross(r[..., :-1, :], r[..., 1:, :])
    return np.concatenate([np.zeros_like(p[..., :1, :]),
                           np.cumsum(increments, axis=-2)], axis=-2)

def cumulative_length(p):
    increments = np.linalg.norm(np.diff(p, axis=-2), axis=-1)
    return np.concatenate([np.zeros_like(increments[..., :1]),
                           np.cumsum(increments, axis=-1)], axis=-1)

class GlobalObjectives:
    def __init__(self, frequency, settings=None, aux=(), transform='loss', epsilon=None, target_z=None):
        self.f = np.asarray(frequency, float); self.z = np.asarray(target_z, complex)
        if self.f.ndim != 1 or len(self.f) < 2 or np.any(self.f <= 0) or np.any(np.diff(self.f) <= 0):
            raise ValueError('Positive increasing frequencies required')
        if self.z.shape != self.f.shape or not np.isfinite(self.z).all() or np.any(abs(self.z) == 0):
            raise ValueError('Finite nonzero target required')
        if settings not in (None, {}) or transform != 'loss' or any(a not in AUX for a in aux) or len(set(aux)) != len(aux):
            raise ValueError('Frozen refinement protocol only')
        self.aux = tuple(aux)
        x = np.log10(self.f); self.u = (x-x[0])/(x[-1]-x[0]); self.du = np.diff(self.u)
        self.a = 20*np.log10(np.maximum(abs(self.z), 1e-300)); self.phase = np.angle(self.z)
        xy = np.column_stack([self.a, self.phase])
        mu = np.sum(self.du[:, None] * (xy[:-1]+xy[1:])/2, axis=0)
        std = np.sqrt(squared_linear_integral((xy-mu).T[..., None], self.du))
        self.references = {}
        for coord, offset, scale in [('logpolar', xy.min(axis=0), np.ptp(xy, axis=0)),
                                     ('stdpolar', mu, std)]:
            if np.any(scale <= 0): raise ValueError('Degenerate target; no epsilon substitution')
            p = np.column_stack([self.u, (xy-offset)/scale])
            length = cumulative_length(p)
            self.references[coord] = dict(offset=offset, scale=scale, p=p, area=area(p),
                s=arc_knots(p), cumarea=cumulative_area(p), cumlength=length)

    def target_summary(self):
        return dict(frequency_points=len(self.f), frequency_coordinate='normalized_log10_frequency',
            coordinates={k:dict(offset=v['offset'].tolist(),scale=v['scale'].tolist()) for k,v in self.references.items()},
            phase='principal_angle_unchanged_no_unwrap', candidate_normalization='fixed_target_only',
            all_original_rows_retained=True, auxiliary_tunable_parameters={},
            standard_deviation='exact_continuous_log_frequency_piecewise_linear_integral')

    def path(self, z, coord='logpolar'):
        if np.any(abs(z) == 0): raise ValueError('Zero outside log-coordinate domain')
        xy = np.stack([20*np.log10(abs(z)), np.angle(z)], axis=-1)
        ref = self.references[coord]; xy = (xy-ref['offset'])/ref['scale']
        return np.concatenate([np.broadcast_to(self.u, xy.shape[:-1])[..., None], xy], axis=-1)

    def evaluate_spectra(self, z):
        z = np.atleast_2d(np.asarray(z, complex))
        if z.shape[1:] != self.z.shape or not np.isfinite(z).all(): raise ValueError('Invalid spectrum')
        output = [np.mean((20*np.log10(np.maximum(abs(z), 1e-300))-self.a)**2, axis=1),
                  np.mean((np.angle(z)-self.phase)**2, axis=1)]
        paths = {c:self.path(z,c) for c in {n.rsplit('_',1)[1] for n in self.aux}}
        for name in self.aux:
            family, coord = name.rsplit('_', 1); p = paths[coord]; ref = self.references[coord]
            if family == 'area': value = np.linalg.norm(area(p)-ref['area'], axis=-1)
            elif family == 'arclength': value = np.array([arc_distance(row,ref['p'],ref['s']) for row in p])
            elif family == 'cumulative_area':
                value = np.sqrt(squared_linear_integral(cumulative_area(p)-ref['cumarea'], self.du))
            elif family == 'cumulative_length':
                e = (cumulative_length(p)-ref['cumlength']) / ref['cumlength'][-1]
                value = np.sqrt(squared_linear_integral(e[..., None], self.du))
            else: raise AssertionError(family)
            output.append(value)
        result = np.column_stack(output)
        if not np.isfinite(result).all() or np.any(result < 0): raise FloatingPointError('Invalid objectives')
        return result

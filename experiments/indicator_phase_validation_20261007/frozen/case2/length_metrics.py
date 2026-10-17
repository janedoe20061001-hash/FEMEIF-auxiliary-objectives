"""Two frozen length refinements, with an unchanged cumulative-length control."""
import numpy as np

LENGTH = 'cumulative_length_logpolar'
INTERVAL = 'interval_length_logpolar'
HELLINGER = 'hellinger_length_logpolar'
CATALOG = [('R-05-C', LENGTH, '对照：累计弧长'),
           ('G-05-I', INTERVAL, '全区间弧长误差'),
           ('G-05-H', HELLINGER, '逐段弧长平方根误差')]
AUX = tuple(row[1] for row in CATALOG)
GROUPS = [()] + [(name,) for name in AUX]
COMPARISONS = [(INTERVAL, LENGTH), (HELLINGER, LENGTH)]
DEFAULTS = {}


def integral_squared(e, h):
    a, b = e[..., :-1, :], e[..., 1:, :]
    return np.sum(h/3*np.sum(a*a+a*b+b*b, axis=-1), axis=-1)


def cumulative_length(p):
    d = np.linalg.norm(np.diff(p, axis=-2), axis=-1)
    return np.concatenate([np.zeros_like(d[..., :1]), np.cumsum(d, axis=-1)], axis=-1)


def interval_loss(e, h):
    mean = np.sum(h*(e[..., :-1]+e[..., 1:])/2, axis=-1)
    centered = e - mean[..., None]
    # Center before integration: avoids subtracting two nearly equal moments.
    return np.sqrt(2*integral_squared(centered[..., None], h))


def hellinger_loss(candidate_segments, target_segments):
    return np.sqrt(np.sum((np.sqrt(candidate_segments)-np.sqrt(target_segments))**2, axis=-1)
                   / np.sum(target_segments))


class GlobalObjectives:
    def __init__(self, frequency, settings=None, aux=(), transform='loss', epsilon=None, target_z=None):
        self.f = np.asarray(frequency, float)
        self.z = np.asarray(target_z, complex)
        if self.f.ndim != 1 or len(self.f) < 2 or np.any(self.f <= 0) or np.any(np.diff(self.f) <= 0):
            raise ValueError('Positive increasing frequency required')
        if self.z.shape != self.f.shape or not np.isfinite(self.z).all() or np.any(abs(self.z) == 0):
            raise ValueError('Nonzero finite target required')
        if settings not in (None, {}) or transform != 'loss' or tuple(aux) not in GROUPS:
            raise ValueError('Frozen baseline and singleton groups only')
        self.aux = tuple(aux)
        x = np.log10(self.f)
        self.u = (x-x[0])/(x[-1]-x[0])
        self.du = np.diff(self.u)
        self.a = 20*np.log10(np.maximum(abs(self.z), 1e-300))
        self.phase = np.angle(self.z)
        xy = np.column_stack([self.a, self.phase])
        self.offset = xy.min(axis=0)
        self.scale = np.ptp(xy, axis=0)
        if np.any(self.scale <= 0):
            raise ValueError('Degenerate target axes; no epsilon substitution')
        self.p = np.column_stack([self.u, (xy-self.offset)/self.scale])
        self.cl = cumulative_length(self.p)
        self.length = self.cl[-1]
        self.segments = np.linalg.norm(np.diff(self.p, axis=0), axis=-1)

    def path(self, z):
        if np.any(abs(z) == 0):
            raise ValueError('Zero outside log-coordinate domain')
        xy = (np.stack([20*np.log10(abs(z)), np.angle(z)], axis=-1)-self.offset)/self.scale
        return np.concatenate([np.broadcast_to(self.u, xy.shape[:-1])[..., None], xy], axis=-1)

    def target_summary(self):
        k = int(np.argmax(self.segments))
        return dict(frequency_points=len(self.f), frequency_coordinate='normalized_log10_frequency',
            magnitude_phase_coordinates='unchanged_target_range_principal_phase', offset=self.offset.tolist(),
            scale=self.scale.tolist(), target_total_length=float(self.length),
            largest_segment_length_share=float(self.segments[k]/self.length),
            largest_segment_frequency_hz=self.f[k:k+2].tolist(), all_original_rows_retained=True,
            candidate_normalization='fixed_target_only', auxiliary_tunable_parameters={},
            interval_comparison='all_continuous_frequency_pairs_exact_integral',
            hellinger_comparison='unnormalized_length_measure_target_total_scale_no_direction_term')

    def evaluate_spectra(self, z):
        z = np.atleast_2d(np.asarray(z, complex))
        if z.shape[1:] != self.z.shape or not np.isfinite(z).all():
            raise ValueError('Invalid spectrum')
        output = [np.mean((20*np.log10(np.maximum(abs(z), 1e-300))-self.a)**2, axis=1),
                  np.mean((np.angle(z)-self.phase)**2, axis=1)]
        if self.aux:
            p = self.path(z)
        for name in self.aux:
            if name in (LENGTH, INTERVAL):
                e = (cumulative_length(p)-self.cl)/self.length
                value = np.sqrt(integral_squared(e[..., None], self.du)) if name == LENGTH else interval_loss(e, self.du)
            elif name == HELLINGER:
                value = hellinger_loss(np.linalg.norm(np.diff(p, axis=-2), axis=-1), self.segments)
            else:
                raise AssertionError(name)
            output.append(value)
        result = np.column_stack(output)
        if not np.isfinite(result).all() or np.any(result < 0):
            raise FloatingPointError('Invalid objectives')
        return result

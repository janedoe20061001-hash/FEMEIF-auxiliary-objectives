"""Measured target and the 15-component network in the English paper, Fig. 15."""
from pathlib import Path
import hashlib
import numpy as np

HERE = Path(__file__).resolve().parent
PARAMETER_NAMES = tuple([f"R{i}" for i in range(1, 8)] +
                        [f"L{i}" for i in range(1, 5)] +
                        [f"C{i}" for i in range(1, 5)])
LOWER = np.array([-2.] * 7 + [-9.] * 4 + [-12.] * 4)
UPPER = np.array([4.5] * 7 + [-4.] * 4 + [-7.] * 4)
N_VAR = len(PARAMETER_NAMES)


def load_target(spec):
    path = HERE / spec["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError("Target CSV changed; refuse to mix data versions")
    if (spec["magnitude_unit"], spec["phase_unit"], spec["sampling"]) != ("ohm", "degree", "all_rows"):
        raise ValueError("Case2 protocol uses every original row, ohms and degrees")
    data = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8-sig")
    if data.dtype.names != ("frequency", "mag", "pha") or data.ndim != 1:
        raise ValueError("Expected frequency,mag,pha CSV columns")
    f, mag, phase = (data[k] for k in data.dtype.names)
    if (len(f) != spec["points"] or len(f) < 5 or
            not np.isfinite(np.column_stack([f, mag, phase])).all() or
            np.any(f <= 0) or np.any(np.diff(f) <= 0) or np.any(mag <= 0) or
            np.any(abs(phase) > 180)):
        raise ValueError("Invalid measured target; rows are never silently removed or sorted")
    return f, mag * np.exp(1j * np.deg2rad(phase))


def impedance(frequency, parameters):
    """Three [C || (L + R_odd) || R_even] blocks, then series C4,R7,L4."""
    p = np.atleast_2d(np.asarray(parameters, float))
    f = np.asarray(frequency, float)
    if (p.ndim != 2 or p.shape[1] != N_VAR or not np.isfinite(p).all() or
            np.any(p <= 0) or f.ndim != 1 or not np.isfinite(f).all() or np.any(f <= 0)):
        raise ValueError("Expected positive finite 15-component parameters and frequencies")
    w = 2 * np.pi * f[None, :]
    r, l, c = p[:, :7], p[:, 7:11], p[:, 11:15]
    z = np.zeros((len(p), len(f)), complex)
    for k in range(3):
        y = 1j * w * c[:, k, None] + 1 / (r[:, 2*k, None] + 1j*w*l[:, k, None]) + 1/r[:, 2*k+1, None]
        z += 1/y
    return z + 1/(1j*w*c[:, 3, None]) + r[:, 6, None] + 1j*w*l[:, 3, None]

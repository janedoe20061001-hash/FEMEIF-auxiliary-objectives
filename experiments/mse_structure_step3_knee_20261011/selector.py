"""Common two-objective selector, frozen before replacement results are inspected."""
import numpy as np

def nondominated(v):
    dominates = np.all(v[:, None, :] <= v[None, :, :], axis=2) & np.any(v[:, None, :] < v[None, :, :], axis=2)
    return np.flatnonzero(~dominates.any(axis=0))

def knee(v):
    v = np.asarray(v, dtype=float)
    assert v.ndim == 2 and v.shape[1] == 2 and len(v) and np.isfinite(v).all()
    front = nondominated(v)
    _, first = np.unique(v[front], axis=0, return_index=True)
    indices = front[first]; values = v[indices]
    low = values.min(axis=0); high = values.max(axis=0); span = high-low
    tolerance = 1e-14*np.maximum(1, np.maximum(abs(low), abs(high)))
    normalized = (values-low)/np.maximum(span, tolerance)
    ideal = np.sum(normalized**2, axis=1)
    distance = (1-normalized.sum(axis=1))/np.sqrt(2)
    if len(values) == 1: reason = 'unique_front_point'
    elif np.any(span <= tolerance): reason = 'degenerate_span'
    elif len(values) < 3: reason = 'insufficient_front_points'
    elif np.max(distance) <= 1e-12: reason = 'no_convex_knee'
    else: reason = 'convex_knee'
    if reason == 'convex_knee':
        eligible = np.flatnonzero(distance >= np.max(distance)-1e-12)
        local = eligible[np.lexsort((indices[eligible], ideal[eligible]))[0]]
    else: local = np.lexsort((indices, ideal))[0]
    return int(indices[local]), dict(reason=reason, front_indices=front.tolist(), unique_indices=indices.tolist(),
        minimum=low.tolist(), maximum=high.tolist(), normalized=normalized.tolist(), signed_distances=distance.tolist(),
        selected_index=int(indices[local]), distance=float(distance[local]))

def checks():
    v=np.array([[0.,1.],[.2,.2],[1.,0.],[.6,.6]])
    assert knee(v)[0] == 1
    assert knee(v*np.array([100,1e-3]))[0] == 1
    order=np.array([2,0,3,1]); assert order[knee(v[order])[0]] == 1
    assert knee(np.array([[.1,.1],[.1,.1],[.2,.3]]))[1]['reason'] == 'unique_front_point'
    assert knee(np.array([[0,1],[.5,.5],[1,0]]))[1]['reason'] == 'no_convex_knee'
    assert knee(np.array([[0,1],[1,0]]))[1]['reason'] == 'insufficient_front_points'


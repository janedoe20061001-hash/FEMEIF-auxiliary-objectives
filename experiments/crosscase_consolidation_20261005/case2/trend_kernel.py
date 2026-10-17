"""Continuous all-interval average residual, for piecewise linear node values.

K is a positive quadratic form. Same-panel triangles are exact; different-panel
rectangles use Gauss-Legendre quadrature. Quadrature order is a numerical setting,
not a smoothing or feature scale. All original frequency nodes are retained.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss


def kernel(u, order):
    u = np.asarray(u, float)
    if u[0] != 0 or u[-1] != 1 or np.any(np.diff(u) <= 0):
        raise ValueError('Need increasing unit-interval nodes')
    h = np.diff(u); n = len(u)
    g, w = leggauss(order); g = (g+1)/2; w = w/2
    prefix = np.zeros((n,n))
    for i in range(n-1):
        prefix[i+1] = prefix[i]
        prefix[i+1,i:i+2] += h[i]/2
    # Antiderivative of each hat basis at each quadrature point.
    values = np.repeat(prefix[:-1,None,:], order, axis=1)
    for i in range(n-1):
        values[i,:,i] += h[i]*(g-g*g/2)
        values[i,:,i+1] += h[i]*g*g/2
    x = u[:-1,None]+h[:,None]*g
    K = np.zeros((n,n))
    for i in range(n-1):
        K[i:i+2,i:i+2] += h[i]**2*np.array([[7.,5.],[5.,7.]])/24
        if i == n-2:
            continue
        # For a in panel i, b in each later panel. Include factor 2 for triangle.
        d = x[i+1:,None,:]-x[i,:,None]
        c = (values[i+1:,None,:,:]-values[i,:,None,:])/d[...,None]
        wt = 2*h[i]*h[i+1:,None,None]*w[None,:,None]*w[None,None,:]
        v = c.reshape(-1,n)*np.sqrt(wt.reshape(-1,1))
        K += v.T@v
    return (K+K.T)/2


def score(residual, K):
    r = np.asarray(residual, float)
    value = np.einsum('...ic,ij,...jc->...',r,K,r,optimize=True)
    if np.any(value < -1e-12):
        raise FloatingPointError('Negative quadratic form')
    return np.sqrt(np.maximum(value,0))


def cumulative_same_coordinates(residual, u):
    """G-01's trapezoidal cumulative formula, in frozen dB/phase coordinates.
    No outer square root, as in the historical G-01 implementation.
    """
    h = np.diff(u)
    A = np.concatenate([np.zeros_like(residual[...,:1,:]),
        np.cumsum(h[:,None]*(residual[...,:-1,:]+residual[...,1:,:])/2,axis=-2)],axis=-2)
    B = A[...,-1:,:]-A
    v = (np.sum(A*A,axis=-1)+np.sum(B*B,axis=-1))/2
    return np.sum(h*(v[...,:-1]+v[...,1:])/2,axis=-1)

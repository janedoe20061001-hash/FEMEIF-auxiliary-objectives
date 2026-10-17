"""Resolve optimizer-created extreme-Q modes and their long tails."""
import numpy as np
from diagnostics import network,pop

def mode_candidates(r,p):
    n=network(r,p);num,den=n.rational();modes=[]
    for coef in [num,den]:
        for root in np.roots(coef[::-1]):
            if abs(root)<1e-12 or root.imag < -1e-9*max(abs(root),1):continue
            center=abs(root)*pop.WREF/(2*np.pi)
            width=.5 if abs(root.imag)<1e-8*abs(root) else min(.5,max(6*abs(root.real)/abs(root.imag),1e-12))
            if np.isfinite(center) and np.isfinite(width):modes.append((center,width))
    # Simple RLC branch frequencies remain accurate when a high-order rational
    # polynomial becomes ill-conditioned. They also resolve widths below 1e-6.
    values=dict(zip(r['parameter_names'],p))
    def visit(node):
        if isinstance(node,str):return
        if len(node)==4 and all(isinstance(c,str) for c in node[1:]) and sorted(c[0] for c in node[1:])==['C','L','R']:
            v={c[0]:values[c] for c in node[1:]};fc=1/(2*np.pi*np.sqrt(v['L']*v['C']))
            q=v['R']*np.sqrt(v['C']/v['L']) if node[0]=='P' else np.sqrt(v['L']/v['C'])/v['R']
            modes.append((fc,min(.5,max(6/q,1e-12))))
        for child in node[1:]:visit(child)
    visit(r['tree']);return modes

def improved_dense_grid(r,parameters,refinement=1):
    lo,hi=np.log10(r['frequency_hz'][0]),np.log10(r['frequency_hz'][-1]);values=[np.linspace(lo,hi,2048*refinement+1),np.log10(r['frequency_hz']),np.linspace(lo,hi,9)]
    for p in [r['target_parameters'],*parameters]:
        for fc,width in mode_candidates(r,np.asarray(p)):
            c=np.log10(fc);dx=width/np.log(10)
            if not lo<=c<=hi:continue
            values.append(np.linspace(max(lo,c-dx),min(hi,c+dx),192*refinement+1))
            # Uniform neighborhoods alone connect an extreme-Q tail to the
            # distant background by a huge trapezoid. Log-spaced offsets close
            # that gap, preserving integrals without resampling objectives.
            offsets=np.geomspace(dx,max(dx,.5),96*refinement+1)
            for sign in [-1,1]:
                a=c+sign*offsets;values.append(a[(a>=lo)&(a<=hi)])
    return 10**np.unique(np.concatenate(values))

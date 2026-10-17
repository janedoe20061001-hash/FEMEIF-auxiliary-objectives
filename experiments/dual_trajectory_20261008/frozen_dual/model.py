"""Frozen scalar losses and fast passive-network evaluation."""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
def response(tree,names,f,parameters):
    p=np.atleast_2d(parameters);w=2*np.pi*np.asarray(f)[None,:];values={n:p[:,i,None] for i,n in enumerate(names)}
    def ev(node):
        if isinstance(node,str):
            v=values[node]
            return np.broadcast_to(v,(len(p),len(f))) if node.startswith('R') else (1j*w*v if node.startswith('L') else 1/(1j*w*v))
        if node[0]=='B':
            y=[1/ev(n) for n in node[1:]]
            det=sum(y[i]*y[j] for i,j in [(0,2),(0,3),(1,2),(1,3),(4,0),(4,1),(4,2),(4,3)])
            num=sum(y[i]*y[j]*y[k] for i,j,k in [(0,1,2),(0,1,3),(0,2,3),(1,2,3),(4,0,1),(4,0,3),(4,1,2),(4,2,3)])
            return det/num
        parts=[ev(c) for c in node[1:]]
        return sum(parts) if node[0]=='S' else 1/sum(1/z for z in parts)
    z=ev(tree)
    if not np.isfinite(z).all() or np.any(z.real<=0):raise FloatingPointError('Invalid passive response')
    return z
class Model:
    def __init__(self,ident):
        self.record=next(r for r in json.loads((HERE/'circuits.json').read_text(encoding='utf-8')) if r['id']==ident)
        r=self.record;self.f=np.asarray(r['frequency_hz']);self.lower=np.asarray(r['lower_log10']);self.upper=np.asarray(r['upper_log10']);self.alpha=r['alpha']
        self.target=np.asarray(r['target_real'])+1j*np.asarray(r['target_imag'])
        self.amp=20*np.log10(abs(self.target));self.phase=np.angle(self.target)
        if ident=='case1':
            self.network=module('scalar_network_case1',HERE/'frozen/case1/legacy_objectives.py').impedance
        elif ident=='case2':
            self.network=module('scalar_network_case2',HERE/'frozen/case2/case2.py').impedance
        else:self.network=lambda f,p:response(r['tree'],r['parameter_names'],f,p)
    def spectrum(self,x):return self.network(self.f,10**np.atleast_2d(x))
    def score_spectra(self,z):
        z=np.atleast_2d(z);a=np.mean((20*np.log10(abs(z))-self.amp)**2,axis=1);p=np.mean((np.angle(z)-self.phase)**2,axis=1)
        residual=z-self.target
        d=np.mean(np.sqrt(residual.real**2+residual.imag**2),axis=1)
        out=np.column_stack([a,p,a+self.alpha*p,d])
        if not np.isfinite(out).all() or np.any(out<0):raise FloatingPointError('Invalid loss')
        return out
    def score(self,x):return self.score_spectra(self.spectrum(x))

"""Three predeclared parameter-free auxiliary losses on relative complex residuals.

Integrals use the original log-frequency grid, trapezoidal quadrature.
G02 considers all intervals whose endpoints are original sample locations.
No model parameter truth is used in an auxiliary objective.
"""
import numpy as np
from scipy.spatial.distance import pdist

CATALOG=[('G-01','bidirectional_cumulative_relative','双向累积相对偏差'),
         ('G-02','max_interval_cumulative_relative','最坏连续频段累积偏差'),
         ('G-05','relative_complex_l2','复阻抗相对平方误差')]
AUX=tuple(x[1] for x in CATALOG)
DEFAULTS={}

class GlobalObjectives:
    def __init__(self,frequency,settings=None,aux=(),transform='loss',epsilon=None,target_z=None):
        self.f=np.asarray(frequency,float)
        self.z=np.asarray(target_z,complex)
        if self.f.ndim!=1 or len(self.f)<2 or np.any(self.f<=0) or np.any(np.diff(self.f)<=0):
            raise ValueError('Positive increasing frequency grid required')
        if self.z.shape!=self.f.shape or not np.isfinite(self.z).all() or np.any(abs(self.z)==0):
            raise ValueError('Finite nonzero target required; no arbitrary epsilon substitution')
        if settings not in (None,{}) or transform!='loss' or any(a not in AUX for a in aux) or len(set(aux))!=len(aux):
            raise ValueError('Predeclared global protocol only')
        self.aux=tuple(aux)
        x=np.log10(self.f);self.u=(x-x[0])/(x[-1]-x[0]);self.du=np.diff(self.u)
        self.a=20*np.log10(np.maximum(abs(self.z),1e-300));self.phase=np.angle(self.z)

    def target_summary(self):
        return dict(frequency_points=len(self.f),target_min_abs=float(abs(self.z).min()),
                    quadrature='trapezoidal_original_log_frequency_grid',
                    interval_endpoints='original_samples',auxiliary_tunable_parameters={})

    def integrate(self,v):
        return np.sum((v[...,:-1]+v[...,1:])*.5*self.du,axis=-1)

    def evaluate_spectra(self,z):
        z=np.atleast_2d(np.asarray(z,complex))
        if z.shape[1:]!=self.z.shape or not np.isfinite(z).all():
            raise ValueError('Invalid candidate spectrum')
        # Identical primary formulas to both historical protocols.
        output=[np.mean((20*np.log10(np.maximum(abs(z),1e-300))-self.a)**2,axis=1),
                np.mean((np.angle(z)-self.phase)**2,axis=1)]
        if self.aux:
            r=z/self.z-1
            need_c=any(a!='relative_complex_l2' for a in self.aux)
            if need_c:
                c=np.concatenate([np.zeros((len(z),1),complex),
                                  np.cumsum(.5*(r[:,:-1]+r[:,1:])*self.du,axis=1)],axis=1)
            for name in self.aux:
                if name=='bidirectional_cumulative_relative':
                    value=.5*self.integrate(abs(c)**2+abs(c[:,-1,None]-c)**2)
                elif name=='max_interval_cumulative_relative':
                    value=np.array([np.sqrt(pdist(np.column_stack([row.real,row.imag]),'sqeuclidean').max()) for row in c])
                else:
                    value=self.integrate(abs(r)**2)
                output.append(value)
        result=np.column_stack(output)
        if not np.isfinite(result).all() or np.any(result<0):
            raise FloatingPointError('Nonfinite or negative objective')
        return result

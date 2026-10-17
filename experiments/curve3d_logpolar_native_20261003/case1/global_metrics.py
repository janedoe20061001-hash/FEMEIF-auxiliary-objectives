"""Second exploratory round: 3D log-magnitude/phase coordinates.

Five existing geometric directions are re-tested in new coordinates, not
claimed as five novel mathematical principles. A sixth direction compares
the orientation of every frequency-pair chord. All choices are frozen.
"""
import numpy as np
from scipy.spatial.distance import pdist
from native_backend import weighted_terms,native_available
from curve_math import covariance,area,srvf,arc_knots,arc_distance

CATALOG=[('D-01-LP','chords_logpolar','全频率对距离'),
         ('D-02-LP','covariance_logpolar','整体坐标变化关系'),
         ('D-03-LP','area_logpolar','三平面有向面积'),
         ('D-04-LP','srvf_logpolar','走向与变化快慢'),
         ('D-05-LP','arclength_logpolar','按弧长对应的形状'),
         ('D-06-LP','orientation_logpolar','全频率对空间方向差')]
AUX=tuple(row[1] for row in CATALOG)
DEFAULTS={}

class GlobalObjectives:
    def __init__(self,frequency,settings=None,aux=(),transform='loss',epsilon=None,target_z=None):
        self.f=np.asarray(frequency,float);self.z=np.asarray(target_z,complex)
        if self.f.ndim!=1 or len(self.f)<2 or np.any(self.f<=0) or np.any(np.diff(self.f)<=0):raise ValueError('Invalid frequency')
        if self.z.shape!=self.f.shape or not np.isfinite(self.z).all() or np.any(abs(self.z)==0):raise ValueError('Invalid target')
        if settings not in (None,{}) or transform!='loss' or any(a not in AUX for a in aux) or len(set(aux))!=len(aux):raise ValueError('Frozen protocol only')
        self.aux=tuple(aux)
        x=np.log10(self.f);self.u=(x-x[0])/(x[-1]-x[0]);self.du=np.diff(self.u)
        self.a=20*np.log10(np.maximum(abs(self.z),1e-300));self.phase=np.angle(self.z)
        xy=np.column_stack([self.a,self.phase]);self.offset=xy.min(axis=0);self.extent=np.ptp(xy,axis=0)
        if np.any(self.extent<=0):raise ValueError('Degenerate target axes; no epsilon substitution')
        self.p=np.column_stack([self.u,(xy-self.offset)/self.extent])
        w=np.r_[self.du[0]/2,(self.du[:-1]+self.du[1:])/2,self.du[-1]/2]
        self.i,self.j=np.triu_indices(len(self.f),1);self.pair_weights=w[self.i]*w[self.j]
        self.d=pdist(self.p);self.chord_norm=np.sum(self.pair_weights*self.d*self.d)
        self.c=covariance(self.p,self.du);self.cov_norm=np.linalg.norm(self.c)
        self.area=area(self.p);self.q=srvf(self.p,self.du);self.s=arc_knots(self.p)
        self.length=np.sum(self.du*np.sum(self.q*self.q,axis=1))
        delta=self.p[self.j]-self.p[self.i];self.directions=delta/np.linalg.norm(delta,axis=1)[:,None]

    def target_summary(self):
        return dict(orientation_backend='native_c' if native_available else 'numpy_reference',frequency_points=len(self.f),frequency_coordinate='normalized_log10_frequency',
            magnitude_coordinate='20_log10_abs_Z_target_range',phase_coordinate='principal_angle_target_range_no_unwrap',
            offset=self.offset.tolist(),extent=self.extent.tolist(),
            target_negative_real_points=int(np.sum(self.z.real<0)),
            target_max_adjacent_phase_step_deg=float(np.rad2deg(abs(np.diff(self.phase))).max()),
            candidate_normalization='target_fixed_not_candidate_specific',
            all_original_rows_retained=True,auxiliary_tunable_parameters={},
            pair_direction_comparison='all_pairs_unit_vector_squared_difference')

    def path(self,z):
        if np.any(abs(z)==0):raise ValueError('Zero impedance outside log-coordinate domain; positive-R circuit cannot produce it')
        xy=np.stack([20*np.log10(abs(z)),np.angle(z)],axis=-1)
        xy=(xy-self.offset)/self.extent
        return np.concatenate([np.broadcast_to(self.u,xy.shape[:-1])[...,None],xy],axis=-1)

    def orientation_loss(self,p,chunk_size=32):
        result=[]
        for start in range(0,len(p),chunk_size):
            terms=weighted_terms(p[start:start+chunk_size],self.i,self.j,self.directions,self.pair_weights)
            result.extend(np.sum(terms,axis=-1)/self.pair_weights.sum())
        return np.asarray(result)

    def evaluate_spectra(self,z):
        z=np.atleast_2d(np.asarray(z,complex))
        if z.shape[1:]!=self.z.shape or not np.isfinite(z).all():raise ValueError('Invalid spectrum')
        output=[np.mean((20*np.log10(np.maximum(abs(z),1e-300))-self.a)**2,axis=1),
                np.mean((np.angle(z)-self.phase)**2,axis=1)]
        if self.aux:p=self.path(z)
        for name in self.aux:
            family=name.rsplit('_',1)[0]
            if family=='chords':value=np.array([np.sqrt(np.sum(self.pair_weights*(pdist(row)-self.d)**2)/self.chord_norm) for row in p])
            elif family=='covariance':value=np.linalg.norm(covariance(p,self.du)-self.c,axis=(-2,-1))/self.cov_norm
            elif family=='area':value=np.linalg.norm(area(p)-self.area,axis=-1)
            elif family=='srvf':value=np.sum(self.du*np.sum((srvf(p,self.du)-self.q)**2,axis=-1),axis=-1)/self.length
            elif family=='arclength':value=np.array([arc_distance(row,self.p,self.s) for row in p])
            elif family=='orientation':value=self.orientation_loss(p)
            else:raise AssertionError(family)
            output.append(value)
        result=np.column_stack(output)
        if not np.isfinite(result).all() or np.any(result<0):raise FloatingPointError('Invalid objective')
        return result

"""Frozen 3D curve protocol: five directions, two target-fixed coordinates.

All integrals are exact for the specified piecewise-linear paths except the
chord matrix, which uses log-frequency trapezoidal vertex quadrature.
There is no event count, fitted threshold, warping window, or tunable loss weight.
"""
import numpy as np
from scipy.spatial.distance import pdist

FAMILIES=[('D-01','chords','全频率对距离'),
          ('D-02','covariance','整体坐标变化关系'),
          ('D-03','area','三平面有向面积'),
          ('D-04','srvf','走向与变化快慢'),
          ('D-05','arclength','按弧长对应的形状')]
CATALOG=[(code+'-'+tag,name+'_'+coord,label+'（'+zh+'）')
         for code,name,label in FAMILIES
         for coord,tag,zh in [('linear','LIN','线性坐标'),('asinh','ASH','有符号对数式坐标')]]
AUX=tuple(row[1] for row in CATALOG)
DEFAULTS={}

def covariance(p,du):
    a=p[...,:-1,:];b=p[...,1:,:]
    mu=np.sum(.5*(a+b)*du[:,None],axis=-2)
    a=a-mu[...,None,:];b=b-mu[...,None,:]
    # Integrate centered outer products directly to avoid subtracting two
    # potentially large nearly equal second moments.
    return np.sum(du[:,None,None]/6*(2*a[..., :,None]*a[...,None,:]+
                  a[..., :,None]*b[...,None,:]+b[..., :,None]*a[...,None,:]+
                  2*b[..., :,None]*b[...,None,:]),axis=-3)

def area(p):
    r=p-p[...,:1,:]
    return .5*np.sum(np.cross(r[...,:-1,:],r[...,1:,:]),axis=-2)

def srvf(p,du):
    v=np.diff(p,axis=-2)/du[:,None]
    return v/np.sqrt(np.linalg.norm(v,axis=-1))[...,None]

def arc_knots(p):
    lengths=np.linalg.norm(np.diff(p,axis=0),axis=1)
    s=np.r_[0.,np.cumsum(lengths)]
    return s/s[-1]

def arc_distance(p,target,s_target):
    s=arc_knots(p)
    knots=np.union1d(s,s_target)
    r=np.column_stack([np.interp(knots,s,p[:,j])-
                       np.interp(knots,s_target,target[:,j]) for j in range(3)])
    left=r[:-1];right=r[1:];middle=.5*(left+right)
    # Simpson is exact on each interval: the squared linear difference is quadratic.
    return np.sum(np.diff(knots)/6*(np.sum(left*left,axis=1)+
                  4*np.sum(middle*middle,axis=1)+np.sum(right*right,axis=1)))

class GlobalObjectives:
    def __init__(self,frequency,settings=None,aux=(),transform='loss',epsilon=None,target_z=None):
        self.f=np.asarray(frequency,float);self.z=np.asarray(target_z,complex)
        if self.f.ndim!=1 or len(self.f)<2 or np.any(self.f<=0) or np.any(np.diff(self.f)<=0):
            raise ValueError('Positive increasing frequency required')
        if self.z.shape!=self.f.shape or not np.isfinite(self.z).all() or np.any(abs(self.z)==0):
            raise ValueError('Finite nonzero target required')
        if settings not in (None,{}) or transform!='loss' or any(a not in AUX for a in aux) or len(set(aux))!=len(aux):
            raise ValueError('Frozen 3D protocol only')
        self.aux=tuple(aux)
        x=np.log10(self.f);self.u=(x-x[0])/(x[-1]-x[0]);self.du=np.diff(self.u)
        self.weights=np.r_[self.du[0]/2,(self.du[:-1]+self.du[1:])/2,self.du[-1]/2]
        self.pair_weights=(self.weights[:,None]*self.weights[None,:])[np.triu_indices(len(x),1)]
        self.a=20*np.log10(np.maximum(abs(self.z),1e-300));self.phase=np.angle(self.z)
        self.scale=float(np.median(abs(self.z)))
        self.reference={}
        for coord in ['linear','asinh']:
            xy=np.column_stack([self.z.real,self.z.imag])
            if coord=='asinh':xy=np.arcsinh(xy/self.scale)
            offset=xy.min(axis=0);extent=np.ptp(xy,axis=0)
            if np.any(extent<=0):raise ValueError('Degenerate target axes; no epsilon substitution')
            p=np.column_stack([self.u,(xy-offset)/extent])
            c=covariance(p,self.du);q=srvf(p,self.du);d=pdist(p)
            self.reference[coord]=dict(offset=offset,extent=extent,p=p,c=c,q=q,d=d,
                area=area(p),s=arc_knots(p),cov_norm=np.linalg.norm(c),
                chord_norm=np.sum(self.pair_weights*d*d),
                length=np.sum(self.du*np.sum(q*q,axis=1)))

    def target_summary(self):
        return dict(frequency_points=len(self.f),frequency_coordinate='normalized_log10_frequency',
            candidate_normalization='target_fixed_not_candidate_specific',asinh_scale=self.scale,
            coordinates={k:dict(offset=v['offset'].tolist(),extent=v['extent'].tolist())
                         for k,v in self.reference.items()},
            auxiliary_tunable_parameters={},arc_correspondence='union_of_all_actual_arc_knots_exact_integral',
            area_closure='straight_endpoint_chord',rotations_or_frequency_warps_optimized=False)

    def path(self,z,coord):
        xy=np.stack([z.real,z.imag],axis=-1)
        if coord=='asinh':xy=np.arcsinh(xy/self.scale)
        ref=self.reference[coord];xy=(xy-ref['offset'])/ref['extent']
        return np.concatenate([np.broadcast_to(self.u,xy.shape[:-1])[...,None],xy],axis=-1)

    def evaluate_spectra(self,z):
        z=np.atleast_2d(np.asarray(z,complex))
        if z.shape[1:]!=self.z.shape or not np.isfinite(z).all():raise ValueError('Invalid spectrum')
        # Exactly the historical primary objectives (including their phase convention).
        output=[np.mean((20*np.log10(np.maximum(abs(z),1e-300))-self.a)**2,axis=1),
                np.mean((np.angle(z)-self.phase)**2,axis=1)]
        paths={coord:self.path(z,coord) for coord in {name.rsplit('_',1)[1] for name in self.aux}}
        for name in self.aux:
            family,coord=name.rsplit('_',1);p=paths[coord];ref=self.reference[coord]
            if family=='chords':
                value=np.array([np.sqrt(np.sum(self.pair_weights*(pdist(row)-ref['d'])**2)/ref['chord_norm']) for row in p])
            elif family=='covariance':
                value=np.linalg.norm(covariance(p,self.du)-ref['c'],axis=(-2,-1))/ref['cov_norm']
            elif family=='area':value=np.linalg.norm(area(p)-ref['area'],axis=-1)
            elif family=='srvf':
                value=np.sum(self.du*np.sum((srvf(p,self.du)-ref['q'])**2,axis=-1),axis=-1)/ref['length']
            elif family=='arclength':value=np.array([arc_distance(row,ref['p'],ref['s']) for row in p])
            else:raise AssertionError(family)
            output.append(value)
        result=np.column_stack(output)
        if not np.isfinite(result).all() or np.any(result<0):raise FloatingPointError('Invalid objective')
        return result

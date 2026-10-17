"""Three targeted variants and one predeclared two-auxiliary combination."""
import numpy as np

AREA='cumulative_area_logpolar';LENGTH='cumulative_length_logpolar'
SYMMETRIC='symmetric_length_logpolar';AXIS='axis_variation_logpolar';DIRECTION='direction_variation_logpolar'
AUX=(AREA,LENGTH,SYMMETRIC,AXIS,DIRECTION)
COMBINATION=AREA+'+'+LENGTH
CATALOG=[('R-03-C',AREA,'对照：累计面积'),('R-05-C',LENGTH,'对照：累计弧长'),
         ('Q-05-B',SYMMETRIC,'两端对称累计弧长'),('Q-05-A',AXIS,'分轴累计变化量'),
         ('Q-05-D',DIRECTION,'分轴上升／下降累计量'),('Q-COMB',COMBINATION,'累计面积＋累计弧长（双辅助）')]
GROUPS=[()]+[(name,) for name in AUX]+[(AREA,LENGTH)]
COMPARISONS=[(name,LENGTH) for name in [SYMMETRIC,AXIS,DIRECTION]]+[(COMBINATION,LENGTH),(COMBINATION,AREA)]
DEFAULTS={}

def integral_squared(e,h):
    a,b=e[...,:-1,:],e[...,1:,:]
    return np.sum(h/3*np.sum(a*a+a*b+b*b,axis=-1),axis=-1)

def cumulative_area(p):
    r=p-p[...,:1,:];increments=.5*np.cross(r[...,:-1,:],r[...,1:,:])
    return np.concatenate([np.zeros_like(p[...,:1,:]),np.cumsum(increments,axis=-2)],axis=-2)

def cumulative_length(p):
    d=np.linalg.norm(np.diff(p,axis=-2),axis=-1)
    return np.concatenate([np.zeros_like(d[...,:1]),np.cumsum(d,axis=-1)],axis=-1)

def variations(p):
    delta=np.diff(p[...,1:],axis=-2)
    def prefix(d):return np.concatenate([np.zeros_like(p[...,:1,1:]),np.cumsum(d,axis=-2)],axis=-2)
    return prefix(np.maximum(delta,0)),prefix(np.maximum(-delta,0))

class GlobalObjectives:
    def __init__(self,frequency,settings=None,aux=(),transform='loss',epsilon=None,target_z=None):
        self.f=np.asarray(frequency,float);self.z=np.asarray(target_z,complex)
        if self.f.ndim!=1 or len(self.f)<2 or np.any(self.f<=0) or np.any(np.diff(self.f)<=0):raise ValueError('Positive increasing frequency required')
        if self.z.shape!=self.f.shape or not np.isfinite(self.z).all() or np.any(abs(self.z)==0):raise ValueError('Nonzero finite target required')
        if settings not in (None,{}) or transform!='loss' or tuple(aux) not in GROUPS:raise ValueError('Frozen groups only')
        self.aux=tuple(aux);x=np.log10(self.f);self.u=(x-x[0])/(x[-1]-x[0]);self.du=np.diff(self.u)
        self.a=20*np.log10(np.maximum(abs(self.z),1e-300));self.phase=np.angle(self.z)
        xy=np.column_stack([self.a,self.phase]);self.offset=xy.min(axis=0);self.scale=np.ptp(xy,axis=0)
        if np.any(self.scale<=0):raise ValueError('Degenerate target axes; no epsilon substitution')
        self.p=np.column_stack([self.u,(xy-self.offset)/self.scale])
        self.ca=cumulative_area(self.p);self.cl=cumulative_length(self.p);self.length=self.cl[-1]
        self.pos,self.neg=variations(self.p);self.tv=self.pos+self.neg;self.axis_total=self.tv[-1]
        if np.any(self.axis_total<=0):raise ValueError('Constant target axis; no epsilon substitution')

    def path(self,z):
        if np.any(abs(z)==0):raise ValueError('Zero outside log-coordinate domain')
        xy=(np.stack([20*np.log10(abs(z)),np.angle(z)],axis=-1)-self.offset)/self.scale
        return np.concatenate([np.broadcast_to(self.u,xy.shape[:-1])[...,None],xy],axis=-1)

    def target_summary(self):
        lengths=np.diff(self.cl);k=int(np.argmax(lengths))
        return dict(frequency_points=len(self.f),frequency_coordinate='normalized_log10_frequency',
            magnitude_phase_coordinates='unchanged_target_range_principal_phase',offset=self.offset.tolist(),scale=self.scale.tolist(),
            target_total_length=float(self.length),target_axis_total_variations=self.axis_total.tolist(),
            largest_segment_length_share=float(lengths[k]/self.length),largest_segment_frequency_hz=self.f[k:k+2].tolist(),
            all_original_rows_retained=True,auxiliary_tunable_parameters={},
            candidate_normalization='fixed_target_only',direction_split='zero_separates_increase_and_decrease_not_tuned_threshold')

    def evaluate_spectra(self,z):
        z=np.atleast_2d(np.asarray(z,complex))
        if z.shape[1:]!=self.z.shape or not np.isfinite(z).all():raise ValueError('Invalid spectrum')
        output=[np.mean((20*np.log10(np.maximum(abs(z),1e-300))-self.a)**2,axis=1),
                np.mean((np.angle(z)-self.phase)**2,axis=1)]
        if self.aux:p=self.path(z)
        for name in self.aux:
            if name==AREA:
                value=np.sqrt(integral_squared(cumulative_area(p)-self.ca,self.du))
            elif name in [LENGTH,SYMMETRIC]:
                e=(cumulative_length(p)-self.cl)/self.length
                forward=integral_squared(e[...,None],self.du)
                if name==LENGTH:value=np.sqrt(forward)
                else:
                    backward=e[...,-1:]-e
                    value=np.sqrt(.5*(forward+integral_squared(backward[...,None],self.du)))
            elif name in [AXIS,DIRECTION]:
                pos,neg=variations(p)
                if name==AXIS:
                    e=(pos+neg-self.tv)/self.axis_total
                else:
                    e=np.concatenate([(pos-self.pos)/self.axis_total,(neg-self.neg)/self.axis_total],axis=-1)
                value=np.sqrt(integral_squared(e,self.du)/2)
            else:raise AssertionError(name)
            output.append(value)
        result=np.column_stack(output)
        if not np.isfinite(result).all() or np.any(result<0):raise FloatingPointError('Invalid objectives')
        return result

"""The Chinese PDF's 18 auxiliary losses, implemented from its explicit formulas.

No target parameter error is used. All target-side quantities depend on sampled Z.
Batch axis is first; the objective vector always starts with magnitude/phase MSE.
"""
import numpy as np
from scipy.spatial.distance import cdist

CATALOG = [
 ("I-01","turning_event_global_count_5","全局转折数量：至少5个"),
 ("I-02","turning_event_coverage_5_of_6","目标转折区域覆盖：5-of-6"),
 ("I-03","reactance_crossing_direction_consistency","电抗过零方向一致性"),
 ("I-04","reactance_crossing_global_count_10","全局电抗过零数量：至少10次"),
 ("I-05","turning_event_global_count_4","全局转折数量：至少4个"),
 ("I-06","paper_ptc","中文PDF峰谷一致性（最近匹配、均值）"),
 ("I-07","reactance_crossing_global_count_8","全局电抗过零数量：至少8次"),
 ("II-01","characteristic_frequency_shift_error_decades","局部特征频移误差"),
 ("II-02","transition_speed_relative_error","转变速度相对误差"),
 ("II-03","constrained_dtw_window_4","窗口约束DTW"),
 ("II-04","soft_dtw_divergence_gamma_0.10","Soft-DTW散度"),
 ("II-05","hausdorff95_error","95% Hausdorff距离"),
 ("II-06","sobolev_h1_error","Sobolev H1误差"),
 ("II-07","translation_aligned_spatial_shape_error","去平移空间形状误差"),
 ("II-08","turning_event_weak_location","转折事件弱位置误差"),
 ("II-09","turning_event_existence_count_guard","转折存在性与过量事件惩罚"),
 ("II-10","normal_shape_error_relative","去局部频移后的相对形变"),
 ("II-11","turning_event_coverage_4_of_6","目标转折区域覆盖：4-of-6"),
]
AUX=tuple(row[1] for row in CATALOG)
DEFAULTS=dict(turn_quantile=.95,fit_turn_ratio=.20,event_min_gap=4,
              ptc_neighbor_db=.2,ptc_gamma=.41,weak_location_weight=.10,
              excess_event_weight=.10,speed_epsilon_ratio=.05,
              dtw_window=4,soft_dtw_window=10,soft_dtw_gamma=.10,hausdorff_quantile=.95)
def derivative(values,coordinate,axis=-1):
    """Appendix B, including second-order one-sided endpoints."""
    return np.gradient(values,coordinate,axis=axis,edge_order=2)


def turning_angles(path):
    segments=np.diff(path,axis=-2)
    tangent=segments/np.maximum(np.linalg.norm(segments,axis=-1,keepdims=True),1e-15)
    return np.arccos(np.clip(np.sum(tangent[...,:-1,:]*tangent[...,1:,:],axis=-1),-1,1))


def select_turns(theta,threshold,min_gap=4):
    """theta[j] belongs to original sample j+1. Exclude two angle endpoints."""
    candidates=np.flatnonzero((theta[1:-1]>theta[:-2])&
                              (theta[1:-1]>=theta[2:])&
                              (theta[1:-1]>=threshold))+1
    order=np.argsort(-theta[candidates],kind="stable")
    kept=[]
    for i in candidates[order]:
        if all(abs(int(i)-j)>=min_gap for j in kept):
            kept.append(int(i))
    return np.array(sorted(kept),int)+1


def region_edges(events,lo=0.,hi=1.):
    return np.r_[lo,(events[:-1]+events[1:])/2,hi]


def region_matches(events,edges):
    # The PDF deliberately includes a shared boundary in BOTH adjacent regions.
    return [(events>=lo)&(events<=hi) for lo,hi in zip(edges[:-1],edges[1:])]


def crossings(imag,u):
    ids=np.flatnonzero(np.signbit(imag[:-1])!=np.signbit(imag[1:]))
    return (u[ids]+u[ids+1])/2,np.where(imag[ids]<imag[ids+1],1,-1)


def peaks_troughs(a,x,threshold=.2):
    hi=np.flatnonzero((a[1:-1]>=a[:-2])&(a[1:-1]>a[2:])&
                      (a[1:-1]-np.maximum(a[:-2],a[2:])>=threshold))+1
    lo=np.flatnonzero((a[1:-1]<=a[:-2])&(a[1:-1]<a[2:])&
                      (np.minimum(a[:-2],a[2:])-a[1:-1]>=threshold))+1
    ids=np.r_[hi,lo];typ=np.r_[np.ones(len(hi),int),-np.ones(len(lo),int)]
    order=np.argsort(ids)
    return x[ids[order]],typ[order]


def nearest_ptc(events,types,target,target_types,edges,span,gamma=.41):
    if not len(target):
        return 0.
    total=0.
    for f,t,mask in zip(target,target_types,region_matches(events,edges)):
        matches=events[mask&(types==t)]
        total+=float(np.expm1(gamma*np.min(abs(matches-f))/span)) if len(matches) else gamma
    return total/len(target)


def band_dtw(p,q,window=4,return_details=False):
    """Eq. 8.1-8.3. Minimize sum first; ties: diagonal, up, left; divide by steps."""
    p,q=np.broadcast_arrays(p,q)  # shape (batch,n,3)
    n=p.shape[1];batch=len(p);bi=np.arange(batch)
    previous=np.full((batch,n+1),np.inf);previous[:,0]=0
    prev_steps=np.zeros((batch,n+1),int)
    for i in range(1,n+1):
        current=np.full_like(previous,np.inf);steps=np.zeros_like(prev_steps)
        for j in range(max(1,i-window),min(n,i+window)+1):
            pred=np.stack([previous[:,j-1],previous[:,j],current[:,j-1]],axis=1)
            choice=np.argmin(pred,axis=1)
            cost=np.linalg.norm(p[:,i-1]-q[:,j-1],axis=1)
            current[:,j]=pred[bi,choice]+cost
            lengths=np.stack([prev_steps[:,j-1],prev_steps[:,j],steps[:,j-1]],axis=1)
            steps[:,j]=lengths[bi,choice]+1
        previous,prev_steps=current,steps
    value=previous[:,-1]/np.maximum(prev_steps[:,-1],1)
    return (value,previous[:,-1],prev_steps[:,-1]) if return_details else value


def band_soft_dtw(p,q,window=10,gamma=.10):
    """Eqs. 8.4-8.6, stable soft minimum; squared local distance."""
    p,q=np.broadcast_arrays(p,q)
    n=p.shape[1];previous=np.full((len(p),n+1),np.inf);previous[:,0]=0
    for i in range(1,n+1):
        current=np.full_like(previous,np.inf)
        for j in range(max(1,i-window),min(n,i+window)+1):
            pred=np.stack([previous[:,j],previous[:,j-1],current[:,j-1]],axis=1)
            m=pred.min(axis=1);valid=np.isfinite(m)
            smooth=np.full(len(p),np.inf)
            smooth[valid]=m[valid]-gamma*np.log(np.exp(-(pred[valid]-m[valid,None])/gamma).sum(axis=1))
            current[:,j]=np.sum((p[:,i-1]-q[:,j-1])**2,axis=1)+smooth
        previous=current
    return previous[:,-1]


def soft_divergence(p,q,window=10,gamma=.10,target_self=None):
    p,q=np.broadcast_arrays(p,q)
    pp=band_soft_dtw(p,p,window,gamma) if target_self is None else target_self
    qq=band_soft_dtw(q,q,window,gamma)
    pq=band_soft_dtw(p,q,window,gamma)
    return np.sqrt(np.maximum((pq-.5*pp-.5*qq)/p.shape[1],0))


class Objectives:
    def __init__(self,frequency,settings=None,aux=(),transform="loss",epsilon=1e-12,target_z=None):
        self.f=np.asarray(frequency,float)
        if (self.f.ndim!=1 or len(self.f)<5 or not np.all(np.isfinite(self.f))
            or np.any(self.f<=0) or np.any(np.diff(self.f)<=0)):
            raise ValueError("At least five finite, strictly increasing positive frequencies required")
        if len(set(aux))!=len(aux) or any(name not in AUX for name in aux):
            raise ValueError("Unknown/repeated auxiliary")
        if transform!="loss":
            raise ValueError("Chinese PDF objectives use raw minimization losses")
        self.aux=tuple(aux)
        self.settings=dict(DEFAULTS if settings is None else settings)
        # Fixed names encode fixed constants. Reject silently changing their formulas.
        if self.settings!=DEFAULTS:
            raise ValueError("This protocol fixes constants to the Chinese PDF; use a separately named protocol for variants")
        self.x=np.log10(self.f);self.span=max(self.x[-1]-self.x[0],1e-12)
        self.u=(self.x-self.x[0])/self.span
        if target_z is None:
            raise ValueError("Case2 requires the measured target spectrum")
        self.z=np.asarray(target_z,complex)
        if self.z.shape!=self.f.shape or not np.all(np.isfinite(self.z)):
            raise ValueError("Target impedance shape/values invalid")
        self.a=20*np.log10(np.maximum(np.abs(self.z),1e-300));self.phase=np.angle(self.z)
        self.scales=np.array([max(np.ptp(v),np.std(v),1e-12) for v in (self.z.real,self.z.imag)])
        self.tr_scales=np.array([max(np.ptp(v),1e-12) for v in (self.z.real,self.z.imag)])
        self.p=self.path(self.z[None,:])[0]
        self.g=derivative(self.z,self.x)
        speed=abs(self.g);self.sz=max(float(np.median(abs(self.z))),1e-12)
        sg=max(float(np.median(speed[speed>0])),1e-12) if np.any(speed>0) else self.sz
        self.ev=.05*sg;self.den=speed**2+self.ev**2
        self.weight=speed**2/self.den
        self.du=derivative(self.p,self.u,axis=0)
        self.du_energy=max(float(np.mean(np.sum(self.du**2,axis=1))),1e-12)
        theta=turning_angles(self.p)
        self.target_threshold=float(np.quantile(theta,.95))
        self.turn_ids=select_turns(theta,self.target_threshold)
        self.turn_events=self.u[self.turn_ids]
        self.turn_edges=region_edges(self.turn_events)
        self.fit_threshold=.20*float(np.min(theta[self.turn_ids-1])) if len(self.turn_ids) else None
        self.cross_events,self.cross_types=crossings(self.z.imag,self.u)
        self.cross_edges=region_edges(self.cross_events)
        self.extrema,self.extrema_types=peaks_troughs(self.a,self.x)
        self.extrema_edges=region_edges(self.extrema,self.x[0],self.x[-1])
        if any(name.startswith(("turning_","reactance_")) for name in aux) and not len(self.turn_events):
            raise ValueError("PDF common event initialization requires at least one target turn")
        if any(name.startswith("reactance_crossing_global_count") for name in aux) and not len(self.cross_events):
            raise ValueError("PDF crossing-count denominator requires a target crossing")
        self.soft_self=(band_soft_dtw(self.p[None],self.p[None])[0]
                        if "soft_dtw_divergence_gamma_0.10" in aux else None)

    def path(self,z):
        return np.stack([np.broadcast_to(self.u,z.shape),z.real/self.scales[0],z.imag/self.scales[1]],axis=-1)

    def target_summary(self):
        return dict(frequency_points=len(self.f),target_turns=len(self.turn_events),
                    turn_u=self.turn_events.tolist(),target_crossings=len(self.cross_events),
                    target_amplitude_extrema=len(self.extrema),path_scales=self.scales.tolist(),
                    impedance_median_scale=self.sz,speed_epsilon=self.ev,
                    target_turn_threshold=self.target_threshold,fit_turn_threshold=self.fit_threshold)

    def evaluate_spectra(self,z):
        z=np.atleast_2d(np.asarray(z,complex))
        if z.shape[1]!=len(self.f) or not np.all(np.isfinite(z)):
            raise ValueError("Invalid candidate spectrum")
        amp=20*np.log10(np.maximum(abs(z),1e-300))
        output=[np.mean((amp-self.a)**2,axis=1),np.mean((np.angle(z)-self.phase)**2,axis=1)]
        cache={}
        def paths():
            if "p" not in cache:cache["p"]=self.path(z)
            return cache["p"]
        def turns():
            if "turns" not in cache:
                cache["turns"]=[self.u[select_turns(t,self.fit_threshold)] for t in turning_angles(paths())]
            return cache["turns"]
        def cross():
            if "cross" not in cache:cache["cross"]=[crossings(v.imag,self.u) for v in z]
            return cache["cross"]
        def shift():
            if "shift" not in cache:
                cache["shift"]=np.real((z-self.z)*np.conj(self.g))/self.den
            return cache["shift"]
        for name in self.aux:
            if name.startswith("turning_event_"):
                v=[];m=len(self.turn_events)
                for events in turns():
                    ne=len(events)
                    masks=region_matches(events,self.turn_edges)
                    covered=sum(mask.any() for mask in masks)
                    if name.startswith("turning_event_global_count_"):
                        q=min(int(name[-1]),m);value=max(q-ne,0)/q
                    elif name.startswith("turning_event_coverage_"):
                        q=min(5 if "_5_" in name else 4,m);value=max(q-covered,0)/q
                    elif name=="turning_event_weak_location":
                        costs=[.10*min(float(np.min(abs(events[mask]-target)))/max(hi-lo,1e-12),1)
                               if mask.any() else 1.
                               for mask,target,lo,hi in zip(masks,self.turn_events,self.turn_edges[:-1],self.turn_edges[1:])]
                        value=float(np.mean(costs))
                    else:
                        value=(m-covered)/m+.10*max(ne-m,0)/m
                    v.append(value)
            elif name.startswith("reactance_crossing_"):
                v=[];m=len(self.cross_events)
                for events,types in cross():
                    if name=="reactance_crossing_direction_consistency":
                        v.append(float(np.mean([not np.any(mask&(types==typ))
                                  for mask,typ in zip(region_matches(events,self.cross_edges),self.cross_types)])) if m else 0.)
                    else:
                        q=min(int(name.rsplit("_",1)[1]),m);v.append(max(q-len(events),0)/q)
            elif name=="paper_ptc":
                a=20*np.log10(np.maximum(abs(z),1e-18))
                v=[nearest_ptc(*peaks_troughs(row,self.x),self.extrema,self.extrema_types,
                               self.extrema_edges,self.span) for row in a]
            elif name=="characteristic_frequency_shift_error_decades":
                v=np.sum(self.weight*abs(shift()),axis=1)/max(self.weight.sum(),1e-12)
            elif name=="transition_speed_relative_error":
                v=np.mean(abs(abs(derivative(z,self.x,axis=1))-abs(self.g))/(abs(self.g)+self.ev),axis=1)
            elif name=="normal_shape_error_relative":
                v=np.mean(abs((z-self.z)-self.g*shift()),axis=1)/self.sz
            elif name=="constrained_dtw_window_4":
                v=band_dtw(self.p[None],paths())
            elif name=="soft_dtw_divergence_gamma_0.10":
                v=soft_divergence(self.p[None],paths(),target_self=self.soft_self)
            elif name=="hausdorff95_error":
                v=[]
                for row in paths():
                    distance=cdist(self.p,row)
                    v.append(max(np.quantile(distance.min(axis=0),.95),np.quantile(distance.min(axis=1),.95)))
            elif name=="sobolev_h1_error":
                e0=np.mean(np.sum((paths()-self.p)**2,axis=2),axis=1)
                du=derivative(paths(),self.u,axis=1)
                e1=np.mean(np.sum((du-self.du)**2,axis=2),axis=1)/self.du_energy
                v=np.sqrt(np.maximum(e0+e1,0))
            elif name=="translation_aligned_spatial_shape_error":
                r=z-self.z
                delta=np.stack([r.real/self.tr_scales[0],r.imag/self.tr_scales[1]],axis=-1)
                v=np.mean(np.linalg.norm(delta-np.median(delta,axis=1,keepdims=True),axis=2),axis=1)
            output.append(np.asarray(v))
        out=np.column_stack(output)
        if not np.all(np.isfinite(out)) or np.any(out<0):
            raise FloatingPointError("Nonfinite/negative loss: no silent penalty substitution")
        return out

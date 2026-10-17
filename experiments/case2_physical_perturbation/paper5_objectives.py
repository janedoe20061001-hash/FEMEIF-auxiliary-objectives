"""Five auxiliary losses from the English paper, adapted to measured Case2 spectra."""
import numpy as np

AUX=("SS","PTC","DTW","LFP","LMEP")
CATALOG=[("P-01","SS","趋势编码相似性"),("P-02","PTC","论文峰谷位置一致性"),
         ("P-03","DTW","60点无窗口DTW"),("P-04","LFP","低频连续达标段数"),
         ("P-05","LMEP","最大局部幅值误差")]
DEFAULTS={"THk":8.0,"beta":0.4,"mu":0.5,"gamma":0.41,"THlf":1.6,"DTW_points":60}


def sigmoid(v):return 1/(1+np.exp(-np.clip(v,-700,700)))


def patterns(a,x,s):
    k=np.diff(a,axis=-1)/np.diff(x);old,new=k[...,:-1],k[...,1:];t,b,m=s["THk"],s["beta"],s["mu"]
    out=np.zeros_like(old)
    for sign,mask in [(-1,((new < -t)&(old <= t)) | ((old < -t)&(new <= t))),
                      (1,((new > t)&(old >= -t)) | ((old > t)&(new >= -t)))]:
        delta=new[mask]-old[mask];out[mask]=sign*(1+b*(2*sigmoid(m*sign*delta)-1))
    for sign,mask in [(1,(old < -t)&(new > t)),(-1,(old > t)&(new < -t))]:
        delta=new[mask]-old[mask];out[mask]=sign*(1+1.5*b*sigmoid(m*sign*(delta-2*sign*t)))
    return out


def extrema(a,x):
    peaks=np.flatnonzero((a[1:-1]>=a[:-2])&(a[1:-1]>a[2:]))+1
    troughs=np.flatnonzero((a[1:-1]<=a[:-2])&(a[1:-1]<a[2:]))+1
    ids=np.r_[peaks,troughs];typ=np.r_[np.ones(len(peaks),int),-np.ones(len(troughs),int)]
    order=np.argsort(ids);return x[ids[order]],typ[order]


def ptc_cost(events,types,target,target_types,edges,gamma):
    total=0.
    for i,(f,t) in enumerate(zip(target,target_types)):
        ok=(events>=edges[i])&(events<edges[i+1])&(types==t)
        total+=float(np.expm1(gamma*np.abs(events[ok]-f)/(edges[-1]-edges[0])).sum()) if ok.any() else gamma
    return total


def dtw(candidates,target):
    prev=np.full((len(candidates),len(target)+1),np.inf);prev[:,0]=0
    for i in range(candidates.shape[1]):
        curr=np.full_like(prev,np.inf)
        for j in range(len(target)):
            curr[:,j+1]=np.minimum(np.minimum(prev[:,j],prev[:,j+1]),curr[:,j])+np.abs(candidates[:,i]-target[j])
        prev=curr
    return prev[:,-1]


class Objectives:
    def __init__(self,frequency,settings=None,aux=(),transform="loss",epsilon=1e-12,target_z=None):
        self.f=np.asarray(frequency,float);self.x=np.log10(self.f);self.settings=DEFAULTS if settings is None else settings
        if self.settings!=DEFAULTS:raise ValueError("paper-five constants changed")
        if target_z is None:raise ValueError("measured Case2 target is required")
        if len(set(aux))!=len(aux) or any(a not in AUX for a in aux):raise ValueError("unknown/repeated auxiliary")
        if transform!="loss":raise ValueError("this protocol uses raw minimization losses")
        self.aux=tuple(aux);self.z=np.asarray(target_z,complex)
        if self.z.shape!=self.f.shape:raise ValueError("target shape mismatch")
        self.a=20*np.log10(np.maximum(abs(self.z),1e-300));self.phase=np.angle(self.z)
        self.shape=patterns(self.a,self.x,self.settings)
        self.events,self.types=extrema(self.a,self.x)
        if len(self.events)==0:raise ValueError("target has no extrema")
        self.edges=np.r_[self.x[0],(self.events[:-1]+self.events[1:])/2,self.x[-1]]
        self.splits=np.searchsorted(self.x,self.edges[1:-1],side="left")
        self.dtw_x=np.linspace(self.x[0],self.x[-1],self.settings["DTW_points"])
        self.ad=np.interp(self.dtw_x,self.x,self.a)

    def target_summary(self):
        return dict(target_extrema=len(self.events),target_segments=len(self.events),dtw_points=len(self.dtw_x),
                    dtw_measured_log_frequency_interpolation=True)

    def evaluate_spectra(self,z):
        z=np.atleast_2d(np.asarray(z,complex))
        if z.shape[1]!=len(self.f) or not np.isfinite(z).all():raise ValueError("invalid candidate spectra")
        a=20*np.log10(np.maximum(abs(z),1e-300));err=a-self.a
        result=[np.mean(err**2,axis=1),np.mean((np.angle(z)-self.phase)**2,axis=1)]
        for name in self.aux:
            if name=="SS":value=np.sum(np.abs(patterns(a,self.x,self.settings)-self.shape),axis=1)/len(self.f)
            elif name=="PTC":value=np.asarray([ptc_cost(*extrema(row,self.x),self.events,self.types,self.edges,self.settings["gamma"]) for row in a])
            elif name=="DTW":value=dtw(np.asarray([np.interp(self.dtw_x,self.x,row) for row in a]),self.ad)
            elif name=="LFP":
                zone=np.stack([np.mean(v**2,axis=1) for v in np.split(err,self.splits,axis=1)],axis=1)
                count=np.cumprod(zone<self.settings["THlf"],axis=1).sum(axis=1);value=1-count/len(self.events)
            elif name=="LMEP":value=np.max(abs(err),axis=1)
            result.append(value)
        out=np.column_stack(result)
        if not np.isfinite(out).all():raise FloatingPointError("non-finite objective")
        return out

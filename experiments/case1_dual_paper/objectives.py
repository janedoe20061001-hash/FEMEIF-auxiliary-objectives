"""Case I terminal impedance and the five paper auxiliaries. All outputs minimize."""
import numpy as np

AUX = ("SS", "PTC", "DTW", "LFP", "LMEP")
TARGET = np.array([[2000,1585e-6,10e-6], [3000,10e-6,3.98e-6],
                   [500,100e-6,1e-6], [1200,2.512e-6,.631e-6],
                   [800,1.259e-6,.501e-6], [50,.251e-6,.158e-6]])
LOWER = np.tile([0.,-7.,-9.],6)   # log10 R, L, C: Table II
UPPER = np.tile([5.,-2.,-4.],6)


def impedance(f, params):
    params=np.asarray(params).reshape(-1,6,3)
    w=2*np.pi*np.asarray(f)[None,:]
    z=np.zeros((len(params),len(f)),complex)
    for j in range(6):
        r,l,c=(params[:,j,k,None] for k in range(3))
        z+=1/(1/r+1j*(w*c-1/(w*l)))
    return z


def sigmoid(v):
    return 1/(1+np.exp(-np.clip(v,-700,700)))


def patterns(a,x,settings):
    """Table I and Eqs. 3-5. N-2 available codes, Eq. 6 denominator remains N."""
    k=np.diff(a,axis=-1)/np.diff(x)
    old,new=k[...,:-1],k[...,1:]
    t,b,m=(settings[v] for v in ("THk","beta","mu"))
    s=np.zeros_like(old)
    for sign,mask in [
        (-1,((new < -t)&(old <= t)) | ((old < -t)&(new <= t))),
        (1,((new > t)&(old >= -t)) | ((old > t)&(new >= -t)))]:
        delta=new[mask]-old[mask]
        s[mask]=sign*(1+b*(2*sigmoid(m*sign*delta)-1))
    for sign,mask in [(1,(old < -t)&(new > t)),(-1,(old > t)&(new < -t))]:
        delta=new[mask]-old[mask]
        s[mask]=sign*(1+1.5*b*sigmoid(m*sign*(delta-2*sign*t)))
    return s


def extrema(a,x):
    peaks=np.flatnonzero((a[1:-1]>=a[:-2])&(a[1:-1]>a[2:]))+1
    troughs=np.flatnonzero((a[1:-1]<=a[:-2])&(a[1:-1]<a[2:]))+1
    ids=np.r_[peaks,troughs]
    typ=np.r_[np.ones(len(peaks),int),-np.ones(len(troughs),int)]
    order=np.argsort(ids)
    return x[ids[order]],typ[order]


def ptc_cost(events,types,target,target_types,edges,gamma):
    result=0.
    for i,(f,t) in enumerate(zip(target,target_types)):
        ok=(events>=edges[i])&(events<edges[i+1])&(types==t)
        result+=float(np.expm1(gamma*np.abs(events[ok]-f)/(edges[-1]-edges[0])).sum()) if ok.any() else gamma
    return result


def dtw(candidates,target):
    """Absolute-cost, unconstrained path sum. Batch dimension is vectorized."""
    prev=np.full((len(candidates),len(target)+1),np.inf)
    prev[:,0]=0
    for i in range(candidates.shape[1]):
        curr=np.full_like(prev,np.inf)
        for j in range(len(target)):
            curr[:,j+1]=np.minimum(np.minimum(prev[:,j],prev[:,j+1]),curr[:,j])+np.abs(candidates[:,i]-target[j])
        prev=curr
    return prev[:,-1]


class Objectives:
    def __init__(self,frequency,settings,aux=(),transform="loss",epsilon=1e-12):
        self.f=np.asarray(frequency,float)
        if self.f.ndim!=1 or len(self.f)<3 or not np.all(np.isfinite(self.f)) or np.any(self.f<=0) or np.any(np.diff(self.f)<=0):
            raise ValueError("Frequency must be finite, positive, strictly increasing, with >=3 points")
        if len(set(aux))!=len(aux) or any(a not in AUX for a in aux):
            raise ValueError("Unknown or repeated auxiliary")
        self.aux=tuple(aux);self.settings=settings
        self.transform=transform;self.epsilon=epsilon
        if transform not in ("loss","negative_reciprocal") or epsilon<=0:
            raise ValueError("Invalid auxiliary transform")
        self.x=np.log10(self.f)
        self.z=impedance(self.f,TARGET)[0]
        self.a=20*np.log10(np.maximum(np.abs(self.z),1e-300))
        self.phase=np.angle(self.z)
        self.shape=patterns(self.a,self.x,settings)
        self.events,self.types=extrema(self.a,self.x)
        if len(self.events)==0:
            raise ValueError("The target grid contains no extrema; PTC/LFP cannot be constructed")
        self.edges=np.r_[self.x[0],(self.events[:-1]+self.events[1:])/2,self.x[-1]]
        self.splits=np.searchsorted(self.x,self.edges[1:-1],side="left")
        self.fd=np.geomspace(self.f[0],self.f[-1],settings["DTW_points"])
        self.ad=20*np.log10(np.abs(impedance(self.fd,TARGET)[0]))

    def evaluate(self,x):
        x=np.atleast_2d(np.asarray(x,float))
        if x.shape[1]!=18 or not np.all(np.isfinite(x)):
            raise ValueError("Expected finite log10 parameters with shape (n,18)")
        p=10**x.reshape(-1,6,3)
        z=impedance(self.f,p)
        a=20*np.log10(np.maximum(np.abs(z),1e-300))
        e=a-self.a
        result=[np.mean(e**2,axis=1),np.mean((np.angle(z)-self.phase)**2,axis=1)]
        for name in self.aux:
            if name=="SS":
                v=np.sum(np.abs(patterns(a,self.x,self.settings)-self.shape),axis=1)/len(self.f)
            elif name=="PTC":
                v=np.array([ptc_cost(*extrema(ai,self.x),self.events,self.types,self.edges,self.settings["gamma"]) for ai in a])
            elif name=="DTW":
                ad=20*np.log10(np.maximum(np.abs(impedance(self.fd,p)),1e-300))
                v=dtw(ad,self.ad)
            elif name=="LFP":
                zone=np.stack([np.mean(v**2,axis=1) for v in np.split(e,self.splits,axis=1)],axis=1)
                count=np.cumprod(zone<self.settings["THlf"],axis=1).sum(axis=1)
                v=1-count/len(self.events)
            elif name=="LMEP":
                v=np.max(np.abs(e),axis=1)
            if self.transform=="negative_reciprocal":
                v=-count.astype(float) if name=="LFP" else -1/(v+self.epsilon)
            result.append(v)
        out=np.column_stack(result)
        if not np.all(np.isfinite(out)):
            raise FloatingPointError("Non-finite objective; refusing silent penalty substitution")
        return out

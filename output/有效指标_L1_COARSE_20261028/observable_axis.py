"""Observable auxiliary for a series sum of parallel RLC branches.

Targets are observed positive-frequency impedance poles and residues in Hz units.
Only this identifiable topology is supported; no target component values are read.
Uses NumPy and SciPy. Score is paired with the original magnitude/phase objectives.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

def strength(p,a):
    return -a.imag/p.imag+1j*(p.real*a.real+p.imag*a.imag)/(p.imag*abs(p))

def port_modes(rlc):
    terms={}
    for R,L,C in np.asarray(rlc,float):
        R,L,C=float(R),float(L),float(C)
        gamma=1/(2*R*C);disc=1/(L*C)-gamma**2
        if disc<=0:continue
        omega=np.sqrt(disc);p=(omega+1j*gamma)/(2*np.pi);a=(gamma-1j*omega)/(4*np.pi*C*omega);terms[p]=terms.get(p,0j)+a
    modes=sorted(terms.items(),key=lambda row:(row[0].real,row[0].imag))
    return np.asarray([v[0] for v in modes],complex),np.asarray([v[1] for v in modes],complex)

METHODS=('L1_COARSE','L1_FINE','L1_BOTH','MAX_COORD_MEAN')

class ObservableAxis:
    def __init__(self,observed_poles_hz,observed_residues_ohm_hz,lower_log10,upper_log10,method):
        if method not in METHODS:raise ValueError('Unknown coordinate norm method')
        self.method=method
        self.p=np.asarray(observed_poles_hz,complex);self.a=np.asarray(observed_residues_ohm_hz,complex);self.n=len(self.p);self.lower=np.asarray(lower_log10,float).reshape(self.n,3);self.upper=np.asarray(upper_log10,float).reshape(self.n,3)
        if len(self.a)!=self.n or not self.n or not np.all(np.isfinite(np.r_[self.p,self.a])) or not np.isfinite(self.lower).all() or not np.isfinite(self.upper).all() or np.any(self.p.real<=0) or np.any(self.p.imag<=0) or np.any(self.lower>=self.upper):raise ValueError('Invalid observed modes or component bounds')
        self.B=strength(self.p,self.a);self.nu=abs(self.p);self.gamma=self.p.imag;self.w=self.gamma/self.nu
        if np.any(abs(self.B)==0):raise ValueError('Unobservable zero-contribution target mode')
        R=abs(self.B);C=1/(4*np.pi*R*self.gamma);L=R*self.gamma/(np.pi*self.nu*self.nu);self.equivalent_target=np.c_[R,L,C]
        lo,hi=10**self.lower,10**self.upper
        def maxlog(v,t):return np.maximum(abs(np.log(v[0]/t)),abs(np.log(v[1]/t)))**2
        er=maxlog((float(np.min(lo[:,0])),float(self.n*np.max(hi[:,0]))),R);el=maxlog((float(np.min(lo[:,1])),float(self.n*np.max(hi[:,1]))),L);ec=maxlog((float(np.min(lo[:,2])/self.n),float(np.max(hi[:,2]))),C)
        r,l,c=np.sqrt(er),np.sqrt(el),np.sqrt(ec);w=self.w
        self.bounds=dict(COARSE=er+el+ec+4,L1_COARSE=r+c+l+2,L2_FINE=np.sqrt(er+(el+ec)/(w*w)+4),L1_FINE=r+(c+l)/w+2,MAX_FINE=np.maximum.reduce([r,c/w,l/w,np.full_like(w,2.)]));self.penalties={k:float(np.nextafter(np.sum(v)+1,np.inf)) for k,v in self.bounds.items()}
        if len(np.unique(self.p))!=self.n:raise ValueError('Distinct observable target modes are required')
    def __call__(self,log10_rlc):
        x=np.asarray(log10_rlc,float).reshape(self.n,3)
        if not np.isfinite(x).all() or np.any(x<self.lower) or np.any(x>self.upper):raise ValueError('Candidate outside frozen bounds')
        p,a=port_modes(10**x);z=strength(p,a)[None,:]/self.B[:,None];u=np.log(abs(p)[None,:]/self.nu[:,None]);g=np.log(p.imag[None,:]/self.gamma[:,None]);b=np.log(abs(z));direction=abs(z/abs(z)-1)**2;ec=-g-b;el=g+b-2*u;r,c,l,d=abs(b),abs(ec),abs(el),np.sqrt(direction);w=self.w[:,None];costs=dict(COARSE=b*b+ec*ec+el*el+direction,L1_COARSE=r+c+l+d,L2_FINE=np.sqrt(b*b+(ec*ec+el*el)/(w*w)+direction),L1_FINE=r+(c+l)/w+d,MAX_FINE=np.maximum.reduce([r,c/w,l/w,d]));ck='L1_COARSE' if self.method in ['L1_COARSE','L1_BOTH'] else 'COARSE';fk='L2_FINE' if self.method=='L1_COARSE' else 'MAX_FINE' if self.method=='MAX_COORD_MEAN' else 'L1_FINE';missing=self.n-len(p);values=[]
        for key in [ck,fk]:
            cost,penalty=costs[key],self.penalties[key]
            full=np.c_[cost,np.zeros((self.n,missing))] if missing else cost;i,j=linear_sum_assignment(full);values.append(float(np.sum(full[i,j])/self.n+missing*penalty))
        value=np.asarray(values)
        if not np.isfinite(value).all():raise ArithmeticError('Nonfinite auxiliary score')
        return value

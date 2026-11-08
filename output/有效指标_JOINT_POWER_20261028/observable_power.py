"""Observable modal power area and three log-frequency quantiles.

Draft API: qualification is determined by the experiment evidence, not this file.
Supports only a series sum of n positive parallel RLC branches, with n distinct
observed oscillatory target modes. Frequencies are Hz, residues are ohm*Hz.
Target component values are never arguments. NumPy and SciPy are required.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

METHODS=('AXIS_POWER','POWER_FINE','JOINT_POWER')

def port_modes(rlc):
    combined={}
    for R,L,C in np.asarray(rlc,float):
        R,L,C=float(R),float(L),float(C)
        gamma=1/(2*R*C);disc=1/(L*C)-gamma**2
        if disc<=0:continue
        omega=np.sqrt(disc);p=(omega+1j*gamma)/(2*np.pi)
        a=(gamma-1j*omega)/(4*np.pi*C*omega)
        combined[p]=combined.get(p,0j)+a
    modes=sorted(combined.items(),key=lambda v:(v[0].real,v[0].imag))
    return np.asarray([v[0] for v in modes],complex),np.asarray([v[1] for v in modes],complex)

def features(p,a):
    nu=np.abs(p);gamma=p.imag;w=gamma/nu;k=p.real/gamma
    B=-a.imag/gamma+1j*(p.real*a.real+gamma*a.imag)/(gamma*nu)
    h=np.log1p(2*w)/2
    log_area=2*np.log(abs(B))+np.log(2*np.arctan(k)/k)
    return nu,gamma,w,B,h,log_area

class ObservablePower:
    def __init__(self,observed_poles_hz,observed_residues_ohm_hz,lower_log10,upper_log10,method):
        if method not in METHODS:raise ValueError('Unknown method')
        self.method=method;self.p=np.asarray(observed_poles_hz,complex);self.a=np.asarray(observed_residues_ohm_hz,complex);self.n=len(self.p)
        self.lower=np.asarray(lower_log10,float).reshape(self.n,3);self.upper=np.asarray(upper_log10,float).reshape(self.n,3)
        if not self.n or len(self.a)!=self.n or not np.all(np.isfinite(np.r_[self.p,self.a])) or np.any(self.p.real<=0) or np.any(self.p.imag<=0) or len(np.unique(self.p))!=self.n:raise ValueError('Distinct observable positive target modes required')
        if not np.isfinite(self.lower).all() or not np.isfinite(self.upper).all() or np.any(self.lower>=self.upper):raise ValueError('Invalid component bounds')
        self.nu,self.gamma,self.w,self.B,self.h,self.log_area=features(self.p,self.a)
        if np.any(abs(self.B)==0):raise ValueError('Unobservable target mode')
        lo,hi=10**self.lower,10**self.upper;R=abs(self.B);C=1/(4*np.pi*R*self.gamma);L=R*self.gamma/(np.pi*self.nu**2)
        def maxlog(low,high,t):return np.maximum(abs(np.log(low/t)),abs(np.log(high/t)))
        r=maxlog(np.min(lo[:,0]),self.n*np.max(hi[:,0]),R)
        c=maxlog(np.min(lo[:,2])/self.n,np.max(hi[:,2]),C)
        l=maxlog(np.min(lo[:,1]),self.n*np.max(hi[:,1]),L)
        nmin=np.min(1/(2*np.pi*np.sqrt(hi[:,1]*hi[:,2])))
        nmax=np.max(1/(2*np.pi*np.sqrt(lo[:,1]*lo[:,2])))
        wmin=np.min(np.sqrt(lo[:,1]/hi[:,2])/(2*hi[:,0]))
        if not 0<wmin<1:raise ValueError('Bounds must permit oscillatory modes')
        kmin=np.sqrt((1-wmin)*(1+wmin))/wmin;hmin=np.log1p(2*wmin)/2;hmax=np.log(3)/2
        amin=np.min(lo[:,0])**2*(2*np.arctan(kmin)/kmin);amax=(self.n*np.max(hi[:,0]))**2*2
        u=maxlog(nmin,nmax,self.nu);eta=maxlog(hmin,hmax,self.h)
        e=np.maximum(abs(np.log(amin)-self.log_area),abs(np.log(amax)-self.log_area))
        dh=np.maximum(abs(hmin/self.h-1),abs(hmax/self.h-1))
        self.row_bounds=dict(AXIS_C=r+c+l+2,AXIS_F=np.sqrt(r*r+(c*c+l*l)/(self.w*self.w)+4),POWER_C=u+eta+e+2,QUANTILE_F=np.sqrt((u/self.h)**2+(2/3)*dh*dh+e*e+4))
        self.row_bounds.update(C_AUG=self.row_bounds['AXIS_C']+e,F_POWER=np.sqrt(self.row_bounds['AXIS_F']**2+(e/self.w)**2),C_JOINT=self.row_bounds['POWER_C']+c+l)
        self.penalties={k:float(np.nextafter(np.sum(v)+1,np.inf)) for k,v in self.row_bounds.items()}

    def __call__(self,log10_rlc):
        x=np.asarray(log10_rlc,float).reshape(self.n,3)
        if not np.isfinite(x).all() or np.any(x<self.lower) or np.any(x>self.upper):raise ValueError('Candidate outside component bounds')
        p,a=port_modes(10**x);nu,gamma,w,B,h,logarea=features(p,a)
        u=np.log(nu[None,:]/self.nu[:,None]);g=np.log(gamma[None,:]/self.gamma[:,None]);z=B[None,:]/self.B[:,None]
        b=np.log(abs(z));d=abs(z/abs(z)-1);ec=-g-b;el=g+b-2*u
        eta=np.log(h[None,:]/self.h[:,None]);e=logarea[None,:]-self.log_area[:,None]
        costs=dict(AXIS_C=abs(b)+abs(ec)+abs(el)+d,AXIS_F=np.sqrt(b*b+(ec*ec+el*el)/(self.w[:,None]**2)+d*d),POWER_C=abs(u)+abs(eta)+abs(e)+d,QUANTILE_F=np.sqrt((u/self.h[:,None])**2+(2/3)*((h[None,:]-self.h[:,None])/self.h[:,None])**2+e*e+d*d))
        costs.update(C_AUG=costs['AXIS_C']+abs(e),F_POWER=np.sqrt(costs['AXIS_F']**2+(e/self.w[:,None])**2),C_JOINT=costs['POWER_C']+abs(ec)+abs(el))
        ck={'AXIS_POWER':'C_AUG','POWER_FINE':'AXIS_C','JOINT_POWER':'C_JOINT'}[self.method];fk='F_POWER' if self.method=='POWER_FINE' else 'AXIS_F'
        missing=self.n-len(p);values=[]
        for key in [ck,fk]:
            matrix=np.c_[costs[key],np.zeros((self.n,missing))] if missing else costs[key]
            i,j=linear_sum_assignment(matrix)
            values.append(float(np.sum(matrix[i,j])/self.n+missing*self.penalties[key]))
        value=np.asarray(values)
        if not np.isfinite(value).all():raise ArithmeticError('Nonfinite score')
        return value

"""Five target-fixed features extracted from ordered (log f, Re Z, Im Z) curves."""
from core import LegacyGeometry, CFG, np
from scipy.special import expit
from scipy.integrate import cumulative_trapezoid

def cumulative(y,x):
    return cumulative_trapezoid(y,x,axis=-1,initial=0.)

def radial_error(radius,target_radius,kind):
    extreme=np.max(radius,axis=-1) if kind=='peak' else np.min(radius,axis=-1)
    truth=np.max(target_radius) if kind=='peak' else np.min(target_radius)
    return abs(np.log(extreme/truth))/(np.log(10)*CFG['family']['radial_db_scale']/20)

def occupancy_error(x,radius,reference,kind):
    sign=1 if kind=='peak' else -1
    extreme=np.max(reference) if sign==1 else np.min(reference)
    scores=[];length=x[-1]-x[0]
    for level in CFG['family']['occupancy_levels_db']:
        threshold=extreme*10**(-sign*level/20)
        tau=np.log(10)*CFG['family']['occupancy_tau_db']/20
        a=cumulative(expit(sign*np.log(radius/threshold)/tau),x)
        ar=cumulative(expit(sign*np.log(reference/threshold)/tau),x)
        assert ar[-1]>0
        scores.append(np.trapezoid(abs(a-ar),x,axis=-1)/(length*ar[-1]))
    return np.mean(scores,axis=0)

def angular_error(x,phase,reference,theta):
    delta=np.diff(phase,axis=-1);dr=np.diff(reference)
    cumulative_error=np.zeros_like(phase)
    for sign in (1,-1):
        a=np.concatenate([np.zeros((*phase.shape[:-1],1)),np.cumsum(np.maximum(sign*delta,0),axis=-1)],axis=-1)
        ar=np.r_[0.,np.cumsum(np.maximum(sign*dr,0))]
        cumulative_error+=abs(a-ar)
    length=x[-1]-x[0]
    orientation=abs(np.exp(1j*phase)-np.exp(1j*reference))
    return np.trapezoid(cumulative_error,x,axis=-1)/(length*theta)+CFG['family']['angular_orientation_coefficient']*np.trapezoid(orientation,x,axis=-1)/length

def backtrack(a):
    return np.maximum((np.sum(abs(np.diff(a,axis=-1)),axis=-1)-abs(a[...,-1]-a[...,0]))/2,0)

def weighted_path_batch(t,mu,candidate,reference,eta):
    """Nonuniform target-fixed measure on both full monotone paths; normalized complex coordinates."""
    z=np.atleast_2d(candidate);n,batch=len(t),len(z)
    previous=np.full((n,batch),np.inf);previous_d=previous.copy()
    for i in range(n):
        current=np.full((n,batch),np.inf);ds=current.copy()
        lo,hi=np.searchsorted(t,[t[i]-eta-1e-12,t[i]+eta+1e-12],side='left')
        for j in range(int(lo),min(n,int(hi)+1)):
            if abs(t[i]-t[j])>eta+1e-12:continue
            d=(t[i]-t[j])**2+abs(z[:,i]-reference[j])**2;ds[j]=d
            if i==j==0:current[j]=0.;continue
            alternatives=[]
            if i:alternatives.append(previous[j]+(previous_d[j]+d)*(mu[i]-mu[i-1])/4)
            if j:alternatives.append(current[j-1]+(ds[j-1]+d)*(mu[j]-mu[j-1])/4)
            if i and j:alternatives.append(previous[j-1]+(previous_d[j-1]+d)*(mu[i]-mu[i-1]+mu[j]-mu[j-1])/4)
            if alternatives:current[j]=np.minimum.reduce(alternatives)
        previous,previous_d=current,ds
    result=np.sqrt(previous[-1]/(mu[-1]-mu[0]))
    if not np.isfinite(result).all():raise AssertionError('No admissible weighted path')
    return result

class FamilyGeometry(LegacyGeometry):
    def __init__(self,data,method='MAG_PHA_INTEGRAL'):
        super().__init__(data,method)
        self.is_legacy=method in ('MAG_PHA','MAG_PHA_MAX','MAG_PHA_INTEGRAL')
        cat={e['event_id']:e for e in data['catalog']}
        for w in self.windows:w['kind']=cat[w['event']]['kind']
        ordered=sorted(self.windows,key=lambda w:w['center'])
        self.span=(min(w['lo'] for w in ordered),max(w['hi'] for w in ordered))
        self.pairs=[]
        for left,right in zip(ordered[:-1],ordered[1:]):
            a=cat[left['event']]['right_3db_x'];b=cat[right['event']]['left_3db_x']
            edges=[left['lo'],a,b,right['hi']] if left['lo']<a<b<right['hi'] else [left['lo'],(left['center']+right['center'])/2,right['hi']]
            x=np.unique(np.r_[left['x'],right['x'],*[np.linspace(lo,hi,CFG['family']['bridge_points_per_block']) for lo,hi in zip(edges[:-1],edges[1:])]])
            self.pairs.append(dict(event=left['event']+'_'+right['event'],lo=edges[0],hi=edges[-1],x=x,edges=edges,left=left,right=right,width=min(left['width'],right['width']),center=(left['center']+right['center'])/2))
        # Full focus span is partitioned into amplitude windows and real intervening gaps.
        boundaries=sorted(set([v for w in ordered for v in (w['lo'],w['hi'])]))
        self.blocks=[]
        scale=max(w['scale'] for w in ordered)
        c=data['circuit']; xx=data['quadrature'].reference_grid(c.p,2)
        xx=xx[(xx>=self.span[0])&(xx<=self.span[1])]
        zz,dz=c.response(10**xx); derivative=np.imag(dz/zz)
        phase_turns=[]
        for i in np.flatnonzero(derivative[:-1]*derivative[1:]<0):
            phase_turns.append(float(xx[i]-derivative[i]*(xx[i+1]-xx[i])/(derivative[i+1]-derivative[i])))
        self.partition=np.unique(np.r_[self.span,[e['x'] for e in data['catalog'] if self.span[0]<e['x']<self.span[1]],phase_turns])
        for i,(lo,hi) in enumerate(zip(boundaries[:-1],boundaries[1:])):
            x=np.unique(np.r_[np.linspace(lo,hi,CFG['geometry']['points_per_window']),*[w['x'][(w['x']>=lo)&(w['x']<=hi)] for w in ordered]])
            z,_=c.response(10**x);turn=float(np.sum(abs(np.diff(np.angle(z)))))
            self.blocks.append(dict(event=f'block_{i}',lo=lo,hi=hi,x=x,scale=scale,target_turn_rad=turn,active=turn>=np.deg2rad(CFG['family']['angular_min_target_turn_deg'])))
        self.segments=[]
        for i,(lo,hi) in enumerate(zip(self.partition[:-1],self.partition[1:])):
            x=np.unique(np.r_[np.linspace(lo,hi,CFG['geometry']['points_per_window']),*[w['x'][(w['x']>=lo)&(w['x']<=hi)] for w in ordered]])
            z,_=c.response(10**x);lr=np.log(np.maximum(abs(z),max(1e-15,scale*1e-12)));phase=np.angle(z)
            denominators=[max(np.sum(abs(np.diff(lr))),np.log(10)*CFG['family']['tv_log_radius_floor_db']/20),max(np.sum(abs(np.diff(phase))),np.deg2rad(CFG['family']['tv_phase_floor_deg']))]
            self.segments.append(dict(event=f'segment_{i}',lo=lo,hi=hi,x=x,scale=scale,denominators=denominators))

    def aggregate(self,events):
        if self.is_legacy:return super().aggregate(events)
        return np.asarray(events).mean(axis=-1)

    def grid(self,win,p,points=None,anchors=False):
        x=win['x']
        if points:x=np.unique(np.r_[x,np.linspace(win['lo'],win['hi'],points)])
        if anchors:
            assert len(p)==1
            extra=[]
            for pars in (self.data['circuit'].p,p[0]):
                for fc,width in self.data['circuit'].modes(pars):
                    center=np.log10(fc)
                    if win['lo']<=center<=win['hi']:extra.extend(center+np.linspace(-width,width,33))
            a=np.asarray(extra);x=np.unique(np.r_[x,a[(a>=win['lo'])&(a<=win['hi'])]])
        return x

    def score(self,logp,points=None,anchors=False,return_events=False,physical_parameters=None):
        if self.is_legacy:return super().score(logp,points,anchors,return_events,physical_parameters)
        if self.method=='MAG_PHA_BRIDGE' and not self.pairs:
            # Exact C fallback is frozen for the one-event control.
            g=LegacyGeometry(self.data,'MAG_PHA_INTEGRAL')
            values=g.score(logp,points,anchors,return_events,physical_parameters);self.queries+=g.queries
            return values
        p=10**np.atleast_2d(logp) if physical_parameters is None else np.atleast_2d(physical_parameters)
        windows={'MAG_PHA_RADIAL':self.windows,'MAG_PHA_OCCUPANCY':self.windows,'MAG_PHA_ANGULAR':[w for w in self.blocks if w['active']],'MAG_PHA_BRIDGE':self.pairs,'MAG_PHA_EXCESS':self.segments}[self.method]
        if not windows:raise ValueError('No reliable target features for active arm')
        scores=[];c=self.data['circuit']
        for w in windows:
            x=self.grid(w,p,points,anchors);z,_=c.response(10**x,p);ref,_=c.response(10**x)
            self.queries+=len(p)*len(x)
            floor=max(1e-15,w.get('scale',max(a['scale'] for a in self.windows))*1e-12)
            r=np.maximum(abs(z),floor);rr=np.maximum(abs(ref),floor)
            if self.method=='MAG_PHA_RADIAL':score=radial_error(r,rr,w['kind'])
            elif self.method=='MAG_PHA_OCCUPANCY':score=occupancy_error(x,r,rr,w['kind'])
            elif self.method=='MAG_PHA_ANGULAR':score=angular_error(x,np.angle(z),np.angle(ref),max(w['target_turn_rad'],np.deg2rad(CFG['family']['angular_floor_deg'])))
            elif self.method=='MAG_PHA_EXCESS':
                score=np.zeros(len(p))
                for a,ar,den in zip([np.log(r),np.angle(z)],[np.log(rr),np.angle(ref)],w['denominators']):
                    score+=np.maximum(backtrack(a)-backtrack(ar)-CFG['family']['tv_noise_allowance'],0)/den
            elif self.method=='MAG_PHA_BRIDGE':
                left,right=w['left'],w['right'];edges=np.asarray(w['edges'])
                mu=np.interp(x,edges,np.linspace(0,1,len(edges)))
                scale=np.exp(np.interp(x,[left['center'],right['center']],np.log([left['scale'],right['scale']])))
                score=weighted_path_batch((x-w['center'])/w['width'],mu,z/scale,ref/scale,CFG['geometry']['frequency_slack_widths'])
            else:raise ValueError(self.method)
            scores.append(score)
        a=np.column_stack(scores)
        return a if return_events else self.aggregate(a)

    def features_description(self):
        clean=lambda w:{k:v for k,v in w.items() if k not in ('x','reference','left','right')}
        return dict(windows=[clean(w) for w in self.windows],angular_blocks=[clean(w) for w in self.blocks],bridge_pairs=[clean(w) for w in self.pairs],tv_segments=[clean(w) for w in self.segments])

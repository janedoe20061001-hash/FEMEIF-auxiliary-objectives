from core import *
from features import weighted_path_batch,radial_error,occupancy_error,angular_error,backtrack
import time

def enumeration(t,mu,z,ref,eta):
    costs=[]
    def d(i,j):return (t[i]-t[j])**2+abs(z[i]-ref[j])**2
    def visit(i,j,cost):
        if i==j==len(t)-1:costs.append(np.sqrt(cost/(mu[-1]-mu[0])));return
        for di,dj in ((1,0),(0,1),(1,1)):
            ii,jj=i+di,j+dj
            if ii>=len(t) or jj>=len(t) or abs(t[ii]-t[jj])>eta+1e-12:continue
            weight=(mu[ii]-mu[i]+mu[jj]-mu[j])/2
            visit(ii,jj,cost+(d(i,j)+d(ii,jj))*weight/2)
    visit(0,0,0.)
    return min(costs)

def check():
    rng=np.random.default_rng(202610112)
    for _ in range(16):
        t=np.array([-1.,-.31,.2,1.]);mu=np.array([0.,.13,.76,1.])
        z=rng.normal(size=4)+1j*rng.normal(size=4);ref=rng.normal(size=4)+1j*rng.normal(size=4)
        np.testing.assert_allclose(weighted_path_batch(t,mu,z,ref,.8)[0],enumeration(t,mu,z,ref,.8),rtol=1e-12)
        assert weighted_path_batch(t,mu,ref,ref,.8)[0]==0
    x=np.linspace(-2,2,1001);r=np.exp(-x*x)+.1
    np.testing.assert_allclose(radial_error(np.array([r*2,r/2]),r,'peak'),np.log(2)/(np.log(10)*3/20))
    assert occupancy_error(x,r[None,:],r,'peak')[0]==0
    assert occupancy_error(x,(np.exp(-(x/.5)**2)+.1)[None,:],r,'peak')[0]>0
    assert occupancy_error(x,(np.exp(-(x-.3)**2)+.1)[None,:],r,'peak')[0]>0
    phase=np.arctan(x);theta=sum(abs(np.diff(phase)))
    assert angular_error(x,phase[None,:],phase,theta)[0]==0
    assert angular_error(x,(phase+.2)[None,:],phase,theta)[0]>0
    assert angular_error(x,(phase+.3*np.sin(12*x))[None,:],phase,theta)[0]>0
    np.testing.assert_allclose(backtrack(np.array([[0.,1.,2.],[0.,1.,0.]])),[0.,1.])
    assert backtrack(np.zeros(5))==0 # known regularizer blindness
    records=[];predictions={}
    # Base predictions are conservative fixed relative setup sizes, not based on new results.
    base_seconds={'case1':210.,'ladder8':180.,'mixed0p25':175.,'three0p25':220.,'control0p25':50.}
    for ident in CFG['targets']:
        data=target(ident);q,c=data['quadrature'],data['circuit'];truth=np.log10(c.p)
        np.testing.assert_allclose(q.score(truth),0,atol=1e-20)
        assert np.all(truth>=data['lower']) and np.all(truth<=data['upper'])
        xx=rng.uniform(data['lower'],data['upper'],(12,len(truth)))
        begin=time.perf_counter();q.score(xx);base_time=time.perf_counter()-begin
        timings={};predictions[ident]={}
        for method in CFG['methods']:
            g=Geometry(data,method)
            exact=float(g.score(truth,physical_parameters=c.p)[0])
            assert exact==0.,(ident,method,exact)
            begin=time.perf_counter();scores=g.score(xx);duration=time.perf_counter()-begin
            assert np.isfinite(scores).all() and np.all(scores>=0)
            parts=g.score(xx,return_events=True)
            np.testing.assert_allclose(scores,g.aggregate(parts),rtol=1e-12,atol=1e-12)
            timings[method]=dict(seconds_12_candidates=duration,exact_identity=exact,roundtrip_score=float(g.score(truth)[0]),components=parts.shape[1])
            predictions[ident][method]=base_seconds[ident]*(1+(0 if method=='MAG_PHA' else duration/max(base_time,1e-9)))
        g=Geometry(data)
        if ident=='control0p25':np.testing.assert_array_equal(g.score(xx),Geometry(data,'MAG_PHA_BRIDGE').score(xx))
        records.append(dict(target=ident,features=g.features_description(),timings=timings,base_seconds_12_candidates=base_time))
    dump(OUT/'PREFLIGHT.json',dict(passed=True,independent_nonuniform_path_enumerations=16,synthetic_checks=['identity','half_double_radius','occupancy_shift_and_width','angle_offset_and_reversal','TV_net_variation','flat_regularizer_blindness','single_event_bridge_equals_C'],targets=records,predicted_seconds=predictions,source_hashes=hashes()))
    print('All seven arms and five targets passed preflight',flush=True)

if __name__=='__main__':check()

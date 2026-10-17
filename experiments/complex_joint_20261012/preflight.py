"""Independent identities/model checks and short cost benchmark, no formal optimizer."""
from core import *
from features import Features,areas,frechet,frechet_python
import time,sys,traceback
from importlib.metadata import version

def installed(name):
    from importlib.metadata import PackageNotFoundError
    try:return version(name)
    except PackageNotFoundError:return None

def main():
    rng=np.random.default_rng(20261012);checks=[];timings={};nodal_fallbacks=[]
    # Area oracle: polygon shoelace for each prefix, independent of vectorized code.
    t=np.linspace(0,1,17);z=rng.normal(size=17)+1j*rng.normal(size=17)
    a=areas(t,z)[0];q=np.column_stack([t,z.real,z.imag]);q=q-q[0]
    oracle=np.zeros_like(a)
    for k in range(1,len(t)):
        for j,(u,v) in enumerate(((0,1),(0,2),(1,2))):
            polygon=q[:k+1][:,[u,v]]
            oracle[j,k]=.5*sum(polygon[i,0]*polygon[(i+1)%len(polygon),1]-polygon[i,1]*polygon[(i+1)%len(polygon),0] for i in range(len(polygon)))
    np.testing.assert_allclose(a,oracle,atol=1e-13)
    np.testing.assert_allclose(areas(t,z+3+4j),areas(t,z),atol=1e-13)
    dm=z[1:-1]-z[:-2];dp=z[2:]-z[1:-1];h=dm.conj()*dp
    np.testing.assert_allclose(h.real,dm.real*dp.real+dm.imag*dp.imag,atol=1e-13)
    np.testing.assert_allclose(h.imag,dm.real*dp.imag-dm.imag*dp.real,atol=1e-13)
    # Identity Frechet, a translated path and native/Python conformance.
    zz=np.stack([z,z+.2+.1j]);fp=frechet_python(t,zz,z)
    np.testing.assert_allclose(frechet(t,zz,z),fp,rtol=1e-12,atol=1e-12)
    assert fp[0]==0 and fp[1]>0
    checks+=['shoelace prefixes','Hermitian dot/cross','documented translation blindness','Frechet oracle/native']
    for ident in target_ids():
        d=target(ident);c=d['circuit'];q=d['quadrature'];g=Features(d)
        f=np.geomspace(100,1e6,11)
        # Verify boundary and random candidates with an independent nodal solve.
        samples=np.vstack([np.log10(c.p),d['lower'],d['upper'],rng.uniform(d['lower'],d['upper'])])
        for lp in samples:
            v,_=c.response(f,10**lp)
            nd=checked_nodal(c,f,10**lp)
            if nd['mode']!='double':nodal_fallbacks.append(dict(target=ident,log_parameters=lp.tolist(),check=nd))
        if d['kind']!='measured':
            truth=np.log10(c.p)[None,:]
            assert np.max(q.score(truth))<1e-15
            for method in CFG['methods'][1:]:
                score=g.score(truth,method)
                if method=='PTC_ref':assert max(score)<1e-8 # same catalogue/grid may carry sub-grid peak displacement
                else:assert max(score)<(1e-10 if method=='C_ref' else 1e-15),(ident,method,score)
        else:
            from case2_legacy import PARAMETER_NAMES,impedance
            for lp in samples:
                p=dict(zip(c.names,10**lp));v,_=c.response(f,10**lp)
                np.testing.assert_allclose(v,impedance(f,np.array([p[n] for n in PARAMETER_NAMES]))[0],rtol=1e-11,atol=1e-8)
        probe=rng.uniform(d['lower'],d['upper'],size=(2,len(c.p)))
        st=time.perf_counter();base=q.score(probe);bt=(time.perf_counter()-st)/len(probe)
        assert np.isfinite(base).all() and (base>=0).all()
        costs={}
        for method in CFG['methods'][1:]:
            st=time.perf_counter();v=g.score(probe,method);costs[method]=(time.perf_counter()-st)/len(probe)
            assert np.isfinite(v).all() and (v>=0).all(),(ident,method)
        timings[ident]={m:(bt+costs.get(m,0))*60000 for m in CFG['methods']}
        dump(INPUTS/ident/'feature_scales.json',g.description())
        print(json.dumps(dict(target=ident,base_seconds_per_candidate=bt,aux_seconds_per_candidate=costs)),flush=True)
    # Compiled Case1 vs independent Python evaluator on identical random parameters.
    d=target('case1');a=d['quadrature'].adaptive
    if (FROZEN/'native.so').exists():
        sample=np.vstack([np.log10(d['circuit'].p),rng.uniform(d['lower'],d['upper'],size=(4,18))])
        np.testing.assert_allclose(a.score(sample),a.score_python(sample),rtol=1e-10,atol=1e-9)
        checks.append('Case1 compiled adaptive vs Python')
    knee.__globals__['checks']()
    rec=dict(passed=True,checks=checks,targets=len(target_ids()),timings_seconds_per_job=timings,
        nodal_fallbacks=nodal_fallbacks,versions={n:installed(n) for n in ('numpy','scipy','pymoo','moocore','matplotlib','psutil','mpmath')},hashes=hashes(),python=sys.version)
    dump(OUT/'PREFLIGHT.json',rec)
    print('PREFLIGHT PASSED',flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        dump(OUT/'PREFLIGHT_FAILED.json',dict(traceback=traceback.format_exc()))
        raise

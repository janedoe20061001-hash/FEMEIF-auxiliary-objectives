"""Explain a frozen local-area counterexample without changing the metric."""
from local_geometry import *

def second_derivative(model,w,t,p):
    lo,hi=w['x'][[0,-1]];k=np.log(10)*(hi-lo);freq=10**(lo+(hi-lo)*t);omega=2*np.pi*freq
    z=zp=zpp=0j
    for r,l,c in np.asarray(p).reshape(6,3):
        b=omega*c-1/(omega*l);bp=k*(omega*c+1/(omega*l));bpp=k*k*b
        v=1/(1/r+1j*b);z+=v;zp+=-1j*bp*v*v;zpp+=-1j*bpp*v*v-2*bp*bp*v*v*v
    return z/w['scale'],zp/w['scale'],zpp/w['scale']

def path_prefix(model,w,p):
    e=model.g.panel_edges(w,model.g.modes(p),2);t,_=nodes(e,64);z,dz=model.response(w,t,p);z0,_=model.response(w,np.array([0.]),p)
    return Prefix(e,.5*np.imag((z-z0[0]).conj()*dz),64),z0[0]

def main():
    model=LocalGeometry();bank=json.loads((OUT/'SCORING_BANK.json').read_text());pair=next(p for p in bank['pairs'] if p['pair_id']=='pair_001')
    w=next(w for w in model.windows if w['block']['label']=='P3');event=next(e for e in model.data['record']['events'] if e['event_id']=='P3')
    lo,hi=w['x'][[0,-1]];center=(event['x']-lo)/(hi-lo);rows=[]
    for side,p in [('target',model.c.p),('better',10**np.array(pair['better']['log_parameters'])),('worse',10**np.array(pair['worse']['log_parameters']))]:
        pref,z0=path_prefix(model,w,p);_,first,second=second_derivative(model,w,center,p)
        # Analytic derivative agrees with the old derivative at the same frequency.
        _,check=model.response(w,np.array([center]),p);np.testing.assert_allclose(first,check[0],rtol=1e-8,atol=1e-8)
        eps=1e-5;_,fplus,_=second_derivative(model,w,center+eps,p);_,fminus,_=second_derivative(model,w,center-eps,p)
        np.testing.assert_allclose((fplus-fminus)/(2*eps),second,rtol=1e-5,atol=1e-6)
        for h in model.steps:
            zz,_=model.response(w,np.array([center-h,center+h]),p)
            actual=float(pref(np.array([center+h]))[0]-pref(np.array([center-h]))[0]-.5*np.imag((zz[0]-z0).conj()*(zz[1]-zz[0])))
            approximation=float(2/3*h**3*np.imag(np.conj(first)*second))
            tt=np.linspace(center-h,center+h,8193);zz,_=model.response(w,tt,p)
            polygon=float(.5*np.sum(np.imag((zz[:-1]-zz[0]).conj()*np.diff(zz))))
            assert abs(actual-polygon)<=1e-10+1e-5*abs(actual)
            rows.append(dict(side=side,h=h,center_t=float(center),center_hz=event['frequency_hz'],signed_local_area=actual,
                derivative_cross=float(np.imag(np.conj(first)*second)),small_h_approximation=approximation,direct_polygon_area=polygon,second_derivative_finite_difference_passed=True))
    ratios=[]
    for h in model.steps:
        ref=next(r['signed_local_area'] for r in rows if r['side']=='target' and r['h']==h)
        ratios.append(dict(h=h,better_area_over_target=next(r['signed_local_area'] for r in rows if r['side']=='better' and r['h']==h)/ref,
            worse_area_over_target=next(r['signed_local_area'] for r in rows if r['side']=='worse' and r['h']==h)/ref))
    dump(OUT/'CURVATURE_COUNTEREXAMPLE.json',dict(pair_id=pair['pair_id'],bank_sha256=sha(OUT/'SCORING_BANK.json'),
        formula='a_local(t,h)=(2/3)*h^3*Im(conj(z_prime)*z_second)+higher-order terms; derived by Taylor expansion',
        evaluation='Same target-centered frequency; no candidate alignment; fixed original h values',rows=rows,ratios=ratios))
    print(json.dumps(ratios),flush=True)

if __name__=='__main__':main()

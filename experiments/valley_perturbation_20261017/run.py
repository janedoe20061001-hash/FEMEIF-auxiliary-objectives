"""Execute fixed physical interventions and retain every numerical result."""
from physical import *

def anchors(c,ref,t):
    x=np.array([t['core'][0],t['center'],t['core'][1]]);z,d=rx(c,x);r,dr=rx(ref,x);tol=1e-12*t['scale']/(t['core'][1]-t['core'][0]);cos=[]
    for a,b in zip(d,dr):cos.append(float(np.real(a.conjugate()*b)/(abs(a)*abs(b))) if abs(a)>tol and abs(b)>tol else None)
    return dict(x=x,real=z.real,imag=z.imag,target_real=r.real,target_imag=r.imag,error_complex=abs(z-r),mean_error_over_target_scale=float(np.mean(abs(z-r))/t['scale']),tangent_cosines=cos)

def compare_scopes(a,b,span):
    rs1,rs2=a['radial_extrema'],b['radial_extrema'];rootok=len(rs1)==len(rs2) and all(u['kind']==v['kind'] and abs(u['x']-v['x'])<=1e-9 for u,v in zip(rs1,rs2))
    quantileok=(a['arc_quantiles'] is None and b['arc_quantiles'] is None) or (a['arc_quantiles'] is not None and b['arc_quantiles'] is not None and all(abs(u-v)<=1e-4*span+1e-3*max(a['arc80_width'],b['arc80_width']) for u,v in zip(a['arc_quantiles'],b['arc_quantiles'])))
    cdfok=(a['cdf_L1'] is None and b['cdf_L1'] is None) or (a['cdf_L1'] is not None and b['cdf_L1'] is not None and close(a['cdf_L1'],b['cdf_L1']))
    return dict(length=close(a['length'],b['length']),root_sequence=rootok,arc_quantiles=quantileok,cdf=cdfok,zero_motion=a['zero_motion']==b['zero_motion'])

def profiles_needed(item):
    f=item['family'];m=item['meta']
    return f in ['baseline','background'] or (f=='shift' and abs(m['shift_widths'])==1) or (f=='offset' and m['relative_offset']==.5) or (f=='Q' and m['factor'] in [.125,8]) or (f=='extra' and m['relative_R']==1 and m['shift_widths']==0 and m['width_fraction'] in [.01,.2]) or (f=='damping' and m['component']=='R3' and m['factor']==4) or (f=='combined' and m['shift_widths']==1 and m['relative_offset']==.5 and m['Q_factor']==4)

def one(item,t):
    c=Physical(item['model']);ref=Physical(t['model']);start=time.time();levels=[];lastprofiles={}
    try:
        for n in [128,256,512,1024]:
            scopes={};pp={}
            for scope in ['core','window']:
                lo,hi=t[scope];x=grid(c,ref,lo,hi,n);x=np.unique(np.r_[x,t['center']] if lo<=t['center']<=hi else x);stat,profile=scope_stats(c,ref,x,t['scale']);scopes[scope]=stat;pp[scope]=profile
            levels.append(dict(n=n,scopes=scopes));lastprofiles=pp
            if len(levels)>1:
                cmp={scope:compare_scopes(levels[-2]['scopes'][scope],scopes[scope],t[scope][1]-t[scope][0]) for scope in ['core','window']}
                if all(all(v.values()) for v in cmp.values()):break
        gaussian={}
        for scope in ['core','window']:
            lo,hi=t[scope];checks=[]
            for n,rf in [(32,1),(64,2),(64,3),(64,4)]:
                L=gauss_length(c,ref,lo,hi,n,rf);checks.append(dict(order=n,refinement=rf,length=L))
                if len(checks)>1 and close(checks[-2]['length'],L) and close(scopes[scope]['length'],L):break
            gaussian[scope]=dict(levels=checks,passed=bool(close(checks[-2]['length'],checks[-1]['length']) and close(scopes[scope]['length'],checks[-1]['length'])))
        identity=analytic_identity(c,ref,item,t)
        # Circuit nodal solve samples exact target anchors, distributed samples and extrema.
        x=lastprofiles['window']['x'];rr=scopes['window']['radial_extrema'];ff=10**np.unique(np.r_[x[np.unique(np.linspace(0,len(x)-1,13).astype(int))],t['center'],t['core'],[q['x'] for q in rr]])
        nodal=checked_nodal(c,ff,c.p);numerical=all(all(v.values()) for v in cmp.values()) and all(v['passed'] for v in gaussian.values()) and identity.get('passed',True)
        result=dict(id=item['id'],target=item['target'],family=item['family'],meta=item['meta'],levels=levels,convergence=cmp,gauss=gaussian,analytic_identity=identity,anchors=anchors(c,ref,t),nodal=nodal,numerical_passed=bool(numerical),seconds=time.time()-start)
    except Exception as exc:
        import traceback
        result=dict(id=item['id'],target=item['target'],family=item['family'],meta=item['meta'],numerical_passed=False,error=repr(exc),traceback=traceback.format_exc(),seconds=time.time()-start)
    dump(OUT/'records'/f"{item['id']}.json",result)
    return result,lastprofiles

def main():
    plan=json.loads((OUT/'PLAN.json').read_text(encoding='utf-8'));targets={t['id']:t for t in plan['targets']};frozen=json.loads((OUT/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'));assert all(sha(ROOT/p)==h for p,h in frozen['files'].items())
    st=time.time();good=0;arrays={};results=[]
    for i,item in enumerate(plan['tasks'],1):
        result,profile=one(item,targets[item['target']]);results.append(result);good+=int(result['numerical_passed'])
        if profiles_needed(item):
            for scope,pp in profile.items():
                for name,v in pp.items():
                    if v is not None:arrays[item['id']+'__'+scope+'__'+name]=v
        dump(OUT/'STATUS.json',dict(completed=i,total=len(plan['tasks']),numerical_passed=good,seconds=time.time()-st))
        if i%16==0 or not result['numerical_passed']:print(i,'/',len(plan['tasks']),'pass',good,'latest',result['id'],result['numerical_passed'],flush=True)
    np.savez_compressed(OUT/'PROFILES.npz',**arrays);dump(OUT/'SUMMARY.json',dict(tasks=len(results),numerical_passed=good,seconds=time.time()-st,failed_ids=[r['id'] for r in results if not r['numerical_passed']],source_unchanged=all(sha(ROOT/p)==h for p,h in frozen['files'].items()),new_optimization=False,new_auxiliary_score=False));print(json.dumps(json.loads((OUT/'SUMMARY.json').read_text())),flush=True)
if __name__=='__main__':main()

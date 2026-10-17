"""Post hoc decomposition of the two frozen P3 local-area counterexamples."""
from curvature_probe import *

def main():
    model=LocalGeometry();bank=json.loads((OUT/'SCORING_BANK.json').read_text());rows=[]
    for pid in ('pair_001','pair_002'):
        pair=next(p for p in bank['pairs'] if p['pair_id']==pid);w=next(w for w in model.windows if w['block']['label']=='P3');j=model.windows.index(w)
        params=[model.c.p,10**np.array(pair['better']['log_parameters']),10**np.array(pair['worse']['log_parameters'])]
        primitives=[path_prefix(model,w,p) for p in params]
        e=np.unique(np.concatenate([model.g.panel_edges(w,model.g.modes(p),2) for p in params]))
        for h in model.steps:
            t,wt=nodes(shifted(e,h),64);wt/=1-2*h;values=[]
            for p,(pref,z0) in zip(params,primitives):
                zz,_=model.response(w,np.r_[t-h,t+h],p);a,b=np.split(zz,2)
                values.append(pref(t+h)-pref(t-h)-.5*np.imag((a-z0).conj()*(b-a)))
            ref,good,bad=values;den=model.denominator(j,h,t)
            lg=((good-ref)/den)**2;lb=((bad-ref)/den)**2;delta=wt*(lg-lb)
            positive=np.maximum(delta,0)
            overshoot=(ref*good>0)&(abs(good)>2*abs(ref))&(abs(ref)>1e-14)
            rows.append(dict(pair_id=pid,h=h,better_loss=float(wt@lg),worse_loss=float(wt@lb),loss_delta=float(sum(delta)),
                fraction_positive_penalty_from_same_sign_more_than_twice_target_area=float(sum(positive[overshoot])/sum(positive)) if sum(positive)>0 else None))
    dump(OUT/'AREA_RESIDUAL_DECOMPOSITION.json',dict(bank_sha256=sha(OUT/'SCORING_BANK.json'),scope='Post hoc mechanism probe on the first two frozen P3 failures; not independent validation or metric tuning',rows=rows))
    print(json.dumps(rows),flush=True)

if __name__=='__main__':main()

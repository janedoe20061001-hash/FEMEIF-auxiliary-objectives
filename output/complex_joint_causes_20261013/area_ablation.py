"""Dissect frozen A into its three components on identical existing candidates."""
from investigate import *
from scipy.stats import spearmanr

rows=[];windows=[];ranks=[];validation=[];matched=[]
for ident in ['mixed_resonators/base','mixed_resonators/lc_reciprocal_8p0']:
    d=target(ident);c=d['circuit'];ff=Features(d);pool=[];labels=[]
    for m in CFG['methods']:
        for s in CFG['seeds']:
            p,F,a=load(ident,m,s);pool.append(p);labels.append((m,s,a))
    pool=np.array(pool);whole=[];half=[];j0fine=[];j0anchors=[]
    dh,_=ff.components(np.log10(pool),names=('D','H'),level=32)
    for w in ff.windows:
        x=np.linspace(w['x'][0],w['x'][-1],4097);t=np.linspace(0,1,len(x));r,_=c.response(10**x);z,_=c.response(10**x,pool)
        aa=areas(t,z/w['scale']);ar=areas(t,r/w['scale'])[0]
        vv=average((aa-ar)**2,t)/w['area_den'][None,:]**2
        whole.append(vv);j0fine.append(average(abs(z-r)**2,t)/w['scale']**2)
        a16=areas(t[::2],z[:,::2]/w['scale']);r16=areas(t[::2],r[::2]/w['scale'])[0]
        half.append(average((a16-r16)**2,t[::2])/w['area_den'][None,:]**2)
        # Linear area expression verified against exact polygon/trapezoid identity.
        re=(z.real-z.real[:,:1])/w['scale'];xx=t[None,:]
        cum=np.column_stack([np.zeros(len(z)),np.cumsum((re[:,1:]+re[:,:-1])/2*np.diff(t),axis=1)])
        ident_err=float(np.max(abs(aa[:,0,:]-(xx*re/2-cum))))
        assert ident_err<1e-8
        validation.append(dict(target=ident,window=w['block']['label'],linear_area_identity_absolute_error=ident_err))
        for i,(m,s,a) in enumerate(labels):
            windows.append(dict(target=ident,method=m,seed=s,window=w['block']['label'],AtR=vv[i,0],AtX=vv[i,1],ARX=vv[i,2],
                denominator_AtR=w['area_den'][0],denominator_AtX=w['area_den'][1],denominator_ARX=w['area_den'][2]))
    val=np.mean(whole,axis=0);v16=np.mean(half,axis=0);j0fine=np.mean(j0fine,axis=0)
    band=(45000,58000) if ident.endswith('/base') else (38500,45000)
    xx=np.linspace(*np.log10(band),4097);ref,_=c.response(10**xx);zz,_=c.response(10**xx,pool)
    local=np.trapezoid(abs(zz-ref)**2,x=xx,axis=1)/(xx[-1]-xx[0])/max(abs(ref))**2
    for i,(m,s,a) in enumerate(labels):
        relevant=[e for e in a['amplitude_events'] if e['event_id'] in ('P2','V3')]
        rows.append(dict(target=ident,method=m,seed=s,AtR=val[i,0],AtX=val[i,1],ARX=val[i,2],
            A_full=float(np.mean(val[i])),A_linear=float(np.mean(val[i,:2])),J0_32=j0fine[i],D_32=dh['D'][i],H_32=dh['H'][i],
            ARX_fraction=float(val[i,2]/max(sum(val[i]),1e-30)),pair_failed=sum(not e['complete'] for e in relevant),
            mag=a['objective'][0],pha=a['objective'][1],local_complex_error=local[i],
            max_component_16_32_relative=float(max(abs(val[i]-v16[i])/np.maximum(abs(val[i]),1e-12))),
            components_converged=bool(np.all(abs(val[i]-v16[i])<=.01+.01*abs(val[i])))))
    rr=[r for r in rows if r['target']==ident]
    for k in ['AtR','AtX','ARX','A_full','A_linear','J0_32','D_32','H_32']:
        ranks.append(dict(target=ident,score=k,n=len(rr),rho_pair_failed=float(spearmanr([r[k] for r in rr],[r['pair_failed'] for r in rr]).statistic),
            rho_local_error=float(spearmanr([r[k] for r in rr],[r['local_complex_error'] for r in rr]).statistic)))
    for threshold in (1.1,1.25):
        pairs=[(a,b) for i,a in enumerate(rr) for b in rr[i+1:]
            if max(a['mag'],b['mag'])<=threshold*min(a['mag'],b['mag'])
            and max(a['pha'],b['pha'])<=threshold*min(a['pha'],b['pha']) and a['pair_failed']!=b['pair_failed']]
        for k in ['AtR','AtX','ARX','A_full','A_linear','J0_32','D_32','H_32']:
            good=sum((a[k]-b[k])*(a['pair_failed']-b['pair_failed'])>0 for a,b in pairs)
            matched.append(dict(target=ident,threshold_ratio=threshold,score=k,comparable_pairs=len(pairs),correct_order=good,
                fraction=good/len(pairs) if pairs else None,note='Overlapping exploratory pairs from developed candidates; not independent trials'))
    print('area ablation',ident,flush=True)
save('A_COMPONENT_SCORES.csv',rows);save('A_COMPONENT_WINDOWS.csv',windows);save('A_COMPONENT_RANKS.csv',ranks);save('AREA_IDENTITY_CHECK.csv',validation);save('MSE_MATCHED_RANKING.csv',matched)

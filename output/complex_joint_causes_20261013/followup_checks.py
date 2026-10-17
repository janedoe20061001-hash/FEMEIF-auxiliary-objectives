from investigate import *
from scipy.stats import spearmanr

def score_reliability():
    rows=[];cors=[];dominance=[];cost=[]
    for ident in ['mixed_resonators/base','mixed_resonators/lc_reciprocal_8p0']:
        d=target(ident);c=d['circuit'];ff=Features(d);pool=[];labels=[]
        for m in CFG['methods']:
            for s in CFG['seeds']:
                p,F,a=load(ident,m,s);pool.append(p);labels.append((m,s,a))
        lp=np.log10(pool)
        r16,details=ff.components(lp,level=16);r32,_=ff.components(lp,level=32)
        for i,(m,s,a) in enumerate(labels):
            ank=a['anchored_geometry'];ok={k:abs(r32[k][i]-r16[k][i])<=.01+.01*abs(r32[k][i]) and abs(r32[k][i]-ank[k])<=.01+.01*abs(ank[k]) for k in ('D','H','A')}
            rows.append(dict(target=ident,method=m,seed=s,**{f'{k}_stable':ok[k] for k in ok},
                **{f'{k}_16_32_relative':abs(r32[k][i]-r16[k][i])/max(abs(r32[k][i]),1e-12) for k in ok},
                **{f'{k}_32_oldanchor_relative':abs(r32[k][i]-ank[k])/max(abs(ank[k]),1e-12) for k in ok}))
            for k in ('D','H','A'):
                vals=np.array([w['scores'][k][i] for w in details]);j=int(np.argmax(vals));local=sum(w['scores'][k][i] for w in details if w['label'] in ('P2','V3'))
                dominance.append(dict(target=ident,method=m,seed=s,score=k,dominant_window=details[j]['label'],dominant_fraction=float(vals[j]/max(sum(vals),1e-30)),
                    P2_V3_fraction=float(local/max(sum(vals),1e-30))))
        print('reliability',ident,flush=True)
    save('SCORE_RELIABILITY.csv',rows);save('SCORE_DOMINANCE.csv',dominance)
    old=list(csv.DictReader((OUT/'FIXED_CANDIDATE_SCORES.csv').open(encoding='utf-8-sig')))
    keys={(r['target'],r['method'],str(r['seed'])):r for r in rows}
    for ident in ['mixed_resonators/base','mixed_resonators/lc_reciprocal_8p0']:
        # Common stable subset to compare all geometric scores on the identical pool.
        rr=[r for r in old if r['target']==ident and all(keys[ident,r['method'],r['seed']][f'{k}_stable'] for k in ('D','H','A'))]
        for k in ['J0','D','A','H','mag','pha']:
            for other in ['pair_failed','J0']:
                rho=spearmanr([float(r[k]) for r in rr],[float(r[other]) for r in rr]).statistic
                cors.append(dict(target=ident,score=k,against=other,n=len(rr),spearman_rho=rho,subset='D,H,A pass 16/32 and archived anchored agreement at original thresholds'))
    save('STABLE_SCORE_CORRELATIONS.csv',cors)

def case1_windows():
    d=target('case1');c=d['circuit'];ff=Features(d);rows=[]
    for m in ['D','H','A','D_H','D_A','H_A','D_H_A']:
        for s in CFG['seeds']:
            p,F,a=load('case1',m,s);pp=p.reshape(6,3);fc=1/(2*np.pi*np.sqrt(pp[:,1]*pp[:,2]))
            for w in ff.windows:
                mask=(np.log10(fc)>=w['x'][0])&(np.log10(fc)<=w['x'][-1])
                if not any(mask):continue
                z,_=c.response(10**w['x'],p);ze,_=c.response(fc[mask],p)
                rows.append(dict(method=m,seed=s,window=w['block']['label'],scale_ohm=w['scale'],
                    coarse_max_ohm=float(max(abs(z))),branch_center_max_ohm=float(max(abs(ze))),
                    center_vs_coarse_db=float(20*np.log10(max(abs(ze))/max(abs(z)))),
                    center_over_target_scale=float(max(abs(ze))/w['scale']),num_branches=int(sum(mask))))
    save('CASE1_WINDOW_PEAKS.csv',rows)

if __name__=='__main__':score_reliability();case1_windows()

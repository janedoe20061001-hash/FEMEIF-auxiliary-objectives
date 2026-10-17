"""Post hoc decomposition of the frozen B score; does not change any gate."""
from common import *
import csv

def main():
    m=Geometry(control_target());c=m.c;w=next(w for w in m.windows if w['block']['label']=='P_control');j=m.windows.index(w)
    rows=[]
    for offset in [0,.5,1,2,4]:
        p=c.params(1,offset,1)
        for hi,h in enumerate(m.steps):
            edges=shifted(m.g.panel_edges(w,m.g.modes(p),2),h);t,wt=nodes(edges,64);wt/=1-2*h
            def bending(parameters):
                z=m.response(w,np.r_[t-h,t,t+h],parameters)[0];zm,zc,zp=np.split(z,3);return zc-(zm+zp)/2
            a=bending(p);b=bending(c.p);den=m.bden[j,hi]**2
            ea=float(wt@abs(a)**2/den);eb=float(wt@abs(b)**2/den);cross=float(wt@np.real(a*b.conj())/den);loss=float(wt@abs(a-b)**2/den)
            assert abs(loss-(ea+eb-2*cross))<=1e-10+1e-10*loss
            rows.append(dict(offset=offset,h=h,candidate_energy=ea,target_energy=eb,real_cross_correlation=cross,loss=loss))
    source=list(csv.DictReader((OUT/'CONTROLLED_SCORES.csv').open(encoding='utf-8-sig')))
    means=[]
    for offset in [0,.5,1,2,4]:
        rr=[r for r in rows if r['offset']==offset];v={k:float(np.mean([r[k] for r in rr])) for k in ['candidate_energy','target_energy','real_cross_correlation','loss']}
        expected=float(next(r for r in source if r['family']=='controlled_grid' and float(r['amplitude'])==1 and float(r['offset'])==offset and float(r['q_ratio'])==1)['B'])
        assert abs(v['loss']-expected)<=1e-10+1e-9*expected;means.append(dict(offset=offset,**v,original_B=expected))
    dump(OUT/'B_TRANSLATION_DECOMPOSITION.json',dict(scope='Post hoc explanation of already-scored controlled shifts; not new validation or tuning',identity='loss = candidate energy + target energy - 2 real cross correlation',per_h=rows,means=means))
    bad=next(r for r in source if r['family']=='controlled_grid' and float(r['amplitude'])==.1 and float(r['offset'])==0 and float(r['q_ratio'])==1)
    near=[r for r in source if r['family']=='controlled_grid' and float(r['offset'])==0 and .75<=float(r['amplitude'])<=1.25 and .5<=float(r['q_ratio'])<=2]
    result={}
    for metric in ['Local_ARX','Root','B','J0']:
        reversed_rows=[dict(amplitude=float(r['amplitude']),q_ratio=float(r['q_ratio']),score_ratio=float(r[metric])/float(bad[metric])) for r in near if float(r[metric])>float(bad[metric])]
        result[metric]=dict(comparisons=len(near),reversals=reversed_rows)
    dump(OUT/'CONTROLLED_EXPLORATORY_CONTRASTS.json',dict(scope='Post hoc slice of the frozen 648-setting grid; not independent trials or a new acceptance gate',reference=dict(amplitude=.1,offset=0,q_ratio=1),results=result))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False})
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained');x=[r['offset'] for r in means]
    axs[0].plot(x,[r['loss'] for r in means],marker='o',color='#2878a4');axs[0].set(xlabel='中心偏移 / 目标对数半功率带宽',ylabel='B局部损失',title='只移动谐振中心：偏得更远，B分数反而降低')
    axs[1].plot(x,[r['candidate_energy'] for r in means],marker='.',label='候选B的平方积分');axs[1].plot(x,[r['target_energy'] for r in means],ls='--',label='目标B的平方积分');axs[1].plot(x,[-2*r['real_cross_correlation'] for r in means],marker='.',label='负两倍实交叉项')
    axs[1].set(xlabel='中心偏移 / 目标对数半功率带宽',ylabel='同一固定尺度下的无量纲项',title='几何平方积分分解；不表示电路储能');axs[1].legend(fontsize=8)
    fig.savefig(OUT/'figures/B_translation_decomposition.png',dpi=160);plt.close(fig)
    for v in means:print(json.dumps(v),flush=True)

if __name__=='__main__':main()

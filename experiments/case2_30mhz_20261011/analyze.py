from core import *
import csv,shutil,re,zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
DATA=HERE/'server_run/retrieved/case2_30mhz_20261011'
OLD=ROOT/'experiments/teacher_doubts_validation_20261011/server_run/retrieved/teacher_doubts_validation_20261011/runs/full/case2'
METHODS=['scalar_ga','scalar_nsga','dual_nsga']
LABEL={'scalar_ga':'WSMSE / 传统GA','scalar_nsga':'WSMSE / 当前框架','dual_nsga':'双目标 / 膝点'}
COL={'scalar_ga':'#d97706','scalar_nsga':'#059669','dual_nsga':'#2563eb'}

def csvwrite(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def verify():
    status=json.loads((DATA/'STATUS.json').read_text());assert status['status']=='complete' and status['completed']==300 and not status['failed']
    m=model('case2','upto30m');rows=[];best={};total=0;individuals=0;stages=[]
    for method in METHODS:
        for seed in range(2300,2400):
            d=DATA/'runs/upto30m/case2'/method/f'seed_{seed}';old=OLD/method/f'seed_{seed}'
            x0=np.load(d/'initial_log_parameters.npy');np.testing.assert_array_equal(x0,np.load(old/'initial_log_parameters.npy'))
            np.testing.assert_array_equal(x0,np.random.default_rng(seed).uniform(m.lower,m.upper,(150,15)))
            rec=json.loads((d/'summary.json').read_text());assert rec['evaluations']==150000 and rec['status']=='complete';total+=rec['evaluations']
            with np.load(d/'trajectory.npz') as a:gens=a['generations'];tx=a['log_parameters'];tv=a['metrics']
            assert np.array_equal(gens,np.arange(20,1001,20))
            np.testing.assert_allclose(tv,metrics(m.spectrum(tx.reshape(-1,15)),m.target,m.alpha).reshape(tv.shape),rtol=1e-11,atol=1e-9);individuals+=100
            for i,g in enumerate(gens):stages.append(dict(method=method,seed=seed,generation=int(g),**dict(zip(METRICS,tv[i,0].tolist()))))
            for g in [300,1000]:
                with np.load(d/f'generation_{g:04}.npz') as a:x=a['log_parameters'];v=a['metrics'];f=a['optimization_objective']
                independent=metrics(m.spectrum(x),m.target,m.alpha)
                np.testing.assert_allclose(v,independent,rtol=1e-11,atol=1e-9);np.testing.assert_allclose(f,v[:,:2] if method=='dual_nsga' else v[:,2:3],rtol=1e-11,atol=1e-9)
                assert x.shape==(150,15) and np.all(x>=m.lower) and np.all(x<=m.upper);individuals+=150
                ki,reason=knee(v[:,:2]) if method=='dual_nsga' else (int(np.argmin(v[:,2])),'scalar_minimum')
                np.testing.assert_array_equal(x[ki],tx[list(gens).index(g),0])
                with np.load(old/f'generation_{g:04}.npz') as a:ox=a['log_parameters'];ov=a['metrics']
                oi,_=knee(ov[:,:2]) if method=='dual_nsga' else (int(np.argmin(ov[:,2])),'scalar_minimum')
                oldcommon=m.score(ox[oi])[0]
                z=m.spectrum(x[ki])[0];amp=20*np.log10(abs(z)/abs(m.target));phase=np.rad2deg(np.angle(z)-m.phase)
                row=dict(method=method,seed=seed,generation=g,selection_index=ki,reason=reason,**dict(zip(METRICS,v[ki].tolist())),
                         old_full_training_common183_wsmse=float(oldcommon[2]),max_amplitude_db=float(abs(amp).max()),max_phase_deg=float(abs(phase).max()))
                rows.append(row);key=(method,g)
                if key not in best or (row['wsmse'],seed)<(best[key][0]['wsmse'],best[key][0]['seed']):best[key]=(row,x[ki].copy())
    assert total==45000000 and individuals==120000
    csvwrite(OUT/'all_seed_metrics.csv',rows);csvwrite(OUT/'all_stage_metrics.csv',stages)
    rng=np.random.default_rng(20261011);summary=[]
    for method in METHODS:
        for g in [300,1000]:
            group=[r for r in rows if r['method']==method and r['generation']==g];a=np.array([r['wsmse'] for r in group]);b=np.array([r['old_full_training_common183_wsmse'] for r in group]);delta=a-b
            boot=delta[rng.integers(0,100,(10000,100))].mean(axis=1);rec,x=best[method,g]
            summary.append(dict(method=method,generation=g,mean_metrics=[float(np.mean([r[k] for r in group])) for k in METRICS],median_wsmse=float(np.median(a)),best={**rec,'log_parameters':x.tolist()},
                                knee_fallbacks=sum(r['reason']!='convex_knee' for r in group) if method=='dual_nsga' else None,
                                old_full_common183_mean=float(b.mean()),mean_cap_minus_full=float(delta.mean()),paired_ci95=np.quantile(boot,[.025,.975]).tolist(),cap_better_seeds=int(sum(delta<0))))
    dump(OUT/'SUMMARY.json',summary);dump(OUT/'VALIDATION.json',dict(passed=True,runs=300,formal_evaluations=total,individuals_recomputed=individuals,initialization_same_as_prior=True,points=183,frequency_cap_hz=30000000))
    return rows,stages,best,summary

def plots(best,stages):
    m=model('case2','upto30m');(OUT/'figures').mkdir(exist_ok=True)
    for g in [300,1000]:
        fig,axes=plt.subplots(2,2,figsize=(13,7.4),layout='constrained');zs={method:m.spectrum(best[method,g][1])[0] for method in METHODS}
        for row in range(2):
            for col in range(2):
                ax=axes[row,col];ax.semilogx(m.f,20*np.log10(abs(m.target)) if row==0 else np.rad2deg(m.phase),color='#172033',lw=1.7,label='目标曲线',zorder=5)
                for method in METHODS:
                    z=zs[method];r=best[method,g][0];label=LABEL[method]+f' · seed {r["seed"]}'
                    if method=='dual_nsga' and r['reason']!='convex_knee':label+='（退化折中）'
                    ax.semilogx(m.f,20*np.log10(abs(z)) if row==0 else np.rad2deg(np.angle(z)),color=COL[method],lw=1.1,label=label)
                ax.set(xlabel='频率 / Hz',ylabel='幅值 / dBΩ' if row==0 else '相位 / °',xlim=(40 if col==0 else 1e6,3e7));ax.grid(True,alpha=.2);ax.spines[['top','right']].set_visible(False)
        axes[0,0].legend(fontsize=8);axes[0,0].set_title('40 Hz–30 MHz');axes[0,1].set_title('1–30 MHz局部放大')
        fig.suptitle(f'CASE2 · 拟合上限30 MHz · {g}代 · 每组100seed最终代表中的最佳WSMSE',fontsize=13)
        fig.savefig(OUT/'figures'/f'case2_30mhz_best_{g}.png',dpi=160);plt.close(fig)
        csvwrite(OUT/f'curves_{g}.csv',[dict(frequency_hz=float(f),target_real=float(m.target[i].real),target_imag=float(m.target[i].imag),
                 **{method+'_real':float(z[i].real) for method,z in zs.items()},**{method+'_imag':float(z[i].imag) for method,z in zs.items()}) for i,f in enumerate(m.f)])
        csvwrite(OUT/f'parameters_{g}.csv',[dict(parameter=n,unit='ohm' if n[0]=='R' else 'H' if n[0]=='L' else 'F',**{method:float(10**best[method,g][1][i]) for method in METHODS}) for i,n in enumerate(m.record['parameter_names'])])
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    for method in METHODS:
        group=[r for r in stages if r['method']==method];gens=sorted(set(r['generation'] for r in group));v=np.array([[r['wsmse'] for r in group if r['generation']==g] for g in gens])
        axes[0].semilogy(gens,v.mean(axis=1),color=COL[method],label=LABEL[method]);axes[1].semilogy(gens,v.min(axis=1),color=COL[method],label=LABEL[method])
    for ax in axes:ax.axvline(300,color='#6b7280',ls=':');ax.set(xlabel='代数',ylabel='WSMSE');ax.grid(True,alpha=.2)
    axes[0].set_title('100 seed平均');axes[1].set_title('100 seed最好');axes[0].legend(fontsize=8)
    fig.suptitle('CASE2 · 40 Hz–30 MHz · 双目标每阶段选膝点');fig.savefig(OUT/'figures/evolution.png',dpi=160);plt.close(fig)

def main():
    for name in ['STATUS.json','PREFLIGHT.json','MANIFEST.json','SERVER_FILES.json']:shutil.copy2(DATA/name,OUT/name)
    rows,stages,best,summary=verify()
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False;plots(best,stages)
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()

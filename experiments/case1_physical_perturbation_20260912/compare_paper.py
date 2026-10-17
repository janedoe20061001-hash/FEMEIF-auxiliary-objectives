"""Evaluate the paper's five auxiliary costs on the existing 1,801 parameter sets.

No fitting/optimization is performed. Original rows and primary losses are preserved.
Run using the same Python as run_study.py. See comparison_report.md for conventions.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import run_study as base

HERE = Path(__file__).resolve().parent
OUT = HERE / 'paper_comparison'
PAPER = ['SS', 'PTC', 'DTW', 'LFP_loss', 'LMEP']
ALL = base.NAMES + PAPER
PRIMARY = ['magnitude_mse_db2', 'phase_mse_rad2']
LABEL = dict(H_resonance='谐振位置', H_antiresonance='抵消位置',
             H_bandwidth='相对带宽', SS='SS 趋势', PTC='PTC 峰谷位置',
             DTW='DTW 对齐距离', LFP_loss='LFP 未通过比例', LMEP='LMEP 最大误差')


def save(name, data):
    (OUT / name).write_text(json.dumps(base.serializable(data), ensure_ascii=False,
                                     indent=2, allow_nan=False), encoding='utf-8')


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -700, 700)))


def pattern(a, x):
    """Paper Table I, Eqs. 3-5. Only N-2 interior patterns are defined."""
    k = np.diff(a) / np.diff(x)
    c, n = k[:-1], k[1:]
    s = np.zeros_like(c)
    down = (n < -8) & (c <= 8) | (c < -8) & (n <= 8)
    up = (n > 8) & (c >= -8) | (c > 8) & (n >= -8)
    for sign, mask in [(-1, down), (1, up)]:
        d = n[mask] - c[mask]
        s[mask] = sign * (1 + .4 * (2 * sigmoid(.5 * sign * d) - 1))
    for sign, mask in [(1, (c < -8) & (n > 8)),
                       (-1, (c > 8) & (n < -8))]:
        d = n[mask] - c[mask]
        s[mask] = sign * (1 + .6 * sigmoid(.5 * sign * (d - 2 * sign * 8)))
    return s


def extrema(a, x):
    # No undocumented 0.2 dB prominence or extra root refinement is introduced.
    peaks = np.flatnonzero((a[1:-1] >= a[:-2]) & (a[1:-1] > a[2:])) + 1
    troughs = np.flatnonzero((a[1:-1] <= a[:-2]) & (a[1:-1] < a[2:])) + 1
    ids = np.r_[peaks, troughs]
    types = np.r_[np.ones(len(peaks), int), -np.ones(len(troughs), int)]
    order = np.argsort(ids)
    return x[ids[order]], types[order]


class Reference:
    def __init__(self, n):
        self.x = np.linspace(2, 6, n)
        self.f = 10 ** self.x
        self.a = 20 * np.log10(np.abs(base.impedance(self.f, base.TARGET)))
        self.s = pattern(self.a, self.x)
        self.events, self.types = extrema(self.a, self.x)
        self.edges = np.r_[2, (self.events[1:] + self.events[:-1]) / 2, 6]
        self.splits = np.searchsorted(self.x, self.edges[1:-1], side='left')

    def evaluate(self, params):
        a = 20 * np.log10(np.abs(base.impedance(self.f, params)))
        e = a - self.a
        events, types = extrema(a, self.x)
        cost, missing = 0., 0
        for i, (f, typ) in enumerate(zip(self.events, self.types)):
            mask = ((events >= self.edges[i]) & (events < self.edges[i + 1])
                    & (types == typ))
            if not np.any(mask):
                cost += .41
                missing += 1
            else:
                cost += np.expm1(.41 * np.abs(events[mask] - f) / 4).sum()
        zone_mse = np.array([np.mean(v ** 2) for v in np.split(e, self.splits)])
        failures = np.flatnonzero(zone_mse >= 1.6)
        count = int(failures[0]) if len(failures) else len(zone_mse)
        # Eq. 6 divides by N_M; uncomputable final two patterns are omitted.
        return dict(SS=float(np.abs(pattern(a, self.x) - self.s).sum() / len(a)),
                    PTC=float(cost), LFP_loss=1-count/len(zone_mse),
                    LFP_count=count, LMEP=float(np.max(np.abs(e))),
                    PTC_missing=missing, extrema_count=len(events))


def batch_dtw(candidates, target):
    """Unconstrained absolute-distance DTW sums, vectorized over parameter sets."""
    count, n = candidates.shape
    prev = np.full((count, len(target)+1), np.inf)
    prev[:, 0] = 0
    for i in range(n):
        curr = np.full_like(prev, np.inf)
        for j in range(len(target)):
            curr[:, j+1] = (np.minimum(np.minimum(prev[:, j+1], curr[:, j]), prev[:, j])
                           + np.abs(candidates[:, i] - target[j]))
        prev = curr
    return prev[:, -1]


def verify(ref):
    checks = {}
    z = ref.evaluate(base.TARGET)
    assert all(abs(z[k]) < 1e-12 for k in PAPER if k != 'DTW')
    assert z['LFP_count'] == 11
    checks['target_zero_cost_and_11_passed_zones'] = True
    # SS is invariant to constant dB offsets.
    shift = np.max(np.abs(pattern(ref.a + 2, ref.x) - ref.s))
    assert shift < 1e-7
    checks['SS_offset_invariance_max_difference'] = shift
    # Table I reversal signs and steady +/- slopes.
    x = np.arange(3.)
    assert pattern(np.array([0., 20., 40.]), x)[0] == 1
    assert pattern(np.array([0., -20., -40.]), x)[0] == -1
    assert 1.4 < pattern(np.array([0., -20., 0.]), x)[0] < 1.6
    assert -1.6 < pattern(np.array([0., 20., 0.]), x)[0] < -1.4
    checks['SS_table_sign_and_reversal_checks'] = True
    assert batch_dtw(np.array([[0., 0., 1.]]), np.array([0., 1., 1.]))[0] == 0
    assert batch_dtw(np.array([[0., 1., 2.]]), np.array([0., 1., 2.]))[0] == 0
    assert batch_dtw(np.array([[3.]]), np.array([1.]))[0] == 2
    checks['DTW_alignment_identity_and_absolute_cost_checks'] = True
    return checks


def analyze(rows):
    stats, correlations, mono = [], [], []
    for fam in ['R', 'L', 'C', 'all']:
        for level in base.LEVELS:
            rs = [r for r in rows if r['kind']=='random' and r['family']==fam and r['level_pct']==level]
            for key in PRIMARY + ALL:
                v = [r[key] for r in rs]
                stats.append(dict(family=fam, level_pct=level, metric=key, n=len(rs),
                                  median=float(np.median(v)), p10=float(np.percentile(v,10)),
                                  p90=float(np.percentile(v,90)), zero_count=int(np.sum(np.abs(v)<=1e-10))))
        for metric in ALL:
            violations, flat = 0, 0
            for i in range(64):
                rs = sorted([r for r in rows if r['kind']=='random' and r['family']==fam
                             and r['direction_id']==f'{fam}_{i}'], key=lambda r:r['level_pct'])
                v = np.array([r[metric] for r in rs])
                violations += int(np.any(np.diff(v) < -1e-10))
                flat += int(np.all(np.abs(np.diff(v)) <= 1e-10))
            mono.append(dict(family=fam, metric=metric, directions=64,
                             nonmonotonic_directions=violations, fully_flat_directions=flat))
    for level in base.LEVELS:
        rs = [r for r in rows if r['kind']=='random' and r['family']=='all' and r['level_pct']==level]
        for a in ALL:
            for b in PRIMARY + ALL:
                correlations.append(dict(level_pct=level, a=a, b=b,
                     spearman=base.spearman([r[a] for r in rs], [r[b] for r in rs])))
    base.write_csv(OUT/'summary.csv', stats)
    base.write_csv(OUT/'correlations.csv', correlations)
    base.write_csv(OUT/'monotonicity.csv', mono)
    return stats, correlations, mono


def figures(rows, stats, correlations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':font_manager.FontProperties(
        fname='C:/Windows/Fonts/msyh.ttc').get_name(), 'axes.unicode_minus':False})
    keys = PRIMARY + ALL
    labels = dict(LABEL, magnitude_mse_db2='幅值 MSE [dB²]', phase_mse_rad2='相位 MSE [rad²]')
    fig, axes = plt.subplots(2,5,figsize=(17,7.4),layout='constrained')
    for ax,key in zip(axes.flat,keys):
        ss = [s for s in stats if s['family']=='all' and s['metric']==key]
        xx=[s['level_pct'] for s in ss]
        ax.plot(xx,[s['median'] for s in ss], 'o-',color='#246a9a')
        ax.fill_between(xx,[s['p10'] for s in ss],[s['p90'] for s in ss],alpha=.2,color='#246a9a')
        ax.set_xscale('log')
        ax.set_xticks(base.LEVELS, ['0.1','0.3','1','3','10','30'])
        if key!='LFP_loss':
            ax.set_yscale('log')
        else:
            ax.set_ylim(-.03,1.03)
        ax.set_title(labels[key],fontsize=12)
        ax.set_xlabel('扰动档位 [%]'); ax.grid(alpha=.2)
    fig.suptitle('同一批 L、C、R 联合随机扰动：64 个方向的中位数与 10%–90% 区间\n'
                 '各指标分别看变化趋势；纵轴数值大小不能用于跨指标排名',fontsize=15)
    fig.savefig(OUT/'response_comparison.png',dpi=175);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,6.5),layout='constrained',
                          gridspec_kw={'width_ratios':[1,1.45]})
    c=np.array([[next(t['spearman'] for t in correlations if t['level_pct']==1 and t['a']==a and t['b']==b)
                 or 0 for b in PRIMARY] for a in ALL])
    im=axes[0].imshow(c,vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto')
    axes[0].set_xticks([0,1],['幅值 MSE','相位 MSE'])
    axes[0].set_yticks(range(8),[LABEL[a] for a in ALL])
    for i in range(8):
        for j in range(2):
            raw=next(t['spearman'] for t in correlations if t['level_pct']==1 and t['a']==ALL[i] and t['b']==PRIMARY[j])
            axes[0].text(j,i,'无变化' if raw is None else f'{raw:.2f}',ha='center',va='center',
                         color='white' if raw is not None and abs(raw)>.7 else 'black')
    axes[0].set_title('固定 1% 档位：与主目标的秩相关\n64 个联合随机方向，未混合不同档位')
    fig.colorbar(im,ax=axes[0],shrink=.5)
    for metric,col in [('PTC','#246a9a'),('LFP_loss','#b66a28')]:
        med=[]; med180=[]
        for level in base.LEVELS:
            rr=[r for r in rows if r['kind']=='random' and r['family']=='all' and r['level_pct']==level]
            med.append(np.mean([abs(r[metric])<=1e-10 for r in rr])*100)
            med180.append(np.mean([abs(r[metric+'_180'])<=1e-10 for r in rr])*100)
        axes[1].plot(base.LEVELS,med,'o-',color=col,label=f'{metric}: 65,537 点')
        axes[1].plot(base.LEVELS,med180,'s--',color=col,label=f'{metric}: 180 点')
    axes[1].set_xscale('log');axes[1].set_xticks(base.LEVELS,['0.1','0.3','1','3','10','30'])
    axes[1].set_ylim(-3,103);axes[1].set_ylabel('指标仍为最优值的样本比例 [%]')
    axes[1].set_xlabel('扰动档位 [%]');axes[1].grid(alpha=.2);axes[1].legend(fontsize=9)
    axes[1].set_title('“指标没变”有多少来自门槛或采样？\n两种网格都由本次构造，并非作者原始频率表')
    fig.savefig(OUT/'resolution_and_correlation.png',dpi=170);plt.close(fig)


def main():
    t=time.perf_counter();OUT.mkdir(exist_ok=True)
    source=HERE/'results/all_runs.jsonl'
    rows=[json.loads(s) for s in source.read_text(encoding='utf-8').splitlines() if s.strip()]
    ref=Reference(65537); coarse=Reference(180)
    checks=verify(ref)
    for i,r in enumerate(rows):
        params=np.array(r['params'])
        r.update(ref.evaluate(params))
        r.update({k+'_180':v for k,v in coarse.evaluate(params).items()})
        r['sample_id']=i
        if i%300==0:
            print(f'{i}/{len(rows)} spectra evaluated',flush=True)
    for n,key in [(60,'DTW'),(180,'DTW_180points')]:
        f=np.geomspace(100,1e6,n)
        target=20*np.log10(np.abs(base.impedance(f,base.TARGET)))
        candidates=np.array([20*np.log10(np.abs(base.impedance(f,np.array(r['params'])))) for r in rows])
        vals=batch_dtw(candidates,target)
        for r,v in zip(rows,vals):
            r[key]=float(v)
    assert len(rows)==1801 and rows[0]['DTW']==0
    (OUT/'all_runs.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False,allow_nan=False)
                                             for r in rows)+'\n',encoding='utf-8')
    scalar=[{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in rows]
    # Optional component exists only for single-parameter rows; unify columns.
    fields=list(dict.fromkeys(k for r in scalar for k in r))
    base.write_csv(OUT/'all_runs.csv',[{k:r.get(k,'') for k in fields} for r in scalar])
    stats,correlations,mono=analyze(rows)
    figures(rows,stats,correlations)
    checks.update(sample_count=len(rows), original_samples_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  copied_original_parameters_and_primary_losses=True,
                  target_extrema_dense=len(ref.events),target_extrema_180=len(coarse.events),
                  elapsed_seconds=time.perf_counter()-t)
    save('checks.json',checks)
    save('config.json',dict(primary='Copied unchanged from original dense study',
        amplitude='20log10(abs(Z)), dB',frequency='log10(f/Hz), [2,6]',
        dense_points=65537,coarse_points=180,DTW_points=60,
        THk=8,beta=.4,mu=.5,gamma=.41,THlf=1.6,
        SS_endpoint_rule='sum over N-2 available patterns, divide by N',
        PTC='sum all same-type within target peak/trough midpoint zones; missing .41; no peak refinement',
        LFP='consecutive magnitude-MSE zones <1.6; report loss 1-count/11',
        DTW='unconstrained absolute-dB-cost path sum; direct evaluation at log-uniform 60 points',
        limitations='Paper does not provide author code or exact frequency coordinates; not exact replication. '
                    'Auxiliary formulas interpreted on magnitude dB. Coarse grid is a sensitivity check.'))
    print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':
    main()

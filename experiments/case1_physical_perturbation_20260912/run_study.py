"""Case I physical-feature perturbation study; no evolutionary optimization.

Run with the bundled Python. Only numpy is needed for calculations.
Plotting additionally uses matplotlib (a workspace-local installation is supported).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL_DEPS = ROOT / 'tmp' / 'perturbation_deps'
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
# Rows retain the component labels in Table III, not resonance-frequency order.
# Columns: R [ohm], L [H], C [F]. Six parallel RLC blocks connected in series.
TARGET = np.array([
    [2000., 1585e-6, 10e-6],
    [3000., 10e-6, 3.98e-6],
    [500., 100e-6, 1e-6],
    [1200., 2.512e-6, .631e-6],
    [800., 1.259e-6, .501e-6],
    [50., .251e-6, .158e-6],
])
LEVELS = [.1, .3, 1., 3., 10., 30.]
NAMES = ['H_resonance', 'H_antiresonance', 'H_bandwidth']
MEASURES = ['magnitude_mse_db2', 'phase_mse_rad2',
            'magnitude_rmse_db', 'phase_rmse_rad'] + NAMES


def impedance(f, params):
    """The candidate's terminal response; no target parameters used here."""
    w = 2 * np.pi * np.asarray(f, dtype=float)
    z = np.zeros_like(w, dtype=complex)
    for r, l, c in params:
        z += 1 / (1 / r + 1j * (w * c - 1 / (w * l)))
    return z


def roots(f, z, level, response=None):
    """Roots of X-level*R. Return log-frequency roots and crossing directions.

    Default: interpolation of the given spectrum, O(n).
    Optional response callback: 28 simultaneous bisections on bracketed roots,
    used ONLY as a numerical-reference refinement for this synthetic study.
    """
    v = z.imag - level * z.real
    ids = np.flatnonzero(np.signbit(v[:-1]) != np.signbit(v[1:]))
    lo, hi = np.log(f[ids]), np.log(f[ids + 1])
    direction = np.where(v[ids + 1] > v[ids], 1, -1)
    if response is not None and len(ids):
        vlo = v[ids].copy()
        for _ in range(28):
            mid = (lo + hi) / 2
            zm = response(np.exp(mid))
            vm = zm.imag - level * zm.real
            left = np.signbit(vlo) != np.signbit(vm)
            hi = np.where(left, mid, hi)
            lo = np.where(left, lo, mid)
            vlo = np.where(left, vlo, vm)
        return (lo + hi) / 2, direction
    a = -v[ids] / (v[ids + 1] - v[ids])
    return lo + a * (hi - lo), direction


def features(f, z, response=None):
    all_zero, directions = roots(f, z, 0, response)
    res = all_zero[directions < 0]  # inductive -> capacitive, high-impedance modes
    anti = all_zero[directions > 0]  # capacitive -> inductive, inter-mode cancellation
    plus, pd = roots(f, z, 1, response)  # phase = +pi/4
    minus, md = roots(f, z, -1, response)  # phase = -pi/4
    plus, minus = plus[pd < 0], minus[md < 0]
    width = np.full(len(res), np.nan)
    endpoints = np.full((len(res), 2), np.nan)
    for k, r in enumerate(res):
        before, after = all_zero[all_zero < r - 1e-12], all_zero[all_zero > r + 1e-12]
        lb = before[-1] if len(before) else np.log(f[0])
        rb = after[0] if len(after) else np.log(f[-1])
        a, b = plus[(plus > lb) & (plus < r)], minus[(minus > r) & (minus < rb)]
        if len(a) and len(b):
            endpoints[k] = [a[-1], b[0]]
            width[k] = np.exp(b[0] - r) - np.exp(a[-1] - r)
    return dict(res=res, anti=anti, width=width, endpoints=endpoints,
                crossing_count=len(all_zero))


def match_locations(candidate, target):
    # Disjoint target windows; an observed event cannot match two targets.
    edges = np.r_[np.log(100), (target[:-1] + target[1:]) / 2, np.log(1e6)]
    matched = np.full(len(target), np.nan)
    which = np.full(len(target), -1, dtype=int)
    for k, t in enumerate(target):
        valid = ((candidate >= edges[k]) &
                 ((candidate < edges[k + 1]) if k + 1 < len(target)
                  else (candidate <= edges[k + 1])))
        ids = np.flatnonzero(valid)
        if len(ids):
            j = ids[np.argmin(np.abs(candidate[ids] - t))]
            matched[k], which[k] = candidate[j], j
    return matched, which


def feature_losses(candidate, target):
    res, ids = match_locations(candidate['res'], target['res'])
    anti, _ = match_locations(candidate['anti'], target['anti'])
    width = np.full(len(ids), np.nan)
    for k, j in enumerate(ids):
        if j >= 0:
            width[k] = candidate['width'][j]
    dr = res - target['res']
    da = anti - target['anti']
    dw = np.log(width / target['width'])
    vals = [dr, da, dw]
    # One missing feature costs 1; log deviations above 1 are saturated.
    losses = [float(np.mean(np.where(np.isfinite(v), np.minimum(np.abs(v), 1), 1)))
              for v in vals]
    return dict(zip(NAMES, losses)) | {
        'missing_resonance': int(np.isnan(res).sum()),
        'missing_antiresonance': int(np.isnan(anti).sum()),
        'missing_bandwidth': int(np.isnan(width).sum()),
        'resonance_count': len(candidate['res']),
        'antiresonance_count': len(candidate['anti']),
        'crossing_count_deficit_10': max(10 - candidate['crossing_count'], 0) / 10,
        'signed_log_resonance': dr.tolist(),
        'signed_log_antiresonance': da.tolist(),
        'signed_log_bandwidth': dw.tolist(),
    }


def mse(z, reference):
    da = 20 * np.log10(np.maximum(np.abs(z), 1e-300)) - 20 * np.log10(np.maximum(np.abs(reference), 1e-300))
    dp = np.angle(z) - np.angle(reference)  # same phase convention as the Chinese PDF
    a, p = float(np.mean(da * da)), float(np.mean(dp * dp))
    return {'magnitude_mse_db2': a, 'phase_mse_rad2': p,
            'magnitude_rmse_db': math.sqrt(a), 'phase_rmse_rad': math.sqrt(p)}


def scenarios(random_directions):
    yield dict(kind='baseline', family='baseline', level_pct=0., direction_id='target', sign=0, params=TARGET.copy())
    for branch in range(6):
        for col, component in enumerate(['R', 'L', 'C']):
            for sign in [-1, 1]:
                for level in LEVELS:
                    p = TARGET.copy()
                    p[branch, col] *= np.exp(sign * np.log1p(level / 100))
                    yield dict(kind='single', family=component, component=f'{component}{branch+1}',
                               level_pct=level, direction_id=f'{component}{branch+1}_{sign}', sign=sign, params=p)
    # Exact circuit transformations serve as mechanistic controls.
    for name in ['global_frequency', 'global_impedance', 'global_LC_ratio', 'global_R']:
        for sign in [-1, 1]:
            for level in LEVELS:
                p = TARGET.copy()
                factor = np.exp(sign * np.log1p(level / 100))
                if name == 'global_frequency':
                    p[:, 1:] *= factor
                elif name == 'global_impedance':
                    p[:, :2] *= factor
                    p[:, 2] /= factor
                elif name == 'global_LC_ratio':
                    p[:, 1] *= factor
                    p[:, 2] /= factor
                else:
                    p[:, 0] *= factor
                yield dict(kind='structured', family=name, level_pct=level,
                           direction_id=f'{name}_{sign}', sign=sign, params=p)
    rng = np.random.default_rng(20260912)
    for fam in ['R', 'L', 'C', 'all']:
        mask = np.ones((6, 3)) if fam == 'all' else np.tile(np.array([fam == x for x in ['R','L','C']]), (6, 1))
        for seed in range(random_directions):
            d = rng.uniform(-1, 1, size=(6, 3)) * mask
            d /= np.max(np.abs(d))  # same fixed direction across all levels
            for level in LEVELS:
                p = TARGET * np.exp(np.log1p(level / 100) * d)
                yield dict(kind='random', family=fam, level_pct=level,
                           direction_id=f'{fam}_{seed}', sign=0, params=p)


def rankdata(a):
    a = np.asarray(a)
    ids = np.argsort(a, kind='stable')
    ranks = np.empty(len(a), dtype=float)
    ordered = a[ids]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    ends = np.r_[starts[1:], len(a)]
    for s, e in zip(starts, ends):
        ranks[ids[s:e]] = (s + e - 1) / 2
    return ranks


def spearman(a, b):
    x, y = rankdata(a), rankdata(b)
    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def serializable(x):
    if isinstance(x, dict):
        return {k: serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [serializable(v) for v in x]
    if isinstance(x, np.ndarray):
        return serializable(x.tolist())
    if isinstance(x, (np.integer, np.floating)):
        return serializable(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def save_json(name, data):
    (OUT / name).write_text(json.dumps(serializable(data), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def summarize(rows):
    summary = []
    for kind, fam, level in sorted({(r['kind'], r['family'], r['level_pct']) for r in rows}):
        rs = [r for r in rows if (r['kind'], r['family'], r['level_pct']) == (kind, fam, level)]
        s = dict(kind=kind, family=fam, level_pct=level, count=len(rs))
        for field in MEASURES:
            a = [r[field] for r in rs]
            for q, label in [(10, 'p10'), (50, 'median'), (90, 'p90')]:
                s[f'{field}_{label}'] = float(np.percentile(a, q))
        s['rows_with_missing_feature'] = sum(any(r[k] for k in ['missing_resonance','missing_antiresonance','missing_bandwidth']) for r in rs)
        summary.append(s)
    write_csv(OUT / 'summary.csv', summary)
    correlations = []
    for level in LEVELS:
        rs = [r for r in rows if r['kind'] == 'random' and r['family'] == 'all' and r['level_pct'] == level]
        for a in NAMES:
            for b in ['magnitude_mse_db2','phase_mse_rad2'] + NAMES:
                correlations.append(dict(level_pct=level, a=a, b=b,
                                         spearman=spearman([r[a] for r in rs], [r[b] for r in rs])))
    write_csv(OUT / 'correlations_fixed_level.csv', correlations)
    monotonic = []
    for family in ['R','L','C','all']:
        rs = [r for r in rows if r['kind']=='random' and r['family']==family]
        for metric in MEASURES:
            violations = []
            for seed in sorted({r['direction_id'] for r in rs}):
                values = [r[metric] for r in sorted((r for r in rs if r['direction_id']==seed), key=lambda r:r['level_pct'])]
                violations.append(any(b + 1e-10 < a for a,b in zip(values[:-1],values[1:])))
            monotonic.append(dict(family=family, metric=metric, directions=len(violations),
                                  nonmonotonic_directions=sum(violations)))
    write_csv(OUT / 'monotonicity.csv', monotonic)
    return summary, correlations, monotonic


def checks(f, reference, ref_features, rows):
    z0 = impedance(f, TARGET)
    assert np.allclose(z0, reference)
    zero = feature_losses(ref_features, ref_features)
    assert all(zero[n] == 0 for n in NAMES)
    assert len(ref_features['res']) == 6 and len(ref_features['anti']) == 5
    assert np.all(np.isfinite(ref_features['width']))
    # Permuting series blocks must not change response.
    permutation_error = float(np.max(np.abs(impedance(f, TARGET[::-1]) - reference)) / np.max(np.abs(reference)))
    assert permutation_error < 1e-13
    # Exact scaling identities independently check circuit and perturbation implementation.
    p = TARGET.copy(); a=1.1
    p[:,:2] *= a; p[:,2] /= a
    scale_error = float(np.max(np.abs(impedance(f,p)-a*reference))/np.max(np.abs(reference)))
    assert scale_error < 1e-11
    p=TARGET.copy();p[:,1:]*=a
    shift_error=float(np.max(np.abs(impedance(f,p)-impedance(a*f,TARGET)))/np.max(np.abs(reference)))
    assert shift_error < 1e-11
    # Single parallel RLC: +/-45-degree relative bandwidth exactly equals 1/Q.
    p=np.array([[100.,1e-3,1e-6]])
    ff=np.geomspace(100,1e6,8193)
    ft=features(ff,impedance(ff,p),lambda x:impedance(x,p))
    known=1/(100*np.sqrt(1e-6/1e-3))
    bandwidth_error=float(abs(ft['width'][0]/known-1))
    assert bandwidth_error<1e-8
    # Do not pool perturbation levels for this numerical convergence check.
    selected=[]
    for family in ['R','L','C','all']:
        for level in [.1,1.,30.]:
            selected.extend([r for r in rows if r['kind']=='random' and r['family']==family and r['level_pct']==level][:2])
    selected += [r for r in rows if r['kind']=='structured' and r['level_pct'] in [.1,30.] and r['sign']==1]
    fd=np.geomspace(100,1e6,2*len(f)-1);zd0=impedance(fd,TARGET)
    convergence=[]
    for r in selected:
        p=np.array(r['params']);zd=impedance(fd,p)
        m=mse(zd,zd0)
        ft=features(fd,zd,lambda x:impedance(x,p))
        h=feature_losses(ft,ref_features)
        z=impedance(f,p)
        hi=feature_losses(features(f,z),features(f,reference))
        record=dict(direction_id=r['direction_id'],family=r['family'],level_pct=r['level_pct'])
        for key in ['magnitude_mse_db2','phase_mse_rad2']:
            record[key+'_relative_grid_change']=abs(m[key]-r[key])/max(abs(m[key]),1e-14)
            record[key+'_absolute_grid_change']=abs(m[key]-r[key])
        for key in NAMES:
            record[key+'_refined_grid_change']=abs(h[key]-r[key])
            record[key+'_interpolation_only_error']=abs(hi[key]-r[key])
        convergence.append(record)
    write_csv(OUT/'numerical_convergence.csv',convergence)
    report=dict(target_zero_passed=True, target_crossings=[6,5],
                permutation_relative_error=permutation_error,
                impedance_scale_relative_error=scale_error,
                frequency_scale_relative_error=shift_error,
                single_RLC_relative_bandwidth_error=bandwidth_error,
                convergence_cases=len(convergence),
                max_magnitude_mse_grid_change=max(r['magnitude_mse_db2_relative_grid_change'] for r in convergence),
                max_phase_mse_grid_change=max(r['phase_mse_rad2_relative_grid_change'] for r in convergence),
                max_refined_feature_grid_change=max(r[n+'_refined_grid_change'] for n in NAMES for r in convergence),
                max_interpolation_only_feature_error=max(r[n+'_interpolation_only_error'] for n in NAMES for r in convergence))
    save_json('checks.json',report)
    return report


def plots(rows, summary, f, reference):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_path=Path('C:/Windows/Fonts/msyh.ttc')
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,
                         'font.size':10,'figure.dpi':160,'savefig.facecolor':'white'})
    labels={'magnitude_mse_db2':'幅值 MSE / dB²','phase_mse_rad2':'相位 MSE / rad²',
            'magnitude_rmse_db':'幅值 RMSE / dB','phase_rmse_rad':'相位 RMSE / rad',
            'H_resonance':'高阻谐振位置偏差','H_antiresonance':'支路间抵消位置偏差','H_bandwidth':'相对带宽偏差'}
    colors=['#177e89','#d1495b','#8b63a5','#df9b25']
    fig,axs=plt.subplots(2,3,figsize=(12.6,7.2),layout='constrained')
    for ax,metric in zip(axs.flat, ['magnitude_mse_db2','phase_mse_rad2']+NAMES):
        for color,fam in zip(colors,['R','L','C','all']):
            rs=sorted([r for r in summary if r['kind']=='random' and r['family']==fam],key=lambda r:r['level_pct'])
            x=[r['level_pct'] for r in rs];y=[r[metric+'_median'] for r in rs]
            ax.plot(x,y,'o-',color=color,label={'all':'R、L、C 同时扰动'}.get(fam,'仅 '+fam))
            ax.fill_between(x,[r[metric+'_p10'] for r in rs],[r[metric+'_p90'] for r in rs],color=color,alpha=.10)
        ax.set(xscale='log',yscale='log',title=labels[metric],xlabel='扰动等级 p / %')
        ax.grid(alpha=.18,which='major')
    axs.flat[5].axis('off')
    handles,legend=axs.flat[0].get_legend_handles_labels()
    axs.flat[5].legend(handles,legend,loc='upper left',frameon=False)
    direction_count = next(s['count'] for s in summary if s['kind']=='random')
    axs.flat[5].text(0,.55,f'点：{direction_count} 个固定方向的中位数\n阴影：10%–90% 分位区间\n\n同一方向逐步增加扰动幅度\n因子范围：[1/(1+p/100), 1+p/100]\n三个辅助指标：平均绝对自然对数偏差\n基准均为零，不能计算相对基准改善率',va='top',transform=axs.flat[5].transAxes)
    fig.suptitle('Case I：参数扰动下的主目标与物理特征响应',fontsize=16)
    fig.savefig(OUT/'perturbation_response.png');plt.close(fig)
    # Equal first-order response units: use RMSE, not MSE, alongside log-feature errors.
    metrics=['magnitude_rmse_db','phase_rmse_rad']+NAMES
    params=[f'{c}{k+1}' for c in ['R','L','C'] for k in range(6)]
    raw=np.array([[np.mean([r[metric] for r in rows if r['kind']=='single' and r.get('component')==par and r['level_pct']==1.]) for par in params] for metric in metrics])
    normalized=raw/np.maximum(np.max(raw,axis=1,keepdims=True),1e-30)
    fig,ax=plt.subplots(figsize=(12.6,4.8),layout='constrained')
    im=ax.imshow(normalized,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
    ax.set_xticks(range(18),params);ax.set_yticks(range(5),[labels[m] for m in metrics])
    for i in range(5):
        for j in range(18):
            value=normalized[i,j]
            annotation='<.01' if 0 < value < .005 else f'{value:.2f}'
            ax.text(j,i,annotation,ha='center',va='center',fontsize=8,color='white' if value>.6 else '#17324d')
    ax.set_title('单参数 1% 扰动：各指标关注哪些参数？\n每行独立除以该行最大值；不能据颜色比较不同指标的绝对大小',pad=16)
    fig.colorbar(im,ax=ax,label='行内相对响应')
    fig.savefig(OUT/'parameter_sensitivity.png');plt.close(fig)
    write_csv(OUT/'single_parameter_sensitivity.csv',[dict(metric=m,**dict(zip(params,raw[i].tolist()))) for i,m in enumerate(metrics)])
    # One reproducible direction, plus an exact scale-invariance counterexample.
    fig,axs=plt.subplots(2,2,figsize=(12.6,7.6),layout='constrained')
    selected=[r for r in rows if r['kind']=='random' and r['direction_id']=='all_0' and r['level_pct'] in [1.,10.,30.]]
    saved={'frequency_hz':f,'target_real_ohm':reference.real,'target_imag_ohm':reference.imag}
    for col,(lo,hi,title) in enumerate([(100,1e6,'全频段'),(24500,26000,'第二支路附近的窄带谐振')]):
        mask=(f>=lo)&(f<=hi)
        for j,(p,label,color) in enumerate([(TARGET,'目标','#162e46')]+[(np.array(r['params']),f"扰动 {r['level_pct']:g}%",c) for r,c in zip(selected,colors)]):
            z=impedance(f,p)
            axs[0,col].plot(f[mask],20*np.log10(abs(z[mask])),color=color,label=label,lw=1.2)
            axs[1,col].plot(f[mask],np.angle(z[mask])*180/np.pi,color=color,lw=1.2)
            if col==0 and j>0:
                saved[f'perturbed_{selected[j-1]["level_pct"]:g}_real_ohm']=z.real
                saved[f'perturbed_{selected[j-1]["level_pct"]:g}_imag_ohm']=z.imag
        axs[0,col].set_title(title)
        for row in [0,1]:axs[row,col].set(xscale='log',xlabel='频率 / Hz');axs[row,col].grid(alpha=.18)
    axs[0,0].set_ylabel('幅值 / dBΩ');axs[1,0].set_ylabel('相位 / °');axs[0,0].legend(frameon=False)
    fig.suptitle('固定随机方向 all_0：参数变化如何改变阻抗谱',fontsize=16)
    fig.savefig(OUT/'example_spectra.png');plt.close(fig)
    np.savez_compressed(OUT/'example_spectra.npz',**saved)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--points',type=int,default=65537)
    parser.add_argument('--random-directions',type=int,default=64)
    parser.add_argument('--plots-only',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    f=np.geomspace(100,1e6,args.points);reference=impedance(f,TARGET)
    target=features(f,reference,lambda x:impedance(x,TARGET))
    if args.plots_only:
        rows=[json.loads(s) for s in (OUT/'all_runs.jsonl').read_text(encoding='utf-8').splitlines()]
        summary,_,_=summarize(rows);plots(rows,summary,f,reference);return
    config=dict(date='2026-09-12',seed=20260912,frequency_points=args.points,
                frequency_min_hz=100,frequency_max_hz=1e6,grid='fixed log-uniform; newly constructed, NOT original 401-point grid',
                target_params_R_L_C=TARGET,levels_pct=LEVELS,random_directions_per_family=args.random_directions,
                parameter_rule='theta=theta_target*exp(log(1+p/100)*direction); max(abs(direction))=1',
                primary='equal-point mean squared dB magnitude difference and direct radian phase difference',
                features='terminal X crossings (down/up) and relative +45 to -45 phase bandwidth',
                refinement='bracket roots using dense sampled spectrum, then 28 simultaneous bisections of the SAME circuit response; reference numerics, not an optimization algorithm',
                missing_penalty=1,absolute_log_error_cap=1,
                source='English PDF Table III/Fig.10; Chinese PDF Sec.2 for primary formulas; original code and frequency CSV are absent')
    save_json('config.json',config)
    target_out={'resonance_hz':np.exp(target['res']),'antiresonance_hz':np.exp(target['anti']),
                'relative_phase_bandwidth':target['width'],'effective_Q':1/target['width'],
                'phase_band_edges_hz':np.exp(target['endpoints'])}
    save_json('target_features.json',target_out)
    rows=[];start=time.perf_counter()
    with (OUT/'all_runs.jsonl').open('w',encoding='utf-8') as output:
        for i,scenario in enumerate(scenarios(args.random_directions)):
            p=scenario['params'];z=impedance(f,p)
            t=time.perf_counter();ft=features(f,z,lambda x:impedance(x,p));elapsed=time.perf_counter()-t
            row=scenario|mse(z,reference)|feature_losses(ft,target)|{'feature_seconds':elapsed}
            rows.append(row)
            output.write(json.dumps(serializable(row),ensure_ascii=False,allow_nan=False)+'\n')
            if i%200==0:print(f'completed {i+1} spectra; elapsed {time.perf_counter()-start:.1f}s',flush=True)
    flat=[]
    for r in rows:
        rr={k:v for k,v in r.items() if k not in ['params','signed_log_resonance','signed_log_antiresonance','signed_log_bandwidth']}
        rr.setdefault('component','')
        for b in range(6):
            for col,c in enumerate(['R','L','C']):rr[f'{c}{b+1}']=float(r['params'][b,col])
        flat.append(rr)
    keys=list(dict.fromkeys(k for r in flat for k in r))
    write_csv(OUT/'all_runs.csv',[{k:r.get(k,'') for k in keys} for r in flat])
    summary,corr,mono=summarize(rows)
    check=checks(f,reference,target,rows)
    plots(rows,summary,f,reference)
    save_json('run_summary.json',dict(total_spectra=len(rows),elapsed_seconds=time.perf_counter()-start,
                                    feature_median_ms=1000*np.median([r['feature_seconds'] for r in rows]),
                                    rows_with_missing=sum(any(r[k] for k in ['missing_resonance','missing_antiresonance','missing_bandwidth']) for r in rows),
                                    checks=check))
    print(json.dumps(serializable(dict(total=len(rows),checks=check)),indent=2),flush=True)


if __name__=='__main__':main()

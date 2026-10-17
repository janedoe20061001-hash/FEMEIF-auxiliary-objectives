"""Local circuit-to-spectrum mechanism atlas. No optimizer or remote access."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.util
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import markdown

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'tmp' / 'curve_refinement_deps'))
OLD = ROOT / 'experiments/crosscase_consolidation_20261005/case1'
SOURCE_CSV = ROOT / 'experiments/case2_extended_pairs_validation/pair_screen_runs/selected_individuals.csv'
TARGET_CSV = ROOT / 'experiments/case2_physical_perturbation/data/targetImpedance_case2.csv'
LEVELS = (.1, .3, 1., 3., 10., 30.)
AUX = ('reactance_crossing_direction_consistency', 'characteristic_frequency_shift_error_decades',
       'transition_speed_relative_error')
METRIC_NAMES = ('magnitude_mse_db2', 'phase_mse_rad2', 'I03', 'II01', 'II02', 'cumulative_length')
BLUE, RED, GREEN = '#155aa8', '#c44536', '#168778'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


legacy = module('mechanism_legacy', OLD / 'legacy_objectives.py')
length_module = module('mechanism_length', OLD / 'length_metrics.py')
case2_source = module('mechanism_case2', ROOT / 'experiments/case2_physical_perturbation/case2.py')
case1_source = module('mechanism_case1', ROOT / 'experiments/case1_physical_perturbation_20260912/run_study.py')


def clean(v):
    if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [clean(x) for x in v]
    if isinstance(v, np.ndarray): return clean(v.tolist())
    if isinstance(v, (np.integer, np.bool_)): return v.item()
    if isinstance(v, (float, np.floating)): return float(v) if np.isfinite(v) else None
    return v


def dump(path, v):
    path.write_text(json.dumps(clean(v), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(clean(rows))


@dataclass
class Circuit:
    slug: str
    title: str
    kind: str
    names: list
    p: np.ndarray
    band: tuple
    topology: str
    seed: int | None = None
    wsmse: float | None = None

    def response(self, f, p=None):
        """Return Z and exact dZ/dln(f), with no finite frequency differences."""
        p = self.p if p is None else np.asarray(p)
        w = 2*np.pi*np.asarray(f)
        z = np.zeros_like(w, dtype=complex); dz = z.copy()
        if self.kind in ('parallel', 'background'):
            n = (len(p)-2)//3 if self.kind == 'background' else len(p)//3
            for r, l, c in p[:3*n].reshape(-1, 3):
                y = 1/r + 1j*(w*c-1/(w*l))
                dy = 1j*(w*c+1/(w*l))
                z += 1/y; dz -= dy/y**2
            if self.kind == 'background':
                z += p[-2]+1j*w*p[-1]; dz += 1j*w*p[-1]
        elif self.kind == 'series':
            r, l, c = p
            z = r+1j*(w*l-1/(w*c)); dz = 1j*(w*l+1/(w*c))
        elif self.kind == 'case2':
            r, l, c = p[:7], p[7:11], p[11:]
            for k in range(3):
                a = r[2*k]+1j*w*l[k]
                y = 1j*w*c[k]+1/a+1/r[2*k+1]
                dy = 1j*w*c[k]-1j*w*l[k]/a**2
                z += 1/y; dz -= dy/y**2
            z += r[6]+1j*w*l[3]+1/(1j*w*c[3])
            dz += 1j*w*l[3]-1/(1j*w*c[3])
        else: raise ValueError(self.kind)
        if not np.isfinite(z).all() or np.any(z.real <= 0):
            raise FloatingPointError('Invalid passive response')
        return z, dz

    def modes(self, p=None):
        """Internal admittance-zero centers and approximate log halfwidths."""
        p = self.p if p is None else p
        modes = []
        if self.kind in ('parallel', 'background', 'series'):
            n = (len(p)-2)//3 if self.kind == 'background' else len(p)//3
            for k, (r, l, c) in enumerate(p[:3*n].reshape(-1, 3)):
                f0 = 1/(2*np.pi*np.sqrt(l*c))
                q = np.sqrt(l/c)/r if self.kind == 'series' else r*np.sqrt(c/l)
                modes.append(dict(block=k+1, f_lc=f0, f_center=f0, q=q, log_halfwidth=1/(2*q)))
        else:
            r, l, c = p[:7], p[7:11], p[11:]
            for k in range(3):
                w2 = 1/(l[k]*c[k])-(r[2*k]/l[k])**2
                f_lc = 1/(2*np.pi*np.sqrt(l[k]*c[k]))
                if w2 <= 0:
                    modes.append(dict(block=k+1, f_lc=f_lc, f_center=None, q=None, log_halfwidth=None))
                    continue
                w = np.sqrt(w2); a = r[2*k]+1j*w*l[k]
                y = 1j*w*c[k]+1/a+1/r[2*k+1]
                dy = 1j*w*c[k]-1j*w*l[k]/a**2
                width = y.real/max(abs(dy.imag), 1e-30)
                modes.append(dict(block=k+1, f_lc=f_lc, f_center=w/(2*np.pi), q=1/(2*width), log_halfwidth=width))
        return modes


def canonical_embedding(p):
    """Remove three exchangeable Case2 block labels for reference selection only."""
    r, l, c = p[:7], p[7:11], p[11:]
    blocks = [[r[2*k], r[2*k+1], l[k], c[k]] for k in range(3)]
    blocks.sort(key=lambda b: 1/np.sqrt(b[2]*b[3]))
    return np.log(np.r_[np.asarray(blocks).ravel(), r[6], l[3], c[3]])


def build_circuits():
    def parallel(slug, title, pars, topology, kind='parallel'):
        n = len(pars)//3
        names = [f'{t}{k+1}' for k in range(n) for t in ('R','L','C')]
        if kind == 'background': names = [f'{t}{k+1}' for k in range((len(pars)-2)//3) for t in ('R','L','C')]+['Rs','Ls']
        return Circuit(slug, title, kind, names, np.asarray(pars, float), (100., 1e6), topology)
    circuits = [
        parallel('simple_parallel', '单并联 RLC：峰位、峰高与宽窄', [1000,1e-3,1e-7], 'R1 || L1 || C1'),
        parallel('simple_series', '单串联 RLC：低阻谷与相位过渡', [10,1e-3,1e-7], 'R1 + L1 + C1', 'series'),
        parallel('two_separated', '两个分离谐振单元', [1000,1e-3,1e-7,800,1e-4,1e-8], '(R1||L1||C1) + (R2||L2||C2)'),
        parallel('two_overlap', '两个接近的谐振单元', [2000,1/((2*np.pi*5e4)**2*1e-8),1e-8,1800,1/((2*np.pi*6e4)**2*1e-8),1e-8], '(R1||L1||C1) + (R2||L2||C2)'),
        parallel('with_background', '谐振单元叠加串联电阻与电感背景', [1500,1e-3,1e-7,800,1e-4,1e-8,100,10e-6], '(R1||L1||C1) + (R2||L2||C2) + Rs + Ls', 'background'),
        parallel('case1', 'Case1：论文六单元电路', case1_source.TARGET.ravel(), 'sum(k=1..6) (Rk || Lk || Ck)'),
    ]
    with SOURCE_CSV.open(encoding='utf-8-sig') as f:
        rows = sorted([r for r in csv.DictReader(f) if r['group']=='baseline'], key=lambda r:(float(r['wsmse_selected_value']),int(r['seed'])))
    assert len(rows) == 100
    pool = [r for r in rows if float(r['wsmse_selected_value']) <= 1.15*float(rows[0]['wsmse_selected_value'])]
    def params(r): return np.asarray([float(r[n]) for n in case2_source.PARAMETER_NAMES])
    selected = [pool[0]]
    while len(selected)<3:
        candidates = [r for r in pool if r not in selected]
        selected.append(max(candidates, key=lambda r:min(np.linalg.norm(canonical_embedding(params(r))-canonical_embedding(params(s))) for s in selected)))
    for j, row in enumerate(selected):
        circuits.append(Circuit(f'case2_ref{j+1}', f'Case2 参考{j+1}（seed {row["seed"]}）', 'case2', list(case2_source.PARAMETER_NAMES), params(row), (100.,1e6),
            'sum(k=1..3) [Ck || (Lk + R(2k-1)) || R(2k)] + C4 + R7 + L4', int(row['seed']), float(row['wsmse_selected_value'])))
    data = np.genfromtxt(TARGET_CSV, delimiter=',', names=True, encoding='utf-8-sig')
    measured_f = data['frequency']; measured_z = data['mag']*np.exp(1j*np.deg2rad(data['pha']))
    assert len(measured_f)==201 and np.all(np.diff(measured_f)>0)
    for model in circuits:
        if model.kind=='case2': model.band = (float(measured_f[0]), float(measured_f[-1]))
    return circuits, measured_f, measured_z, len(pool)


def grid(model, pars=(), refinement=1):
    lo, hi = np.log(model.band)
    chunks = [np.linspace(lo,hi,4096*refinement+1)]
    for p in [model.p]+list(pars):
        for mode in model.modes(p):
            if mode['f_center'] is None: continue
            center = np.log(mode['f_center']); width = min(mode['log_halfwidth']*10, .6)
            if lo <= center <= hi:
                chunks.append(np.linspace(max(lo,center-width),min(hi,center+width),256*refinement+1))
    return np.exp(np.unique(np.concatenate(chunks)))


def weights(f):
    x = np.log(f); d = np.diff(x); w = np.zeros(len(x))
    w[:-1] += d/2; w[1:] += d/2
    return w/(x[-1]-x[0])


def sensitivity(model):
    h = 1e-5
    pms = [model.p*np.exp(h*np.eye(len(model.p))[j]) for j in range(len(model.p))]
    f = grid(model, pms); z = model.response(f)[0]
    def calc(step):
        return np.column_stack([(model.response(f,model.p*np.exp(step*np.eye(len(model.p))[j]))[0]-model.response(f,model.p*np.exp(-step*np.eye(len(model.p))[j]))[0])/(2*step) for j in range(len(model.p))])
    j = calc(h); jhalf = calc(h/2)
    relerr = np.linalg.norm((j-jhalf)/z[:,None])/max(np.linalg.norm(jhalf/z[:,None]),1e-30)
    if relerr > .001:
        h /= 10; j = calc(h); jhalf = calc(h/2)
        relerr = np.linalg.norm((j-jhalf)/z[:,None])/np.linalg.norm(jhalf/z[:,None])
    assert relerr < .001, (model.slug,relerr)
    j = jhalf; s = j/z[:,None]; w = weights(f)
    a = np.vstack([s.real*np.sqrt(w[:,None]),s.imag*np.sqrt(w[:,None])])
    _, singular, vh = np.linalg.svd(a, full_matrices=False)
    strong, weak = vh[0]/max(abs(vh[0])), vh[-1]/max(abs(vh[-1]))
    cos = (a.T@a)/np.maximum(np.linalg.norm(a,axis=0)[:,None]*np.linalg.norm(a,axis=0)[None,:],1e-30)
    rows = []
    for k,name in enumerate(model.names):
        energy = w*abs(s[:,k])**2; order = np.cumsum(energy)/max(energy.sum(),1e-30)
        rows.append(dict(circuit=model.slug,parameter=name,value=model.p[k],
            magnitude_db_rms_per_logparam=np.sqrt(np.sum(w*s[:,k].real**2))*20/np.log(10),
            phase_deg_rms_per_logparam=np.rad2deg(np.sqrt(np.sum(w*s[:,k].imag**2))),
            relative_complex_rms_per_logparam=np.sqrt(energy.sum()),
            max_response_frequency_hz=f[np.argmax(abs(s[:,k]))],
            energy_10pct_frequency_hz=f[min(np.searchsorted(order,.1),len(f)-1)],
            energy_90pct_frequency_hz=f[min(np.searchsorted(order,.9),len(f)-1)]))
    return dict(f=f,z=z,j=j,s=s,cos=cos,strong=strong,weak=weak,singular=singular,rows=rows,
                step=h/2,half_step_relative_error=relerr,condition=singular[0]/max(singular[-1],1e-300))


def direction_plan(model, sens):
    n = len(model.p); result = []
    for j, name in enumerate(model.names): result.append(('single',name,np.eye(n)[j]))
    for family in ('R','L','C'):
        result.append(('structured',family+'_all',np.array([float(x.startswith(family)) for x in model.names])))
    result.append(('structured','impedance_scale',np.array([-1. if x.startswith('C') else 1. for x in model.names])))
    result.append(('structured','frequency_shift',np.array([0. if x.startswith('R') else 1. for x in model.names])))
    v = np.zeros(n); v[model.names.index('L1')]=1; v[model.names.index('C1')]=-1
    result.append(('structured','L1_C1_reciprocal',v))
    result.extend([('svd','local_strong',sens['strong']),('svd','local_weak',sens['weak'])])
    return result


def bisection(model, p, left, right, kind, threshold=None):
    left, right = np.asarray(left).copy(), np.asarray(right).copy()
    def val(x):
        z,dz = model.response(np.exp(x),p)
        if kind=='zero': return z.imag
        if kind=='extreme': return (np.conj(z)*dz).real
        return abs(z)-threshold
    initial = val(left)
    for _ in range(34):
        mid = (left+right)/2; v = val(mid); same = np.signbit(v)==np.signbit(initial)
        left = np.where(same,mid,left); right=np.where(same,right,mid)
    return np.exp((left+right)/2)


def features(model, p, f, z=None):
    z,dz = model.response(f,p) if z is None else (z,model.response(f,p)[1])
    x=np.log(f)
    def roots(v,kind):
        idx=np.flatnonzero(np.signbit(v[:-1])!=np.signbit(v[1:]))
        roots=bisection(model,p,x[idx],x[idx+1],kind)
        directions=np.where(v[idx+1]>v[idx],1,-1)
        return roots,directions
    zero,zero_dir=roots(z.imag,'zero')
    extreme,extreme_dir=roots((np.conj(z)*dz).real,'extreme')
    peaks=extreme[extreme_dir<0]; valleys=extreme[extreme_dir>0]
    heights=abs(model.response(peaks,p)[0]); widths=[]
    for peak,height in zip(peaks,heights):
        left=max([model.band[0]]+list(valleys[valleys<peak]))
        right=min([model.band[1]]+list(valleys[valleys>peak]))
        xx=np.unique(np.r_[x[(f>=left)&(f<=right)],np.log([left,peak,right])]); ff=np.exp(xx)
        v=abs(model.response(ff,p)[0])-height/np.sqrt(2)
        ids=np.flatnonzero(np.signbit(v[:-1])!=np.signbit(v[1:]))
        roots=bisection(model,p,xx[ids],xx[ids+1],'halfpower',height/np.sqrt(2))
        before,after=roots[roots<peak],roots[roots>peak]
        widths.append((after[0]-before[-1])/peak if len(before) and len(after) else np.nan)
    return dict(zero_hz=zero,zero_direction=zero_dir,peak_hz=peaks,peak_ohm=heights,
                peak_relative_halfpower_width=np.array(widths),valley_hz=valleys)


def compare_features(base, candidate, band, expected_frequencies=None):
    shifts=[]; heights=[]; widths=[]; missing=0; ambiguous=0
    ref=base['peak_hz']; cand=candidate['peak_hz']
    centers=ref if expected_frequencies is None else np.asarray(expected_frequencies)
    edges=np.r_[np.log(band[0]),(np.log(centers[:-1])+np.log(centers[1:]))/2,np.log(band[1])]
    for k,f0 in enumerate(ref):
        ids=np.flatnonzero((np.log(cand)>=edges[k])&(np.log(cand)<edges[k+1]))
        if not len(ids): missing+=1; continue
        ambiguous+=int(len(ids)>1)
        j=ids[np.argmin(abs(np.log(cand[ids]/centers[k])))]
        if expected_frequencies is not None:
            assert abs(np.log(cand[j]/centers[k])) < 1e-7, (cand[j],centers[k])
        shifts.append(np.log10(cand[j]/f0)); heights.append(np.log(candidate['peak_ohm'][j]/base['peak_ohm'][k]))
        bw0=base['peak_relative_halfpower_width'][k]; bw1=candidate['peak_relative_halfpower_width'][j]
        if np.isfinite(bw0) and np.isfinite(bw1): widths.append(np.log(bw1/bw0))
    return dict(port_peak_count=len(cand),port_valley_count=len(candidate['valley_hz']),
        reactance_zero_count=len(candidate['zero_hz']),reference_peak_missing_window_count=missing,
        multiple_peaks_in_reference_window_count=ambiguous,
        peak_shift_signed_mean_decades=np.mean(shifts) if shifts else np.nan,
        peak_shift_max_abs_decades=max(abs(np.array(shifts))) if shifts else np.nan,
        peak_height_signed_mean_logratio=np.mean(heights) if heights else np.nan,
        peak_width_signed_mean_logratio=np.mean(widths) if widths else np.nan,
        matched_width_count=len(widths))


def curve_errors(z,z0,f):
    w=weights(f); da=20*np.log10(abs(z)/abs(z0)); dp=np.angle(z)-np.angle(z0)
    return dict(magnitude_integral_rmse_db=np.sqrt(np.sum(w*da**2)),
        phase_integral_rmse_deg=np.rad2deg(np.sqrt(np.sum(w*dp**2))),
        relative_complex_integral_rmse=np.sqrt(np.sum(w*abs((z-z0)/z0)**2)),
        magnitude_max_abs_db=max(abs(da)),phase_max_abs_deg=np.rad2deg(max(abs(dp))))


def frozen_evaluator(f,target):
    first=legacy.Objectives(f,aux=AUX,target_z=target)
    second=length_module.GlobalObjectives(f,aux=('cumulative_length_logpolar',),target_z=target)
    def evaluate(z):
        return np.r_[first.evaluate_spectra(z)[0],second.evaluate_spectra(z)[0,2]]
    return evaluate


def style():
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists(): font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False,
        'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'#fafcfe',
        'axes.labelcolor':'#213a55','text.color':'#213a55','xtick.color':'#536578','ytick.color':'#536578'})


def savefig(fig,path):
    fig.savefig(path,dpi=155,bbox_inches='tight'); plt.close(fig)


def sensitivity_plot(model,sens,path,normalized=False):
    f,s=sens['f'],sens['s']; mag=s.real*20/np.log(10); pha=np.rad2deg(s.imag)
    fig,axes=plt.subplots(1,2,figsize=(14, max(4.5,len(model.p)*.28)),sharey=True)
    for ax,values,title in zip(axes,[mag,pha],['幅值 dB / Δln参数','相位 ° / Δln参数']):
        if normalized:
            values=values/np.maximum(np.max(abs(values),axis=0),1e-30)
        v=1. if normalized else max(np.quantile(abs(values),.99),1e-8)
        im=ax.pcolormesh(f,np.arange(len(model.p)),values.T,cmap='RdBu_r',vmin=-v,vmax=v,shading='nearest',rasterized=True)
        ax.set_xscale('log'); ax.set_yticks(np.arange(len(model.p)),model.names); ax.set_xlabel('频率 / Hz'); ax.set_title(title)
        fig.colorbar(im,ax=ax,label='本行最大绝对导数归一化' if normalized else '正值：参数增大时响应上升')
    fig.suptitle(model.title+('｜各参数分别归一化：只比较作用频段和方向' if normalized else '｜局部带符号敏感性（色限取99%分位，极值可能饱和）'),y=1.015)
    fig.tight_layout(); savefig(fig,path)


def curves_plot(model,direction,title,path,measured=None):
    ps=[model.p*np.exp(sign*np.log1p(level/100)*direction) for level in (1.,10.,30.) for sign in (-1,1)]
    f=grid(model,ps); z0=model.response(f)[0]
    fig,axes=plt.subplots(2,2,figsize=(12.8,8))
    colors=['#9bc1e4','#e9b8b1','#5090c9','#d67466',BLUE,RED]
    styles=[':']*2+['--']*2+['-']*2
    centers=[m['f_center'] for m in model.modes() if m['f_center'] is not None]
    nyquist_lo=max(model.band[0],min(centers)*.3) if model.kind=='case2' and centers else model.band[0]
    nmask=f>=nyquist_lo
    all_z=[z0]
    for p,col,ls,(level,sign) in zip(ps,colors,styles,[(l,s) for l in (1.,10.,30.) for s in (-1,1)]):
        z=model.response(f,p)[0]; all_z.append(z)
        fac=np.exp(sign*np.log1p(level/100)); label=f'{level:g}档 {sign:+d}（最大分量×{fac:.4g}）'
        axes[0,0].semilogx(f,20*np.log10(abs(z)),color=col,ls=ls,lw=1.2,label=label)
        axes[0,1].semilogx(f,np.rad2deg(np.angle(z)),color=col,ls=ls,lw=1.2)
        axes[1,0].semilogx(f,20*np.log10(abs(z)/abs(z0)),color=col,ls=ls,lw=1.2)
        axes[1,1].plot(z.real[nmask],-z.imag[nmask],color=col,ls=ls,lw=1.2)
    axes[0,0].semilogx(f,20*np.log10(abs(z0)),color='#172b40',lw=1.8,label='模型参考')
    axes[0,1].semilogx(f,np.rad2deg(np.angle(z0)),color='#172b40',lw=1.8)
    axes[1,0].axhline(0,color='#172b40',lw=.8)
    axes[1,1].plot(z0.real[nmask],-z0.imag[nmask],color='#172b40',lw=1.8)
    if measured is not None:
        mf,mz=measured
        axes[0,0].semilogx(mf,20*np.log10(abs(mz)),'.',color=GREEN,ms=3,label='201点实测')
        axes[0,1].semilogx(mf,np.rad2deg(np.angle(mz)),'.',color=GREEN,ms=3)
        axes[1,1].plot(mz.real[mf>=nyquist_lo],-mz.imag[mf>=nyquist_lo],'.',color=GREEN,ms=3)
    axes[0,0].set_ylabel('20log10|Z| / dBΩ'); axes[0,1].set_ylabel('相位 / °')
    axes[1,0].set_ylabel('相对模型参考的幅值变化 / dB'); axes[1,1].set_ylabel('−Im Z / Ω')
    for ax in axes.flat: ax.grid(alpha=.2)
    for ax in (axes[0,0],axes[0,1],axes[1,0]): ax.set_xlabel('频率 / Hz')
    axes[1,1].set_xlabel('Re Z / Ω'); axes[1,1].set_title(f'Nyquist（f ≥ {nyquist_lo:.3g} Hz；两轴独立缩放）' if model.kind=='case2' else 'Nyquist（全频带；两轴独立缩放）')
    axes[0,0].legend(fontsize=8,loc='best')
    fig.suptitle(model.title+'｜'+title,y=1.01); fig.tight_layout(); savefig(fig,path)


def local_plots(model,sens,rows,out):
    fig,axes=plt.subplots(1,2,figsize=(12.5,5))
    im=axes[0].imshow(sens['cos'],vmin=-1,vmax=1,cmap='RdBu_r')
    axes[0].set_xticks(range(len(model.p)),model.names,rotation=90,fontsize=8); axes[0].set_yticks(range(len(model.p)),model.names,fontsize=8)
    axes[0].set_title('敏感性方向余弦：相近方向可能补偿'); fig.colorbar(im,ax=axes[0],shrink=.8)
    axes[1].semilogy(np.arange(1,len(sens['singular'])+1),sens['singular'],'o-',color=BLUE)
    axes[1].set_xlabel('奇异值序号'); axes[1].set_ylabel('带频率积分权重的相对响应'); axes[1].set_title('局部响应的强弱方向'); axes[1].grid(alpha=.2)
    fig.suptitle(model.title,y=1.02); fig.tight_layout(); savefig(fig,out/'figures'/f'{model.slug}_local.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.2))
    for name,col in [('local_strong',RED),('local_weak',BLUE)]:
        subset=sorted([r for r in rows if r['direction']==name and r['sign']==1],key=lambda r:r['level_pct'])
        axes[0].loglog([r['level_pct'] for r in subset],[r['relative_complex_integral_rmse'] for r in subset],'o-',color=col,label=name)
        axes[1].loglog([r['level_pct'] for r in subset],[max(r['linearization_relative_error'],1e-12) for r in subset],'o-',color=col,label=name)
    axes[0].set_ylabel('相对复阻抗变化 RMS'); axes[1].set_ylabel('线性预测残差 / 实际变化')
    for ax in axes: ax.set_xlabel('乘性扰动档位 / %'); ax.grid(alpha=.2); ax.legend()
    fig.suptitle(model.title+'｜局部强弱方向与有限扰动',y=1.02); fig.tight_layout(); savefig(fig,out/'figures'/f'{model.slug}_nonlinear.png')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=ROOT/'output/parameter_curve_mechanism_20261006')
    args=ap.parse_args(); out=args.output.resolve(); (out/'data').mkdir(parents=True,exist_ok=True); (out/'figures').mkdir(exist_ok=True)
    style(); start=time.time(); circuits,mf,mz,pool_count=build_circuits()
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE_CSV,TARGET_CSV,Path(__file__),OLD/'legacy_objectives.py',OLD/'length_metrics.py',
        ROOT/'experiments/case1_physical_perturbation_20260912/run_study.py',ROOT/'experiments/case2_physical_perturbation/case2.py',
        Path(__file__).with_name('README.md'),ROOT/'Feature-Engineering_Enabled_Multi-Objective_Evolutionary_Impedance_Fitting_Technique.pdf']}
    dump(out/'data/PROTOCOL.json',dict(date='2026-10-06',local_only=True,optimizer_runs=0,levels_pct=LEVELS,
        negative_factor='1/(1+level/100)',integration='trapezoidal normalized ln(f)',case2_reference_pool_count=pool_count,
        case2_reference_selection='best + canonical-log-parameter farthest points within 1.15*best',source_sha256=hashes))
    all_rows=[]; all_sens=[]; all_params=[]; all_checks=[]; events=[]; summaries=[]; gallery=[]; svd_data=[]
    for model in circuits:
        print(f'分析 {model.slug}: {model.title}',flush=True)
        for n,p in zip(model.names,model.p): all_params.append(dict(circuit=model.slug,parameter=n,value=p,unit='ohm' if n.startswith('R') else 'H' if n.startswith('L') else 'F',reference_seed=model.seed,wsmse=model.wsmse))
        f0=grid(model); z0=model.response(f0)[0]; bf=features(model,model.p,f0,z0); events.append(dict(circuit=model.slug,direction='reference',level_pct=0,sign=0,features=bf,modes=model.modes()))
        if model.slug=='simple_parallel':
            r,l,c=model.p; analytic=np.array([1/(2*np.pi*np.sqrt(l*c)),r,np.sqrt(l/c)/r])
            numeric=np.array([bf['peak_hz'][0],bf['peak_ohm'][0],bf['peak_relative_halfpower_width'][0]])
            feature_error=max(abs(numeric/analytic-1)); assert feature_error<1e-8
            all_checks.append(dict(circuit=model.slug,check='isolated_parallel_peak_and_bandwidth',value=feature_error,limit=1e-8,passed=True))
        sens=sensitivity(model); all_sens+=sens['rows']; plan=direction_plan(model,sens)
        svd_data.append(dict(circuit=model.slug,condition_number=sens['condition'],singular_values=sens['singular'],parameter_names=model.names,
            strong_direction=sens['strong'],weak_direction=sens['weak'],ln_parameter_step=sens['step'],half_step_relative_error=sens['half_step_relative_error']))
        all_checks.append(dict(circuit=model.slug,check='jacobian_half_step',value=sens['half_step_relative_error'],limit=.001,passed=True))
        # Both source implementations and analytical derivatives must agree.
        checkf=np.geomspace(*model.band,301); zz,dz=model.response(checkf)
        if model.slug=='case1': sourcez=case1_source.impedance(checkf,model.p.reshape(6,3))
        elif model.kind=='case2': sourcez=case2_source.impedance(checkf,model.p)[0]
        else: sourcez=zz
        err=max(abs(zz-sourcez))/max(abs(zz)); assert err<1e-12
        all_checks.append(dict(circuit=model.slug,check='source_circuit_agreement',value=err,limit=1e-12,passed=True))
        fd=(model.response(checkf*np.exp(1e-6))[0]-model.response(checkf*np.exp(-1e-6))[0])/2e-6
        derr=np.linalg.norm(fd-dz)/np.linalg.norm(dz); assert derr<1e-4
        all_checks.append(dict(circuit=model.slug,check='analytic_frequency_derivative',value=derr,limit=1e-4,passed=True))
        evf=mf if model.kind=='case2' else np.geomspace(*model.band,180)
        evalref=frozen_evaluator(evf,model.response(evf)[0])
        evalmeas=frozen_evaluator(mf,mz) if model.kind=='case2' else None
        baseline_measured=evalmeas(model.response(mf)[0]) if evalmeas else np.full(6,np.nan)
        if model.kind=='case2':
            assert abs(baseline_measured[0]+17.38*baseline_measured[1]-model.wsmse)<1e-9
        rows=[]
        for kind,name,v in plan:
            for level in LEVELS:
                for sign in (-1,1):
                    delta=sign*np.log1p(level/100)*v; p=model.p*np.exp(delta)
                    f=grid(model,[p]); z=model.response(f,p)[0]; base=model.response(f)[0]
                    err=curve_errors(z,base,f); cf=features(model,p,f,z)
                    window_feat=compare_features(bf,cf,model.band)
                    feat=compare_features(bf,cf,model.band,bf['peak_hz']/np.exp(sign*np.log1p(level/100))) if name=='frequency_shift' else window_feat
                    # First-order prediction at the sensitivity grid, weighted equally in log frequency.
                    sj=sens['j']@delta; actual=model.response(sens['f'],p)[0]-sens['z']; w=weights(sens['f'])
                    norm=np.sqrt(np.sum(w*abs(actual/sens['z'])**2))
                    linear=np.sqrt(np.sum(w*abs((actual-sj)/sens['z'])**2))/max(norm,1e-30)
                    sampled_z=model.response(evf,p)[0]; metrics=evalref(sampled_z)
                    measured=evalmeas(sampled_z) if evalmeas else np.full(6,np.nan)
                    row=dict(circuit=model.slug,kind=kind,direction=name,level_pct=level,sign=sign,max_log_parameter_change=max(abs(delta)),
                        **err,**feat,linearization_relative_error=linear,
                        feature_matching_method='analytic_frequency_continuation' if name=='frequency_shift' else 'fixed_reference_windows',
                        fixed_window_missing_peaks=window_feat['reference_peak_missing_window_count'],
                        fixed_window_peak_shift_max_abs_decades=window_feat['peak_shift_max_abs_decades'],
                        fixed_window_peak_width_signed_mean_logratio=window_feat['peak_width_signed_mean_logratio'])
                    row.update({'reference_'+n:val for n,val in zip(METRIC_NAMES,metrics)})
                    row.update({'measured_'+n:val for n,val in zip(METRIC_NAMES,measured)})
                    row.update({'measured_delta_'+n:val-b for n,val,b in zip(METRIC_NAMES,measured,baseline_measured)})
                    rows.append(row); events.append(dict(circuit=model.slug,direction=name,level_pct=level,sign=sign,features=cf,modes=model.modes(p)))
                    # Numerical convergence at key finite directions, without changing original sampled metrics.
                    if name in ('L1','C1','frequency_shift','L1_C1_reciprocal') and level in (1.,30.):
                        ff=grid(model,[p],2); fine=curve_errors(model.response(ff,p)[0],model.response(ff)[0],ff)
                        for key in ('magnitude_integral_rmse_db','phase_integral_rmse_deg','relative_complex_integral_rmse'):
                            diff=abs(fine[key]-err[key])/max(fine[key],1e-12)
                            all_checks.append(dict(circuit=model.slug,check=f'grid_{name}_{level:g}_{sign}_{key}',value=diff,limit=.005,passed=bool(diff<.005)))
                            assert diff<.005,(model.slug,name,level,sign,key,diff)
        # Exact RLC homogeneity: Z(R*a,L*a,C/a)=aZ and Z(R,L*a,C*a)(f)=Z(af).
        for name in ('impedance_scale','frequency_shift'):
            v=next(v for _,n,v in plan if n==name); a=1.3; p=model.p*np.exp(np.log(a)*v)
            z=model.response(checkf,p)[0]; expected=a*zz if name=='impedance_scale' else model.response(checkf*a)[0]
            err=np.linalg.norm(z-expected)/np.linalg.norm(expected); assert err<1e-12
            all_checks.append(dict(circuit=model.slug,check='identity_'+name,value=err,limit=1e-12,passed=True))
        if model.kind=='case2' or model.kind=='background':
            name='R7' if model.kind=='case2' else 'Rs'; k=model.names.index(name); p=model.p.copy(); shift=.3*p[k]; p[k]+=shift
            err=max(abs((model.response(checkf,p)[0]-zz)-shift))/max(max(abs(zz)),shift); assert err<1e-12
            all_checks.append(dict(circuit=model.slug,check='real_series_shift_identity',value=err,limit=1e-12,passed=True))
            m=evalref(model.response(evf,p)[0]); assert m[2]==0 and m[4]<1e-10
            all_checks.append(dict(circuit=model.slug,check='I03_II02_real_shift_null',value=m[4],limit=1e-10,passed=True))
        all_rows+=rows
        summaries.append(dict(circuit=model.slug,title=model.title,topology=model.topology,seed=model.seed,wsmse=model.wsmse,baseline_features=bf,
            grid_reference_points=len(f0),base_measured_metrics=baseline_measured,sensitivity=sens['rows'],spectra_evaluated=len(rows),band_hz=model.band))
        sensitivity_plot(model,sens,out/'figures'/f'{model.slug}_sensitivity.png')
        sensitivity_plot(model,sens,out/'figures'/f'{model.slug}_sensitivity_normalized.png',True)
        local_plots(model,sens,rows,out)
        items=[(f'{model.slug}_sensitivity.png','局部带符号敏感性（统一尺度）'),(f'{model.slug}_sensitivity_normalized.png','每行归一化：作用频段与方向'),(f'{model.slug}_local.png','参数补偿与局部强弱方向'),(f'{model.slug}_nonlinear.png','有限扰动与线性近似')]
        if model.kind=='case2':
            fig,axes=plt.subplots(1,2,figsize=(12.5,5))
            for ax in axes:
                ax.plot(z0.real,-z0.imag,color=BLUE,label='模型参考'); ax.plot(mz.real,-mz.imag,'.',color=GREEN,ms=3,label='201点实测')
                ax.set_xlabel('Re Z / Ω'); ax.set_ylabel('−Im Z / Ω'); ax.grid(alpha=.2)
            axes[0].set_title('完整原频带'); axes[0].legend()
            centers=[v['f_center'] for v in model.modes() if v['f_center'] is not None]; threshold=min(centers)*.3
            mask=f0>=threshold; mm=mf>=threshold
            axes[1].clear(); axes[1].plot(z0.real[mask],-z0.imag[mask],color=BLUE); axes[1].plot(mz.real[mm],-mz.imag[mm],'.',color=GREEN,ms=3)
            axes[1].set_title(f'谐振频段：f ≥ {threshold:.3g} Hz'); axes[1].set_xlabel('Re Z / Ω'); axes[1].set_ylabel('−Im Z / Ω'); axes[1].grid(alpha=.2)
            fig.suptitle(model.title+'｜Nyquist 全频带与局部图（两轴独立缩放）',y=1.02); fig.tight_layout()
            file=f'{model.slug}_nyquist_full.png'; savefig(fig,out/'figures'/file); items.append((file,'Nyquist完整频带与局部放大'))
        # Detailed all-parameter atlas for Case1 and primary Case2; secondary fits focus on robustness.
        for kind,name,v in plan:
            if kind=='svd': continue
            if kind=='single' and model.slug in ('case2_ref2','case2_ref3') and name not in ('R7','L1','C1'): continue
            if kind=='single' and model.slug in ('two_separated','two_overlap','with_background') and name not in ('R1','L1','C1','Rs','Ls'): continue
            if kind=='structured' and name not in ('impedance_scale','frequency_shift','L1_C1_reciprocal'): continue
            file=f'{model.slug}_{name}.png'; title=direction_title(name)
            curves_plot(model,v,title,out/'figures'/file, (mf,mz) if model.kind=='case2' else None); items.append((file,title))
        gallery.append((model,items))
        print(f'完成 {model.slug}: {len(rows)} 条扰动谱，累计 {time.time()-start:.1f}s',flush=True)
    write_csv(out/'data/perturbations.csv',all_rows); write_csv(out/'data/sensitivity.csv',all_sens); write_csv(out/'data/reference_parameters.csv',all_params)
    write_csv(out/'data/numerical_checks.csv',all_checks); dump(out/'data/summary.json',summaries); dump(out/'data/local_directions.json',svd_data)
    with (out/'data/feature_events.jsonl').open('w',encoding='utf-8') as f:
        for event in events: f.write(json.dumps(clean(event),ensure_ascii=False,allow_nan=False)+'\n')
    report=build_report(circuits,summaries,all_rows,all_checks,svd_data,pool_count)
    (out/'REPORT.md').write_text(report,encoding='utf-8')
    guide=build_parameter_guide(circuits,all_rows,all_sens)
    (out/'PARAMETER_GUIDE.md').write_text(guide,encoding='utf-8')
    write_html(out,gallery,report)
    artifacts={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='INTEGRITY.json'}
    dump(out/'INTEGRITY.json',dict(completed=True,local_only=True,optimizer_runs=0,circuits=len(circuits),perturbed_spectra=len(all_rows),
        figures=len(list((out/'figures').glob('*.png'))),numerical_checks=len(all_checks),all_checks_passed=all(r['passed'] for r in all_checks),
        elapsed_seconds=time.time()-start,python=platform.python_version(),numpy=np.__version__,artifacts_sha256=artifacts))
    print(json.dumps(dict(output=str(out),spectra=len(all_rows),checks=len(all_checks),seconds=round(time.time()-start,1)),ensure_ascii=False),flush=True)


def direction_title(name):
    return {'impedance_scale':'整体阻抗缩放：R、L ×a；C ÷a','frequency_shift':'整体频移：L、C ×a；R 不变',
        'L1_C1_reciprocal':'L1 ×a、C1 ÷a：保持该单元 LC 乘积'}.get(name,'单参数 '+name)


def build_report(models,summaries,rows,checks,svd,pool_count):
    def subset(slug,direction,level=30.,sign=1):
        return next(r for r in rows if r['circuit']==slug and r['direction']==direction and r['level_pct']==level and r['sign']==sign)
    def fmt(x,d=4): return '未定义' if x is None or not np.isfinite(x) else f'{x:.{d}g}'
    def table(headers,data):
        escaped=lambda r:[str(v).replace('|',r'\|') for v in r]
        return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(escaped(r))+' |\n' for r in data)
    lines=['# 参数如何改变阻抗曲线：第一轮本地机制分析','',
        '2026-10-06。本轮全部在本地计算，没有新增优化，也没有使用服务器。分析五种自设计电路、论文 Case1 和 Case2 三个拟合参考点。',
        '', '## 先看结论','',
        '1. 参数与曲线变化可以建立清楚的关系，但应以拓扑、参考参数和频带为条件。一个参数通常同时影响峰位、峰高、宽度和相位；复杂端口响应的实际峰位不能直接当成内部 LC 频率。',
        '2. 两种精确整体变化可由真实 RLC 元件组合实现：R、L 同乘 a 且 C 除以 a，使 Z 整体乘 a；R 不变而 L、C 同乘 a，使 Z_new(f)=Z_old(af)，曲线向低频移动。这两种变化适合当作指标审查的可解释对照。',
        '3. 保持某单元 LC 乘积只固定该单元的自然频率，不能固定它的宽度；多个单元串联时也不能保证端口峰位固定。局部敏感性与有限扰动必须分别判断。',
        '4. Case2 的串联 R7 变化是可实现的实轴平移。I-03 与 II-02 对这种变化保持零损失（相对模型参考），但幅值、相位仍改变。这验证的是辅助项盲区，不是整个目标系统无效。',
        '5. 本轮不能根据局部响应宣布哪个新指标会改善优化。更合适的下一步是以具体变化模式审查现有指标，再做小规模优化对照。',
        '', '## 电路与参考点','',
        '五种自设计电路均使用理想正值 R/L/C 元件。Case1 复用论文 Table III 的已知生成参数；Case2 复用论文 Fig.15 的15元件拟合模型和原201点实测目标。', '',
        table(['配置','拓扑','参数数','参考'],[[m.title,m.topology,len(m.p),'已知生成参数' if m.slug=='case1' else f'seed {m.seed}；WSMSE={m.wsmse:.6g}' if m.kind=='case2' else '自设计、已知生成参数'] for m in models]),'',
        f'Case2 在原100个 baseline 代表解中取最低 WSMSE 者，再从 WSMSE ≤1.15×最低值的 {pool_count} 个参考候选中，按消除三组可交换支路标签后的 log 参数距离进行最远点选择。三组点仅覆盖这一优质拟合子集。选择规则提前写在 README；不将其称为真实元件参数。',
        '', '完整数值与单位见 [参考参数](data/reference_parameters.csv)。所有负向扰动使用正向因子的倒数，例如30档为×1.3或÷1.3（约-23.08%），不是对称的±30%。',
        '', '## 单个并联 RLC：解析规律与数值验证','',
        '对于孤立理想并联 RLC：f0=1/(2π√LC)，峰值=R，Q=R√(C/L)，相对半功率带宽 b=1/Q。这些公式只直接解释该单元，不能无条件推广到多单元端口。', '',
        table(['参数正向×1.3','峰位移动 / decade','峰高比','相对带宽比','相位积分 RMSE / °'],[[n,fmt(subset('simple_parallel',n)['peak_shift_signed_mean_decades']),fmt(np.exp(subset('simple_parallel',n)['peak_height_signed_mean_logratio'])),fmt(np.exp(subset('simple_parallel',n)['peak_width_signed_mean_logratio'])),fmt(subset('simple_parallel',n)['phase_integral_rmse_deg'])] for n in ('R1','L1','C1')]),'',
        'L、C 增大都让自然频率降低；L 增大使相对带宽变宽，C 增大使相对带宽变窄；R 增大提高峰高并使相对带宽变窄。串联 RLC 对应低阻谷，其谷底 R、阻尼与相位变化另有关系，图册保留该对照。',
        '', '单并联 RLC 的 L 或 C 改变时，Nyquist 轨迹可能仍沿同一个圆走过，而每个频率对应的位置已经改变。完整频移也可能保持几何轨迹相同。因此比较曲线几何形状时必须保留频率对应关系；不能只凭复平面上的重合判断拟合正确。',
        '', '![单并联RLC敏感性](figures/simple_parallel_sensitivity.png)','',
        '## 组合变化：曲线有哪些可实现的独立变化','',
        '下表将三类组合在30档正向下相对各自模型参考的响应分开统计。峰位和宽度采用端口特征，不使用真值参数给候选打分。', '',
        table(['配置','组合','幅值积分RMSE / dB','相位积分RMSE / °','最大峰位偏移 / decade','平均带宽比'],[[m.slug,n,fmt(r['magnitude_integral_rmse_db']),fmt(r['phase_integral_rmse_deg']),fmt(r['peak_shift_max_abs_decades']),fmt(np.exp(r['peak_width_signed_mean_logratio']))] for m in models if m.slug in ('simple_parallel','two_overlap','case1','case2_ref1') for n in ('impedance_scale','frequency_shift','L1_C1_reciprocal') for r in [subset(m.slug,n)]]),'',
        '整体缩放应具有 20log10(1.3)≈2.279 dB 的恒定幅值变化、零相位变化；累计弧长使用固定目标归一化，对此常数幅值平移不敏感。整体频移使特征频率乘1/1.3；若峰移出频带，事件数量可能变化，不能自动解释成谐振消失。',
        '', '![Case1保持单元LC乘积](figures/case1_L1_C1_reciprocal.png)','',
        '## Case1：哪些参数影响哪些频段','',
        '采用带符号的 ln 参数敏感性，分别保留幅值与相位。颜色为正意味着增大参数会在该频点抬高幅值或相位；同一参数可能在峰两侧产生相反变化。下图每行单独归一化，只比较作用频段及正负方向；实际强弱请看数据或图册中的统一尺度图。', '',
        '![Case1敏感性](figures/case1_sensitivity_normalized.png)','',
        table(['配置','参数','幅值敏感性RMS / dB','相位敏感性RMS / °','最大相对响应频率 / Hz'],[[r['circuit'],r['parameter'],fmt(r['magnitude_db_rms_per_logparam']),fmt(r['phase_deg_rms_per_logparam']),fmt(r['max_response_frequency_hz'],6)] for r in sorted(next(s for s in summaries if s['circuit']=='case1')['sensitivity'],key=lambda r:r['relative_complex_rms_per_logparam'],reverse=True)[:8]]),'',
        '上述 RMS 是每单位 Δln参数的局部导数，不是实际1%误差。Case1 的尖锐谐振会使很小参数变化造成明显局部差异；30档不能由小扰动敏感性直接外推。每个参数的1、10、30档曲线在图册中可单独查看。',
        '', '## Case2：模型参考变化与实测误差必须分开','',
        '以下以 R7 ×1.3 为例。相对模型参考的 I-03、II-02 为零；相对实测目标的损失有非零基线，因此只看候选值是否下降会混淆“模型变化”与“朝目标改善”。', '',
        table(['参考','R7 / Ω','相对参考幅值RMSE / dB','相对参考相位RMSE / °','参考 I-03','参考 II-02','实测幅值MSE变化 / dB²','实测相位MSE变化 / rad²'],[[m.slug,fmt(m.p[6]),fmt(r['magnitude_integral_rmse_db']),fmt(r['phase_integral_rmse_deg']),fmt(r['reference_I03']),fmt(r['reference_II02']),fmt(r['measured_delta_magnitude_mse_db2']),fmt(r['measured_delta_phase_mse_rad2'])] for m in models if m.kind=='case2' for r in [subset(m.slug,'R7')]]),'',
        'R7 位于全电路串联位置，增大 ΔR7 精确产生 Z_new=Z_old+ΔR7；电抗和复导数不变。幅值通常上升，相位向0靠近。是否更接近实测相位，取决于参考电路原先的偏差。',
        '', '这组三参考点已经给出不同方向：R7 ×1.3 在 seed7、seed19 下使实测相位误差下降，在 seed35 下使其上升。不能将某一参考点的改善方向写成 Case2 的普遍规律。',
        '', '![Case2串联电阻变化](figures/case2_ref1_R7.png)','',
        '![Case2参考2敏感性](figures/case2_ref2_sensitivity.png)','',
        '## 参数补偿、强弱方向与非线性','',
        '局部矩阵为 [Re(J/Z); Im(J/Z)]，J=∂Z/∂ln参数，按等 log 频率积分加权。两部分分别为相对幅值和弧度相位，采用相同无量纲权重，不等同于正式优化的幅相权重。SVD 的强弱方向统一按最大参数分量归一化为1，避免用参数单位比较敏感性。', '',
        table(['配置','局部条件数','强方向1档相对复响应RMS','弱方向1档相对复响应RMS','弱方向30档线性残差/实际变化'],[[v['circuit'],fmt(v['condition_number']),fmt(subset(v['circuit'],'local_strong',1)['relative_complex_integral_rmse']),fmt(subset(v['circuit'],'local_weak',1)['relative_complex_integral_rmse']),fmt(subset(v['circuit'],'local_weak',30)['linearization_relative_error'])] for v in svd]),'',
        '条件数受参考点、频带和响应权重影响。本轮没有噪声下参数置信区间或全局结构可辨识性证明。局部弱方向只表示一阶响应弱，有限扰动可能暴露二阶变化；方向线性误差在实际响应极弱时也会有较大的比值。完整向量见 local_directions.json。',
        '', '![Case1非线性](figures/case1_nonlinear.png)','',
        '## 现有指标审查：哪些机制被看到、哪些被遗漏','',
        '直接载入已有冻结实现，在原采样网格计算 I-03、II-01、II-02 与累计弧长。下表只以相对模型参考的损失解释可实现的变化；所有参考点的基线为零。', '',
        table(['配置','30档变化','幅值MSE / dB²','相位MSE / rad²','I-03','II-01','II-02','累计弧长'],[[slug,n]+[fmt(subset(slug,n)['reference_'+k]) for k in METRIC_NAMES] for slug in ('case1','case2_ref1') for n in ('impedance_scale','frequency_shift','L1_C1_reciprocal')]+[['case2_ref1','R7']+[fmt(subset('case2_ref1','R7')['reference_'+k]) for k in METRIC_NAMES]]),'',
        '诊断意义：I-03 主要检查目标区域内的同方向事件存在性，并不连续约束峰高、宽度或相位角度；II-02 不保留复导数的方向且忽略常数实平移；累计弧长对固定幅相坐标中的平移不敏感。各项盲区不同，不能据此把所有敏感指标都加入 Pareto 目标。',
        '', '另一项具体发现：纯整体阻抗缩放没有移动任何峰，II-01 仍有非零响应（Case1约0.04348，Case2参考1约0.09383）。其公式是曲线残差在目标切向上的投影，幅值变化也可能进入这个投影；因此名称中的“频移”不能解释成它只测真实频移。这是语义与机制的审查线索，尚不证明它会损害优化。',
        '', '## 固定匹配窗口可能制造假的形状变化','',
        '整体频移有精确已知的峰对应关系：每个峰都随频率乘1/a，峰高与相对半功率宽度保持不变（端点仍在观察频带内时）。本轮对此采用解析连续对应并数值核验，同时保存固定参考窗口的结果作为匹配方法诊断。其他一般轨迹仍使用固定参考窗口，不能把大扰动下的匹配结果当成已证明的同一峰延续。', '',
        table(['配置','真实对应最大偏移 / decade','固定窗最大偏移 / decade','真实对应宽度比','固定窗宽度比','固定窗未匹配峰数'],[[slug,fmt(r['peak_shift_max_abs_decades']),fmt(r['fixed_window_peak_shift_max_abs_decades']),fmt(np.exp(r['peak_width_signed_mean_logratio'])),fmt(np.exp(r['fixed_window_peak_width_signed_mean_logratio'])),r['fixed_window_missing_peaks']] for slug in ('two_overlap','case1','case2_ref1') for r in [subset(slug,'frequency_shift')]]),'',
        '上表为整体频移30档正向。这说明观察到“带宽变了”时，首先应检查事件是否配对正确。两个接近的峰整体移动时尤其容易跨窗配错；这种误差如果被用作优化目标，可能鼓励优化器修正一个实际上并不存在的形状问题。这里证明的是该匹配机制的反例，没有直接归因为此前正式优化的失败原因。',
        '', '## 数值质量与可复现范围','',
        f'共计算 {len(rows)} 条有限扰动谱（另含参考、局部差分与数值核验），生成 {len(checks)} 条数值检查记录，全部通过。检查包括原电路代码一致性、解析频率导数、两种整体缩放恒等式、半步参数敏感性、关键1/30档轨迹的频率网格加密，以及可实现实轴平移的辅助项零响应。',
        '',
        '密集机制统计按等 log 频率梯形积分，不能与旧报告逐采样点平均 MSE 混用。原采样指标仍使用 Case1 180点、Case2 原201点。加密收敛检查覆盖 L1、C1、整体频移和 LC补偿的正反1/30档三项积分误差，要求相对变化小于0.5%；不声称对全部参数空间实现严格误差界。事件定位采用固定网格括区及34步二分；大扰动后的窗口未匹配、同窗口多峰和带宽缺失均保留。',
        '', 'Case2 图中201个绿色点是原实测数据，密集线是模型预测，未添加或插值新测量。Case2单参数图的Nyquist面板明确限制在主要谐振频段，以免低频大阻抗淹没局部变化；图册另保留完整原频带与局部放大对照。幅频、相频与幅值变化面板均保留完整原频带。',
        '', '## 下一步应如何使用这些结果','',
        '优先对三类明确机制做指标审查：峰位基本正确而宽度或幅值错误；背景电阻导致相位角度偏差；相邻谐振靠近造成事件匹配变化。下一轮可从本轮图册中挑出代表性参数轨迹，用完整幅相误差及分频段响应作为共同评判，再确定是否修改指标。',
        '', '随后扩展参考参数区域和具有新拓扑机制的文献电路。只有在形成明确改动假设后，才启动多随机种子的优化验证；本轮不改动既有正式结论。',
        '', '数据入口：[逐参数变化速查](PARAMETER_GUIDE.md)、[全部扰动](data/perturbations.csv)、[逐参数敏感性](data/sensitivity.csv)、[数值检查](data/numerical_checks.csv)、[局部方向](data/local_directions.json)、[图册](index.html)。',
        '', '来源：本项目英文论文 Table III/Fig.10/Fig.15、既有 Case1/Case2 电路代码、原 Case2 CSV 和2026-10-05冻结指标代码。所有输入文件哈希保存在 data/PROTOCOL.json；输出哈希保存在 INTEGRITY.json。',
    ]
    return '\n'.join(lines)+'\n'


def build_parameter_guide(models,rows,sens_rows):
    def fmt(x): return '—' if not np.isfinite(x) else f'{x:.4g}'
    text=['# 每个参数改变时，曲线怎么变化','',
        '以下均为单参数正向×1.1，相对各自模型参考。负向和其他档位见图册及 perturbations.csv。峰位、峰高、宽度统计只涉及固定参考窗口内匹配的端口峰，跨窗移动可能配错；单串联 RLC 主要表现为谷，因此峰统计记—。多峰的平均变化可能抵消，必须与最大偏移和具体曲线一起阅读。',
        '', '相位/幅值 RMS 按等 log 频率积分；峰高和宽度列是成功匹配峰的平均对数比再取指数（几何平均比值）。敏感频段是局部相对复响应能量10%–90%分位覆盖区间，不表示区间内每个频点都同样敏感。', '']
    for model in models:
        text.extend(['## '+model.title,'',model.topology,'',
            '| 参数×1.1 | 幅值 RMS / dB | 相位 RMS / ° | 最大峰位偏移 / decade | 峰高比 | 相对宽度比 | 未匹配峰数 | 敏感频段 / Hz |',
            '| --- | --- | --- | --- | --- | --- | --- | --- |'])
        for name in model.names:
            r=next(r for r in rows if r['circuit']==model.slug and r['direction']==name and r['level_pct']==10 and r['sign']==1)
            s=next(s for s in sens_rows if s['circuit']==model.slug and s['parameter']==name)
            text.append('| '+' | '.join([name,fmt(r['magnitude_integral_rmse_db']),fmt(r['phase_integral_rmse_deg']),
                fmt(r['peak_shift_max_abs_decades']),fmt(np.exp(r['peak_height_signed_mean_logratio'])),fmt(np.exp(r['peak_width_signed_mean_logratio'])),
                str(r['reference_peak_missing_window_count']),f"{s['energy_10pct_frequency_hz']:.4g}–{s['energy_90pct_frequency_hz']:.4g}"])+' |')
        text.append('')
    text.extend(['完整数值：[全部扰动](data/perturbations.csv)。机制解释：[主报告](REPORT.md)。曲线：[图册](index.html)。',''])
    return '\n'.join(text)


def write_html(out,gallery,report):
    sections=[]
    for model,items in gallery:
        pictures=''.join(f'<details><summary>{html.escape(title)}</summary><a href="figures/{file}" target="_blank"><img loading="lazy" src="figures/{file}" alt="{html.escape(title)}"></a></details>' for file,title in items)
        sections.append(f'<section id="{model.slug}"><h2>{html.escape(model.title)}</h2><p>{html.escape(model.topology)}</p>{pictures}</section>')
    nav=''.join(f'<a href="#{model.slug}">{html.escape(model.title)}</a>' for model,_ in gallery)
    def render(text): return markdown.markdown(text,extensions=['tables']).replace('<table>','<div class="tablewrap"><table>').replace('</table>','</table></div>')
    rendered=render(report)
    guide_rendered=render((out/'PARAMETER_GUIDE.md').read_text(encoding='utf-8'))
    content=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>参数如何改变阻抗曲线</title><style>
body{{margin:0;background:#f3f6fa;color:#213a55;font:16px/1.7 "Microsoft YaHei",sans-serif}}main{{max-width:1220px;margin:auto;padding:28px}}h1{{font-size:30px}}h2{{font-size:23px}}nav{{display:flex;flex-wrap:wrap;gap:10px}}nav a,.button{{background:white;padding:8px 12px;border:1px solid #d9e4ef;border-radius:8px;text-decoration:none;color:#155aa8}}section{{margin-top:30px;background:white;border-radius:12px;padding:22px}}summary{{cursor:pointer;font-weight:bold;padding:12px;border-bottom:1px solid #e7edf5}}details{{margin:10px 0}}img{{width:100%;height:auto}}.lead{{background:#e8f1fb;padding:18px;border-radius:10px}}a{{color:#155aa8}}pre{{white-space:pre-wrap;font:14px/1.7 "Microsoft YaHei",sans-serif}}.footer{{font-size:13px;color:#536578}}</style><main>
<style>.tablewrap{{overflow-x:auto}}table{{border-collapse:collapse;font-size:13px;min-width:650px}}th,td{{border:1px solid #d9e4ef;padding:8px;text-align:left}}th{{background:#e8f1fb}}</style><h1>参数如何改变阻抗曲线</h1><p>2026-10-06 · 第一轮本地机制分析 · 5种自设计电路 + Case1 + Case2的3组拟合参考</p>
<div class="lead">先查看带符号敏感性，再展开单参数或组合曲线。1、10、30档使用乘性因子，负向为正向因子的倒数。Case2绿色点是实测，曲线是模型预测。点击图可打开原图。<br><a href="#report">中文报告</a> · <a href="#guide">逐参数速查</a> · <a href="data/perturbations.csv">全部扰动数据</a> · <a href="data/reference_parameters.csv">元件参数</a> · <a href="data/numerical_checks.csv">数值检查</a></div>
<h2>电路图册</h2><nav>{nav}</nav>{''.join(sections)}
<section id="report"><h2>完整报告</h2>{rendered}</section>
<section id="guide"><h2>逐参数速查</h2>{guide_rendered}</section>
<p class="footer">本轮全部本地计算，无新增优化。Case2元件参数是拟合参考，不是真值。局部敏感性不能直接代表有限变化或全局可辨识性。</p></main></html>'''
    (out/'index.html').write_text(content,encoding='utf-8')


if __name__=='__main__': main()

"""Deterministic, target-fixed curve diagnostics; no optimization or plotting."""
from pathlib import Path
import sys
import numpy as np
from scipy.signal import find_peaks,peak_widths
from scipy.optimize import linear_sum_assignment
from model import module
HERE=Path(__file__).resolve().parent
pop=module('structure_population_network',HERE/'frozen/population_source.py')

def weights(f):
    x=np.log10(f); h=np.diff(x); w=np.zeros(len(x));w[:-1]+=h/2;w[1:]+=h/2
    return w/(x[-1]-x[0])
def norm(y,w): return float(np.sqrt(np.sum(w*abs(y)**2)))
def network(r,params):
    return pop.Network(r['id'],r['family'],r['title'],r['tree'],r['parameter_names'],np.asarray(params),'DIAG',0,0,{})
def dense_grid(r,parameters,refinement=1):
    lo,hi=np.log10(r['frequency_hz'][0]),np.log10(r['frequency_hz'][-1])
    values=[np.linspace(lo,hi,2048*refinement+1),np.log10(r['frequency_hz']),np.linspace(lo,hi,9)]
    for p in [r['target_parameters'],*parameters]:
        for fc,width in network(r,p).modes():
            c=np.log10(fc); dx=width/np.log(10)
            if lo<=c<=hi: values.append(np.linspace(max(lo,c-dx),min(hi,c+dx),192*refinement+1))
    return 10**np.unique(np.concatenate(values))
def events(x,z,prominence_db=1.):
    a=20*np.log10(abs(z));phase=np.rad2deg(np.angle(z)); result=[]
    for field,y,prom in [('amp',a,prominence_db),('phase',phase,5.)]:
        for direction in [1,-1]:
            ids,props=find_peaks(direction*y,prominence=prom)
            widths=peak_widths(direction*y,ids,rel_height=.5,prominence_data=(props['prominences'],props['left_bases'],props['right_bases'])) if len(ids) else None
            for k,i in enumerate(ids):
                width=float(np.interp(widths[3][k],np.arange(len(x)),x)-np.interp(widths[2][k],np.arange(len(x)),x))
                result.append(dict(kind=field+'_peak' if direction==1 else field+'_valley',x=float(x[i]),value=float(y[i]),prominence=float(props['prominences'][k]),width=width))
    tol=1e-12*float(np.max(abs(z))); ids=np.flatnonzero(abs(z.imag)>tol)
    for i,j in zip(ids[:-1],ids[1:]):
        if np.signbit(z.imag[i])==np.signbit(z.imag[j]):continue
        pos=x[i]+(x[j]-x[i])*abs(z.imag[i])/(abs(z.imag[i])+abs(z.imag[j]))
        result.append(dict(kind='cross_up' if z.imag[j]>0 else 'cross_down',x=float(pos),value=0.,prominence=0.,width=0.))
    return result
def match_events(target,fit):
    result={};details=[]
    for prefix,kinds in [('amp',['amp_peak','amp_valley']),('phase',['phase_peak','phase_valley']),('cross',['cross_up','cross_down'])]:
        missed=extra=matches=0; shifts=[]; widths=[]; heights=[]
        for kind in kinds:
            t=[e for e in target if e['kind']==kind];f=[e for e in fit if e['kind']==kind]
            used=set(); count=0
            if t:
                cost=np.full((len(t),len(f)+len(t)),.250001)
                for i,a in enumerate(t):
                    for j,b in enumerate(f):
                        d=abs(a['x']-b['x']);cost[i,j]=d if d<=.25 else 1e6
                rr,cc=linear_sum_assignment(cost)
                for i,j in zip(rr,cc):
                    if j>=len(f) or cost[i,j]>.25:continue
                    count+=1;used.add(int(j));shifts.append(float(f[j]['x']-t[i]['x']))
                    if prefix!='cross':
                        heights.append(float(f[j]['value']-t[i]['value']))
                        if t[i]['width']>0 and f[j]['width']>0:widths.append(float(np.log2(f[j]['width']/t[i]['width'])))
                    details.append(dict(kind=kind,target_x=t[i]['x'],fit_x=f[j]['x'],shift_decades=shifts[-1]))
            missed+=len(t)-count;extra+=len(f)-len(used);matches+=count
        result.update({prefix+'_target_events':sum(e['kind'] in kinds for e in target),prefix+'_fit_events':sum(e['kind'] in kinds for e in fit),prefix+'_missed':missed,prefix+'_extra':extra,prefix+'_matched':matches,
            prefix+'_mean_abs_shift_decades':float(np.mean(np.abs(shifts))) if shifts else None,
            prefix+'_mean_abs_width_log2_ratio':float(np.mean(np.abs(widths))) if widths else None,
            prefix+'_mean_abs_event_value_error':float(np.mean(np.abs(heights))) if heights else None})
    return result,details
def band_integral(x,values,lo,hi):
    mask=(x>lo)&(x<hi); xb=np.r_[lo,x[mask],hi];yb=np.r_[np.interp(lo,x,values),values[mask],np.interp(hi,x,values)]
    return float(np.trapezoid(yb,xb)/(hi-lo))
def diagnose(f,target,z):
    x=np.log10(f);w=weights(f);da=20*np.log10(abs(z)/abs(target)); dp=np.rad2deg(np.angle(z)-np.angle(target));wrapped=np.rad2deg(np.angle(z/target));dr=abs(z-target)
    gain=float(np.sum(w*da)); phase_offset=float(np.sum(w*dp)); amp=norm(da,w); phase=norm(dp,w)
    out=dict(amplitude_rmse_db=amp,phase_rmse_deg=phase,phase_wrapped_rmse_deg=norm(wrapped,w),complex_rms_ohm=norm(z-target,w),complex_rms_target_relative=norm(z-target,w)/norm(target,w),mean_distance_logweighted_ohm=float(np.sum(w*dr)),gain_offset_db=gain,
        amplitude_shape_rmse_db=norm(da-gain,w),phase_offset_deg=phase_offset,phase_shape_rmse_deg=norm(dp-phase_offset,w),amplitude_bad_span_fraction=float(np.sum(w*(abs(da)>3))),phase_bad_span_fraction=float(np.sum(w*(abs(dp)>15))),max_amplitude_error_db=float(np.max(abs(da))),max_phase_error_deg=float(np.max(abs(dp))))
    out['gain_squared_error_fraction']=gain**2/amp**2 if amp>1e-12 else 0.
    edges=np.linspace(x[0],x[-1],9);bands=[]
    for k,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        bands.append(dict(band=k+1,lo_hz=float(10**lo),hi_hz=float(10**hi),amplitude_rmse_db=np.sqrt(band_integral(x,da**2,lo,hi)),phase_rmse_deg=np.sqrt(band_integral(x,dp**2,lo,hi)),mean_distance_ohm=band_integral(x,dr,lo,hi),complex_rms_ohm=np.sqrt(band_integral(x,dr**2,lo,hi))))
    out['worst_band_amplitude_rmse_db']=max(b['amplitude_rmse_db'] for b in bands);out['worst_band_phase_rmse_deg']=max(b['phase_rmse_deg'] for b in bands)
    out['worst_amplitude_band']=int(np.argmax([b['amplitude_rmse_db'] for b in bands])+1);out['worst_phase_band']=int(np.argmax([b['phase_rmse_deg'] for b in bands])+1)
    matched,details=match_events(events(x,target),events(x,z));out.update(matched)
    for p in [.5,2.]:
        m,_=match_events(events(x,target,p),events(x,z,p));key=str(p).replace('.','p');out['amp_missed_prom_'+key]=m['amp_missed'];out['amp_extra_prom_'+key]=m['amp_extra']
    return out,bands,details
def loss_location(target,z):
    rho=abs(z);rho0=abs(target);delta=np.angle(z)-np.angle(target); d=abs(z-target); total=d.sum();q=np.zeros(len(target),bool);q[np.argsort(rho0,kind='stable')[-int(np.ceil(len(target)/4)):]]=True
    ma=(20*np.log10(rho/rho0))**2;mp=(np.angle(z)-np.angle(target))**2
    radial=(rho-rho0)**2;angular=2*rho*rho0*(1-np.cos(delta));sq=radial+angular
    fractions=np.divide(angular,sq,out=np.zeros_like(sq),where=sq>np.finfo(float).tiny)
    def share(y):return float(y[q].sum()/y.sum()) if y.sum()>1e-24 else 0.
    return dict(distance_high_target_quartile_share=share(d),magnitude_mse_high_target_quartile_share=share(ma),phase_mse_high_target_quartile_share=share(mp),distance_angular_share=float((d*fractions).sum()/total) if total>1e-12 else 0.,phase_largest_point_squared_error_share=float(mp.max()/mp.sum()) if mp.sum()>1e-24 else 0.,phase_target_extreme_point_squared_error_share=float(mp[int(np.argmax(abs(np.angle(target))))]/mp.sum()) if mp.sum()>1e-24 else 0.)
def interaction(m,log_parameters):
    p0=m.record['target_parameters']
    if p0 is None:return {'interaction_eta':None,'additive_cancellation':None,'interaction_rms_ohm':None,'total_rms_ohm':None,'additive_rms_ohm':None,'parameter_log10_rms':None}
    p0=np.array(p0);p=10**np.array(log_parameters);hybrids=np.tile(p0,(len(p),1));hybrids[np.arange(len(p)),np.arange(len(p))]=p
    z=m.network(m.f,p)[0];singles=m.network(m.f,hybrids)-m.target;add=singles.sum(axis=0);total=z-m.target;rem=total-add;w=weights(m.f)
    a,b,c=norm(total,w),norm(add,w),norm(rem,w);sum_single=sum(norm(y,w) for y in singles)
    tolerance=1e-12*norm(m.target,w)
    return dict(interaction_eta=c/(a+b) if a+b>tolerance else 0.,additive_cancellation=float(max(0.,1-b/sum_single)) if sum_single>tolerance else 0.,interaction_rms_ohm=c,total_rms_ohm=a,additive_rms_ohm=b,parameter_log10_rms=float(np.sqrt(np.mean((np.log10(p/p0))**2))))

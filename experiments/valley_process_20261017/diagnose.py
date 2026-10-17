"""Local, read-only cohort diagnosis. No new scoring, label or optimizer."""
from pathlib import Path
import sys,json,time,datetime
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/variation_transport_20261017'))
from metric import *
from core import checked_nodal
from scipy.optimize import brentq
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=ROOT/'output/valley_process_20261017'
OLD=ROOT/'output/geometry_followup_20261016/HOLDOUT_BANK.json'
MODELS={}

def get(tid):
    if tid not in MODELS:MODELS[tid]=Variation(tid)
    return MODELS[tid]

def rx(m,x,p):
    z,d=m.c.response(10**np.asarray(x),p)
    return z,d*np.log(10)

def radial(m,x,p):
    z,d=rx(m,np.asarray(x),p)
    return np.real(np.conj(z)*d)/abs(z)

def roots(m,p,x):
    v=radial(m,x,p);result=[]
    for i in np.where(v[:-1]*v[1:]<0)[0]:
        q=brentq(lambda xx:float(radial(m,np.array([xx]),p)[0]),x[i],x[i+1],xtol=5e-15)
        if result and abs(q-result[-1]['x'])<1e-13:continue
        z,d=rx(m,np.array([q]),p);r=z[0].real;xx=z[0].imag;dr=d[0].real;dx=d[0].imag
        result.append(dict(x=float(q),frequency_hz=float(10**q),kind='valley' if v[i]<v[i+1] else 'peak',real_ohm=float(r),imag_ohm=float(xx),magnitude_ohm=float(abs(z[0])),phase_deg=float(np.rad2deg(np.angle(z[0]))),R_dR=float(r*dr),X_dX=float(xx*dx),radial_balance_relative=float(abs(r*dr+xx*dx)/max(abs(r*dr)+abs(xx*dx),1e-30)),normalized_radial_dot=float(np.real(np.conj(z[0])*d[0])/max(abs(z[0])*abs(d[0]),1e-30))))
    return result

def path_stats(m,p,x):
    z,d=rx(m,x,p);ds=abs(np.diff(z));length=float(sum(ds));cs=np.r_[0.,np.cumsum(ds)]
    qs=np.interp(np.array([.1,.5,.9])*length,cs,x) if length>0 else np.array([x[0]]*3)
    return dict(length_ohm=length,endpoint_displacement_ohm=float(abs(z[-1]-z[0])),endpoint_efficiency=float(abs(z[-1]-z[0])/max(length,1e-30)),q10_x=float(qs[0]),q50_x=float(qs[1]),q90_x=float(qs[2]),arc80_width_decades=float(qs[2]-qs[0]),radial_extrema=roots(m,p,x))

def integral_length(m,w,p,lo,hi,n,ref):
    e=m.g.panel_edges(w,m.g.modes(p),ref);xx=w['x'][0]+e*(w['x'][-1]-w['x'][0]);xx=np.unique(np.r_[lo,xx[(xx>lo)&(xx<hi)],hi]);t,wt=nodes(xx,n);z,d=rx(m,t,p)
    return float(wt@abs(d))

def widths(m,p,event,global_x,global_roots):
    if event is None:return None
    candidates=[e for e in global_roots if e['kind']=='valley']
    if not candidates:return dict(resolved=False,reason='No analytic valley root')
    e=min(candidates,key=lambda e:abs(e['x']-event['x']));center=e['x'];height=20*np.log10(e['magnitude_ohm'])
    x=np.unique(np.r_[global_x,center]);z,_=rx(m,x,p);db=20*np.log10(abs(z));out={}
    for level in [1.,3.]:
        v=db-(height+level);ii=np.where(v[:-1]*v[1:]<0)[0];left=[i for i in ii if x[i+1]<=center];right=[i for i in ii if x[i]>=center]
        fun=lambda xx:float(20*np.log10(abs(rx(m,np.array([xx]),p)[0][0]))-height-level)
        if not left or not right:out[str(int(level))]=None;continue
        i,j=left[-1],right[0];l=brentq(fun,x[i],x[i+1],xtol=5e-15);r=brentq(fun,x[j],x[j+1],xtol=5e-15)
        out[str(int(level))]=dict(left_x=float(l),right_x=float(r),width_decades=float(r-l),left_extent=float(center-l),right_extent=float(r-center),asymmetry_right_over_left=float((r-center)/max(center-l,1e-30)))
    return dict(resolved=True,source_sample_x=event['x'],refined_x=center,refinement_shift=float(center-event['x']),height_db=float(height),levels=out)

def one(ident,tid,p,source):
    m=get(tid);w=m.window('V3');event=next(e for e in m.data['record']['events'] if e['event_id']=='V3');lo,hi=w['x'][[0,-1]];cl,cc,ch=event['left_3db_x'],event['x'],event['right_3db_x'];levels=[];profiles={}
    anchors=np.array([cl,cc,ch]);az,ad=rx(m,anchors,p);tz,td=rx(m,anchors,m.c.p)
    anchor_record=[dict(label=label,x=float(x),frequency_hz=float(10**x),real_ohm=float(z.real),imag_ohm=float(z.imag),target_real_ohm=float(r.real),target_imag_ohm=float(r.imag),complex_error_ohm=float(abs(z-r)),tangent_cosine_to_target=float(np.real(np.conj(v)*v0)/max(abs(v)*abs(v0),1e-30))) for label,x,z,r,v,v0 in zip(['left3dB','target_minimum','right3dB'],anchors,az,tz,ad,td)]
    for n in [128,256]:
        t=m.polygrid(w,p,n);x=np.unique(np.r_[lo+(hi-lo)*t,anchors]);core=x[(x>=cl)&(x<=ch)]
        stat=dict(n=n,window=path_stats(m,p,x),core=path_stats(m,p,core));levels.append(stat)
        if n==256:
            z,d=rx(m,x,p);profiles=dict(x=x,z=z,dz_dx=d)
    checks={}
    for scope,l,r in [('window',lo,hi),('core',cl,ch)]:
        a,b=levels[0][scope],levels[1][scope];g1=integral_length(m,w,p,l,r,32,1);g2=integral_length(m,w,p,l,r,64,2)
        numeric=lambda x,y:abs(x-y)<=1e-4+1e-3*abs(y)
        rr1,rr2=a['radial_extrema'],b['radial_extrema'];rootok=len(rr1)==len(rr2) and all(x['kind']==y['kind'] and abs(x['x']-y['x'])<=1e-9 for x,y in zip(rr1,rr2))
        checks[scope]=dict(polygon_length_agrees=numeric(a['length_ohm'],b['length_ohm']),gauss_length_agrees=numeric(g1,g2),polygon_gauss_agrees=numeric(b['length_ohm'],g2),gauss32=g1,gauss64=g2,roots_stable=rootok,quantiles_stable=all(abs(a[key]-b[key])<=1e-4*(r-l)+1e-3*max(a['arc80_width_decades'],b['arc80_width_decades']) for key in ['q10_x','q50_x','q90_x']))
    # A global view is only used to interpret the source-matched event's width.
    glob=dict(x=np.array([2.,6.]));e=m.g.panel_edges(glob,m.g.modes(p),0);gx=np.unique((2+4*(e[:-1,None]+np.diff(e)[:,None]*np.linspace(0,1,65))).ravel());gr=roots(m,p,gx)
    if source is None:source_event=event;oldlabel=None
    else:
        oldlabel=next(e for e in source['candidate']['events'] if e['event_id']=='V3');source_event=next((e for e in source['candidate']['fit'] if e['event_id']==oldlabel.get('fit_event_id')),None)
    width=widths(m,p,source_event,gx,gr)
    ix=np.unique(np.r_[np.linspace(0,len(profiles['x'])-1,17).astype(int),np.searchsorted(profiles['x'],anchors)]);nodal=checked_nodal(m.c,10**profiles['x'][ix],p)
    ok=all(all(v for k,v in c.items() if k not in ['gauss32','gauss64']) for c in checks.values())
    result=dict(id=ident,target=tid,parameters=p,source=source,old_V3_label=oldlabel,target_event=event,fixed_window=[float(lo),float(hi)],fixed_core=[cl,ch],anchors=anchor_record,levels=levels,matched_valley_widths=width,numerical_checks=checks,numerical_passed=ok,nodal=nodal)
    return result,profiles

def physical_reference():
    # Series RLC: R constant; valley caused by crossing the imaginary axis level X=0.
    R=10.;f0=25000.;Q=10.;w0=2*np.pi*f0;L=Q*R/w0;C=1/(w0*Q*R);f=np.geomspace(f0/2,f0*2,1001);om=2*np.pi*f;z=R+1j*(om*L-1/(om*C));d=1j*np.log(10)*(om*L+1/(om*C));v=np.real(np.conj(z)*d)/abs(z)
    assert np.all(z.real==R);assert abs(z[len(f)//2].imag)<1e-10;assert v[100]<0<v[-100]
    return dict(R=R,f0=f0,Q=Q,L=L,C=C,real_constant=True,valley_imag_zero=True,complex_plane_path='straight line',no_loop_required=True)

def main():
    OUT.mkdir(parents=True,exist_ok=True);bank=json.loads(OLD.read_text());selected=[p for p in bank['selected'] if p['target']!='case1'];assert len(selected)==16
    tasks=[(f"{p['pair_id']}_{s}",p['target'],(10**np.array(p[s]['log_parameters'])).tolist(),dict(pair_id=p['pair_id'],side=s,original_region=p['region'],original_family=p['family'],candidate=p[s])) for p in selected for s in ['better','worse']]
    frozen=dict(tasks=tasks,scope='All 32 previously frozen mixed-network candidates, diagnosed at V3; no new validation',bank_sha256=sha(OLD),source_sha256=sha(HERE/'diagnose.py'),protocol_sha256=sha(ROOT/'docs/混合网络谷结构完整过程_诊断方案_20261017.md'))
    dump(OUT/'COHORT.json',frozen);dump(OUT/'PHYSICAL_REFERENCE.json',physical_reference());allresults=[];arrays={};st=time.time()
    for tid in sorted({t[1] for t in tasks}):
        result,profile=one('target_'+tid.replace('/','_'),tid,get(tid).c.p,None);allresults.append(result)
        for key,v in profile.items():arrays[result['id']+'__'+key]=v
    for ident,tid,p,src in tasks:
        result,profile=one(ident,tid,np.array(p),src);allresults.append(result)
        for key,v in profile.items():arrays[ident+'__'+key]=v
        print(ident,'numerical',result['numerical_passed'],flush=True)
    dump(OUT/'DIAGNOSTICS.json',dict(records=allresults,numerical_passed=sum(r['numerical_passed'] for r in allresults),seconds=time.time()-st,source_sha256=sha(HERE/'diagnose.py')));np.savez_compressed(OUT/'PROFILES.npz',**arrays)
    print('Finished',len(allresults),'records',sum(r['numerical_passed'] for r in allresults),'passed',flush=True)
if __name__=='__main__':main()

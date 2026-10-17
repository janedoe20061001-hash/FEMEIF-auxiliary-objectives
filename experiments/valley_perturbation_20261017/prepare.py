"""Freeze every target, physical parameter and intervention before evaluation."""
from physical import *
import itertools

def main():
    OUT.mkdir(parents=True,exist_ok=True);targets=[];old=ROOT/'output/valley_process_20261017/DIAGNOSTICS.json';prior=json.loads(old.read_text(encoding='utf-8'))
    for tid in ['mixed_resonators/base','mixed_resonators/lc_reciprocal_8p0']:
        c=target(tid)['circuit'];q=next(r for r in prior['records'] if r['target']==tid and r['source'] is None);e=next(e for e in q['levels'][-1]['core']['radial_extrema'] if e['kind']=='valley');targets.append(dict(id=tid,kind='mixed',model=record(c),core=q['fixed_core'],window=q['fixed_window'],center=e['x'],scale=e['magnitude_ohm']))
    f0=25000.;w0=2*np.pi*f0
    for kind,Q in [('series',10.),('notch',20.),('notch',2000.)]:
        if kind=='series':
            R=25.;values=dict(R0=R,L0=Q*R/w0,C0=1/(w0*Q*R));tree=['S','R0','L0','C0'];distance=np.arcsinh(np.sqrt(10**.3-1)/(2*Q))/np.log(10);minimum=R
        else:
            Rs,Rp,Rm=5.,200.,20.;T=Rp+Rm;values=dict(Rs=Rs,Rp=Rp,Rm=Rm,L0=T*Q/w0,C0=1/(w0*Q*T));tree=['S','Rs',['P','Rp',['S','Rm','L0','C0']]];B=Rs+Rp;K=Rp*Rp;minimum=B-K/T;X2=(2*B*K*T-K*K)/(B*B-10**.3*minimum**2)-T*T;distance=np.arcsinh(np.sqrt(X2)/(2*T*Q))/np.log(10)
        center=np.log10(f0);width=2*distance;tid=f'{kind}_Q{int(Q)}';targets.append(dict(id=tid,kind=kind,Q=Q,model=dict(id=tid,tree=tree,values=values),core=[center-distance,center+distance],window=[center-4*width,center+4*width],center=center,scale=minimum))
    tasks=[]
    def add(t,family,values=None,tree=None,**meta):
        m=copy.deepcopy(t['model']);m['id']=f"setting_{len(tasks):04}";m['values']=values if values is not None else m['values'];m['tree']=tree if tree is not None else m['tree'];Physical(m)
        tasks.append(dict(id=m['id'],target=t['id'],family=family,meta=meta,model=m))
    for t in targets:
        base=t['model']['values'];width=t['core'][1]-t['core'][0];center=t['center'];offset_resistor='R0' if t['kind']=='series' else 'Rs'
        add(t,'baseline')
        for shift in [-2,-1,-.5,-.25,.25,.5,1,2]:
            s=10**(shift*width);v={n:value/s if n[0] in 'LC' else value for n,value in base.items()};add(t,'shift',v,shift_widths=shift)
        for amount in [.05,.1,.25,.5]:
            v=base.copy();v[offset_resistor]+=amount*t['scale'];add(t,'offset',v,relative_offset=amount,delta_R=amount*t['scale'])
        if t['kind']!='mixed':
            for k in [.125,.25,.5,1.5,2,4,8]:
                v=base.copy();v['L0']*=k;v['C0']/=k;add(t,'Q',v,factor=k)
        for amp,shift,fraction in itertools.product([.1,.5,1.],[-.1,0,.1],[.01,.05,.2]):
            R=amp*t['scale'];f=10**(center+shift*width);Q=1/(2*np.sinh(np.log(10)*fraction*width/2));v={**base,'Re':R,'Le':R/(2*np.pi*f*Q),'Ce':Q/(2*np.pi*f*R)};tree=['S',t['model']['tree'],['P','Re','Le','Ce']];add(t,'extra',v,tree,relative_R=amp,shift_widths=shift,width_fraction=fraction,extra_R=R,extra_f0=f,extra_Q=Q)
        c=Physical(t['model']);z=c.response(np.array([10**center]))[0][0];v=dict(Rb=float(z.real));tree='Rb'
        if t['kind']=='mixed':v['Lb']=float(z.imag/(2*np.pi*10**center));tree=['S','Rb','Lb']
        add(t,'background',v,tree)
        if t['kind']=='mixed':
            for name,k in itertools.product(['R1','R3'],[.25,.5,2.,4.]):
                v=base.copy();v[name]*=k;add(t,'damping',v,component=name,factor=k)
        else:
            for shift,offset,k in itertools.product([-1,1],[.1,.5],[.5,4.]):
                s=10**(shift*width);v=base.copy();v['L0']*=k/s;v['C0']/=k*s;v[offset_resistor]+=offset*t['scale'];add(t,'combined',v,shift_widths=shift,relative_offset=offset,Q_factor=k,delta_R=offset*t['scale'])
    assert len(targets)==5 and len(tasks)==266
    plan=dict(targets=targets,tasks=tasks,scope='Physical mechanism development; new perturbation parameters, not independent method validation',prior_input_sha256=sha(old),protocol_sha256=sha(ROOT/'docs/谷过程判据_物理扰动检验方案_20261017.md'))
    dump(OUT/'PLAN.json',plan)
    files=list(HERE.glob('*.py'))+[ROOT/'docs/谷过程判据_物理扰动检验方案_20261017.md',old,OUT/'PLAN.json',ROOT/'experiments/valley_process_20261017/diagnose.py',ROOT/'experiments/variation_transport_20261017/metric.py',ROOT/'experiments/arx_light_20261015/core.py',ROOT/'experiments/arx_light_20261015/frozen/circuits.py',ROOT/'experiments/arx_light_20261015/reference_features.py']
    dump(OUT/'SOURCE_MANIFEST.json',dict(files={p.relative_to(ROOT).as_posix():sha(p) for p in files},frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print('Frozen',len(targets),'targets and',len(tasks),'physical settings',flush=True)
if __name__=='__main__':main()

"""Freeze fresh physical settings and legacy candidates before scoring."""
from shape import *
import itertools
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    prior=ROOT/'output/valley_perturbation_20261017/PLAN.json';p=json.loads(prior.read_text(encoding='utf-8'));tasks=[]
    for t in p['targets']:
        base=t['model'];W=t['core'][1]-t['core'][0];S=t['scale'];res='R0' if t['kind']=='series' else 'Rs'
        def add(family,v=None,tree=None,**meta):
            m=copy.deepcopy(base);m['id']=f'new_{len(tasks):03}';m['values']=v if v is not None else m['values'];m['tree']=tree if tree is not None else m['tree'];Physical(m)
            tasks.append(dict(id=m['id'],target=t['id'],cohort='new',family=family,meta=meta,model=m))
        def changed(k,shift):
            scale=10**(shift*W);v=base['values'].copy()
            for name in v:
                if name[0] in 'LC':v[name]/=scale
            if t['kind']!='mixed':v['L0']*=k;v['C0']/=k
            return v
        add('baseline',factor=1.,shift_widths=0.)
        for shift in [-1.4,-.65,.65,1.4]:add('shift',changed(1.,shift),factor=1.,shift_widths=shift)
        for amount in [.075,.2,.4,.8]:
            v=base['values'].copy();v[res]+=amount*S;add('offset',v,relative_offset=amount,delta_R=amount*S)
        for amp,shift,frac in itertools.product([.2,.8],[-.17,0.,.17],[.025,.12]):
            R=amp*S;f=10**(t['center']+shift*W);Q=1/(2*np.sinh(np.log(10)*frac*W/2));v={**base['values'],'Re':R,'Le':R/(2*np.pi*f*Q),'Ce':Q/(2*np.pi*f*R)}
            add('extra',v,['S',base['tree'],['P','Re','Le','Ce']],relative_R=amp,shift_widths=shift,width_fraction=frac,extra_R=R,extra_f0=f,extra_Q=Q)
        z=rx(Physical(base),np.array([t['center']]))[0][0];v=dict(Rb=float(z.real));tree='Rb'
        if t['kind']=='mixed':v['Lb']=float(z.imag/(2*np.pi*10**t['center']));tree=['S','Rb','Lb']
        add('background',v,tree)
        if t['kind']!='mixed':
            for k in [.3,.6,1.7,3.3]:add('Q',changed(k,0.),factor=k,shift_widths=0.)
            for k,shift in itertools.product([.3,3.3],[-.65,.65]):add('combined',changed(k,shift),factor=k,Q_factor=k,shift_widths=shift,delta_R=0.)
    assert len(tasks)==134
    oldpath=ROOT/'output/valley_attribution_20261017/PLAN.json';old=json.loads(oldpath.read_text(encoding='utf-8'))
    for q in old['tasks']:
        if q['cohort']=='old_candidate':tasks.append({**q,'cohort':'old','family':'old','meta':{}})
    assert len(tasks)==166
    dump(OUT/'PLAN.json',dict(targets=p['targets'],tasks=tasks,scope='Observable radial-section representation controls, not general mode attribution or optimization'))
    files=list(HERE.glob('*.py'))+[OUT/'PLAN.json',prior,oldpath,ROOT/'docs/复轨迹形态与频率展开_控制实验方案_20261017.md']
    previous=json.loads((ROOT/'output/valley_attribution_20261017/PROVENANCE.json').read_text(encoding='utf-8'))
    files += [ROOT/k for k in previous['sources']]
    dump(OUT/'SOURCE_MANIFEST.json',dict(files={f.relative_to(ROOT).as_posix():sha(f) for f in sorted(set(files))},frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print('Frozen 134 fresh controls + 32 old candidates',flush=True)
if __name__=='__main__':main()

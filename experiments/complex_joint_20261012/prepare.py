"""Copy existing sources and export all 23 target conditions; no optimization."""
from pathlib import Path
import sys,json,shutil,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

def freeze():
    frozen=HERE/'frozen';frozen.mkdir(exist_ok=True)
    copies={
      'circuits.py':'experiments/mse_structure_complex_20261010/circuits.py',
      'sampling.py':'experiments/mse_structure_complex_20261010/sampling.py',
      'diagnostics.py':'experiments/mse_structure_complex_20261010/diagnostics.py',
      'schematic.py':'experiments/mse_structure_complex_20261010/schematic.py',
      'nested_config.json':'experiments/mse_structure_complex_20261010/config.json',
      'case1_adaptive.py':'experiments/mse_structure_step2_20261009/adaptive.py',
      'native.c':'experiments/mse_structure_step1_20261008/native.c',
      'case1_grid.npz':'experiments/mse_structure_step2_20261009/targets/s1/grid.npz',
      'case1_record.json':'experiments/mse_structure_step2_20261009/targets/s1/record.json',
      'selector.py':'experiments/mse_structure_step3_knee_20261011/selector.py',
      'case2_legacy.py':'experiments/case2_dual_pdf18/case2.py',
      'case2.csv':'experiments/case2_dual_pdf18/data/targetImpedance_case2.csv'}
    provenance={}
    for dest,source in copies.items():
        p=ROOT/source;q=frozen/dest
        if q.exists():assert q.read_bytes()==p.read_bytes(),f'Frozen source changed: {dest}'
        else:shutil.copy2(p,q)
        provenance[dest]=dict(source=source,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    (frozen/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')

def partition(catalog,lo,hi):
    events=sorted(catalog,key=lambda e:e['x']);windows=[]
    for i,e in enumerate(events):
        width=e['width_3db_decades']
        if not width or width<=0:continue
        a=max(lo,e['x']-3*width,(events[i-1]['x']+e['x'])/2 if i else lo)
        b=min(hi,e['x']+3*width,(events[i+1]['x']+e['x'])/2 if i+1<len(events) else hi)
        if b-a>1e-12:windows.append(dict(lo=a,hi=b,label=e['event_id'],event=True))
    blocks=[];last=lo
    for w in windows:
        assert w['lo']>=last-1e-12
        if w['lo']-last>1e-12:blocks.append(dict(lo=last,hi=w['lo'],label=f'gap_{len(blocks)}',event=False))
        blocks.append(w);last=w['hi']
    if hi-last>1e-12:blocks.append(dict(lo=last,hi=hi,label='end_gap',event=False))
    if not blocks:blocks=[dict(lo=lo,hi=hi,label='full_band',event=False)]
    assert abs(sum(b['hi']-b['lo'] for b in blocks)-(hi-lo))<1e-10
    return blocks

def main():
    freeze()
    from core import np,Circuit,S,P,dump,INPUTS,BASECFG,CFG,target,diag,sha
    from circuits import library
    from schematic import schematic
    INPUTS.mkdir(exist_ok=True)
    records=[]
    for name,base in library().items():
        lower=np.log10(base.p)-BASECFG['bound_log10_half_width'];upper=np.log10(base.p)+BASECFG['bound_log10_half_width']
        for factor in CFG['nested_lc_factors']:
            key='base' if factor==1 else 'lc_reciprocal_'+str(factor).replace('.','p')
            ident=f'{name}/{key}';c=base if factor==1 else base.changed('lc_reciprocal',factor)
            group='other';source=ROOT/f'output/mse_structure_complex_20261010_server/retrieved/{group}/inputs_core/{ident}/target.json'
            old=json.loads(source.read_text(encoding='utf-8'))
            np.testing.assert_array_equal(c.p,old['circuit']['parameters'])
            records.append(dict(id=ident,kind='theoretical',circuit=c.record(),lower_log10=lower,upper_log10=upper,
                events=old['events'],phase_events=old['phase_events'],factor=factor,
                source=dict(path=source.relative_to(ROOT).as_posix(),sha256=sha(source))))
    old=json.loads((HERE/'frozen/case1_record.json').read_text(encoding='utf-8'))
    c=Circuit('case1','Case1',old['tree'],dict(zip(old['parameter_names'],old['target_parameters'])),('L2','C2'),'R2','R1')
    records.append(dict(id='case1',kind='case1',circuit=c.record(),lower_log10=old['lower_log10'],upper_log10=old['upper_log10'],source='frozen/case1_record.json'))
    from case2_legacy import PARAMETER_NAMES,LOWER,UPPER,impedance
    tree=S(*[P(f'C{k+1}',S(f'L{k+1}',f'R{2*k+1}'),f'R{2*k+2}') for k in range(3)],'C4','R7','L4')
    nominal=dict(zip(PARAMETER_NAMES,10**((LOWER+UPPER)/2)))
    c=Circuit('case2','Case2：实测目标与15元件模型',tree,nominal,('L1','C1'),'R1','R7')
    bounds={n:(lo,hi) for n,lo,hi in zip(PARAMETER_NAMES,LOWER,UPPER)}
    rows=np.genfromtxt(HERE/'frozen/case2.csv',delimiter=',',names=True,encoding='utf-8-sig')
    assert len(rows)==201 and sha(HERE/'frozen/case2.csv')=='ec50906f6cd3e5fdc5792734f7339f8368d2a3280b0a0d14451b0420acc7b725'
    x=np.log10(rows['frequency']);z=rows['mag']*np.exp(1j*np.deg2rad(rows['pha']))
    reference,_=c.response(10**x)
    original=impedance(10**x,np.array([nominal[n] for n in PARAMETER_NAMES]))[0]
    np.testing.assert_allclose(reference,original,rtol=1e-12,atol=1e-9)
    records.append(dict(id='case2',kind='measured',circuit=c.record(),lower_log10=[bounds[n][0] for n in c.names],
                        upper_log10=[bounds[n][1] for n in c.names],known_true_parameters=False,
                        adapter_parameters_note='Circuit parameters here are adapter nominal values, NOT measured ground truth',
                        source='frozen/case2.csv',events=diag.events(x,z,BASECFG),phase_events=[]))
    for rec in records:
        d=INPUTS/rec['id'];d.mkdir(parents=True,exist_ok=True)
        dump(d/'target.json',rec)
        if rec['kind']=='measured':np.savez_compressed(d/'target_curve.npz',x=x,z=z)
        data=target(rec['id']);q=data['quadrature'];c=data['circuit']
        if rec['kind']=='measured':
            # Save before loading the measured quadrature on the next pass.
            pass
        else:
            xx=q.reference_grid(c.p,4);zz,dz=c.response(10**xx)
            rec['events']=diag.events(xx,zz,BASECFG)
            rec['phase_events']=diag.phase_events(xx,zz,dz,BASECFG)
            rec['blocks']=partition(rec['events'],float(xx[0]),float(xx[-1]))
            np.savez_compressed(d/'target_curve.npz',x=xx,z=zz)
            (d/'circuit.svg').write_text(schematic(c),encoding='utf-8')
        if rec['kind']=='measured':
            rec['blocks']=[]
            starts=list(range(0,len(x)-24,24))
            if len(x)-starts[-1]<33:starts=starts[:-1]
            for j,start in enumerate(starts):
                end=min(start+25,len(x)) if j+1<len(starts) else len(x)
                rec['blocks'].append(dict(start=start,end=end,label=f'rows_{start}_{end-1}',event=False))
            (d/'circuit.svg').write_text(schematic(c),encoding='utf-8')
        dump(d/'target.json',rec)
    dump(INPUTS/'manifest.json',dict(targets=[r['id'] for r in records],circuits=9,target_count=len(records),scope='all 7 nested circuits x 3 LC conditions plus Case1 and Case2'))
    assert len(records)==CFG['target_count']
    print(f'Prepared {len(records)} complete target conditions; no optimizer invoked',flush=True)

if __name__=='__main__':main()

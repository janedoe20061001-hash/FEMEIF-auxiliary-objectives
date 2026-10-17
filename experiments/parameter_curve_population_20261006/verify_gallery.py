"""Check curve-to-parameter correspondence and schematic impedance independently."""
from pathlib import Path
from html.parser import HTMLParser
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
import gallery_site
from run_population import load_record, BAND

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/parameter_curve_population_20261006'


class Page(HTMLParser):
    def __init__(self): super().__init__(); self.capture=False; self.data=''; self.ids=[]; self.assets=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a: self.ids.append(a['id'])
        if tag=='script' and a.get('id')=='study-data': self.capture=True
        if tag in ('a','link','script','img'):
            ref=a.get('src') or a.get('href')
            if ref: self.assets.append(ref)
    def handle_endtag(self,tag):
        if tag=='script': self.capture=False
    def handle_data(self,data):
        if self.capture: self.data+=data


def nodal(edges,record,frequencies):
    params=dict(zip(record['names'],record['parameters']))
    nodes=sorted({e[k] for e in edges for k in ('start','end')}-{'A','B'})
    result=[]
    for frequency in frequencies:
        omega=2*np.pi*frequency
        ys={e['name']:1/params[e['name']] if e['name'].startswith('R') else 1/(1j*omega*params[e['name']]) if e['name'].startswith('L') else 1j*omega*params[e['name']] for e in edges}
        mat=np.zeros((len(nodes),len(nodes)),complex); rhs=np.zeros(len(nodes),complex)
        for e in edges:
            a,b=e['start'],e['end']; y=ys[e['name']]
            for start,end in ((a,b),(b,a)):
                if start in nodes:
                    i=nodes.index(start); mat[i,i]+=y
                    if end in nodes: mat[i,nodes.index(end)]-=y
                    elif end=='A': rhs[i]+=y
        values=np.linalg.solve(mat,rhs) if nodes else []
        voltages={'A':1,'B':0,**dict(zip(nodes,values))}; current=0j
        for e in edges:
            if e['start']=='A': current+=ys[e['name']]*(1-voltages[e['end']])
            elif e['end']=='A': current+=ys[e['name']]*(1-voltages[e['start']])
        result.append(1/current)
    return np.array(result)


page=Page(); page.feed((OUT/'index.html').read_text(encoding='utf-8'))
assert len(page.ids)==len(set(page.ids)),'Duplicate HTML IDs'
for asset in page.assets:
    if not asset.startswith(('#','http://','https://')): assert (OUT/asset).is_file(),asset
data=json.loads(page.data)
refs=json.loads((OUT/'data/reference_circuits.json').read_text(encoding='utf-8'))
by_id={r['id']:r for r in refs}
netlists=json.loads((OUT/'data/schematic_netlists.json').read_text(encoding='utf-8'))
net_by_family={n['family']:n for n in netlists}
assert len(data['families'])==len(netlists)==16
freq=np.geomspace(*BAND,23); checked=0; exports=0; max_error=0.
for family in data['families']:
    code=family['code']; record=family['default']
    assert record==by_id[record['id']] and record['split']=='A'
    assert record['damping_bin']==record['spacing_bin']==1 and record['id'].endswith('_00')
    assert len(family['overview'])==9 and {(r['damping_bin'],r['spacing_bin']) for r in family['overview']}=={(i,j) for i in range(3) for j in range(3)}
    for r in family['overview']: assert r==by_id[r['id']] and r['id'].endswith('_00')
    for p in family['plots']:
        assert p['parameter'] in record['names'] and (OUT/p['src']).is_file()
    svg=ET.fromstring(family['svg'])
    drawn=[dict(name=g.attrib['data-element'],start=g.attrib['data-from'],end=g.attrib['data-to']) for g in svg.iter() if 'data-element' in g.attrib]
    assert drawn==net_by_family[code]['elements']
    assert set(e['name'] for e in drawn)==set(record['names']) and len(drawn)==len(record['names'])
    assert family['svg']==(OUT/'schematics'/f'{code}.svg').read_text(encoding='utf-8')
    for r in family['overview']:
        independent=nodal(drawn,r,freq); expected=load_record(r).response(freq)[0]
        error=float(np.max(abs(independent-expected)/abs(expected))); max_error=max(max_error,error)
        np.testing.assert_allclose(independent,expected,rtol=1e-7,atol=1e-8); checked+=len(freq)
    with (OUT/'data'/f'{code}_parameters.csv').open(encoding='utf-8-sig',newline='') as file:
        rows=list(csv.reader(file)); assert len(rows)==145
        for row in rows[1:]:
            r=by_id[row[0]]; assert r['family']==code and r['split']==row[1]
            assert list(map(float,row[4:]))==r['parameters']; exports+=1
for i,e in enumerate(data['examples']):
    r=e['reference']; assert r==by_id[r['id']] and e['direction'] in r['names']
    assert (OUT/'figures'/f'counterexample_{i+1:02d}.png').is_file()
    independent=nodal(net_by_family[r['family']]['elements'],r,freq); expected=load_record(r).response(freq)[0]
    error=float(np.max(abs(independent-expected)/abs(expected))); max_error=max(max_error,error)
    np.testing.assert_allclose(independent,expected,rtol=1e-7,atol=1e-8); checked+=len(freq)
protocol=json.loads((OUT/'data/GALLERY_PROTOCOL.json').read_text(encoding='utf-8'))
for file,sha in protocol['sources'].items(): assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==sha,file
for file,sha in protocol['inputs'].items(): assert hashlib.sha256((OUT/file).read_bytes()).hexdigest()==sha,file
result=dict(passed=True,schematic_templates=16,overview_baselines=144,counterexample_baselines=12,parameter_exports=exports,
            independent_nodal_frequency_checks=checked,max_relative_impedance_discrepancy=max_error,
            curve_baseline_parameters_matched=True,source_and_input_hashes_verified=True)
(OUT/'GALLERY_VALIDATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
gallery_site.refresh_manifest(OUT)
print(json.dumps(result,ensure_ascii=False))

"""Readable gallery and electrical schematics, generated only from saved study data."""
from pathlib import Path
import csv
import hashlib
import html
import json
import math
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))


class Schematic:
    """Draw a two-terminal circuit with standard R/L/C symbols; preserve its netlist."""
    def __init__(self): self.parts=[]; self.edges=[]; self.serial=0

    def wire(self,x1,y1,x2,y2):
        self.parts.append(f'<path class="wire" d="M{x1:g},{y1:g} L{x2:g},{y2:g}"/>')

    def dot(self,x,y): self.parts.append(f'<circle class="junction" cx="{x:g}" cy="{y:g}" r="3"/>')

    def component(self,name,x1,y1,x2,y2,a,b,label_side=-1):
        dx,dy=x2-x1,y2-y1; length=math.hypot(dx,dy); angle=math.degrees(math.atan2(dy,dx))
        cx,cy=(x1+x2)/2,(y1+y2)/2; half=length/2
        self.edges.append(dict(name=name,start=a,end=b))
        if name.startswith('R'): symbol='<rect class="symbol" x="-23" y="-8" width="46" height="16" rx="1"/>'; cut=23
        elif name.startswith('C'): symbol='<path class="symbol" d="M-5,-17 V17 M5,-17 V17"/>'; cut=5
        else:
            symbol='<path class="symbol" d="M-24,0 C-24,-18 -12,-18 -12,0 C-12,-18 0,-18 0,0 C0,-18 12,-18 12,0 C12,-18 24,-18 24,0"/>'; cut=24
        self.parts.append(f'<g class="component" data-element="{name}" data-from="{a}" data-to="{b}"><g transform="translate({cx:g} {cy:g}) rotate({angle:g})"><path class="wire" d="M{-half:g},0 H{-cut:g} M{cut:g},0 H{half:g}"/>{symbol}</g>')
        # Keep labels upright, including bridge elements on diagonal wires.
        nx,ny=-dy/length,dx/length; offset=27*label_side
        self.parts.append(f'<text class="element-label" x="{cx+nx*offset:g}" y="{cy+ny*offset:g}" text-anchor="middle" dominant-baseline="central">{name}</text></g>')

    def size(self,tree):
        if isinstance(tree,str): return 116,76
        sizes=[self.size(c) for c in tree[1:]]
        if tree[0]=='S': return sum(w for w,h in sizes)+24*(len(sizes)-1),max(h for w,h in sizes)
        return max(w for w,h in sizes)+48,sum(h for w,h in sizes)+20*(len(sizes)-1)

    def draw(self,tree,x,y,a,b):
        w,h=self.size(tree)
        if isinstance(tree,str): self.component(tree,x,y,x+w,y,a,b); return
        if tree[0]=='S':
            xx=x; start=a
            for i,c in enumerate(tree[1:]):
                cw,ch=self.size(c)
                if i==len(tree)-2: end=b
                else: self.serial+=1; end=f'n{self.serial}'
                self.draw(c,xx,y,start,end); xx+=cw
                if i<len(tree)-2: self.wire(xx,y,xx+24,y); xx+=24
                start=end
        else:
            top=y-h/2; ys=[]
            for c in tree[1:]:
                cw,ch=self.size(c); yy=top+ch/2; ys.append(yy)
                self.wire(x,yy,x+24,yy); self.draw(c,x+24,yy,a,b)
                self.wire(x+24+cw,yy,x+w,yy); top+=ch+20
            self.wire(x,ys[0],x,ys[-1]); self.wire(x+w,ys[0],x+w,ys[-1])
            for yy in ys: self.dot(x,yy); self.dot(x+w,yy)

    def render(self,tree,title):
        if tree[0]=='B':
            w,h=620,386; pts={'A':(310,36),'B':(310,338),'u':(110,187),'v':(510,187)}
            for name,a,b,side in [('R1','A','u',1),('L1','u','B',1),('C1','A','v',-1),('R2','v','B',-1),('R3','u','v',-1)]:
                self.component(name,*pts[a],*pts[b],a,b,side)
            for name in ('u','v'):
                xx,yy=pts[name]; self.dot(xx,yy)
                self.parts.append(f'<text class="node-label" x="{xx}" y="{yy+24}" text-anchor="middle">{name}</text>')
            terminals=[(310,36,'A',-19),(310,338,'B',26)]
        else:
            cw,ch=self.size(tree); w,h=cw+96,ch+64; yy=h/2
            self.draw(tree,48,yy,'A','B'); self.wire(24,yy,48,yy); self.wire(w-48,yy,w-24,yy)
            terminals=[(24,yy,'A',-24),(w-24,yy,'B',-24)]
        for x,y,label,offset in terminals:
            self.parts.append(f'<circle class="terminal" cx="{x:g}" cy="{y:g}" r="5"/><text class="node-label" x="{x:g}" y="{y+offset:g}" text-anchor="middle">{label}</text>')
        style='.wire,.symbol{fill:none;stroke:#25394e;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}.symbol{fill:white}.junction{fill:#25394e}.terminal{fill:white;stroke:#25394e;stroke-width:2}.element-label{font:17px Arial,sans-serif;fill:#25394e}.node-label{font:15px Arial,sans-serif;fill:#748497}.component.active .symbol{stroke:#1769ab;stroke-width:3}.component.active .element-label{fill:#1769ab;font-weight:bold}'
        return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(title)}电路结构"><title>{html.escape(title)}；输入阻抗在端口A和B之间测量</title><style>{style}</style>'+''.join(self.parts)+'</svg>'


def render(out,families,report=None):
    out=Path(out); (out/'assets').mkdir(exist_ok=True); (out/'schematics').mkdir(exist_ok=True)
    refs=json.loads((out/'data/reference_circuits.json').read_text(encoding='utf-8'))
    examples=json.loads((out/'data/counterexamples.json').read_text(encoding='utf-8'))
    inventory=[]; netlists=[]
    for code,title,tree in families:
        rr=[r for r in refs if r['family']==code]
        overview=[r for r in rr if r['split']=='A' and r['id'].endswith('_00')]
        default=next((r for r in overview if r['damping_bin']==1 and r['spacing_bin']==1),overview[0])
        drawings=Schematic(); svg=drawings.render(tree,title)
        (out/'schematics'/f'{code}.svg').write_text(svg,encoding='utf-8')
        netlists.append(dict(family=code,terminals=['A','B'],elements=drawings.edges))
        plots=[dict(parameter=p.stem.split('_parameter_')[1],src=f'figures/{p.name}') for p in sorted((out/'figures').glob(f'{code}_parameter_*.png'))]
        plots.sort(key=lambda p:default['names'].index(p['parameter']))
        inventory.append(dict(code=code,title=title,tree=tree,svg=svg,default=default,overview=overview,plots=plots,count=len(rr)))
        headers=['circuit_id','batch','construction_bin','spacing_bin']+[n+('_ohm' if n.startswith('R') else '_H' if n.startswith('L') else '_F') for n in default['names']]
        with (out/'data'/f'{code}_parameters.csv').open('w',encoding='utf-8-sig',newline='') as file:
            writer=csv.writer(file); writer.writerow(headers)
            for r in rr: writer.writerow([r['id'],r['split'],r['damping_bin']+1,r['spacing_bin']+1,*r['parameters']])
    (out/'data/schematic_netlists.json').write_text(json.dumps(netlists,ensure_ascii=False,indent=2),encoding='utf-8')
    def selected(filename):
        path=out/'data'/filename
        return [r for r in read_csv(path) if float(r['level_pct'])==10] if path.is_file() else []
    payload=dict(families=inventory,examples=examples,
                 rules=selected('rule_summary.csv'),positions=selected('parameter_rule_summary.csv'),
                 strata=[r for r in selected('rule_strata_summary.csv') if r['rule']=='R_amp_up'],responses=selected('response_distribution.csv'))
    # Embed data so the gallery also works when index.html is opened directly from disk.
    data=json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    first=inventory[0]
    options=''.join(f'<option value="{f["code"]}">{html.escape(f["title"])}</option>' for f in inventory)
    family_buttons=''.join(f'<button class="family-button" data-family="{f["code"]}"><span>{i+1:02d}</span>{html.escape(f["title"])}</button>' for i,f in enumerate(inventory))
    page=(HERE/'gallery_template.html').read_text(encoding='utf-8')
    page=page.replace('__FAMILY_OPTIONS__',options).replace('__FAMILY_BUTTONS__',family_buttons).replace('__INITIAL_SVG__',first['svg']).replace('__DATA__',data)
    (out/'index.html').write_text(page,encoding='utf-8')
    for filename in ('gallery.css','gallery.js'): shutil.copyfile(HERE/filename,out/'assets'/filename)
    inputs=['data/reference_circuits.json','data/counterexamples.json','data/rule_summary.csv','data/parameter_rule_summary.csv','data/rule_strata_summary.csv','data/response_distribution.csv']
    inputs=[f for f in inputs if (out/f).is_file()]
    for file in ('REPORT.md','CONDITIONAL_RULES.md'):
        if (out/file).is_file(): inputs.append(file)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    protocol=dict(no_new_simulation=True,schematic_templates=len(inventory),curve_parameter_selection='Same A/d2/s2/draw00 baseline as original parameter_plot; overview has nine A/draw00 baselines; counterexamples retain their exact saved baseline.',
                  sources={str((HERE/f).relative_to(ROOT)):sha(HERE/f) for f in ('gallery_site.py','gallery_template.html','gallery.css','gallery.js')},
                  inputs={f:sha(out/f) for f in inputs})
    (out/'data/GALLERY_PROTOCOL.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')


def refresh_manifest(out):
    out=Path(out); info=json.loads((out/'INTEGRITY.json').read_text(encoding='utf-8'))
    info['schematic_templates']=len(list((out/'schematics').glob('*.svg')))
    info['artifacts_sha256']={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='INTEGRITY.json'}
    (out/'INTEGRITY.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')

"""Ordinary RLC schematics: series left-to-right, parallel on separate branches."""
from collections import defaultdict
from io import StringIO
from pathlib import Path
import numpy as np

LEAF_W, LEAF_H, SERIES_GAP, BRANCH_GAP, RAIL_MARGIN = 150., 86., 24., 24., 32.


def dimensions(node):
    if isinstance(node, str):
        return LEAF_W, LEAF_H
    sizes = [dimensions(child) for child in node[1:]]
    if node[0] == "S":
        return sum(w for w,h in sizes) + SERIES_GAP*(len(sizes)-1), max(h for w,h in sizes)
    return max(w for w,h in sizes)+2*RAIL_MARGIN, sum(h for w,h in sizes)+BRANCH_GAP*(len(sizes)-1)


def engineering(name, value):
    unit = {"R":"Ω", "L":"H", "C":"F"}[name[0]]
    for scale,prefix in [(1e6,"M"),(1e3,"k"),(1.,""),(1e-3,"m"),(1e-6,"µ"),(1e-9,"n"),(1e-12,"p")]:
        if value >= scale or scale == 1e-12:
            return f"{value/scale:.4g} {prefix}{unit}"


class Layout:
    def __init__(self, circuit):
        self.circuit = circuit
        self.wires, self.components, self.junctions = [], {}, set()
        self.width, self.height = dimensions(circuit.tree)
        self.place(circuit.tree,0.,0.)
        self.terminals = {"A":(-30.,self.height/2), "B":(self.width+30.,self.height/2)}
        self.wire(self.terminals["A"],(0.,self.height/2))
        self.wire((self.width,self.height/2),self.terminals["B"])

    def wire(self,a,b):
        if a != b:
            if a[0] != b[0] and a[1] != b[1]:
                raise ValueError("Schematic routing must be orthogonal")
            self.wires.append((a,b))

    def place(self,node,x,y):
        w,h = dimensions(node)
        if isinstance(node,str):
            cx,cy = x+w/2,y+h/2
            left,right = (cx-28,cy),(cx+28,cy)
            self.wire((x,cy),left)
            self.wire(right,(x+w,cy))
            self.components[node] = (left,right)
            return
        if node[0] == "S":
            cursor = x
            for i,child in enumerate(node[1:]):
                cw,ch = dimensions(child)
                self.place(child,cursor,y+(h-ch)/2)
                cursor += cw
                if i < len(node)-2:
                    self.wire((cursor,y+h/2),(cursor+SERIES_GAP,y+h/2))
                    cursor += SERIES_GAP
            return
        rail_l,rail_r = x+RAIL_MARGIN/2,x+w-RAIL_MARGIN/2
        mid = y+h/2
        self.wire((x,mid),(rail_l,mid))
        self.wire((rail_r,mid),(x+w,mid))
        cursor,levels = y,[]
        for child in node[1:]:
            cw,ch = dimensions(child)
            xx = x+(w-cw)/2
            yy = cursor+ch/2
            self.place(child,xx,cursor)
            self.wire((rail_l,yy),(xx,yy))
            self.wire((xx+cw,yy),(rail_r,yy))
            levels.append(yy)
            cursor += ch+BRANCH_GAP
        self.wire((rail_l,min(levels)),(rail_l,max(levels)))
        self.wire((rail_r,min(levels)),(rail_r,max(levels)))
        # Only degree >=3 points receive a junction dot; corners do not.
        for rail in (rail_l,rail_r):
            for yy in set(levels+[mid]):
                degree = int(yy>min(levels))+int(yy<max(levels))+int(yy in levels)+int(yy==mid)
                if degree >= 3:
                    self.junctions.add((rail,yy))

    def connectivity(self):
        """Contract drawn wires and return named component/port incidences."""
        points = set(self.terminals.values())
        for a,b in self.wires:
            points.update((a,b))
        for a,b in self.components.values():
            points.update((a,b))
        parent = {p:p for p in points}
        def find(p):
            while parent[p] != p:
                parent[p] = parent[parent[p]]
                p = parent[p]
            return p
        for a,b in self.wires:
            on = [p for p in points if (a[0]==b[0]==p[0] and min(a[1],b[1])<=p[1]<=max(a[1],b[1]))
                  or (a[1]==b[1]==p[1] and min(a[0],b[0])<=p[0]<=max(a[0],b[0]))]
            for p in on:
                parent[find(p)] = find(a)
        groups = defaultdict(set)
        for name,(a,b) in self.components.items():
            groups[find(a)].add(name+":a")
            groups[find(b)].add(name+":b")
        for name,p in self.terminals.items():
            groups[find(p)].add(name)
        return {frozenset(items) for items in groups.values()}

    def validate(self):
        expected = defaultdict(set)
        for name,a,b in self.circuit.netlist():
            expected[a].add(name+":a")
            expected[b].add(name+":b")
        expected[1].add("A")
        expected[0].add("B")
        if self.connectivity() != {frozenset(items) for items in expected.values()}:
            raise AssertionError("Drawn schematic differs from circuit netlist")


def figure(circuit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    layout = Layout(circuit)
    layout.validate()
    width,height = layout.width+130,layout.height+110
    fig = plt.figure(figsize=(width/100,height/100),facecolor="white")
    ax = fig.add_axes([0,0,1,1])
    ax.set(xlim=(-65,layout.width+65),ylim=(layout.height+55,-55),aspect="equal")
    ax.axis("off")
    style = dict(color="#18212b",linewidth=1.9,solid_capstyle="round")
    for a,b in layout.wires:
        ax.plot([a[0],b[0]],[a[1],b[1]],**style)
    for name,(a,b) in layout.components.items():
        x,y = (a[0]+b[0])/2,a[1]
        if name[0] == "R":
            xx = np.linspace(a[0],b[0],9)
            yy = np.array([0,-8,8,-8,8,-8,8,-8,0])+y
            ax.plot(xx,yy,**style)
        elif name[0] == "L":
            # Four smooth arches form the conventional inductor symbol.
            for k in range(4):
                theta = np.linspace(np.pi,0,40)
                ax.plot(a[0]+7+14*k+7*np.cos(theta),y-12*np.sin(theta),**style)
        else:
            ax.plot([a[0],x-7],[y,y],**style)
            ax.plot([x+7,b[0]],[y,y],**style)
            for xx in (x-7,x+7):
                ax.plot([xx,xx],[y-15,y+15],**style)
        ax.text(x,y-24,name,ha="center",va="center",fontsize=11.5,color="#18212b",fontfamily="DejaVu Sans")
        ax.text(x,y+29,engineering(name,circuit.values[name]),ha="center",va="center",fontsize=10,color="#475364",fontfamily="DejaVu Sans")
    for x,y in layout.junctions:
        ax.plot(x,y,"o",color="#18212b",markersize=4)
    for name,(x,y) in layout.terminals.items():
        ax.plot(x,y,"o",markerfacecolor="white",markeredgecolor="#18212b",markersize=6,markeredgewidth=1.8)
        ax.text(x,y-21,name,ha="center",va="center",fontsize=12,fontweight="bold",fontfamily="DejaVu Sans")
    titles = {"control_rlc":"Parallel RLC", "user_nested":"Nested RLC network", "rlc_ladder":"Two-stage RLC ladder",
              "loaded_tank":"Loaded resonant network", "dual_nested":"Two nested parallel branches",
              "three_level":"Three-stage RLC ladder", "mixed_resonators":"Mixed resonant branches"}
    ax.text(layout.width/2,-33,titles.get(circuit.ident,circuit.ident),ha="center",va="center",fontsize=12.5,fontfamily="DejaVu Sans",color="#18212b")
    return fig


def schematic(circuit):
    import matplotlib.pyplot as plt
    fig = figure(circuit)
    stream = StringIO()
    fig.savefig(stream,format="svg")
    plt.close(fig)
    return stream.getvalue()


def save_schematic(circuit,stem):
    import matplotlib.pyplot as plt
    stem = Path(stem)
    fig = figure(circuit)
    fig.savefig(stem.with_suffix(".svg"),format="svg")
    fig.savefig(stem.with_suffix(".png"),dpi=180)
    plt.close(fig)

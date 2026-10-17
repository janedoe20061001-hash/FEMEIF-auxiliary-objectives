"""Explicit series/parallel trees; no ambiguous text-expression parser."""
from dataclasses import dataclass, replace
import numpy as np
from numpy.polynomial import polynomial as poly

WREF = 2 * np.pi * 1e4


def S(*parts):
    return ["S", *parts]


def P(*parts):
    return ["P", *parts]


def leaves(tree):
    if isinstance(tree, str):
        if not tree or tree[0] not in "RLC":
            raise ValueError(f"Unsupported component: {tree}")
        return [tree]
    if not isinstance(tree, list) or len(tree) < 3 or tree[0] not in ("S", "P"):
        raise ValueError(f"Invalid topology: {tree}")
    return [name for child in tree[1:] for name in leaves(child)]


def expression(tree):
    return tree if isinstance(tree, str) else "(" + (" + " if tree[0] == "S" else " // ").join(map(expression, tree[1:])) + ")"


def normalize(num, den):
    num, den = poly.polytrim(num), poly.polytrim(den)
    while len(num) > 1 and len(den) > 1 and num[0] == den[0] == 0:
        num, den = num[1:], den[1:]
    scale = max(np.max(abs(num)), np.max(abs(den)))
    if not np.isfinite(scale) or scale <= 0:
        raise FloatingPointError("Invalid rational coefficients")
    return num / scale, den / scale


@dataclass
class Circuit:
    ident: str
    title: str
    tree: list
    values: dict
    lc: tuple
    damping: str
    background: str

    def __post_init__(self):
        self.names = leaves(self.tree)
        if len(set(self.names)) != len(self.names) or set(self.names) != set(self.values):
            raise ValueError("Each physical component must have a unique name and value")
        self.p = np.array([self.values[n] for n in self.names], float)
        if not np.isfinite(self.p).all() or np.any(self.p <= 0):
            raise ValueError("Parameters must be finite and positive")
        if self.lc[0][0] != "L" or self.lc[1][0] != "C":
            raise ValueError("Invalid LC intervention")
        if any(n not in self.names for n in (*self.lc, self.damping, self.background)):
            raise ValueError("Unknown intervention component")

    def response(self, f, parameters=None):
        """Broadcasts parameter batches; derivative is dZ/d(ln f)."""
        p = np.asarray(self.p if parameters is None else parameters, float)
        single = p.ndim == 1
        p = np.atleast_2d(p)
        f = np.asarray(f, float)
        if p.shape[1] != len(self.names) or not np.isfinite(p).all() or np.any(p <= 0):
            raise ValueError("Invalid physical parameters")
        if f.ndim not in (1, 2) or np.any(f <= 0) or not np.isfinite(f).all():
            raise ValueError("Invalid frequency array")
        w = 2 * np.pi * (f[None, :] if f.ndim == 1 else f)
        values = {n: p[:, j, None] for j, n in enumerate(self.names)}

        def evaluate(node):
            if isinstance(node, str):
                v = values[node]
                if node[0] == "R":
                    z = np.broadcast_to(v, np.broadcast_shapes(v.shape, w.shape)).astype(complex)
                    return z, np.zeros_like(z)
                z = 1j * w * v if node[0] == "L" else 1 / (1j * w * v)
                return z, z if node[0] == "L" else -z
            parts = [evaluate(c) for c in node[1:]]
            if node[0] == "S":
                return sum(z for z, _ in parts), sum(d for _, d in parts)
            y = sum(1 / z for z, _ in parts)
            dy = -sum(d / z**2 for z, d in parts)
            return 1 / y, -dy / y**2

        z, dz = evaluate(self.tree)
        if not np.isfinite(z).all() or not np.isfinite(dz).all() or np.any(z.real <= 0):
            raise FloatingPointError(f"Invalid passive response: {self.ident}")
        return (z[0], dz[0]) if single and f.ndim == 1 else (z, dz)

    def netlist(self):
        """Terminal 0 is ground, terminal 1 is driven; internal nodes >=2."""
        edges, counter = [], [2]

        def visit(node, a, b):
            if isinstance(node, str):
                edges.append((node, a, b))
            elif node[0] == "P":
                for c in node[1:]:
                    visit(c, a, b)
            else:
                here = a
                for i, c in enumerate(node[1:]):
                    there = b if i == len(node) - 2 else counter[0]
                    if there != b:
                        counter[0] += 1
                    visit(c, here, there)
                    here = there
        visit(self.tree, 1, 0)
        return edges

    def nodal_response(self, f, parameters=None):
        """Independent Kirchhoff nodal solve with port voltage fixed at 1 V."""
        p = self.p if parameters is None else np.asarray(parameters)
        values = dict(zip(self.names, p))
        edges = self.netlist()
        internal = sorted({k for _, a, b in edges for k in (a, b) if k >= 2})
        result = []
        for freq in f:
            y = {}
            for n, a, b in edges:
                v = values[n]
                z = v if n[0] == "R" else (2j * np.pi * freq * v if n[0] == "L" else 1 / (2j * np.pi * freq * v))
                y[n] = 1 / z
            mat = np.zeros((len(internal), len(internal)), complex)
            rhs = np.zeros(len(internal), complex)
            loc = {node: i for i, node in enumerate(internal)}
            for n, a, b in edges:
                for node, other in ((a, b), (b, a)):
                    if node in loc:
                        i = loc[node]
                        mat[i, i] += y[n]
                        if other in loc:
                            mat[i, loc[other]] -= y[n]
                        elif other == 1:
                            rhs[i] += y[n]
            volt = {0: 0, 1: 1, **dict(zip(internal, np.linalg.solve(mat, rhs)))}
            current = sum(y[n] * (1 - volt[b if a == 1 else a]) for n, a, b in edges if a == 1 or b == 1)
            result.append(1 / current)
        return np.array(result)

    def rational(self, parameters=None):
        values = dict(zip(self.names, self.p if parameters is None else parameters))

        def build(node):
            if isinstance(node, str):
                v = values[node]
                if node[0] == "R":
                    return np.array([v]), np.array([1.])
                if node[0] == "L":
                    return np.array([0., v * WREF]), np.array([1.])
                return np.array([1.]), np.array([0., v * WREF])
            parts = [build(c) for c in node[1:]]
            if node[0] == "P":
                parts = [(d, n) for n, d in parts]
            n, d = parts[0]
            for nn, dd in parts[1:]:
                n, d = normalize(poly.polyadd(poly.polymul(n, dd), poly.polymul(nn, d)), poly.polymul(d, dd))
            return normalize(d, n) if node[0] == "P" else normalize(n, d)
        return build(self.tree)

    def modes(self, parameters=None):
        """Pole/zero anchors, including cancellations; these are NOT port events."""
        modes = []
        for coef in self.rational(parameters):
            roots = np.roots(coef[::-1])
            for root in roots:
                if abs(root) < 1e-14:
                    continue
                residual = abs(poly.polyval(root, coef)) / max(poly.polyval(abs(root), abs(coef)), 1e-300)
                if not np.isfinite(residual) or residual > 1e-7:
                    raise FloatingPointError("Unreliable polynomial root; evaluation blocked")
                if root.imag < -1e-9 * abs(root):
                    continue
                oscillating = abs(root.imag) > 1e-8 * abs(root)
                fc = (abs(root.imag) if oscillating else abs(root.real)) * WREF / (2 * np.pi)
                width = min(.5, max(6 * abs(root.real / root.imag) / np.log(10), 1e-12)) if oscillating else .3
                modes.append((float(fc), float(width)))
        return modes

    def record(self):
        return dict(id=self.ident, title=self.title, tree=self.tree, expression=expression(self.tree), names=self.names,
                    parameters=self.p.tolist(), units=[{"R": "ohm", "L": "H", "C": "F"}[n[0]] for n in self.names],
                    netlist=[dict(component=n, a=a, b=b) for n, a, b in self.netlist()])

    def changed(self, kind, factor):
        values = self.values.copy()
        if kind == "lc_reciprocal":
            values[self.lc[0]] *= factor
            values[self.lc[1]] /= factor
        elif kind == "damping":
            values[self.damping] *= factor
        elif kind == "background":
            values[self.background] *= factor
        else:
            raise ValueError(kind)
        return replace(self, values=values)


def library():
    # Three nominal LC scales near 10, 22 and 40 kHz, impedance scale 100 ohm.
    w = 2 * np.pi * np.array([1e4, 2.2e4, 4e4])
    common = {"Rs": 5., "R1": 2., "R2": 3., "R3": 7., "R4": 200.}
    for i in range(3):
        common.update({f"L{i+1}": 100 / w[i], f"C{i+1}": 1 / (100 * w[i])})
    definitions = [
        ("control_rlc", "单并联RLC对照", P("R1", "L1", "C1"), {"R1": 3000.}, ("L1", "C1"), "R1", "R1"),
        ("user_nested", "用户提出的嵌套电路", S("R1", P("L1", S("R2", P(S("R3", "C1"), "L2")))), {}, ("L2", "C1"), "R3", "R1"),
        ("rlc_ladder", "两级RLC梯形网络", S("Rs", P("C1", S("R1", "L1", P("C2", S("R2", "L2"))))), {}, ("L2", "C2"), "R2", "Rs"),
        ("loaded_tank", "谐振单元嵌入负载支路", S("Rs", P(S("R1", "L1"), S("C1", P("R2", "L2", "C2")))), {"R2": 1500.}, ("L2", "C2"), "R2", "Rs"),
        ("dual_nested", "两个不同的嵌套支路并联", S("Rs", P(S("R1", P("L1", S("R2", "C1"))), S("R3", P("C2", S("R4", "L2"))))), {"R4": 4.}, ("L2", "C2"), "R3", "Rs"),
        ("three_level", "三级交替串并联网络", S("Rs", P("C1", S("R1", "L1", P("C2", S("R2", "L2", P("R3", "L3", "C3")))))), {"R3": 2000.}, ("L3", "C3"), "R3", "Rs"),
        ("mixed_resonators", "串联和并联谐振结构混合嵌套", S("Rs", P(S("R1", "L1", "C1"), S("L2", P("R2", "C2", S("R3", "L3", "C3"))))), {"R2": 1500.}, ("L3", "C3"), "R3", "Rs"),
    ]
    return {ident: Circuit(ident, title, tree, {n: overrides.get(n, common[n]) for n in leaves(tree)}, lc, damping, background)
            for ident, title, tree, overrides, lc, damping, background in definitions}

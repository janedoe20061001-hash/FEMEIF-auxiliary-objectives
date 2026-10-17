"""Compiled RX-only evaluator; retains candidate anchors and convergence guard."""
from core import *
from legacy_features import Features as Legacy
from reference_features import rule,mode_points
import ctypes

class Features:
    def __init__(self,data):
        self.data=data;self.c=data['circuit'];self.legacy=Legacy(data)
        assert data['kind']!='measured'
        self.windows=self.legacy.windows;self.queries=self.calls=self.refined=self.max_order=0
        self.target_modes=self.modes(self.c.p)
        self.lo=min(w['x'][0] for w in self.windows);self.hi=max(w['x'][-1] for w in self.windows)
        self.global_scale=max(w['scale'] for w in self.windows)
        ops=[];args=[]
        def visit(node):
            if isinstance(node,str):ops.append({'R':1,'L':2,'C':3}[node[0]]);args.append(self.c.names.index(node))
            else:
                for child in node[1:]:visit(child)
                ops.append(10 if node[0]=='S' else 11);args.append(len(node)-1)
        visit(self.c.tree)
        self.ops=np.array(ops,dtype=np.int32);self.args=np.array(args,dtype=np.int32)
        self.nt=-len(self.ops) if data['kind']=='case1' else len(self.ops)
        self.lib=None;self._buffers={}
        self._base_edges={}
        self._scales=None
    def native(self):
        if self.lib is None:
            self.lib=ctypes.CDLL(str(HERE/'area_native.so'))
            ptr=ctypes.c_void_p;integer=ctypes.c_int
            self.lib.integrate.argtypes=[integer,ptr,ptr,ptr,ptr,integer,ptr,ptr,ptr,ptr,ptr,integer,ptr,ptr,ptr,ptr,ptr,integer]
            self.lib.integrate.restype=integer
            self.lib.model.argtypes=[integer,ptr,ptr,ptr,integer,ptr,ptr,ptr];self.lib.model.restype=integer
        return self.lib
    def modes(self,p):return case1_modes(p) if self.data['kind']=='case1' else self.c.modes(p)
    def panel_edges(self,w,modes,refinement):
        lo,hi=w['x'][[0,-1]];span=hi-lo
        key=(float(lo),float(hi))
        if key not in self._base_edges:
            self._base_edges[key]=np.r_[np.linspace(0,1,9),(mode_points(self.target_modes,lo,hi)-lo)/span]
        edges=np.unique(np.r_[self._base_edges[key],(mode_points(modes,lo,hi)-lo)/span])
        for _ in range(refinement):edges=np.sort(np.r_[edges,(edges[1:]+edges[:-1])/2])
        return edges
    def response(self,f,p):
        f=np.ascontiguousarray(f,dtype=float);p=np.ascontiguousarray(p,dtype=float)
        z=np.empty(f.shape,complex);d=np.empty_like(z)
        rc=self.native().model(self.nt,self.ops.ctypes.data,self.args.ctypes.data,p.ctypes.data,len(f),f.ctypes.data,z.ctypes.data,d.ctypes.data)
        if rc:raise FloatingPointError(f'Native model code {rc}')
        return z,d
    def grid(self,modes,refinement,global_only=False):
        windows=self.windows if not global_only else [dict(x=np.array([self.lo,self.hi]),scale=self.global_scale)]
        ee=[self.panel_edges(w,modes,refinement) for w in windows]
        edges=np.ascontiguousarray(np.concatenate(ee));offset=np.array(np.r_[0,np.cumsum([len(e) for e in ee])],dtype=np.int32)
        lo=np.array([w['x'][0] for w in windows]);span=np.array([w['x'][-1]-w['x'][0] for w in windows]);scale=np.array([w['scale'] for w in windows])
        return lo,span,scale,offset,edges
    def raw(self,p,modes,n,refinement,global_only=False,grid=None,flags=7):
        lo,span,scale,offset,edges=self.grid(modes,refinement,global_only) if grid is None else grid
        u,weights,anti=rule(n);u=np.ascontiguousarray(u);weights=np.ascontiguousarray(weights);anti=np.ascontiguousarray(anti)
        p=np.ascontiguousarray(p);ref=np.ascontiguousarray(self.c.p)
        out=np.empty((len(lo),3));queries=np.zeros(1,np.int64)
        rc=self.native().integrate(self.nt,self.ops.ctypes.data,self.args.ctypes.data,p.ctypes.data,ref.ctypes.data,len(lo),lo.ctypes.data,span.ctypes.data,scale.ctypes.data,offset.ctypes.data,edges.ctypes.data,n,u.ctypes.data,weights.ctypes.data,anti.ctypes.data,out.ctypes.data,queries.ctypes.data,flags)
        self.queries+=int(queries[0])
        if rc:raise FloatingPointError(f'Native integral code {rc}')
        return out
    def scales(self):
        if self._scales is None:
            a=self.raw(self.c.p,[],32,1,flags=2)[:,2];b=self.raw(self.c.p,[],64,2,flags=2)[:,2]
            if not np.all(abs(a-b)<=1e-9+1e-6*abs(b)):raise RuntimeError('Target arclength not converged')
            b2=b*b;ref=float(np.sqrt(np.mean(b2*b2)))
            self._scales=np.maximum(b2,max(CFG['light']['scale_tau']*ref,CFG['features']['area_scale_floor']))
        return self._scales
    def convert(self,raw,names):
        old=np.array([w['area_den'][2] for w in self.windows]);new=self.scales() if any(n in ('ARX_S_L','J0_W') for n in names) else None
        out=[]
        for name in names:
            if name=='G0':v=raw[:,1]
            elif name=='J0':v=raw[:,1]
            elif name=='J0_W':v=raw[:,1]*(old/new)**2
            elif name=='ARX_L':v=raw[:,0]/old**2
            elif name=='ARX_S_L':v=raw[:,0]/new**2
            else:raise KeyError(name)
            out.append(v)
        return np.column_stack(out)
    def fixed(self,lp,names,n=32,refinement=1):
        p=10**np.asarray(lp);modes=self.modes(p)
        flags=(1 if any(x.startswith('ARX') for x in names) else 0)|(4 if any(x in ('G0','J0','J0_W') for x in names) else 0)
        return self.convert(self.raw(p,modes,n,refinement,names==('G0',),flags=flags),names)
    def one(self,lp,names):
        cfg=CFG['quadrature'];prev=None
        p=10**np.asarray(lp);modes=self.modes(p)
        flags=(1 if any(x.startswith('ARX') for x in names) else 0)|(4 if any(x in ('G0','J0','J0_W') for x in names) else 0)
        for refine in range(cfg['max_panel_refinements']+1):
            grid=self.grid(modes,refine,names==('G0',))
            for n in cfg['orders']:
                now=self.convert(self.raw(p,modes,n,refine,names==('G0',),grid,flags),names)
                if prev is not None and np.all(abs(now-prev)<=cfg['atol']+cfg['rtol']*abs(now)):
                    self.calls+=1;self.refined+=int(n>16 or refine>0);self.max_order=max(self.max_order,n)
                    return now.mean(axis=0)
                prev=now
        raise RuntimeError('Unresolved light integral; logp='+repr(np.asarray(lp).tolist()))
    def score(self,logp,method,level=1):return np.array([self.one(lp,(method,))[0] for lp in np.atleast_2d(logp)])
    def description(self):
        s=self.scales()
        return [dict(label=w['block']['label'],frequency_hz=(10**w['x'][[0,-1]]).tolist(),scale_ohm=w['scale'],old_area_den=float(w['area_den'][2]),new_area_den=float(s[i]),matched_weight=float((w['area_den'][2]/s[i])**2)) for i,w in enumerate(self.windows)]

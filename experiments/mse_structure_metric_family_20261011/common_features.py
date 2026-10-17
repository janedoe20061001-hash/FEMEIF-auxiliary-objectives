"""Shared angular progress diagnostics for primary curves, not retrospective selection."""
from core import *
from concurrent.futures import ProcessPoolExecutor,as_completed

def one(job):
    ident,method,seed=job
    data=target(ident);c=data['circuit'];g=Geometry(data,'MAG_PHA_ANGULAR')
    d=json.loads((OUT/f'analysis/details/{ident}_{method}_{seed}_primary.json').read_text(encoding='utf-8'))
    p=np.array(d['log_parameters']);x=data['quadrature'].reference_grid(10**p,max(4,d['diagnostic']['refinement']))
    lo,hi=g.span;x=np.unique(np.r_[lo,x[(x>lo)&(x<hi)],hi])
    z,_=c.response(10**x,10**p);ref,_=c.response(10**x)
    phase=np.angle(z);truth=np.angle(ref)
    delta=np.diff(phase);dt=np.diff(truth)
    parts=g.score(p,points=1025,anchors=True,return_events=True)[0]
    rec=dict(target=ident,method=method,seed=seed,common_angular_error=float(g.aggregate(parts)),target_positive_turn_deg=float(np.rad2deg(np.maximum(dt,0).sum())),target_negative_turn_deg=float(np.rad2deg(np.maximum(-dt,0).sum())),candidate_positive_turn_deg=float(np.rad2deg(np.maximum(delta,0).sum())),candidate_negative_turn_deg=float(np.rad2deg(np.maximum(-delta,0).sum())),extra_phase_events=len(d['diagnostic']['extra_phase_events']),span_low_hz=10**lo,span_high_hz=10**hi)
    blocks=[]
    for w in g.blocks:
        xx=np.unique(np.r_[w['lo'],x[(x>w['lo'])&(x<w['hi'])],w['hi']]);a,_=c.response(10**xx,10**p);b,_=c.response(10**xx)
        pa,pb=np.angle(a),np.angle(b);error=np.trapezoid((pa-pb)**2,xx)/(xx[-1]-xx[0])
        is_gap=not any(v['lo']<=((w['lo']+w['hi'])/2)<=v['hi'] for v in g.windows)
        blocks.append(dict(target=ident,method=method,seed=seed,block=w['event'],is_connecting_gap=is_gap,active_angular_target=w['active'],low_hz=10**w['lo'],high_hz=10**w['hi'],phase_rmse_deg=float(np.rad2deg(np.sqrt(error))),target_total_turn_deg=float(np.rad2deg(abs(np.diff(pb)).sum())),candidate_total_turn_deg=float(np.rad2deg(abs(np.diff(pa)).sum()))))
    return rec,blocks

def main():
    if __name__!='__main__':
        # Windows spawn needs an importable script module for worker functions.
        import subprocess
        subprocess.run([sys.executable,str(Path(__file__).resolve())],cwd=ROOT,check=True)
        return
    rows=[];blocks=[]
    with ProcessPoolExecutor(max_workers=min(32,os.cpu_count() or 1)) as pool:
        for future in as_completed([pool.submit(one,(i,m,s)) for i in CFG['targets'] for m in CFG['methods'] for s in CFG['seeds']]):
            r,b=future.result();rows.append(r);blocks.extend(b)
    rows.sort(key=lambda r:(r['target'],r['method'],r['seed']))
    load('family_common_csv',HERE/'analyze.py').csvout(OUT/'analysis/COMMON_ANGULAR_FEATURES.csv',rows)
    load('family_blocks_csv',HERE/'analyze.py').csvout(OUT/'analysis/PHASE_BLOCKS.csv',blocks)
    dump(OUT/'analysis/COMMON_FEATURES_PROVENANCE.json',dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),curves=len(rows),purpose='same fixed angular diagnostic for every primary curve; never selects curves',grid_points=1025,candidate_and_target_anchors=True))
    print('Shared angular diagnostics ready',flush=True)

if __name__=='__main__':main()

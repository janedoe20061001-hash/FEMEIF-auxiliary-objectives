"""Reuse download/stat audit; substitute independently derived new formulas."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
src=root/'scripts/verify_curve_refinement.py';dst=root/'scripts/verify_curve_directional.py'
if dst.exists():raise FileExistsError(dst)
text=src.read_text(encoding='utf-8')
text=text.replace("run=root/'experiments/curve3d_refinement_20261003'","run=root/'experiments/curve3d_directional_20261003'")
text=text.replace('start=100','start=200').replace('range(100,200)','range(200,300)').replace('100<=seed<200','200<=seed<300')
text=text.replace("experiments/curve3d_logpolar_native_20261003/{case}/config.json","experiments/curve3d_refinement_20261003/{case}/config.json")
start=text.index("        group=directory.parent.name\n");end=text.index('        checked+=1;',start)
text=text[:start]+'''        group=directory.parent.name
        names=[] if group=='baseline' else group.split('+')
        assert stored.shape==(cfg['population'],2+len(names))
        if names:
            offset,scale=refs['logpolar']
            t=np.column_stack([u,(xy-offset)/scale]);candidate=np.stack([20*np.log10(abs(z)),np.angle(z)],axis=-1)
            candidate=(candidate-offset)/scale;p=np.concatenate([np.broadcast_to(u,candidate.shape[:-1])[...,None],candidate],axis=-1)
            for column,name in enumerate(names,2):
                family=name.rsplit('_',1)[0]
                if family=='cumulative_area':value=np.sqrt(integrate_square(prefixes(p)-prefixes(t),h))
                elif family in ['cumulative_length','symmetric_length']:
                    bp,bt=lengths(p),lengths(t);err=(bp-bt)/bt[-1]
                    forward=integrate_square(err[...,None],h)
                    if family=='cumulative_length':value=np.sqrt(forward)
                    else:
                        # Independently form remaining length, not e_N-e.
                        backward=((bp[...,-1:]-bp)-(bt[-1]-bt))/bt[-1]
                        value=np.sqrt((forward+integrate_square(backward[...,None],h))/2)
                elif family in ['axis_variation','direction_variation']:
                    delta=p[...,1:,1:]-p[...,:-1,1:];dt=t[1:,1:]-t[:-1,1:]
                    def prefix(d):return np.concatenate([np.zeros_like(d[...,:1,:]),np.cumsum(d,axis=-2)],axis=-2)
                    vp,vt=prefix(abs(delta)),prefix(abs(dt));total=vt[-1]
                    if family=='axis_variation':error=(vp-vt)/total
                    else:
                        # Jordan identity rather than max used in production.
                        disp=p[...,1:]-p[...,:1,1:];target_disp=t[:,1:]-t[:1,1:]
                        positive=(vp+disp)/2;negative=(vp-disp)/2
                        tp=(vt+target_disp)/2;tn=(vt-target_disp)/2
                        error=np.concatenate([(positive-tp)/total,(negative-tn)/total],axis=-1)
                    value=np.sqrt(integrate_square(error,h)/2)
                else:raise AssertionError(family)
                np.testing.assert_allclose(stored[:,column],value,rtol=2e-9,atol=1e-10)
                max_aux_abs=max(max_aux_abs,float(np.max(abs(stored[:,column]-value))))
'''+text[end:]
text=text.replace('len(parent_rows)==8','len(parent_rows)==10')
start=text.index('parent_names=');end=text.index('bootstrap=',start)
text=text[:start]+"parent_names={'R-03-C':'cumulative_area_logpolar','R-05-C':'cumulative_length_logpolar'}\n"+text[end:]
text=text.replace("else parent_names[r['id']]","else parent_names[r['parent']]")
text=text.replace('assert len(pvals)==40','assert len(pvals)==44').replace('np.zeros(40)','np.zeros(44)').replace('(40-rank)','(44-rank)')
text=text.replace('p_40_primary','p_44_primary').replace('all_20_comparisons','all_22_comparisons').replace('new_seed_block=[100,199]','new_seed_block=[200,299]')
dst.write_text(text,encoding='utf-8');print(dst)

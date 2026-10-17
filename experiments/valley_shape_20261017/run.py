"""Run fixed extraction and independent physical control checks; keep failures."""
from shape import *
def main():
    plan=json.loads((OUT/'PLAN.json').read_text(encoding='utf-8'));targets={t['id']:t for t in plan['targets']};manifest=json.loads((OUT/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))['files'];assert all(sha(ROOT/k)==v for k,v in manifest.items())
    refs={};results=[];start=time.time()
    for item in plan['tasks']:
        t=targets[item['target']];levels=[]
        try:
            for n in [128,256,512,1024]:
                level=sections(item['model'],t,n)
                key=(t['id'],n)
                if key not in refs:refs[key]=target_section(t,n)
                for q in level['candidates']:q['diagnostics']=compare(q,refs[key],t)
                levels.append(level)
                if len(levels)>1 and stable(levels[-2],levels[-1],t):break
            ok=stable(levels[-2],levels[-1],t)
            # Ground truth metadata is used only after observable extraction.
            control=None
            if item['cohort']=='new':
                im=dict(item);im['family']='shift' if item['family']=='baseline' else item['family']
                identity=analytic_identity(Physical(item['model']),Physical(t['model']),im,t)
                if item['family'] in ['baseline','shift','Q','combined']:
                    m=item['meta'];W=t['core'][1]-t['core'][0];expected=t['center']+m['shift_widths']*W
                    near=[q for q in levels[-1]['candidates'] if abs(q['center']-expected)<=1e-9]
                    control=dict(expected_center=expected,matching_center_count=len(near),passed=False)
                    if len(near)==1 and near[0]['status']=='bounded':
                        q=near[0];d=q['diagnostics'];b0=refs[(t['id'],n)]['width'];k=m['factor'];width=2*np.arcsinh(np.sinh(np.log(10)*b0/2)/k)/np.log(10)
                        # Mixed controls only translate; simple circuit formula used for Q changes.
                        if t['kind']=='mixed':width=b0
                        control.update(observed_center=q['center'],G=d['G'],length_ratio=d['length_ratio'],width_ratio_error=abs(q['width']/b0-width/b0),expected_width_ratio=width/b0,observed_width_ratio=d['width_ratio'])
                        control['passed']=bool(ok and d['G']<=.003 and abs(d['length_ratio']-1)<=.003 and control['width_ratio_error']<=1e-6)
                xs=np.unique(np.r_[np.linspace(*levels[-1]['search'],13),[q['center'] for q in levels[-1]['candidates']],*[q['bounds'] for q in levels[-1]['candidates'] if q['status']=='bounded']])
                nodal=checked_nodal(Physical(item['model']),10**xs,Physical(item['model']).p)
            else:identity=None;nodal=None
            result=dict(id=item['id'],target=t['id'],cohort=item['cohort'],family=item['family'],numerical_passed=bool(ok),levels=levels,identity=identity,control=control,nodal=nodal)
        except Exception as e:
            import traceback
            result=dict(id=item['id'],target=t['id'],cohort=item['cohort'],family=item['family'],numerical_passed=False,levels=levels,error=repr(e),traceback=traceback.format_exc())
        dump(OUT/'records'/(item['id']+'.json'),result);results.append(result)
        if len(results)%20==0 or not result['numerical_passed']:print(len(results), '/',len(plan['tasks']), 'numeric',sum(v['numerical_passed'] for v in results),flush=True)
    dump(OUT/'TARGET_SECTIONS.json',{tid:refs[(tid,256)] for tid in targets})
    dump(OUT/'RUN_STATUS.json',dict(records=len(results),numerical_passed=sum(r['numerical_passed'] for r in results),seconds=time.time()-start,source_unchanged=all(sha(ROOT/k)==v for k,v in manifest.items())))
    print('Done',flush=True)
if __name__=='__main__':main()

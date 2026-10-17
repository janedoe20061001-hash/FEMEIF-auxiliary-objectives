"""Execute all fixed records; expected truth excluded from matching function."""
from correspondence import *
def main():
    p=json.loads((OUT/'PLAN.json').read_text(encoding='utf-8'));targets={t['id']:t for t in p['targets']};manifest=json.loads((OUT/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'));assert all(sha(ROOT/f)==h for f,h in manifest['files'].items());start=time.time();good=0
    for i,item in enumerate(p['tasks'],1):
        levels=[]
        try:
            for n in [128,256,512,1024]:
                levels.append(level(item['model'],targets[item['target']],n))
                if len(levels)>1:
                    checks=stable(levels[-2],levels[-1])
                    if all(checks.values()):break
            result=dict(id=item['id'],target=item['target'],cohort=item['cohort'],levels=levels,stability_checks=checks,numerical_passed=all(checks.values()))
        except Exception as exc:
            import traceback
            result=dict(id=item['id'],target=item['target'],cohort=item['cohort'],numerical_passed=False,error=repr(exc),traceback=traceback.format_exc(),levels=levels)
        dump(OUT/'records'/f"{item['id']}.json",result);good+=int(result['numerical_passed'])
        if i%32==0 or not result['numerical_passed']:print(i,'/',len(p['tasks']),'numerical',good,flush=True)
    dump(OUT/'RUN_STATUS.json',dict(records=len(p['tasks']),numerical_passed=good,seconds=time.time()-start,source_unchanged=all(sha(ROOT/f)==h for f,h in manifest['files'].items())));print('DONE',good,'/',len(p['tasks']),flush=True)
if __name__=='__main__':main()

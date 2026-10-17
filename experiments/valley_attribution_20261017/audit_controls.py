"""Independent nodal response checks for newly generated physical controls."""
from correspondence import *
def main():
    p=json.loads((OUT/'PLAN.json').read_text(encoding='utf-8'));targets={t['id']:t for t in p['targets']};rows=[]
    for item in p['tasks']:
        if item['cohort']!='new_control':continue
        r=json.loads((OUT/'records'/(item['id']+'.json')).read_text(encoding='utf-8'));q=r['levels'][-1];t=targets[item['target']];c=Physical(item['model']);x=np.unique(np.r_[np.linspace(*q['search'],13),t['core'],q['signature_center_x']]);info=checked_nodal(c,10**x,c.p)
        rows.append(dict(id=item['id'],points=len(x),nodal=info))
    dump(OUT/'NODAL_CONTROL_AUDIT.json',dict(records=len(rows),passed=True,rows=rows));print('Fresh control nodal audits passed:',len(rows),flush=True)
if __name__=='__main__':main()

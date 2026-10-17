from core import *
import re,csv,time

def main():
    manifest=json.loads((OUT/'batch/manifest.json').read_text());assert manifest['hashes']==hashes()
    integrity=json.loads((OUT/'INTEGRITY.json').read_text());assert integrity['checked_jobs']==80 and integrity['evaluations']==4800000
    audit=json.loads((OUT/'analysis/AUDIT.json').read_text())
    with (OUT/'analysis/REPRESENTATIVES.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    assert sum(r['rule']=='primary' for r in rows)==80
    equality=[]
    for seed in CFG['seeds']:
        paths=[OUT/f'batch/control0p25/{m}/seed_{seed}/generation_0300.npz' for m in ('MAG_PHA_MAX','MAG_PHA_MEAN')]
        with np.load(paths[0]) as a,np.load(paths[1]) as b:
            np.testing.assert_array_equal(a['log_parameters'],b['log_parameters']);np.testing.assert_array_equal(a['objectives'],b['objectives'])
        equality.append(seed)
    refs=[]
    notes=[OUT/n for n in ('REPORT.md','FINDINGS.md','ISSUES.md','TARGETS.md')]
    notes+=[ROOT/'docs/三维曲线指标_新一轮候选与实验约定_20261011.md']
    delivery=OUT/'DELIVERY_VERIFICATION.json'
    for note in notes:
        for url in re.findall(r'\]\(([^)]+)\)',note.read_text(encoding='utf-8')):
            if re.match(r'^[A-Za-z]:/',url):
                refs.append(url)
                if Path(url)!=delivery:assert Path(url).exists(),(note,url)
    figures=list((OUT/'figures').glob('*.png'));assert len(figures)==6
    dump(delivery,dict(time_unix=time.time(),formal_tasks=80,evaluations=4800000,representative_records=len(rows),control_max_mean_exactly_equal_seeds=equality,source_inputs_verified=True,links_checked=len(refs),figures=6,audit_failures=audit['failures'],numerical_audit_passed=audit['passed'],helper_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},artifact_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*notes,*figures]}))
    # A report may link to this check's own output. Verify it after creation.
    for url in refs:assert Path(url).exists(),url
    print('Delivery checks passed; numerical audit failures remain separately recorded',flush=True)

if __name__=='__main__':main()

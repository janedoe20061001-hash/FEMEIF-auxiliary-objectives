from core import *
import re,csv,time

def main():
    assert json.loads((OUT/'batch/manifest.json').read_text())['hashes']==hashes()
    integrity=json.loads((OUT/'INTEGRITY.json').read_text());assert integrity['checked_jobs']==140 and integrity['evaluations']==8400000
    audit=json.loads((OUT/'analysis/AUDIT.json').read_text())
    with (OUT/'analysis/REPRESENTATIVES.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    assert sum(r['rule']=='primary' for r in rows)==140
    stages=0
    for ident in CFG['targets']:
        data=target(ident)
        for method in CFG['methods']:
            for seed in CFG['seeds']:
                directory=OUT/f'batch/{ident}/{method}/seed_{seed}'
                history=json.loads((directory/'history.json').read_text())
                assert len(history)==300
                assert all(r['generation']==i and r['evaluations']==200*i for i,r in enumerate(history,1))
                for generation in range(20,301,20):
                    with np.load(directory/f'generation_{generation:04}.npz') as a:
                        x,f=a['log_parameters'],a['objectives']
                        assert x.shape==(200,len(data['lower'])) and f.shape==(200,2 if method=='MAG_PHA' else 3)
                        assert np.isfinite(x).all() and np.isfinite(f).all()
                        assert np.all(x>=data['lower']) and np.all(x<=data['upper'])
                        assert int(a['primary'])==int(np.argmin(f[:,0]+25.12*f[:,1]))
                    stages+=1
    assert stages==2100
    equality=[]
    for seed in CFG['seeds']:
        paths=[OUT/f'batch/control0p25/{m}/seed_{seed}/generation_0300.npz' for m in ('MAG_PHA_INTEGRAL','MAG_PHA_BRIDGE')]
        with np.load(paths[0]) as a,np.load(paths[1]) as b:
            np.testing.assert_array_equal(a['log_parameters'],b['log_parameters']);np.testing.assert_array_equal(a['objectives'],b['objectives'])
        equality.append(seed)
    notes=[OUT/n for n in ('REPORT.md','FINDINGS.md','ISSUES.md','TARGETS.md','PROTOCOL.md')]
    refs=[];delivery=OUT/'DELIVERY_VERIFICATION.json'
    for note in notes:
        for url in re.findall(r'\]\(([^)]+)\)',note.read_text(encoding='utf-8')):
            if re.match(r'^[A-Za-z]:/',url):
                refs.append(url)
                if Path(url)!=delivery:assert Path(url).exists(),(note,url)
    figures=list((OUT/'figures').glob('*.png'));assert len(figures)==7
    assert json.loads((OUT/'RETRIEVAL.json').read_text())['crc_passed']
    dump(delivery,dict(time_unix=time.time(),formal_tasks=140,evaluations=8400000,all_saved_population_stages_checked=stages,all_300_generation_evaluation_histories_checked=True,representative_records=len(rows),control_C_bridge_exactly_equal_seeds=equality,source_inputs_verified=True,links_checked=len(refs),figures=7,numerical_audit_passed=audit['passed'],audit_failures=audit['failures'],helper_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},artifact_hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [*notes,*figures]}))
    for url in refs:assert Path(url).exists(),url
    print('Delivery checks passed; numerical exceptions remain separately flagged',flush=True)

if __name__=='__main__':main()

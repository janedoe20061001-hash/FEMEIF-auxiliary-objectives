from run import *
if __name__=='__main__':
    status=json.loads((OUT/'STATUS.json').read_text());assert status['status']=='failed' and status['completed']==798
    failed=sorted(status['failed'],key=lambda x:x['seed']);assert [r['seed'] for r in failed]==[2334,2365]
    dump(OUT/'INITIAL_BATCH_STATUS.json',status)
    for r in failed:
        d=OUT/f"runs/full/case2/dual_nsga/seed_{r['seed']}";dest=OUT/f"failed_attempts/full/case2/dual_nsga/seed_{r['seed']}"
        assert d.resolve().is_relative_to(OUT.resolve()) and dest.resolve().is_relative_to(OUT.resolve()) and not dest.exists()
        dest.parent.mkdir(parents=True,exist_ok=True);d.rename(dest)
    results=[one(('full','case2','dual_nsga',r['seed'])) for r in failed]
    dump(OUT/'RECOVERY.json',results);assert all(r['status']=='complete' for r in results)
    dump(OUT/'STATUS.json',dict(status='complete',total=800,completed=800,failed=[],
                              initial_replay_assertion_failures=2,failed_attempt_extra_evaluations=90000,
                              recovery_note='Same two seeds rerun to 1000 with first attempt 300 population equality required. Historical mismatch retained, no optimizer change.'))

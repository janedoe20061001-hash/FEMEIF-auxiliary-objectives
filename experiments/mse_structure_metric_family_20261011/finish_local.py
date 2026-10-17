from core import *
import time

def main():
    while not (OUT/'RETRIEVAL.json').exists():
        if (OUT/'WATCHER_STATUS.json').exists():
            state=json.loads((OUT/'WATCHER_STATUS.json').read_text())
            if state['status']=='server_failed':raise RuntimeError('Server failure retained; inspect before continuing')
        time.sleep(5)
    for module in ('targets_note','plot_results','report','common_features'):
        load('family_'+module,HERE/(module+'.py')).main()
    dump(OUT/'LOCAL_POSTPROCESSING.json',dict(status='complete',helper_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}))
    print('Local postprocessing complete; interpretation and image review still required',flush=True)

if __name__=='__main__':main()

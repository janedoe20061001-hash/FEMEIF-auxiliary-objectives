"""Wait only for retrieval, then build local illustrations and numerical report."""
from core import *
import time

def main():
    while not (OUT/'RETRIEVAL.json').exists():
        if (OUT/'WATCHER_STATUS.json').exists():
            state=json.loads((OUT/'WATCHER_STATUS.json').read_text())
            if state['status']=='server_failed':raise RuntimeError('Server failure preserved; repair required')
        time.sleep(5)
    load('revision_target_notes',HERE/'targets_note.py').main()
    load('revision_plots',HERE/'plot_results.py').main()
    load('revision_report',HERE/'report.py').main()
    dump(OUT/'LOCAL_POSTPROCESSING.json',dict(status='complete',helper_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}))
    print('Local figures and evidence report ready for human interpretation',flush=True)

if __name__=='__main__':main()

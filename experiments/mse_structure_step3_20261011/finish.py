"""Wait for this one local batch, then independently analyze and plot it once."""
from core import *
import time
analysis_module=load("step3_analysis_tool",HERE/"analyze.py")
plots_module=load("step3_plots_tool",HERE/"plots.py")

def main():
    file=OUT/"batch/status.json"
    while True:
        if file.exists():
            status=json.loads(file.read_text(encoding="utf-8"))["status"]
            if status=="failed":
                raise RuntimeError("Optimization failed; raw failure records preserved")
            if status=="complete":
                break
        time.sleep(5)
    # Reload postprocessing only after optimization is complete so recorded
    # helper hashes describe exactly the executed sources, not startup copies.
    analysis=load("step3_analysis_tool",HERE/"analyze.py")
    analysis.main()
    plots=load("step3_plots_tool",HERE/"plots.py")
    plots.main()
    dump(OUT/"POSTPROCESSING.json",dict(status="complete",helper_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob("*.py")}))

if __name__=="__main__":
    main()

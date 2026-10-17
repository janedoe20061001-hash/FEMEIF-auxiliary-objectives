# Case2 independent champion validation

This experiment compares ten finalists against a paired baseline on entirely new
seeds 200--399.  Every seed uses the same initial population across all eleven
groups.  Population size, generations, operators, measured target, and final
WSMSE reporting rule are unchanged from the prior Case2 experiments.

The final group is a genuine cross-family NSGA-II run combining Chinese-PDF
I-03 and II-02 with English-paper P-03.  `objectives.py` delegates formulas to
the previously audited implementations and only merges their objective columns.

Formal plan: 11 groups × 200 seeds = 2200 runs; 99,000,000 evaluations.

Run locally:

```bash
python experiment.py plan
python -m unittest -v test_protocol.py
python launch_batch.py --run --workers 28 --config config.json --output runs
python report_results.py runs
```


# Case2 final-method robustness validation

This experiment compares the dual-primary baseline with the final I-03+II-02
method under six observation perturbations.  Optimization and final selection
use the perturbed or subsampled observation, while reported primary errors are
always recomputed against the original full 201-point Case2 measurement.

Three Gaussian-noise conditions operate in dB magnitude and degree phase.
Two deterministic stride conditions retain about one half or one quarter of the
frequency points, including the final endpoint.  The gap condition removes one
seed-specific contiguous block containing 20% of points.  The same observation
and initial population are used by baseline and I-03+II-02 for every paired
condition/seed comparison.

Formal plan: 6 conditions × 2 groups × 100 seeds = 1200 runs and 54,000,000
individual evaluations.

